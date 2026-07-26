from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import TypeAlias

import torch
from torch import Tensor, nn


KeyValue: TypeAlias = tuple[Tensor, Tensor]
KeyValueCache: TypeAlias = tuple[KeyValue, ...]


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_len: int = 1024
    embedding_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_len <= 0:
            raise ValueError("context_len must be positive")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive")
        if self.embedding_dim <= 0 or self.embedding_dim % 2:
            raise ValueError("embedding_dim must be a positive even number")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def sinusoidal_positions(context_len: int, embedding_dim: int) -> Tensor:
    positions = torch.arange(context_len, dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(0, embedding_dim, 2, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * dimensions / embedding_dim)
    encoding = torch.zeros(context_len, embedding_dim)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding.unsqueeze(0)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        self.Q = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.K = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.V = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.context_len, config.context_len),
                diagonal=1,
            ).bool().unsqueeze(0),
        )

        self.attn_norm = nn.LayerNorm(config.embedding_dim)
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.mlp_dropout = nn.Dropout(config.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim),
        )

    def forward(
        self,
        inputs: Tensor,
        past_key_value: KeyValue | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KeyValue]:
        batch_size, query_len, embedding_dim = inputs.shape
        normed = self.attn_norm(inputs)
        query = self.Q(normed).reshape(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.K(normed).reshape(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.V(normed).reshape(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        past_len = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.shape != past_value.shape:
                raise ValueError("cached keys and values must have identical shapes")
            if past_key.shape[:2] != (batch_size, self.num_heads):
                raise ValueError("cache batch size or head count does not match input")
            past_len = past_key.size(-2)
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        key_len = key.size(-2)
        query_positions = torch.arange(
            past_len, past_len + query_len, device=inputs.device
        ).unsqueeze(1)
        key_positions = torch.arange(key_len, device=inputs.device).unsqueeze(0)
        causal_mask = key_positions > query_positions

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probabilities = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = probabilities @ value
        attended = attended.transpose(1, 2).reshape(
            batch_size, query_len, embedding_dim
        )
        output = inputs + self.residual_dropout(attended)
        output = output + self.mlp_dropout(self.mlp(self.mlp_norm(output)))

        if use_cache:
            return output, (key, value)
        return output


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.input_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "pos_embedding",
            sinusoidal_positions(config.context_len, config.embedding_dim),
        )
        self.blocks = nn.ModuleList(
            Block(config) for _ in range(config.num_layers)
        )
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KeyValueCache]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("cache must contain one key/value pair per layer")

        if past_key_values is None:
            past_len = 0
        else:
            cache_lengths = {key.size(-2) for key, _ in past_key_values}
            if len(cache_lengths) != 1:
                raise ValueError("all layer caches must have the same sequence length")
            past_len = cache_lengths.pop()
        end_position = past_len + input_ids.size(1)
        if end_position > self.config.context_len:
            raise ValueError(
                f"sequence plus cache length {end_position} exceeds context length "
                f"{self.config.context_len}"
            )

        positions = self.pos_embedding[:, past_len:end_position, :]
        hidden = self.embedding_dropout(self.input_embedding(input_ids) + positions)
        next_cache: list[KeyValue] = []

        for index, block in enumerate(self.blocks):
            layer_cache = None if past_key_values is None else past_key_values[index]
            block_output = block(hidden, layer_cache, use_cache)
            if use_cache:
                hidden, present = block_output
                next_cache.append(present)
            else:
                hidden = block_output

        logits = self.lm_head(hidden)
        if use_cache:
            return logits, tuple(next_cache)
        return logits


@torch.inference_mode()
def generate(
    model: GPT,
    prompt_tokens: list[int],
    max_new_tokens: int,
    *,
    use_cache: bool,
    temperature: float = 1.0,
    do_sample: bool = True,
    generator: torch.Generator | None = None,
) -> list[int]:
    if not prompt_tokens:
        raise ValueError("prompt_tokens cannot be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(prompt_tokens) + max_new_tokens > model.config.context_len:
        raise ValueError("requested generation exceeds model context length")

    was_training = model.training
    model.eval()
    tokens = list(prompt_tokens)
    try:
        if max_new_tokens == 0:
            return tokens

        if use_cache:
            input_ids = torch.tensor(
                [tokens], dtype=torch.long, device=next(model.parameters()).device
            )
            logits, cache = model(input_ids, use_cache=True)
            for index in range(max_new_tokens):
                probabilities = torch.softmax(
                    logits[:, -1, :] / temperature, dim=-1
                )
                next_token = (
                    torch.multinomial(probabilities, 1, generator=generator)
                    if do_sample
                    else probabilities.argmax(dim=-1, keepdim=True)
                )
                tokens.append(next_token.item())
                if index + 1 < max_new_tokens:
                    logits, cache = model(
                        next_token, past_key_values=cache, use_cache=True
                    )
        else:
            for _ in range(max_new_tokens):
                input_ids = torch.tensor(
                    [tokens],
                    dtype=torch.long,
                    device=next(model.parameters()).device,
                )
                logits = model(input_ids)
                probabilities = torch.softmax(
                    logits[:, -1, :] / temperature, dim=-1
                )
                next_token = (
                    torch.multinomial(probabilities, 1, generator=generator)
                    if do_sample
                    else probabilities.argmax(dim=-1, keepdim=True)
                )
                tokens.append(next_token.item())
        return tokens
    finally:
        model.train(was_training)

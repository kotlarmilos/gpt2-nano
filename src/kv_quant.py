from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn

from src.model import (
    Block,
    GPT,
    GPTConfig,
    KeyValue,
    KeyValueCache,
    sinusoidal_positions,
)

# Supported bit widths for K/V cache quantization.
VALID_BITS: frozenset[int] = frozenset({16, 8, 4})

_INT8_MAX: int = 127
_INT4_MAX: int = 7


def _validate_bits(bits: int) -> None:
    if bits not in VALID_BITS:
        raise ValueError(
            f"bits must be one of {sorted(VALID_BITS)}, got {bits}"
        )


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------


def quantize_sym_int8(x: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric INT8 quantization with per-token per-head scale over head_dim.

    This is a controlled per-token per-head quantizer, not KIVI-equivalent.

    Input:  x      - float, shape (batch, seq_len, head_dim)
    Output: q      - int8,  shape (batch, seq_len, head_dim)
            scales - float16, shape (batch, seq_len)
                     one scale per (batch index, sequence position)

    scale[b, t] = max(|x[b, t, :]|), stored as float16.
    Step size = scale / 127.  Max rounding error = 0.5 * scale / 127.
    """
    scale = x.abs().amax(dim=-1).clamp_min(1e-8).to(torch.float16)
    q = (x.float() * _INT8_MAX / scale.float().unsqueeze(-1)).round().clamp(
        -_INT8_MAX, _INT8_MAX
    )
    return q.to(torch.int8), scale


def dequantize_sym_int8(
    q: Tensor, scales: Tensor, target_dtype: torch.dtype
) -> Tensor:
    """Dequantize INT8 tensor produced by quantize_sym_int8.

    q:      int8,   shape (batch, seq_len, head_dim)
    scales: float16, shape (batch, seq_len)  [stores max_abs, not step size]
    """
    return (q.float() * scales.float().unsqueeze(-1) / _INT8_MAX).to(target_dtype)


def quantize_sym_int4(x: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric INT4 quantization packing two 4-bit signed values into one uint8.

    This is a controlled per-token per-head quantizer, not KIVI-equivalent.
    Odd head_dim is supported: the padding nibble stores zero.

    Input:  x      - float, shape (batch, seq_len, head_dim)
    Output: packed - uint8, shape (batch, seq_len, ceil(head_dim / 2))
                     low nibble = first element, high nibble = second element
            scales - float16, shape (batch, seq_len)

    scale[b, t] = max(|x[b, t, :]|), stored as float16.
    Step size = scale / 7.  Max rounding error = 0.5 * scale / 7.
    """
    scale = x.abs().amax(dim=-1).clamp_min(1e-8).to(torch.float16)
    q = (x.float() * _INT4_MAX / scale.float().unsqueeze(-1)).round().clamp(
        -_INT4_MAX, _INT4_MAX
    )
    q_int8 = q.to(torch.int8)

    batch, seq_len, head_dim = x.shape
    packed_dim = (head_dim + 1) // 2

    if head_dim % 2 == 1:
        pad = q_int8.new_zeros(batch, seq_len, 1)
        q_int8 = torch.cat([q_int8, pad], dim=-1)

    # (batch, seq_len, packed_dim, 2) -> lo and hi nibbles
    q_pairs = q_int8.reshape(batch, seq_len, packed_dim, 2)
    lo = q_pairs[..., 0]  # (batch, seq_len, packed_dim), int8
    hi = q_pairs[..., 1]

    # Mask to low 4 bits via int16 to avoid sign issues, then cast to uint8
    lo_nib = lo.to(torch.int16).bitwise_and(0x0F).to(torch.uint8)
    hi_nib = hi.to(torch.int16).bitwise_and(0x0F).to(torch.uint8)
    packed = lo_nib | (hi_nib << 4)
    return packed, scale


def dequantize_sym_int4(
    packed: Tensor,
    scales: Tensor,
    head_dim: int,
    target_dtype: torch.dtype,
) -> Tensor:
    """Unpack and dequantize INT4 tensor produced by quantize_sym_int4.

    packed:   uint8, shape (batch, seq_len, packed_dim)
    scales:   float16, shape (batch, seq_len)
    head_dim: original head dimension (used to strip padding nibble)
    """
    batch, seq_len, packed_dim = packed.shape

    lo_raw = packed.bitwise_and(0x0F).to(torch.int16)
    hi_raw = packed.bitwise_right_shift(4).bitwise_and(0x0F).to(torch.int16)

    # Sign-extend 4-bit two's complement to int: values >= 8 map to x - 16
    lo_signed = torch.where(lo_raw >= 8, lo_raw - 16, lo_raw)
    hi_signed = torch.where(hi_raw >= 8, hi_raw - 16, hi_raw)

    # Reconstruct interleaved sequence: lo, hi, lo, hi, ...
    q_pairs = torch.stack([lo_signed, hi_signed], dim=-1)
    q = q_pairs.reshape(batch, seq_len, packed_dim * 2)
    q = q[..., :head_dim]  # strip padding if head_dim was odd

    return (q.float() * scales.float().unsqueeze(-1) / _INT4_MAX).to(target_dtype)


# ---------------------------------------------------------------------------
# Quantized cache storage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantizedHeadTensor:
    """Quantized storage for one attention head's K or V activations.

    payload shape:
      16-bit -> (batch, seq_len, head_dim),          dtype = model float dtype
       8-bit -> (batch, seq_len, head_dim),          dtype = int8
       4-bit -> (batch, seq_len, ceil(head_dim/2)),  dtype = uint8

    scales shape (when present): (batch, seq_len), dtype = float16
    """

    bits: int
    head_dim: int
    payload: Tensor
    scales: Tensor | None  # None for 16-bit

    def dequantize(self, target_dtype: torch.dtype | None = None) -> Tensor:
        """Return float tensor of shape (batch, seq_len, head_dim)."""
        if self.bits == 16:
            t = self.payload
            return t if target_dtype is None else t.to(target_dtype)
        dtype = target_dtype if target_dtype is not None else torch.float32
        if self.bits == 8:
            assert self.scales is not None
            return dequantize_sym_int8(self.payload, self.scales, dtype)
        if self.bits == 4:
            assert self.scales is not None
            return dequantize_sym_int4(
                self.payload, self.scales, self.head_dim, dtype
            )
        raise ValueError(f"unsupported bits {self.bits}")

    def scale_nbytes(self) -> int:
        """Exact bytes for scale storage; 0 for 16-bit heads."""
        if self.scales is None:
            return 0
        return self.scales.numel() * self.scales.element_size()

    def payload_nbytes(self) -> int:
        """Exact bytes for quantized payload storage."""
        return self.payload.numel() * self.payload.element_size()

    def total_nbytes(self) -> int:
        return self.scale_nbytes() + self.payload_nbytes()

    def append(self, other: QuantizedHeadTensor) -> QuantizedHeadTensor:
        if self.bits != other.bits or self.head_dim != other.head_dim:
            raise ValueError("appended cache tensors must use the same format")
        if self.payload.shape[0] != other.payload.shape[0]:
            raise ValueError("appended cache tensors must use the same batch size")
        payload = torch.cat((self.payload, other.payload), dim=1)
        if self.scales is None and other.scales is None:
            scales = None
        elif self.scales is not None and other.scales is not None:
            scales = torch.cat((self.scales, other.scales), dim=1)
        else:
            raise ValueError("appended cache tensors must both have scales or neither")
        return QuantizedHeadTensor(
            bits=self.bits,
            head_dim=self.head_dim,
            payload=payload,
            scales=scales,
        )

    @staticmethod
    def from_float(x: Tensor, bits: int) -> QuantizedHeadTensor:
        """Quantize a float tensor of shape (batch, seq_len, head_dim).

        Note: 16-bit stores a reference to x, not a copy.  The caller is
        responsible for not mutating x afterwards.
        """
        _validate_bits(bits)
        head_dim = x.size(-1)
        if bits == 16:
            return QuantizedHeadTensor(
                bits=16, head_dim=head_dim, payload=x, scales=None
            )
        if bits == 8:
            q, sc = quantize_sym_int8(x)
            return QuantizedHeadTensor(bits=8, head_dim=head_dim, payload=q, scales=sc)
        # bits == 4
        packed, sc = quantize_sym_int4(x)
        return QuantizedHeadTensor(bits=4, head_dim=head_dim, payload=packed, scales=sc)


@dataclass(frozen=True)
class QuantizedLayerKV:
    """Per-head quantized K and V tensors for one transformer layer."""

    keys: tuple[QuantizedHeadTensor, ...]
    values: tuple[QuantizedHeadTensor, ...]

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values):
            raise ValueError(
                f"keys and values must have equal head counts, "
                f"got {len(self.keys)} vs {len(self.values)}"
            )

    @property
    def num_heads(self) -> int:
        return len(self.keys)

    def dequantize(self, target_dtype: torch.dtype | None = None) -> KeyValue:
        """Return (key, value) tensors of shape (batch, num_heads, seq_len, head_dim)."""
        k_heads = [qt.dequantize(target_dtype) for qt in self.keys]
        v_heads = [qt.dequantize(target_dtype) for qt in self.values]
        key = torch.stack(k_heads, dim=1)
        value = torch.stack(v_heads, dim=1)
        return key, value

    def scale_nbytes(self) -> int:
        return sum(qt.scale_nbytes() for qt in self.keys) + sum(
            qt.scale_nbytes() for qt in self.values
        )

    def payload_nbytes(self) -> int:
        return sum(qt.payload_nbytes() for qt in self.keys) + sum(
            qt.payload_nbytes() for qt in self.values
        )

    def total_nbytes(self) -> int:
        return self.scale_nbytes() + self.payload_nbytes()

    def append(self, other: QuantizedLayerKV) -> QuantizedLayerKV:
        if self.num_heads != other.num_heads:
            raise ValueError("appended layer caches must have equal head counts")
        return QuantizedLayerKV(
            keys=tuple(
                current.append(new)
                for current, new in zip(self.keys, other.keys, strict=True)
            ),
            values=tuple(
                current.append(new)
                for current, new in zip(self.values, other.values, strict=True)
            ),
        )

    @staticmethod
    def from_key_value(
        key: Tensor,
        value: Tensor,
        head_bits: tuple[int, ...],
    ) -> QuantizedLayerKV:
        """Quantize key and value tensors with per-head precision.

        key, value: (batch, num_heads, seq_len, head_dim)
        head_bits:  precision per head, length must equal num_heads
        """
        if key.size(1) != len(head_bits):
            raise ValueError(
                f"key has {key.size(1)} heads but head_bits has {len(head_bits)} entries"
            )
        if key.shape != value.shape:
            raise ValueError("key and value must have identical shapes")
        keys = tuple(
            QuantizedHeadTensor.from_float(key[:, h, :, :].contiguous(), head_bits[h])
            for h in range(len(head_bits))
        )
        values = tuple(
            QuantizedHeadTensor.from_float(value[:, h, :, :].contiguous(), head_bits[h])
            for h in range(len(head_bits))
        )
        return QuantizedLayerKV(keys=keys, values=values)


QuantizedKVCache = tuple[QuantizedLayerKV, ...]


# ---------------------------------------------------------------------------
# Cache policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachePolicy:
    """Per-layer, per-head precision policy.  bits_per_layer[l][h] in {16, 8, 4}."""

    bits_per_layer: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for l_idx, layer_bits in enumerate(self.bits_per_layer):
            for h_idx, bits in enumerate(layer_bits):
                _validate_bits(bits)

    @property
    def num_layers(self) -> int:
        return len(self.bits_per_layer)

    @property
    def num_heads(self) -> int:
        return len(self.bits_per_layer[0]) if self.bits_per_layer else 0

    @staticmethod
    def all_sixteen(num_layers: int, num_heads: int) -> CachePolicy:
        """Return a policy with all heads at 16-bit precision."""
        return CachePolicy(
            bits_per_layer=tuple(
                tuple(16 for _ in range(num_heads)) for _ in range(num_layers)
            )
        )

    def with_head(self, layer: int, head: int, bits: int) -> CachePolicy:
        """Return a new policy identical to this one but with one head changed."""
        _validate_bits(bits)
        layers = [list(lb) for lb in self.bits_per_layer]
        layers[layer][head] = bits
        return CachePolicy(bits_per_layer=tuple(tuple(lb) for lb in layers))

    def avg_payload_bits(self) -> float:
        """Average bits per K or V payload element across all layers and heads."""
        n = self.num_layers * self.num_heads
        if n == 0:
            return 0.0
        return sum(b for lb in self.bits_per_layer for b in lb) / n

    def exact_cache_bytes(
        self,
        batch: int,
        seq_len: int,
        head_dim: int,
        model_dtype_bytes: int = 4,
    ) -> dict[str, int]:
        """Exact byte accounting for combined K and V storage.

        scale_bytes and payload_bytes are reported separately to distinguish
        persistent packed storage from transient dequantization peak memory.
        """
        payload = 0
        scales = 0
        packed_dim = (head_dim + 1) // 2
        for lb in self.bits_per_layer:
            for bits in lb:
                for _ in range(2):  # K and V
                    if bits == 16:
                        payload += batch * seq_len * head_dim * model_dtype_bytes
                    elif bits == 8:
                        payload += batch * seq_len * head_dim * 1  # int8
                        scales += batch * seq_len * 2              # float16 scale
                    elif bits == 4:
                        payload += batch * seq_len * packed_dim * 1  # uint8 packed
                        scales += batch * seq_len * 2                # float16 scale
        return {
            "payload_bytes": payload,
            "scale_bytes": scales,
            "total_bytes": payload + scales,
        }

    def scale_overhead_bytes_per_token(self, batch: int = 1) -> float:
        """Scale bytes per token (summed over all layers and heads, K+V combined)."""
        n_quantized = sum(
            1 for lb in self.bits_per_layer for b in lb if b < 16
        )
        return float(n_quantized * 2 * batch * 2)  # 2 for K+V, 2 for float16

    def to_list(self) -> list[list[int]]:
        return [list(lb) for lb in self.bits_per_layer]


# ---------------------------------------------------------------------------
# QuantizedCacheGPT adapter
# ---------------------------------------------------------------------------


class QuantizedCacheGPT(nn.Module):
    """GPT adapter that stores the KV cache in quantized form.

    The wrapped GPT model is never modified.  Dequantization happens before
    passing past_key_values to the underlying model; requantization happens
    after receiving the returned cache.  This ensures the wrapped GPT always
    sees full-precision K and V tensors.

    Persistent cache bytes differ from transient dequantization peak memory.
    The peak memory during a forward step includes full-precision K, V
    temporarily in flight even for low-bit heads.
    """

    def __init__(self, gpt: GPT, policy: CachePolicy) -> None:
        super().__init__()
        if policy.num_layers != len(gpt.blocks):
            raise ValueError(
                f"policy has {policy.num_layers} layers but model has {len(gpt.blocks)}"
            )
        if policy.num_heads != gpt.config.num_heads:
            raise ValueError(
                f"policy has {policy.num_heads} heads but model has {gpt.config.num_heads}"
            )
        self.gpt = gpt
        self.policy = policy

    @property
    def config(self) -> GPTConfig:
        return self.gpt.config

    def forward(
        self,
        input_ids: Tensor,
        past_kv_cache: QuantizedKVCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, QuantizedKVCache]:
        past_key_values: KeyValueCache | None = None
        if past_kv_cache is not None:
            target_dtype = next(self.gpt.parameters()).dtype
            past_key_values = tuple(
                lkv.dequantize(target_dtype) for lkv in past_kv_cache
            )

        result = self.gpt(
            input_ids, past_key_values=past_key_values, use_cache=use_cache
        )

        if not use_cache:
            assert isinstance(result, Tensor)
            return result

        logits, new_kv = result
        appended_tokens = input_ids.shape[1]
        quantized_updates = tuple(
            QuantizedLayerKV.from_key_value(
                key[:, :, -appended_tokens:, :],
                value[:, :, -appended_tokens:, :],
                self.policy.bits_per_layer[index],
            )
            for index, (key, value) in enumerate(new_kv)
        )
        new_quant = (
            tuple(
                previous.append(update)
                for previous, update in zip(
                    past_kv_cache, quantized_updates, strict=True
                )
            )
            if past_kv_cache is not None
            else quantized_updates
        )
        return logits, new_quant


# ---------------------------------------------------------------------------
# NoisyBlock and NoisyKVGPT for cache-noise robustness fine-tuning
# ---------------------------------------------------------------------------


class NoisyBlock(Block):
    """Block subclass that injects quantization-matched noise into K and V.

    Noise is injected only when self.training is True.  The distribution is
    zero-mean uniform or Gaussian with variance matched to the expected
    rounding error of the target bit width.

    For symmetric INT8 (max_int=127): step = |x|_inf / 127.
    For symmetric INT4 (max_int=7):  step = |x|_inf / 7.
    Uniform rounding noise lies on [-step/2, step/2]; Gaussian uses
    equivalent variance std = step / (2*sqrt(3)).

    Inheriting from Block guarantees parameter key compatibility with GPT
    checkpoints so load_state_dict(gpt.state_dict()) works without remapping.
    """

    def __init__(
        self, config: GPTConfig, noise_bits: int, noise_type: str
    ) -> None:
        super().__init__(config)
        if noise_bits not in {4, 8}:
            raise ValueError(f"noise_bits must be 4 or 8, got {noise_bits}")
        if noise_type not in {"uniform", "gaussian"}:
            raise ValueError(
                f"noise_type must be 'uniform' or 'gaussian', got {noise_type}"
            )
        self.noise_bits = noise_bits
        self.noise_type = noise_type

    def _sample_noise(self, x: Tensor) -> Tensor:
        max_int = _INT8_MAX if self.noise_bits == 8 else _INT4_MAX
        with torch.no_grad():
            scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        half_step = scale / (2.0 * max_int)
        if self.noise_type == "uniform":
            return torch.empty_like(x).uniform_(-1.0, 1.0) * half_step
        # gaussian: same variance as uniform on [-half_step, half_step]
        std = half_step / (3.0 ** 0.5)
        return torch.randn_like(x) * std

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

        if self.training:
            key = key + self._sample_noise(key)
            value = value + self._sample_noise(value)

        past_len = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.shape != past_value.shape:
                raise ValueError("cached keys and values must have identical shapes")
            if past_key.shape[:2] != (batch_size, self.num_heads):
                raise ValueError(
                    "cache batch size or head count does not match input"
                )
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


class NoisyKVGPT(nn.Module):
    """Standalone GPT with NoisyBlock layers for cache-noise robustness training.

    The state dict is parameter-key-compatible with GPT checkpoints: all
    Block parameter names are preserved under blocks.N.*, and the top-level
    embedding and head names are identical.  Loading is done via:

        noisy_model.load_state_dict(gpt.state_dict())

    noise_bits and noise_type are plain Python attributes and are not part of
    the state dict.  The teacher model must be kept separately, frozen, and
    evaluated under torch.no_grad().
    """

    def __init__(
        self,
        config: GPTConfig,
        noise_bits: int,
        noise_type: str = "uniform",
    ) -> None:
        super().__init__()
        if noise_bits not in {4, 8}:
            raise ValueError(f"noise_bits must be 4 or 8, got {noise_bits}")
        self.config = config
        self.noise_bits = noise_bits
        self.noise_type = noise_type

        self.input_embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "pos_embedding",
            sinusoidal_positions(config.context_len, config.embedding_dim),
        )
        self.blocks = nn.ModuleList(
            NoisyBlock(config, noise_bits, noise_type)
            for _ in range(config.num_layers)
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
            raise ValueError("past_key_values must have one entry per transformer block")
        past_len = (
            past_key_values[0][0].size(-2)
            if past_key_values is not None
            else 0
        )
        seq_len = input_ids.size(1)
        if past_len + seq_len > self.config.context_len:
            raise ValueError(
                f"sequence length {past_len + seq_len} exceeds context length "
                f"{self.config.context_len}"
            )
        positions = self.pos_embedding[:, past_len : past_len + seq_len, :]
        hidden = self.embedding_dropout(
            self.input_embedding(input_ids) + positions
        )
        new_key_values: list[KeyValue] = []
        for index, block in enumerate(self.blocks):
            past = past_key_values[index] if past_key_values is not None else None
            block_output = block(hidden, past_key_value=past, use_cache=use_cache)
            if use_cache:
                hidden, key_value = block_output
                new_key_values.append(key_value)
            else:
                hidden = block_output
        logits = self.lm_head(hidden)
        if use_cache:
            return logits, tuple(new_key_values)
        return logits

from glob import glob
import math
import time

import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from data.bpe_tokenizer import decode, encode, load_shard, load_tokenizer, SHARDS_PATH

# ---- Hyperparameters ----
context_len = 1024
embedding_dim = 512
num_heads = 8
num_layers = 12
batch_size = 32
learning_rate = 3e-4
weight_decay = 0.1
num_steps = 10000
log_every = 10
generate_every = 500
checkpoint_every = 1000
checkpoint_dir = "checkpoints"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- Model ----

class Block(nn.Module):
    def __init__(self, embedding_dim, context_len, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.Q = nn.Linear(embedding_dim, embedding_dim)
        self.K = nn.Linear(embedding_dim, embedding_dim)
        self.V = nn.Linear(embedding_dim, embedding_dim)

        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool().unsqueeze(0)
        )

        self.attn_norm = nn.LayerNorm(embedding_dim)
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, inputs):
        B, C, E = inputs.shape

        # Pre-norm: normalize before attention
        normed = self.attn_norm(inputs)
        q = self.Q(normed).reshape(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.K(normed).reshape(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.V(normed).reshape(B, C, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:, :C, :C], float('-inf'))
        scores = torch.softmax(scores, dim=-1)

        out = scores @ v
        out = out.transpose(1, 2).reshape(B, C, E)
        out = inputs + out

        # Pre-norm: normalize before MLP
        out = out + self.mlp(self.mlp_norm(out))
        return out


class GPT(nn.Module):
    def __init__(self, vocab_size, context_len, embedding_dim, num_layers, num_heads):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Sinusoidal positional encoding (not learned)
        pos_embedding = torch.zeros(context_len, embedding_dim)
        for pos in range(context_len):
            for i in range(0, embedding_dim, 2):
                angle = pos / (10000 ** (i / embedding_dim))
                pos_embedding[pos][i] = math.sin(angle)
                pos_embedding[pos][i + 1] = math.cos(angle)
        self.register_buffer('pos_embedding', pos_embedding.unsqueeze(0))

        self.blocks = nn.ModuleList([
            Block(embedding_dim, context_len, num_heads) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        out = self.input_embedding(x) + self.pos_embedding[:, :x.size(1), :]
        for block in self.blocks:
            out = block(out)
        return self.lm_head(out)


def generate(gpt, merges, vocab, prompt_tokens, context_len, max_new_tokens=100):
    gpt.eval()
    tokens = list(prompt_tokens)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([tokens[-context_len:]], dtype=torch.long, device=next(gpt.parameters()).device)
            logits = gpt(x)
            probs = torch.softmax(logits[0, -1], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
    gpt.train()
    return decode(tokens, vocab)

def save_checkpoint(gpt, optimizer, step, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": gpt.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "config": {
            "vocab_size": vocab_size,
            "context_len": context_len,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
        },
    }, path)
    print(f"Checkpoint saved: {path}")

# ---- Data ----

merges, vocab = load_tokenizer()
vocab_size = len(vocab)
tokens = np.concatenate([load_shard(f) for f in sorted(glob("data/bpe-shards/*.bin"))])

val_split = int(len(tokens) * 0.9)
train_tokens = tokens[:val_split]
val_tokens = tokens[val_split:]

# ---- Training ----

gpt = GPT(vocab_size, context_len, embedding_dim, num_layers, num_heads).to(device)
optimizer = torch.optim.AdamW(gpt.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
num_params = sum(p.numel() for p in gpt.parameters())

print(f"=== Training GPT ({device}) ===")
print(f"Vocab: {vocab_size} | Dim: {embedding_dim} | Heads: {num_heads} | Layers: {num_layers}")
print(f"Context: {context_len} | Batch: {batch_size} | Params: {num_params:,}")
print(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")
print()

train_start = time.time()
for step in range(num_steps):
    start = (step * batch_size * context_len) % (len(train_tokens) - batch_size * context_len)
    batch = train_tokens[start:start + batch_size * context_len]
    x = torch.tensor(batch.reshape(batch_size, context_len), dtype=torch.long, device=device)

    logits = gpt(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), x[:, 1:].reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if step % log_every == 0:
        elapsed = time.time() - train_start
        perplexity = math.exp(loss.item())

        with torch.no_grad():
            val_batch = val_tokens[:batch_size * context_len]
            vx = torch.tensor(val_batch.reshape(batch_size, context_len), dtype=torch.long, device=device)
            vlogits = gpt(vx)
            vloss = F.cross_entropy(vlogits[:, :-1].reshape(-1, vocab_size), vx[:, 1:].reshape(-1))
            val_perplexity = math.exp(vloss.item())

        print(f"step {step:>5d}/{num_steps} | "
              f"loss: {loss.item():.4f} | val: {vloss.item():.4f} | "
              f"ppl: {perplexity:.1f} | val_ppl: {val_perplexity:.1f} | "
              f"{elapsed:.1f}s")

    if step > 0 and step % generate_every == 0:
        prompt = encode("The ", merges, vocab)
        text = generate(gpt, merges, vocab, prompt, context_len, max_new_tokens=50)
        print(f"\n--- Generated (step {step}) ---\n{text}\n---\n")

    if step > 0 and step % checkpoint_every == 0:
        save_checkpoint(gpt, optimizer, step, loss.item(),
                        f"{checkpoint_dir}/step_{step}.pt")

save_checkpoint(gpt, optimizer, num_steps, loss.item(),
                f"{checkpoint_dir}/final.pt")
print(f"\n=== Done in {time.time() - train_start:.1f}s ===")
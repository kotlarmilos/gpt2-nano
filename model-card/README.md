---
library_name: pytorch
license: mit
language:
  - en
tags:
  - causal-lm
  - from-scratch
  - apple-silicon
  - kv-cache
---

# gpt2-nano

This is a 44M parameter decoder-only transformer implemented from scratch in
PyTorch. The expanded release adds configurable dropout, incremental decoding
with a KV-cache, and tools for measuring and regularizing token-distribution
shift with KL divergence.

The published checkpoint predates these experiments. A trained-checkpoint
KV-cache benchmark is measured locally on Apple M1. Dropout and KL publication
runs remain pending, as recorded in the artifact manifest.

## Intended use

The model and code are educational. They are intended for studying transformer
training and inference on modest hardware, not for production text generation.

## Base configuration

| Setting | Value |
|---|---:|
| Parameters | approximately 44M |
| Layers | 12 |
| Hidden size | 512 |
| Attention heads | 8 |
| Context | 1,024 |
| Vocabulary | 9,157 custom BPE tokens |
| Training data | FineWeb-edu subset |

## Local extension result

A development-scale KV-cache numerical equivalence run uses
`checkpoints/final.pt`, three prompts, a 128-token greedy horizon, and Apple MPS.
It records no token flips for `float32` or `bfloat16`. The maximum absolute logit
deviation is `2.09808349609375e-05` for `float32` and `0.125` for `bfloat16`.
The maximum relative logit deviation is `1.32924469653517e-06` for `float32` and
`0.008438818156719208` for `bfloat16`. This is not a publication-scale result.
The artifact path is `artifacts/kv-equivalence-dev-local/results.json`, with
SHA-256 `bc47aba423ba538a698ca0c040b6a21fef7e9cada13f662ffe9cb08b308cd54a`
recorded in `artifacts/manifest.json`.

## Limitations

The original run used one seed and approximately 99M tokens. The model is too
small and under-trained for reliable factual generation. Cache benchmarks on a
random smoke model validate the implementation but are not evidence of
publication-scale speedups.

---
library_name: pytorch
language:
  - en
tags:
  - causal-lm
  - from-scratch
  - kv-cache
  - quantization
---

# gpt2-nano

This is a 44M-parameter decoder-only transformer implemented in PyTorch. The
published checkpoint predates the mixed-precision extension.

## Measured extension

The extension assigns BF16, INT8, or packed INT4 KV-cache storage to each of
the model's 96 attention heads. A greedy allocator uses isolated single-head
output-KL measurements as a calibration proxy.

On an RTX A5000, the selected policy uses 70 INT4 heads, 19 INT8 heads, and
7 BF16 heads. It stores 579,840 bytes for a batch-one, 64-token cache, compared
with 1,572,864 bytes for BF16. Held-out teacher-forced output KL is `0.001528`,
and next-token agreement is 97.85%.

The fixed cache-noise adaptation does not establish a robustness gain.

## Intended use

The model and code are intended for education and controlled cache experiments.
They are not intended for production text generation or factual question
answering.

## Limits

- Agreement is measured on shared teacher-forced prefixes, not generated text.
- The additive calibration score is not a bound on joint held-out KL.
- Persistent cache bytes exclude model weights and temporary dequantized tensors.
- The implementation dequantizes before attention and does not provide a speedup.
- The study uses one checkpoint, one held-out window, and one GPU.

The measured evidence is in
`artifacts/kv-cache-publication/results.json`.

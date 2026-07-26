---
title: "Dropout, KV caching, and KL divergence in my small transformer"
thumbnail: /blog/assets/nano-dropout-kv-kl/thumbnail.png
authors:
- user: kotlarmilos
---

# Dropout, KV caching, and KL divergence in my small transformer

## TL;DR

I went back to the 44M parameter transformer I built from scratch and added
three pieces I had deliberately left out of the first version. Dropout gives me a
controlled way to test whether the model is overfitting. A KV-cache stops
generation from recomputing the same keys and values for every token. KL
divergence gives me a direct measurement of how far an adapted model moves from
the original. They solve different problems, but implementing all three in the
same small model made the training and inference trade-offs much easier to see.

## The problem

The original project was useful because every layer was visible. Replacing it
with a library model would make these additions shorter, but it would hide the
parts I wanted to understand. Keeping the architecture and data fixed means the
comparison is about the technique rather than a new model.

The experiments are designed around that constraint. Dropout is compared at a
fixed token budget. Cached and uncached decoding use identical weights and
sampling seeds. KL divergence is measured on the same held-out token batches.

## Intuition

### Dropout regularizes only when there is overfitting to prevent

The expanded model applies configurable dropout to embeddings, attention
probabilities, residual outputs, and the MLP. A value of zero reproduces the
original computation. The useful number is not training loss by itself. It is the
gap between training and validation loss. If the zero-dropout model is still
improving on validation data, regularization can make it worse by slowing a model
that is already under-trained.

### The cache removes repeated work

Without a cache, generating token t runs the complete prefix through every layer
again. The keys and values for the first t-1 positions do not change, so
recomputing them is wasted work. The cached path stores one key and one value
tensor per layer and only projects the new token.

### KL divergence measures movement, not quality

A value near zero means the two models assign similar probabilities. A larger
value means the adaptation moved the distribution more strongly. That does not say
whether the movement was useful, so the experiment reports task loss and held-out
KL together.

## The math

### KV-cache memory

For batch \(B\), layers \(L\), heads \(H\), head dimension \(d_h\), context \(C\),
and element size \(s\), the cache occupies

$$
M_{\mathrm{KV}} = 2BLHCd_hs.
$$

The speedup therefore costs memory and grows linearly with context. The benchmark
reports both sides of that trade at several prompt lengths.

### KL divergence

For the adapted distribution \(p\) and reference distribution \(q\), I measure

$$
D_{\mathrm{KL}}(p \parallel q)
=
\sum_x p(x)\log\frac{p(x)}{q(x)}.
$$

### The KL-regularized objective

I also add a regularized objective

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\beta D_{\mathrm{KL}}(p_\theta \parallel p_{\mathrm{ref}}).
$$

The coefficient \(\beta\) controls how much movement the adaptation is allowed to
make. The point is to show the trade rather than select one universal value.

## In the code

Dropout knobs live in the model configuration and are applied across embeddings,
attention probabilities, residual outputs, and the MLP in `src/model.py`. The
cached decode path and the key-value tensors live alongside the attention block in
`src/gpt.py` and `src/model.py`. The divergence and the regularized objective live
in `src/objectives.py`. `LEARNING.md` derives each result from first principles
and points at the exact functions and lines.

## Results and honesty boundary

I ran the cache implementation against the published 44M parameter checkpoint on
the same Apple M1. Each row is the median of three runs generating 16 tokens with
greedy decoding.

| Prompt | Uncached tokens/s | Cached tokens/s | Speedup | Full sequence cache |
|---:|---:|---:|---:|---:|
| 32 | 64.8 | 66.8 | 1.03x | 2.25 MiB |
| 128 | 49.4 | 63.1 | 1.28x | 6.75 MiB |
| 512 | 16.7 | 49.0 | 2.93x | 24.75 MiB |

The cache memory column is the exact tensor size needed to retain the prompt and
all 16 generated positions for continued decoding. The short prompt is expected to
benefit less because model launch and sampling overhead still dominate. Longer
prompts expose more repeated prefix work.

### EXTENSION. KV-cache numerical equivalence under dtype

I ran a development-scale local extension with `python -m src.kv_equivalence`
using `configs/kv_equivalence.json`, the restored `checkpoints/final.pt`, three
prompts, a 128-token greedy horizon, and Apple MPS. The run compares cached
logits against full-recompute logits on one shared greedy trajectory. In
`float32`, no prompt shows a token flip within 128 generated tokens. The maximum
absolute logit deviation is `2.09808349609375e-05`, and the maximum relative
logit deviation is `1.32924469653517e-06`. In `bfloat16`, no prompt shows a
token flip within the same horizon. The maximum absolute logit deviation is
`0.125`, and the maximum relative logit deviation is
`0.008438818156719208`. This falsifies the stronger expectation that `bfloat16`
produces a token flip within this small horizon, while it supports the narrower
numerical claim that the lower-precision path accumulates larger logit
differences. The artifact lives at
`artifacts/kv-equivalence-dev-local/results.json`.

The dropout and KL tables come from the publication runs. The publication run
sweeps 0.00, 0.05, 0.10, and 0.20 over three seeds for dropout, and reports task
loss with held-out KL together across the same held-out token batches. Those
three-seed runs remain pending accelerator execution. The repository already
contains the implementation, deterministic smoke checks, benchmark format, and
artifact manifest. I will not fill the dropout or KL result tables from a random
model or a single smoke run because those numbers would not answer the questions
above.

<!-- Insert the signed-off dropout table and loss figure here. -->
<!-- Insert the signed-off KL table here. -->

## References

1. Vaswani, A. et al. Attention Is All You Need. NeurIPS, 2017.
2. Hinton, G. et al. Distilling the Knowledge in a Neural Network. 2015.
3. Schulman, J. et al. Proximal Policy Optimization Algorithms. 2017.
4. Pope, R. et al. Efficiently Scaling Transformer Inference.
   arXiv:2211.05102, 2022.

Code and model.

- Code: [gpt2-nano](https://github.com/kotlarmilos/gpt2-nano)
- Model: [gpt2-nano](https://huggingface.co/kotlarmilos/gpt2-nano)
- Demo: [gpt2-nano KV-cache](https://huggingface.co/spaces/kotlarmilos/gpt2-nano-kv-cache)

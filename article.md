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

For batch \\(B\\), layers \\(L\\), heads \\(H\\), head dimension \\(d_h\\), context \\(C\\),
and element size \\(s\\), the cache occupies

$$
M_{\mathrm{KV}} = 2BLHCd_hs.
$$

The speedup therefore costs memory and grows linearly with context. The benchmark
reports both sides of that trade at several prompt lengths.

### KL divergence

For the adapted distribution \\(p\\) and reference distribution \\(q\\), I measure

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

The coefficient \\(\beta\\) controls how much movement the adaptation is allowed to
make. The point is to show the trade rather than select one universal value.

## In the code

Dropout knobs live in the model configuration and are applied across embeddings,
attention probabilities, residual outputs, and the MLP in `src/model.py`. The
cached decode path and the key-value tensors live alongside the attention block in
`src/gpt.py` and `src/model.py`. The divergence and the regularized objective live
in `src/objectives.py`. `LEARNING.md` derives each result from first principles
and points at the exact functions and lines.

The publication extension in `src/kv_cache_study.py` measures the output KL caused
by quantizing one cache head at a time. A greedy allocator then lowers heads from
BF16 to INT8 and INT4 under a sum of those single-head KL measurements. That sum
is an additive calibration proxy. It is not a bound on held-out joint output KL
and it does not model interactions between heads.

The quantized cache in `src/kv_quant.py` keeps history packed and quantizes only
new positions. Every policy, including the BF16 reference, follows the same
incremental execution path. This avoids attributing ordinary cached-versus-
uncached rounding differences to quantization.

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

### KV-cache numerical equivalence under dtype

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

### Allocating KV precision under a KL calibration budget

I ran the mixed-precision study on an NVIDIA RTX A5000 with PyTorch 2.4.1 and
BF16 model execution. Calibration uses the first 512 corpus tokens. Final
validation uses 2,048 tokens beginning at position 9,000,000. The checkpoint,
corpus, config, split boundaries, source commit, and deterministic CUDA settings
are recorded in `artifacts/kv-cache-publication/results.json`.

| Policy | Mean held-out KL | Token agreement | Cache size | Compression versus BF16 |
|---|---:|---:|---:|---:|
| Uniform INT8 | 0.000258 | 99.07% | 811,008 bytes | 1.94x |
| Mixed 0.01 proxy budget | 0.001528 | 97.85% | 579,840 bytes | 2.71x |
| Uniform INT4 | 0.009261 | 94.34% | 417,792 bytes | 3.76x |
| All BF16 | 0 | 100% | 1,572,864 bytes | 1.00x |

The mixed policy assigns 70 heads to INT4, 19 to INT8, and 7 to BF16. It creates
an intermediate memory-quality point that the uniform policies cannot express.
The allocator predicts calibration proxy `0.009986`, while held-out joint output
KL is `0.001528`. The difference shows why the proxy must guide allocation rather
than serve as a claimed KL guarantee.

The larger configured budgets all select uniform INT4 because no precision below
four bits is available. The five-value budget grid therefore produces only one
nontrivial mixed policy. The result demonstrates controlled heterogeneous
allocation, not a dense Pareto frontier or a globally optimal policy.

Quantized execution is slower than BF16 in this readable implementation because
Python code reconstructs full-precision keys and values for attention. Those
timings measure simulation overhead and do not support an inference-speed claim.
The defensible systems result is persistent cache size.

### Cache-noise training is a null result

I selected a conservative learning rate and KL weight on reserved tokens from
positions 4,096 through 5,120, then trained three seeds for 50 steps without
looking at final validation. The mixed-policy KL after noise training is
`0.001515 ± 0.000001`, compared with `0.001528` before training. Token agreement
is `97.82% ± 0.09%`, compared with `97.85%` before training. The tuned full-
precision model also drifts from the original checkpoint by KL
`0.000189 ± 0.000002`.

The change is too small and confounded by weight drift to support a robustness
improvement. This negative result still answers the intervention question. At
this scale, the simple cache-noise objective does not materially improve the
measured mixed-precision policy.

The result artifact is committed. The three 101.8 MB tuned checkpoints are
excluded from Git and can be regenerated from the pinned source inputs and
configuration.

The dropout and KL tables come from the publication runs. The publication run
sweeps 0.00, 0.05, 0.10, and 0.20 over three seeds for dropout, and reports task
loss with held-out KL together across the same held-out token batches. Those
three-seed runs remain pending accelerator execution. The repository already
contains the implementation, deterministic smoke checks, benchmark format, and
artifact manifest. I will not fill the dropout or KL result tables from a random
model or a single smoke run because those numbers would not answer the questions
above.

## References

1. Vaswani, A. et al. Attention Is All You Need. NeurIPS, 2017.
2. Hinton, G. et al. Distilling the Knowledge in a Neural Network. 2015.
3. Schulman, J. et al. Proximal Policy Optimization Algorithms. 2017.
4. Pope, R. et al. Efficiently Scaling Transformer Inference.
   arXiv:2211.05102, 2022.
5. Hooper, C. et al. KVQuant. arXiv:2401.18079, 2024.
6. Li, X. et al. KVTuner. arXiv:2502.04420, 2025.

Code and model.

- Code: [gpt2-nano](https://github.com/kotlarmilos/gpt2-nano)
- Model: [gpt2-nano](https://huggingface.co/kotlarmilos/gpt2-nano)
- Demo: [gpt2-nano KV-cache](https://huggingface.co/spaces/kotlarmilos/gpt2-nano-kv-cache)

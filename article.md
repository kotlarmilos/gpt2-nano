---
title: "Shrinking my transformer's KV cache with a KL calibration budget"
thumbnail: /blog/assets/kv-cache-kl-budget/thumbnail.png
authors:
- user: kotlarmilos
---

# Shrinking my transformer's KV cache with a KL calibration budget

## TL;DR

I extend my trained 44M-parameter transformer with a per-head mixed-precision
KV cache. Instead of assigning every head the same precision, I measure how
quantizing each head changes the next-token distribution, then allocate BF16,
INT8, and INT4 under a KL calibration budget.

On an RTX A5000, the selected policy uses 579,840 bytes of persistent cache
storage instead of 1,572,864 bytes for BF16, a 2.71x compression factor. It
preserves 97.85% next-token agreement with the BF16 reference on the evaluated
positions, with mean output KL of `0.001528`.

This is a storage result, not a speedup or a guarantee of unchanged generation.
The implementation is slower than BF16, the calibration budget is only a proxy,
and a fixed three-seed cache-noise adaptation does not establish a robustness
gain.

## The problem

A KV cache avoids recomputing earlier keys and values during autoregressive
decoding. That saves repeated work, but the stored tensors grow with context
length and batch size. For batch size `B`, layers `L`, heads per layer `H`,
context length `C`, head dimension `d_h`, and bytes per element `s`, the dense
cache occupies

$$
M_{\mathrm{KV}} = 2BLHCd_hs.
$$

The factor of two accounts for keys and values. Reducing `s` saves storage, but
quantization can change attention and therefore the output distribution.
Uniform INT8 and uniform INT4 give two simple trade-offs. The question I test
is whether assigning different precisions to different heads gives a useful
intermediate point.

Low-bit caches and sensitivity-aware mixed precision are established research
directions. KVQuant studies KV-cache quantization [1]. KVTuner studies
sensitivity-aware layer-wise mixed-precision KV-cache quantization [2]. I am not
claiming to invent either idea or to outperform those systems. I implement a
small per-head experiment using output KL as the calibration signal, then test
whether training with cache noise improves the selected policy.

## Measuring sensitivity one head at a time

The model has 12 layers with 8 heads per layer, giving 96 layer-head pairs. I
keep its weights fixed and measure two interventions for each pair. In one,
that head's keys and values use INT8. In the other, they use INT4. Every other
head remains BF16.

For the quantized next-token distribution `p` and BF16 reference `q`, I measure

$$
D_{\mathrm{KL}}(p \parallel q)
= \sum_x p(x)\log\frac{p(x)}{q(x)}.
$$

The sensitivity score is the mean of this quantity over calibration positions.
It measures the effect on the model's output, rather than reconstruction error
in the key and value tensors alone.

Both paths receive the same ground-truth token at every position. This is
teacher-forced evaluation, not two independently generated continuations.
Both also use the same incremental cache execution path, including the BF16
reference. Otherwise, cached-versus-uncached numerical differences could become
part of the apparent quantization error.

Low KL means the distributions are similar. It does not establish language
quality. I therefore also report next-token cross entropy and agreement between
the two distributions' highest-probability tokens.

## Allocating precision under a calibration budget

The allocator starts with every head in BF16. It considers two transitions for
each head.

```text
BF16 -> INT8 -> INT4
```

Each transition has a storage saving and an estimated increase in the
single-head KL score. The allocator chooses the feasible transition with the
lowest estimated increase per byte saved, updates the policy, and repeats.

```text
priority = added calibration KL / bytes saved
```

For the INT8-to-INT4 transition, the added score is the INT4 sensitivity minus
the INT8 sensitivity, clipped at zero. Storage savings include the scales needed
by the quantizer, not just the nominal payload bit width.

The budget limits the accumulated single-head scores. It is an additive
calibration proxy, not a bound on the jointly quantized model. Heads interact
through attention outputs, residual connections, and later layers. Their joint
effect does not have to equal the sum of isolated interventions. The greedy
search is also not globally optimal.

That distinction matters in the result. The selected policy has calibration
proxy `0.009986` under a budget of `0.01`, while its measured held-out joint KL
is `0.001528`. A smaller observed value in this run does not turn the proxy into
a guarantee.

## What the cache stores

`src/kv_quant.py` implements symmetric quantization with a separate scale for
each token and head, independently for keys and values. INT8 stores one byte per
value. INT4 packs two values into one byte. Both store their scales in FP16.
BF16 heads retain the model-dtype representation.

Previously stored positions remain packed. Each step quantizes only the new
positions before appending them. Attention still consumes reconstructed BF16
keys and values, so a decode step also allocates transient dequantized tensors.

The reported compression covers persistent tensor payloads and scales. It is
not a reduction in model-weight storage, Python object overhead, or total peak
GPU memory. This implementation is deliberately inspectable rather than a
fused low-bit attention kernel.

## The measured experiment

I run the study against the existing trained checkpoint on an NVIDIA RTX A5000
with PyTorch `2.4.1+cu124`, BF16 model execution, and deterministic CUDA settings.
Dropout is zero. The checkpoint, tokenizer, and token shard are pinned to the
existing `gpt2-nano` artifacts rather than regenerated for this experiment.

| Setting | Value |
| --- | --- |
| Calibration tokens | Positions 0 through 511 |
| Noise-training tokens | Positions 512 through 4,095 |
| Configured reserved tokens, unused by the study | Positions 4,096 through 5,119 |
| Final validation tokens | Positions 9,000,000 through 9,002,047 |
| Evaluation block length | 64 tokens |
| Calibration / validation blocks | 8 / 32 |
| Cache batch size | 1 |
| Calibration budgets | 0.01, 0.05, 0.10, 0.25, 0.50 |

These windows do not overlap. The final validation window is held out from this
study's calibration and adaptation. This does not establish that the original
model never saw those tokens during pretraining.

The 2,048 validation tokens are split into 32 independent 64-token blocks. They
are not one 2,048-token generation or a long-context benchmark. Each policy uses
the same blocks and reference logits.

### Storage and output-distribution results

The storage column below is for a single 64-token cache, including quantization
scales. Agreement measures matching next-token argmax choices at the same
teacher-forced positions. It is not accuracy against the corpus, a human quality
score, or agreement between complete generated sequences.

| Policy | Mean held-out KL | Token agreement | Next-token CE | Cache bytes | Compression versus BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All BF16 | 0 | 100% | 2.74403 | 1,572,864 | 1.00x |
| Uniform INT8 | 0.000258 | 99.07% | 2.74390 | 811,008 | 1.94x |
| Mixed 0.01 proxy budget | 0.001528 | 97.85% | 2.74538 | 579,840 | 2.71x |
| Uniform INT4 | 0.009261 | 94.34% | 2.75178 | 417,792 | 3.76x |

The mixed policy assigns 70 heads to INT4, 19 to INT8, and 7 to BF16. It stores
less than uniform INT8 and changes the reference distribution less than uniform
INT4. That gives an intermediate storage-versus-distribution-change point.

The other four budgets all select uniform INT4 because no precision below four
bits is available. The five-value grid therefore yields only one nontrivial
mixed policy. It does not establish a dense Pareto frontier.

The study also lacks a random or alternative allocation baseline at the same
579,840-byte budget. It therefore does not establish that sensitivity-guided
selection is better than another mixed allocation with identical storage.

### No decoding-speed claim

The mixed-precision path is slower than BF16 in this implementation. Python code
unpacks and reconstructs BF16 tensors before attention, and the evaluation
includes other host-side overhead. Those timings do not represent the
performance of a fused quantized-cache kernel.

The supported systems claim is smaller persistent cache storage. This study
does not use an ordinary cached-versus-uncached benchmark to claim a speedup.

## Cache-noise adaptation did not establish a robustness gain

The second question is whether a short adaptation makes the model less
sensitive to low-bit cache errors. During training, I perturb keys and values
with uniform noise scaled to approximate quantization error. The model learns
from next-token cross entropy plus KL to a frozen copy of the original model.

$$
\mathcal{L}
= \mathcal{L}_{\mathrm{task}}
+ \beta D_{\mathrm{KL}}(p_{\mathrm{student}} \parallel p_{\mathrm{reference}}).
$$

The fixed recipe uses a learning rate of `1e-6` and KL weight of `1.0`, then
runs 50 optimizer steps for each of three seeds. The implementation does not
evaluate candidate settings or use the configured reserved window to choose
them. Final validation only measures the fixed recipe. I evaluate the original
mixed policy with each adapted model and compare its outputs with the original
BF16 model.

Across the three seeds, mean mixed-policy KL is `0.001515`, compared with
`0.001528` before training. Mean token agreement is `97.82%`, compared with
`97.85%` before training. The adapted models also differ from the original when
their caches remain BF16, with mean KL `0.000189`.

The small quantized-KL change includes both weight drift and changed response
to quantization. Agreement does not improve. Without an equal-step adaptation
control that omits cache noise, I cannot attribute that change to improved
robustness. This fixed recipe is a negative result, not evidence that
noise-based adaptation can never work.

## Code and reproducibility

| File | Role |
| --- | --- |
| `src/model.py` | Transformer and incremental KV-cache execution |
| `src/kv_quant.py` | Packed cache tensors, quantization, and cache-noise injection |
| `src/kv_cache_study.py` | Sensitivity measurement, allocation, validation, and fixed adaptation |
| `src/objectives.py` | Output KL |
| `configs/kv_cache_publication.json` | Hardware-run settings and disjoint token windows |

The raw measurements are in `artifacts/kv-cache-publication/results.json`.
`artifacts/manifest.json` records their SHA-256 along with the input and
configuration hashes. The run records clean source commit
`3a7e9e2f94a6407fe90eff9ec877455b664c644e` at study start.

From a clean checkout of that source with the project dependencies installed,
the recorded workflow is

```bash
python scripts/prepare_publication_inputs.py
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m src.kv_cache_study --config configs/kv_cache_publication.json
```

The input-preparation script verifies the downloaded files against the manifest.
The study command requires CUDA and reruns calibration, evaluation, and
three-seed adaptation. The three 101.8 MB adapted checkpoints are excluded from
Git, while their hashes and measured results remain in the evidence.

## What this establishes

On one small trained model and a short held-out evaluation window, a per-head
allocation gives a measured intermediate trade-off between uniform INT8 and
INT4. Persistent cache storage falls by a factor of 2.71 relative to BF16 while
the recorded next-token agreement remains 97.85%.

It does not establish faster inference, long-context reliability, unchanged
generation quality, or superiority over prior quantization methods. The value
of this extension is the explicit allocation experiment, its measured
trade-off, and the negative result from the fixed cache-noise adaptation.

## References

1. KVQuant. *Towards 10 Million Context Length LLM Inference with KV Cache
   Quantization*. arXiv:2401.18079.
2. KVTuner. *Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization
   for Efficient and Nearly Lossless LLM Inference*. arXiv:2502.04420.

The existing source and base model are available at
[gpt2-nano on GitHub](https://github.com/kotlarmilos/gpt2-nano) and
[gpt2-nano on Hugging Face](https://huggingface.co/kotlarmilos/gpt2-nano).

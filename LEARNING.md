# Building up dropout, KV caching, and KL divergence from scratch

This note builds three ideas from first principles and ties each one to the exact
lines that implement it in `src/model.py` and `src/objectives.py`. The goal is
that you can rederive every formula yourself and then see where it lives in the
code.

## Dropout

### The problem dropout solves

A network with enough parameters can fit the training set by memorizing it. What
we want instead is a function that still behaves when the input shifts a little.
Dropout attacks this by refusing to let the network rely on any single
activation. During training it randomly deletes activations, so a feature that is
only useful when a specific partner feature is present becomes unreliable, and
the network is pushed toward features that work on their own.

### The mechanism

Take one activation `a`. Draw a mask `m` from a Bernoulli distribution that is one
with keep probability `q = 1 - p` and zero with probability `p`. The dropped
activation is `m * a`. If we stop here, the expected activation is `q * a`, so the
training-time scale and the test-time scale disagree, and every downstream weight
sees a different input magnitude at test time.

Inverted dropout fixes the scale during training. Divide the kept activations by
`q`.

```
a_dropped = (m / q) * a
E[a_dropped] = q * (a / q) + p * 0 = a
```

The expectation is now `a`, so test time needs no rescaling at all. This is what
`torch.nn.Dropout` does, which is why the model applies dropout only through
`nn.Dropout` modules and never rescales by hand.

### Why this regularizes

Look at a single linear unit `y = sum_i w_i x_i` with dropout on its inputs.

```
y = sum_i (m_i / q) w_i x_i
E[y] = sum_i w_i x_i
Var[y] = sum_i w_i^2 x_i^2 * Var[m_i / q] = (p / q) * sum_i w_i^2 x_i^2
```

The mean is the clean output, so dropout does not bias the unit. The variance is
the interesting part. Under a squared-error objective the expected loss picks up a
term proportional to the output variance, which here is `(p / q) * sum_i w_i^2
x_i^2`. That is an input-scaled `L2` penalty on the weights, and it appears
without anyone writing a penalty term. Dropout is data-dependent weight decay in
disguise.

There is a second reading. Each mask defines a thinned subnetwork, training
visits a different subnetwork on every step, and test-time inference with the full
weights approximates an average over that exponential family of subnetworks. The
inverted-dropout scaling is what makes the single full-weight pass a sensible
stand-in for the ensemble.

### Where it lives in the code

`GPTConfig.dropout` is validated to lie in `[0, 1)` in `__post_init__`, so a rate
of one, which would delete everything, is rejected. The `Block` holds three
dropouts and the `GPT` holds one, each placed where a specific reliance should be
discouraged.

- `attn_dropout` acts on the attention weights after the softmax on line 118.
  Dropping attention weights randomly severs some query-to-key links, so a head
  cannot depend on one exact context position. The distribution no longer sums to
  one after dropping, and the `1 / q` scaling is exactly what keeps the attended
  vector unbiased in expectation.
- `residual_dropout` acts on the attention output before it re-enters the
  residual stream on line 123.
- `mlp_dropout` acts on the feed-forward output before its residual add on line
  124.
- `embedding_dropout` acts on the summed token and position embeddings on line
  172, so the very first representation is already noisy.

The default rate is `0.0`, which makes dropout the identity and keeps the small
smoke runs deterministic. Regularization is something you switch on for the
larger publication configuration, not a fixed cost on every run.

## KV caching

### The cost of naive generation

Autoregressive generation emits one token at a time. To choose token `t` the model
needs the attention of the query at position `t` over the keys and values at every
position up to `t`. A naive loop rebuilds the entire prefix on every step. Line
237 to 252 is that loop. It retokenizes the growing sequence, runs a full forward
pass, and keeps only the last position's logits.

Count the work. At step `t` the projections cost about `t * d^2` and the attention
scores cost about `t^2 * d`. Summing over `T` generated tokens gives roughly `T^2
* d^2` for projections and `T^3 * d` for attention. The cubic term is the one that
hurts, and almost all of it is recomputation, because the keys and values of the
earlier tokens are identical to what they were on the previous step.

### The invariant that makes caching correct

Causal masking is the reason the recomputation is wasteful. The key and value at
position `i` are functions of the layer input at position `i` and the positions
before it, never of a future token. So once token `i` has been processed its key
`k_i` and value `v_i` are frozen for the rest of the generation. The query at
position `i` is also never needed again once its own output has been produced.

That gives the cache. Keep every past `k_i` and `v_i`. On each new step compute
the query, key, and value for the single new token only, append the new key and
value to the store, and attend the one new query over the whole store.

```
per-step projection cost: d^2
per-step attention cost:   t * d
total over T tokens:       T * d^2 + T^2 * d
```

The cubic attention term collapses to quadratic, and the quadratic projection term
collapses to linear. The cost of storing the cache is `2 * L * B * H * T *
head_dim`, which equals `2 * L * B * d * T` floats for `L` layers, batch `B`, and
model width `d`. You are trading memory for the removed recomputation.

### Getting the positions right

The subtle part is that a cached step feeds only the new token, so the model no
longer knows the absolute position from the input length. The code carries the
position explicitly.

`GPT.forward` reads `past_len` from the cache on lines 157 to 163, checks that all
layer caches share one length, and slices the sinusoidal position table from
`past_len` to `end_position` on line 171. Inside `Block.forward` the query
positions start at `past_len` on lines 110 to 112, the key positions run over the
full cached length on line 113, and the mask keeps a query from attending to any
strictly later key on line 114.

```
causal_mask = key_positions > query_positions
```

During incremental decoding the query length is one and its position is
`past_len`, while the keys run from zero to `past_len`, so no key is later than the
query and the mask hides nothing. The same expression also handles the first call,
where a whole prompt is processed at once and the mask must hide the future inside
that prompt. `generate` uses this by prefilling the prompt on line 221 and then
feeding one token at a time on lines 233 to 234.

Because the cache changes only the arithmetic, not the function being computed, the
cached and uncached paths must return identical logits under identical sampling,
and that equivalence is what the tests pin down.

### Why exact equivalence can break under `bfloat16`

The equality claim starts from one recurrence. In `src/model.py`, `GPT.forward`
slices the position table from `past_len` to `end_position` on line 171. In
`Block.forward`, lines 98 to 117 concatenate cached keys and values with the new
key and value, then build the same causal mask from absolute query and key
positions. For a fixed prefix, the cached path and the full path present the same
key and value sequence to each new query. Exact arithmetic gives the same real
number for every projection, residual add, softmax, and MLP application on those
same operands.

`src/kv_equivalence.py` turns that claim into a falsifiable measurement.
`measure_prompt_equivalence` pre-fills the cache on lines 143 to 146, recomputes
the full prefix on lines 149 to 152, compares the two last-position logit vectors
on lines 154 to 158, and records the first greedy-token disagreement on lines 159
to 167. It extends both paths with the uncached greedy token on line 181, so every
later deviation still compares one shared prefix. `max_logit_deviation` on lines
84 to 97 reports the maximum absolute deviation and scales it by the larger
maximum logit magnitude, clamped by `relative_epsilon`.

`bfloat16` breaks the exact-arithmetic premise because it keeps an 8-bit exponent
but only 7 explicit fraction bits. Each projection, attention score, softmax
output, residual add, and MLP output can round to a coarser grid than `float32`
when the operation stores a `bfloat16` result. The cached path rounds old keys and
values once and reuses them, while the uncached path recomputes and rounds the
whole prefix at each step. These are different rounding histories. The real-number
function is the same, but the floating-point programs need not produce identical
logits. The runner records the dtype, prompt count, horizon, seed, per-step
deviations, flip positions, environment, and hash records through `run_study` and
`write_artifacts` on lines 298 to 420.

## KL divergence

### Defining the quantity

For two distributions `p` and `q` over the same vocabulary the relative entropy,
or Kullback-Leibler divergence, is

```
KL(p || q) = sum_v p(v) * log( p(v) / q(v) ).
```

It is never negative, and it is zero only when `p` equals `q`. The proof is one
line of Jensen's inequality applied to the concave logarithm.

```
-KL(p || q) = sum_v p(v) * log( q(v) / p(v) )
            <= log sum_v p(v) * ( q(v) / p(v) )
            =  log sum_v q(v) = log 1 = 0.
```

So `KL >= 0`. It measures how many extra nats you pay to encode samples from `p`
using a code built for `q`. It is not symmetric, and the asymmetry is a feature we
use below.

### The code that computes it

`categorical_kl` in `src/objectives.py` computes `KL(model || reference)`. It takes
log-softmax of both logit tensors, exponentiates the model log-probabilities to
recover `p`, and sums `p * (log p - log q)` over the vocabulary. The reference
logits are detached first, so `q` is a fixed target and no gradient flows back into
whatever produced it. The reductions expose the two conventions that matter in
practice, mean over the batch and `batchmean`, which is the mathematically correct
per-distribution average that matches the sum-over-vocabulary definition.

The direction is deliberate. Because the model is the first argument, the term
`p * log(p / q)` grows large wherever the model places mass that the reference does
not support. Minimizing it therefore pulls the model to stay inside the support of
the reference, which is the behavior you want from an anchor that says do not
wander far from this distribution.

### Turning it into a training objective

`kl_regularized_loss` returns `task_loss + beta * kl_loss`, where the task loss is
the ordinary cross-entropy to the target tokens and the KL loss anchors the model
to the reference distribution.

```
total(theta) = CrossEntropy(model_theta, targets) + beta * KL(model_theta || reference)
```

Cross-entropy is itself a KL. Against a one-hot target the entropy of the target is
zero, so `CrossEntropy(model, target) = KL(target || model)`. Reading the two
terms together, the objective is one KL that pushes the model toward the data and
a second KL, weighted by `beta`, that pulls it toward the reference, and the two
divergences point in opposite directions.

```
total = KL(data || model) + beta * KL(model || reference)
```

This single structure is the backbone of three techniques. With the reference set
to a larger teacher it is knowledge distillation. With the reference set to a
frozen copy of the policy it is the KL penalty that keeps reinforcement learning
from human feedback near its starting point. With a small `beta` it is a trust
region that limits how far one update may move the output distribution. The article
makes the honest point that the KL term measures movement and not quality. A model
can lower its KL to the reference while getting no better at the task, so `beta`
sets a leash length, not a target.

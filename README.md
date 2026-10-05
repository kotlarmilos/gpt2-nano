# gpt2-nano

`gpt2-nano` is a 44M-parameter decoder-only transformer implemented in
PyTorch. This repository contains the original model and one measured
extension: allocating BF16, INT8, and packed INT4 KV-cache precision per
attention head.

The full writeup is in `article.md`.

## Result

The experiment measures each head's output-KL sensitivity, then greedily lowers
precision under an additive calibration budget. On an NVIDIA RTX A5000, the
selected policy uses 70 INT4 heads, 19 INT8 heads, and 7 BF16 heads.

| Policy | Held-out KL | Agreement with BF16 | Cache bytes |
| --- | ---: | ---: | ---: |
| BF16 | 0 | 100% | 1,572,864 |
| INT8 | 0.000258 | 99.07% | 811,008 |
| Mixed | 0.001528 | 97.85% | 579,840 |
| INT4 | 0.009261 | 94.34% | 417,792 |

The mixed policy reduces persistent cache storage by 2.71x. Agreement compares
next-token choices on the same teacher-forced prefixes. The implementation
dequantizes before attention, so this is a storage result rather than a speedup.

The fixed three-seed cache-noise adaptation did not improve agreement and does
not establish a robustness gain.

## Repository layout

- `src/model.py` implements the transformer and incremental KV cache.
- `src/kv_quant.py` implements packed INT8 and INT4 cache storage.
- `src/kv_cache_study.py` implements calibration, allocation, validation, and
  the fixed cache-noise adaptation.
- `configs/kv_cache_publication.json` is the exact measured configuration.
- `artifacts/kv-cache-publication/results.json` is the measured result.
- `artifacts/manifest.json` records hashes and input locations.
- `tests/test_kv_experiment.py` contains the focused experiment tests.

## Local checks

Python 3.10 through 3.13 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[data]"
python -m unittest -v tests.test_kv_experiment
```

## Reproduce the GPU study

The preparation script downloads the published checkpoint, tokenizer, and
token shard and verifies their hashes.

```bash
python scripts/prepare_publication_inputs.py
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m src.kv_cache_study --config configs/kv_cache_publication.json
```

The command requires CUDA and reruns calibration, held-out evaluation, and the
three fixed adaptation seeds. The adapted checkpoints are not stored in Git;
their hashes and measurements are recorded in the result JSON.

## Original model

The base checkpoint was trained for 10,000 steps on approximately 99M
FineWeb-Edu tokens. It is educational and too small for reliable factual text
generation.

```bash
python -m data.bpe_tokenizer
python -m src.gpt
```

Model artifacts are available at
`https://huggingface.co/kotlarmilos/gpt2-nano`.

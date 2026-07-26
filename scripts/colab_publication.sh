#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/publication.json}"
EXPERIMENT="${EXPERIMENT:-}"
SEED="${SEED:-1337}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-checkpoints/final.pt}"
cd "$ROOT"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e '.[data]'
"$PYTHON" scripts/prepare_publication_inputs.py

case "$EXPERIMENT" in
  dropout)
    : "${DROPOUT:?Set DROPOUT to 0.0, 0.05, 0.1, or 0.2}"
    "$PYTHON" scripts/run_dropout_sweep.py \
      --config "$CONFIG" \
      --output-dir artifacts/dropout \
      --dropout "$DROPOUT" \
      --seed "$SEED"
    ;;
  kl)
    : "${KL_BETA:?Set KL_BETA to 0.0, 0.01, 0.05, or 0.1}"
    "$PYTHON" scripts/run_kl_sweep.py \
      --config "$CONFIG" \
      --output-dir artifacts/kl \
      --reference-checkpoint "$REFERENCE_CHECKPOINT" \
      --beta "$KL_BETA" \
      --seed "$SEED"
    ;;
  cache)
    "$PYTHON" scripts/benchmark_cache.py \
      --checkpoint "$REFERENCE_CHECKPOINT" \
      --prompt-lengths 32 128 512 \
      --new-tokens 16 \
      --repeats 5 \
      --output artifacts/cache-publication-colab.json
    ;;
  plan)
    "$PYTHON" scripts/run_dropout_sweep.py \
      --config "$CONFIG" \
      --output-dir artifacts/dropout \
      --dry-run
    "$PYTHON" scripts/run_kl_sweep.py \
      --config "$CONFIG" \
      --output-dir artifacts/kl \
      --reference-checkpoint "$REFERENCE_CHECKPOINT" \
      --dry-run
    ;;
  *)
    echo "Set EXPERIMENT to dropout, kl, cache, or plan." >&2
    exit 2
    ;;
esac

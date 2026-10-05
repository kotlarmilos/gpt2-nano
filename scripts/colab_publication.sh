#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/kv_cache_publication.json}"
cd "$ROOT"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e '.[data]'
"$PYTHON" scripts/prepare_publication_inputs.py
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
"$PYTHON" -m src.kv_cache_study --config "$CONFIG"

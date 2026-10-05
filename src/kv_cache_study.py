"""Mixed-precision KV-cache calibration, allocation, and validation study.

This module implements:
  1. Sensitivity calibration: per-(layer, head) KL measured with one head
     quantized at a time, all others retained in the model dtype.
  2. Greedy marginal-KL-per-byte policy allocation under explicit KL budgets.
  3. Validation: held-out CE, KL percentiles, token agreement, timing, memory.
  4. A fixed three-seed cache-noise adaptation that did not establish a
     robustness gain.

Per-head mixed-precision KV quantization is prior art. This study measures a
small model using output KL as an additive calibration proxy and reports the
joint held-out result separately. The proxy is not a bound.

CLI usage:
  python -m src.kv_cache_study --config configs/kv_cache_publication.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.kv_quant import (
    CachePolicy,
    NoisyKVGPT,
    QuantizedCacheGPT,
    QuantizedKVCache,
)
from src.model import GPT, GPTConfig
from src.objectives import categorical_kl


# ---------------------------------------------------------------------------
# Helpers: hashing, environment, device
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_source_state(cwd: Path) -> dict[str, Any]:
    revision: str | None = None
    dirty: bool | None = None
    status_sha256: str | None = None
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=5
        )
        if revision_result.returncode == 0:
            revision = revision_result.stdout.strip()
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            cwd=cwd,
            timeout=5,
        )
        if status_result.returncode == 0:
            dirty = bool(status_result.stdout)
            status_sha256 = sha256_bytes(status_result.stdout)
    except (OSError, subprocess.TimeoutExpired):
        return {
            "git_revision": revision,
            "git_dirty": dirty,
            "git_status_sha256": status_sha256,
        }
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "git_status_sha256": status_sha256,
    }


def collect_environment(device: torch.device) -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if device.type == "cuda":
        env["cuda_version"] = torch.version.cuda
        env["cuda_device_name"] = torch.cuda.get_device_name(device)
    return env


def configure_deterministic_execution(device: torch.device) -> None:
    torch.use_deterministic_algorithms(True)
    if (
        device.type == "cuda"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        not in {":4096:8", ":16:8"}
    ):
        raise RuntimeError(
            "CUDA publication runs require CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "or CUBLAS_WORKSPACE_CONFIG=:16:8"
        )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_from_name(name: str) -> torch.dtype:
    n = name.lower()
    if n in {"float32", "fp32"}:
        return torch.float32
    if n in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if n in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype name: {name}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> GPT:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = GPTConfig(**payload["config"])
    model = GPT(config)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device=device, dtype=dtype).eval()


def build_random_model(
    cfg_dict: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> GPT:
    torch.manual_seed(seed)
    cfg = GPTConfig(**cfg_dict)
    return GPT(cfg).to(device=device, dtype=dtype).eval()


# ---------------------------------------------------------------------------
# Token shard loading
# ---------------------------------------------------------------------------


def load_token_shard(path: Path) -> np.ndarray:
    """Load and validate the repository's headered uint16 BPE shard format."""
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256:
        raise ValueError(f"token shard {path} has a truncated header")
    if int(header[0]) != 20250320 or int(header[1]) != 1:
        raise ValueError(f"token shard {path} has an unsupported header")
    num_tokens = int(header[2])
    expected_size = 256 * 4 + num_tokens * 2
    if num_tokens < 1 or path.stat().st_size != expected_size:
        raise ValueError(
            f"token shard {path} size does not match its token count"
        )
    return np.fromfile(
        path,
        dtype="<u2",
        count=num_tokens,
        offset=256 * 4,
    ).astype(np.int64)


def token_windows(
    tokens: np.ndarray,
    start: int,
    end: int,
    seq_len: int,
) -> list[list[int]]:
    """Slice non-overlapping token windows of length seq_len from tokens[start:end]."""
    windows = []
    pos = start
    while pos + seq_len <= end:
        windows.append(tokens[pos : pos + seq_len].tolist())
        pos += seq_len
    return windows


# ---------------------------------------------------------------------------
# Teacher-forced incremental evaluation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def teacher_forced_logits(
    model: GPT,
    sequences: list[list[int]],
    device: torch.device,
) -> list[Tensor]:
    """Full-sequence teacher-forced helper without a KV cache.

    Returns logits[i] of shape (seq_len, vocab_size) for each sequence i.
    The publication study instead uses the model-dtype incremental path as its
    reference so cached-versus-uncached arithmetic is not part of the result.
    """
    model.eval()
    results = []
    for seq in sequences:
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(ids)  # (1, seq_len, vocab)
        results.append(logits[0].cpu())
    return results


@torch.inference_mode()
def quantized_incremental_logits(
    qmodel: QuantizedCacheGPT,
    sequences: list[list[int]],
    device: torch.device,
) -> list[Tensor]:
    """Incremental token-by-token evaluation using quantized KV cache.

    At each position, the ground-truth token is fed (teacher forcing).
    Returns logits[i] of shape (seq_len, vocab_size) for each sequence i.
    """
    qmodel.eval()
    results = []
    for seq in sequences:
        position_logits = []
        cache: QuantizedKVCache | None = None
        for pos in range(len(seq)):
            token_id = torch.tensor([[seq[pos]]], dtype=torch.long, device=device)
            out = qmodel(token_id, past_kv_cache=cache, use_cache=True)
            logits, cache = out
            position_logits.append(logits[0, 0].cpu())  # (vocab,)
        seq_logits = torch.stack(position_logits, dim=0)  # (seq_len, vocab)
        results.append(seq_logits)
    return results


# ---------------------------------------------------------------------------
# Sensitivity calibration
# ---------------------------------------------------------------------------


def compute_sensitivity_matrices(
    model: GPT,
    sequences: list[list[int]],
    device: torch.device,
) -> dict[str, Any]:
    """Compute per-head INT8 and INT4 output-KL sensitivity matrices.

    For each (layer, head) pair independently, quantize only that head to
    INT8 or INT4 and retain all other heads in the model dtype. Measure mean
    output KL against the all-model-dtype incremental baseline.

    Sensitivity assumes per-head independence.  Cross-head interactions are
    not modeled.  This is a greedy approximation baseline.

    Returns a dict with keys:
      int8_matrix   - list[list[float]], shape [num_layers][num_heads]
      int4_matrix   - list[list[float]], shape [num_layers][num_heads]
    """
    num_layers = len(model.blocks)
    num_heads = model.config.num_heads

    int8_matrix = [
        [0.0] * num_heads for _ in range(num_layers)
    ]
    int4_matrix = [
        [0.0] * num_heads for _ in range(num_layers)
    ]

    base_policy = CachePolicy.all_sixteen(num_layers, num_heads)
    baseline_logits_list = quantized_incremental_logits(
        QuantizedCacheGPT(model, base_policy),
        sequences,
        device,
    )

    for bits, matrix in ((8, int8_matrix), (4, int4_matrix)):
        for l_idx in range(num_layers):
            for h_idx in range(num_heads):
                policy = base_policy.with_head(l_idx, h_idx, bits)
                qmodel = QuantizedCacheGPT(model, policy)
                quant_logits_list = quantized_incremental_logits(
                    qmodel, sequences, device
                )
                kl_values = []
                for baseline_logits, quant_logits in zip(
                    baseline_logits_list, quant_logits_list
                ):
                    # Shape: (seq_len, vocab) -> KL per position, then mean
                    kl = categorical_kl(
                        quant_logits.unsqueeze(0),
                        baseline_logits.unsqueeze(0),
                        reduction="none",
                    )
                    kl_values.append(float(kl.mean().item()))
                matrix[l_idx][h_idx] = float(
                    sum(kl_values) / len(kl_values)
                )

    return {
        "int8_matrix": int8_matrix,
        "int4_matrix": int4_matrix,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_sequences": len(sequences),
    }


# ---------------------------------------------------------------------------
# Policy allocation: greedy marginal KL-per-byte
# ---------------------------------------------------------------------------


def bytes_saved(
    layer: int,
    head: int,
    current_bits: int,
    target_bits: int,
    batch: int,
    seq_len: int,
    head_dim: int,
    model_dtype_bytes: int,
) -> int:
    """Bytes saved for one K+V head when downgrading from current_bits to target_bits."""
    def head_bytes(bits: int) -> int:
        if bits == 16:
            return 2 * batch * seq_len * head_dim * model_dtype_bytes
        if bits == 8:
            packed_dim = head_dim
            return 2 * (batch * seq_len * packed_dim + batch * seq_len * 2)
        # bits == 4
        packed_dim = (head_dim + 1) // 2
        return 2 * (batch * seq_len * packed_dim + batch * seq_len * 2)
    return head_bytes(current_bits) - head_bytes(target_bits)


def allocate_policy_greedy(
    int8_matrix: list[list[float]],
    int4_matrix: list[list[float]],
    kl_budget: float,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    model_dtype_bytes: int = 4,
    batch: int = 1,
    seq_len: int = 1,
) -> dict[str, Any]:
    """Greedy marginal-KL-per-byte policy under a KL budget.

    Starting from all-16, at each step selects the (layer, head, transition)
    with the smallest KL increase per byte saved.  The constraint
    16->8 before 8->4 is enforced per head.

    This procedure is greedy and NOT globally optimal.
    Predicted KL uses per-head independence (sum of individual sensitivities).

    Returns:
      policy:          CachePolicy
      predicted_kl:    float (sum over downgraded heads)
      steps:           list of downgrade records
      avg_payload_bits: float
    """
    current_bits: list[list[int]] = [
        [16] * num_heads for _ in range(num_layers)
    ]
    # marginal_kl[l][h][16->8] and [l][h][8->4]
    def marginal_kl(l: int, h: int, current: int, target: int) -> float:
        if current == 16 and target == 8:
            return int8_matrix[l][h]
        if current == 8 and target == 4:
            return max(int4_matrix[l][h] - int8_matrix[l][h], 0.0)
        raise ValueError(f"invalid transition {current}->{target}")

    predicted_kl = 0.0
    steps = []

    while True:
        # Collect all valid next transitions
        candidates = []
        for l in range(num_layers):
            for h in range(num_heads):
                cb = current_bits[l][h]
                transitions = []
                if cb == 16:
                    transitions = [(16, 8)]
                elif cb == 8:
                    transitions = [(8, 4)]
                for cur, tgt in transitions:
                    delta_kl = marginal_kl(l, h, cur, tgt)
                    delta_bytes = bytes_saved(
                        l, h, cur, tgt, batch, seq_len, head_dim, model_dtype_bytes
                    )
                    if delta_bytes <= 0:
                        continue
                    cost = delta_kl / delta_bytes
                    candidates.append((cost, delta_kl, delta_bytes, l, h, cur, tgt))

        if not candidates:
            break

        candidates.sort(key=lambda c: c[0])
        feasible = [
            candidate
            for candidate in candidates
            if predicted_kl + candidate[1] <= kl_budget
        ]
        if not feasible:
            break
        best = feasible[0]
        _, delta_kl, delta_bytes, l, h, cur, tgt = best

        predicted_kl += delta_kl
        current_bits[l][h] = tgt
        steps.append(
            {
                "layer": l,
                "head": h,
                "from_bits": cur,
                "to_bits": tgt,
                "delta_kl": delta_kl,
                "delta_bytes": delta_bytes,
                "cumulative_kl": predicted_kl,
            }
        )

    policy = CachePolicy(
        bits_per_layer=tuple(tuple(lb) for lb in current_bits)
    )
    return {
        "policy": policy,
        "predicted_kl": predicted_kl,
        "steps": steps,
        "avg_payload_bits": policy.avg_payload_bits(),
    }


def average_storage_bits(
    policy: CachePolicy,
    model_dtype_bytes: int,
) -> float:
    values = [
        model_dtype_bytes * 8 if bits == 16 else bits
        for layer in policy.bits_per_layer
        for bits in layer
    ]
    return sum(values) / len(values) if values else 0.0


def build_named_policies(
    int8_matrix: list[list[float]],
    int4_matrix: list[list[float]],
    num_layers: int,
    num_heads: int,
    head_dim: int,
    kl_budgets: list[float],
    model_dtype_bytes: int = 4,
    model_dtype_label: str | None = None,
    seq_len: int = 1,
) -> list[dict[str, Any]]:
    """Build all named policies: uniform INT8, uniform INT4, sensitivity budgets."""
    policies = []

    # Uniform INT8
    p_int8 = CachePolicy(
        bits_per_layer=tuple(
            tuple(8 for _ in range(num_heads)) for _ in range(num_layers)
        )
    )
    policies.append(
        {
            "label": "uniform_int8",
            "policy": p_int8,
            "predicted_kl": sum(
                int8_matrix[l][h]
                for l in range(num_layers)
                for h in range(num_heads)
            ),
            "avg_payload_bits": p_int8.avg_payload_bits(),
            "steps": [],
        }
    )

    # Uniform INT4
    p_int4 = CachePolicy(
        bits_per_layer=tuple(
            tuple(4 for _ in range(num_heads)) for _ in range(num_layers)
        )
    )
    policies.append(
        {
            "label": "uniform_int4",
            "policy": p_int4,
            "predicted_kl": sum(
                int4_matrix[l][h]
                for l in range(num_layers)
                for h in range(num_heads)
            ),
            "avg_payload_bits": p_int4.avg_payload_bits(),
            "steps": [],
        }
    )

    if model_dtype_label is None:
        model_dtype_label = (
            "all_fp32" if model_dtype_bytes == 4 else "all_16bit"
        )

    # The policy value 16 means that a head stays in the model dtype.
    p_model_dtype = CachePolicy.all_sixteen(num_layers, num_heads)
    policies.append(
        {
            "label": model_dtype_label,
            "policy": p_model_dtype,
            "predicted_kl": 0.0,
            "avg_payload_bits": float(model_dtype_bytes * 8),
            "steps": [],
        }
    )

    # Sensitivity-guided greedy for each KL budget
    for budget in kl_budgets:
        result = allocate_policy_greedy(
            int8_matrix,
            int4_matrix,
            kl_budget=budget,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            model_dtype_bytes=model_dtype_bytes,
            seq_len=seq_len,
        )
        policies.append(
            {
                "label": f"greedy_kl_budget_{budget:.4g}",
                "policy": result["policy"],
                "predicted_kl": result["predicted_kl"],
                "avg_payload_bits": average_storage_bits(
                    result["policy"], model_dtype_bytes
                ),
                "steps": result["steps"],
            }
        )

    return policies


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------


def compute_kl_percentiles(kl_tensor: Tensor) -> dict[str, float]:
    kl_flat = kl_tensor.flatten()
    return {
        "mean_kl": float(kl_flat.mean().item()),
        "p50_kl": float(kl_flat.quantile(0.50).item()),
        "p95_kl": float(kl_flat.quantile(0.95).item()),
        "p99_kl": float(kl_flat.quantile(0.99).item()),
        "max_kl": float(kl_flat.max().item()),
    }


def first_disagreement(
    quant_logits: list[Tensor],
    baseline_logits: list[Tensor],
) -> dict[str, Any]:
    """Find first position where greedy token differs between quantized and baseline.

    Returns mean first-disagreement position across all sequences, and agreement rate.
    """
    agreements = 0
    total = 0
    first_positions = []

    for ql, bl in zip(quant_logits, baseline_logits):
        q_tokens = ql.argmax(dim=-1)  # (seq_len,)
        b_tokens = bl.argmax(dim=-1)
        matches = (q_tokens == b_tokens)
        agreements += int(matches.sum().item())
        total += len(matches)
        diff = torch.where(~matches)[0]
        if len(diff) > 0:
            first_positions.append(int(diff[0].item()))

    return {
        "token_agreement_rate": agreements / total if total > 0 else 1.0,
        "first_disagreement_position_mean": (
            float(sum(first_positions) / len(first_positions))
            if first_positions else None
        ),
        "sequences_with_disagreement": len(first_positions),
        "total_sequences": len(quant_logits),
    }


def max_logit_deviation_all(
    quant_logits: list[Tensor],
    baseline_logits: list[Tensor],
) -> float:
    max_dev = 0.0
    for ql, bl in zip(quant_logits, baseline_logits):
        dev = float((ql - bl).abs().max().item())
        if dev > max_dev:
            max_dev = dev
    return max_dev


@torch.inference_mode()
def evaluate_policy(
    model: GPT,
    policy: CachePolicy,
    sequences: list[list[int]],
    baseline_logits_list: list[Tensor],
    device: torch.device,
    *,
    time_sequences: int = 0,
    policy_label: str = "",
) -> dict[str, Any]:
    """Evaluate one policy on a set of sequences, returning all metrics."""
    model.eval()
    qmodel = QuantizedCacheGPT(model, policy)
    num_layers = len(model.blocks)
    num_heads = model.config.num_heads
    head_dim = model.config.embedding_dim // num_heads
    dtype_bytes = next(model.parameters()).element_size()
    seq_len = len(sequences[0]) if sequences else 0

    # Cross-entropy and perplexity on the quantized-cache outputs
    total_ce = 0.0
    n_tokens = 0
    all_kl = []

    quant_logits_list = quantized_incremental_logits(qmodel, sequences, device)

    for quant_logits, baseline_logits, seq in zip(
        quant_logits_list, baseline_logits_list, sequences
    ):
        # CE: predict tokens 1..N given 0..N-1
        targets = torch.tensor(seq[1:], dtype=torch.long)
        pred_logits = quant_logits[:-1]  # (seq_len-1, vocab)
        ce = F.cross_entropy(
            pred_logits.float(), targets, reduction="sum"
        )
        total_ce += float(ce.item())
        n_tokens += len(seq) - 1

        # KL at each position
        kl = categorical_kl(
            quant_logits.unsqueeze(0),
            baseline_logits.unsqueeze(0),
            reduction="none",
        )  # (1, seq_len)
        all_kl.append(kl.squeeze(0))

    mean_ce = total_ce / max(n_tokens, 1)
    kl_tensor = torch.cat(all_kl, dim=0)
    kl_stats = compute_kl_percentiles(kl_tensor)
    disagree = first_disagreement(quant_logits_list, baseline_logits_list)
    max_dev = max_logit_deviation_all(quant_logits_list, baseline_logits_list)

    cache_bytes = policy.exact_cache_bytes(
        batch=1,
        seq_len=seq_len,
        head_dim=head_dim,
        model_dtype_bytes=dtype_bytes,
    )
    fp32_bytes = CachePolicy.all_sixteen(num_layers, num_heads).exact_cache_bytes(
        batch=1,
        seq_len=seq_len,
        head_dim=head_dim,
        model_dtype_bytes=4,
    )
    model_dtype_cache_bytes = CachePolicy.all_sixteen(
        num_layers, num_heads
    ).exact_cache_bytes(
        batch=1,
        seq_len=seq_len,
        head_dim=head_dim,
        model_dtype_bytes=dtype_bytes,
    )
    compression_ratio = (
        fp32_bytes["total_bytes"] / cache_bytes["total_bytes"]
        if cache_bytes["total_bytes"] > 0
        else float("inf")
    )
    compression_factor = (
        model_dtype_cache_bytes["total_bytes"] / cache_bytes["total_bytes"]
        if cache_bytes["total_bytes"] > 0
        else float("inf")
    )

    # Timing measurement
    timing: dict[str, Any] = {}
    if time_sequences > 0 and sequences:
        synchronize(device)
        t0 = time.perf_counter()
        for seq in sequences[:time_sequences]:
            quantized_incremental_logits(qmodel, [seq], device)
        synchronize(device)
        elapsed = time.perf_counter() - t0
        n_timed_tokens = sum(len(s) for s in sequences[:time_sequences])
        timing = {
            "timed_sequences": min(time_sequences, len(sequences)),
            "timed_tokens": n_timed_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": n_timed_tokens / max(elapsed, 1e-9),
        }

    # CUDA peak memory
    cuda_memory: dict[str, Any] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        quantized_incremental_logits(qmodel, sequences[:1], device)
        synchronize(device)
        cuda_memory = {
            "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device)
        }

    return {
        "policy_label": policy_label,
        "bits_per_layer": policy.to_list(),
        "held_out_ce": mean_ce,
        "perplexity": float(torch.tensor(mean_ce).exp().item()),
        **kl_stats,
        **disagree,
        "max_logit_deviation": max_dev,
        "cache_payload_bytes": cache_bytes["payload_bytes"],
        "cache_scale_bytes": cache_bytes["scale_bytes"],
        "cache_total_bytes": cache_bytes["total_bytes"],
        "fp32_cache_bytes": fp32_bytes["total_bytes"],
        "compression_ratio_vs_fp32": compression_ratio,
        "model_dtype_cache_bytes": model_dtype_cache_bytes["total_bytes"],
        "compression_factor_vs_model_dtype": compression_factor,
        # Retained for compatibility with the recorded v1 artifact.
        "full_precision_cache_bytes": model_dtype_cache_bytes["total_bytes"],
        "compression_factor_vs_full_precision": compression_factor,
        "avg_payload_bits": average_storage_bits(policy, dtype_bytes),
        **timing,
        **cuda_memory,
    }


# ---------------------------------------------------------------------------
# Cache-noise adaptation
# ---------------------------------------------------------------------------


def run_noise_adaptation(
    checkpoint_path: Path,
    train_sequences: list[list[int]],
    val_sequences: list[list[int]],
    baseline_logits_list: list[Tensor],
    policy: CachePolicy,
    adaptation_cfg: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run the fixed cache-noise adaptation for multiple seeds.

    For each seed, creates a NoisyKVGPT initialized from the checkpoint and
    trains with next-token CE + KL to the frozen teacher.  The teacher
    receives no gradients.

    Saves adapted checkpoints to output_dir. They are not committed.

    Returns per-seed results (curves, checkpoint hashes, validation metrics).
    """
    seeds: list[int] = list(adaptation_cfg.get("seeds", [1337]))
    num_steps: int = int(adaptation_cfg.get("num_steps", 100))
    kl_beta: float = float(adaptation_cfg.get("kl_beta", 0.1))
    lr: float = float(adaptation_cfg.get("lr", 1e-4))
    noise_bits: int = int(adaptation_cfg.get("noise_bits", 4))
    noise_type: str = str(adaptation_cfg.get("noise_type", "uniform"))
    context_len: int = int(
        adaptation_cfg.get("context_len", len(train_sequences[0]))
    )
    batch_size: int = int(adaptation_cfg.get("batch_size", 1))
    if context_len < 2 or context_len > len(train_sequences[0]):
        raise ValueError(
            "noise_finetune.context_len must be between 2 and the sampled "
            "training sequence length"
        )
    if batch_size < 1:
        raise ValueError("noise_finetune.batch_size must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen teacher once
    teacher = load_checkpoint(checkpoint_path, device=device, dtype=dtype)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    all_seed_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        student = NoisyKVGPT(teacher.config, noise_bits=noise_bits, noise_type=noise_type)
        # Load pretrained weights
        teacher_sd = teacher.state_dict()
        student.load_state_dict(teacher_sd)
        student = student.to(device=device, dtype=dtype).train()

        optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
        curves: list[dict[str, float]] = []

        for step in range(num_steps):
            batch_sequences = [
                train_sequences[
                    (step * batch_size + index) % len(train_sequences)
                ][:context_len]
                for index in range(batch_size)
            ]
            ids = torch.tensor(
                batch_sequences,
                dtype=torch.long,
                device=device,
            )

            student_output = student(ids)
            assert isinstance(student_output, Tensor)
            student_logits = student_output[:, :-1]
            targets = ids[:, 1:]

            with torch.no_grad():
                teacher_logits = teacher(ids)[:, :-1]

            ce = F.cross_entropy(
                student_logits.float().reshape(
                    -1, student_logits.size(-1)
                ),
                targets.reshape(-1),
            )
            kl = categorical_kl(student_logits, teacher_logits)
            loss = ce + kl_beta * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            curves.append(
                {
                    "step": step,
                    "ce": float(ce.item()),
                    "kl": float(kl.item()),
                    "loss": float(loss.item()),
                }
            )

        # Save checkpoint
        ckpt_path = output_dir / f"noise_tuned_seed{seed}.pt"
        student_state = student.state_dict()
        torch.save(
            {
                "model_state_dict": student_state,
                "config": teacher.config.to_dict(),
                "noise_bits": noise_bits,
                "noise_type": noise_type,
                "seed": seed,
                "num_steps": num_steps,
                "kl_beta": kl_beta,
            },
            ckpt_path,
        )
        ckpt_hash = sha256_file(ckpt_path)

        # Validation: convert NoisyKVGPT to GPT temporarily for evaluation
        eval_gpt = GPT(teacher.config).to(device=device, dtype=dtype).eval()
        eval_gpt.load_state_dict(student_state)

        quantized_validation = evaluate_policy(
            eval_gpt,
            policy,
            val_sequences,
            baseline_logits_list,
            device,
            policy_label="noise_adapted_quantized",
        )
        model_dtype_validation = evaluate_policy(
            eval_gpt,
            CachePolicy.all_sixteen(
                len(eval_gpt.blocks), eval_gpt.config.num_heads
            ),
            val_sequences,
            baseline_logits_list,
            device,
            policy_label="noise_adapted_model_dtype",
        )

        all_seed_results.append(
            {
                "seed": seed,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": ckpt_hash,
                "train_curves": curves,
                "validation": quantized_validation,
                "model_dtype_validation": model_dtype_validation,
                # Retained for compatibility with the recorded v1 artifact.
                "full_precision_validation": model_dtype_validation,
            }
        )

    return all_seed_results


# ---------------------------------------------------------------------------
# Full study runner
# ---------------------------------------------------------------------------


def run_study(cfg: dict[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    """Run the full KV-cache quantization study and return the artifact payload."""
    cwd = Path.cwd()
    source_state = git_source_state(cwd)
    seed: int = int(cfg.get("seed", 1337))
    device = select_device(str(cfg.get("device", "auto")))
    configure_deterministic_execution(device)
    dtype_name: str = str(cfg.get("dtype", "float32"))
    dtype = dtype_from_name(dtype_name)

    checkpoint_path = Path(str(cfg["checkpoint"])) if "checkpoint" in cfg else None
    corpus_path = Path(str(cfg["corpus"])) if "corpus" in cfg else None

    calib_start: int = int(cfg.get("calib_start", 0))
    calib_end: int = int(cfg.get("calib_end", 256))
    val_start: int = int(cfg.get("val_start", 256))
    val_end: int = int(cfg.get("val_end", 512))
    seq_len: int = int(cfg.get("seq_len", 32))

    if calib_end > val_start:
        raise ValueError(
            f"calibration end {calib_end} overlaps validation start {val_start}"
        )

    kl_budgets: list[float] = [float(b) for b in cfg.get("kl_budgets", [0.01, 0.05, 0.1])]
    time_sequences_n: int = int(cfg.get("time_sequences", 2))
    # The v1 config uses `noise_finetune`; this is the fixed cache-noise
    # adaptation, and its `selection_*` window was reserved but unused.
    adaptation_cfg: dict[str, Any] | None = cfg.get("noise_finetune")
    if adaptation_cfg is not None:
        has_reserved_start = "selection_start" in adaptation_cfg
        has_reserved_end = "selection_end" in adaptation_cfg
        if has_reserved_start != has_reserved_end:
            raise ValueError(
                "noise_finetune selection_start and selection_end must be set together"
            )
        if has_reserved_start:
            noise_train_start = int(
                adaptation_cfg.get("train_start", calib_start)
            )
            noise_train_end = int(
                adaptation_cfg.get("train_end", calib_end)
            )
            reserved_start = int(adaptation_cfg["selection_start"])
            reserved_end = int(adaptation_cfg["selection_end"])
            if not (
                calib_start
                < calib_end
                <= noise_train_start
                < noise_train_end
                <= reserved_start
                < reserved_end
                <= val_start
                < val_end
            ):
                raise ValueError(
                    "calibration, noise training, reserved, and validation "
                    "windows must be ordered and non-overlapping"
                )

    torch.manual_seed(seed)

    # --- Model ---
    if checkpoint_path is not None:
        model = load_checkpoint(checkpoint_path, device=device, dtype=dtype)
        model_source = str(checkpoint_path)
        checkpoint_hash: str | None = sha256_file(checkpoint_path)
    else:
        model = build_random_model(
            dict(cfg["model_config"]), seed=seed, device=device, dtype=dtype
        )
        model_source = "random"
        checkpoint_hash = None

    num_layers = len(model.blocks)
    num_heads = model.config.num_heads
    head_dim = model.config.embedding_dim // num_heads
    model_dtype_bytes = next(model.parameters()).element_size()
    model_dtype_label = {
        "float32": "all_fp32",
        "float16": "all_fp16",
        "bfloat16": "all_bf16",
    }[dtype_name]

    # --- Corpus ---
    corpus_hash: str | None = None
    noise_train_seqs: list[list[int]] | None = None
    if corpus_path is not None:
        tokens = load_token_shard(corpus_path)
        if int(tokens.max()) >= model.config.vocab_size:
            raise ValueError(
                f"token shard contains id {int(tokens.max())}, but model "
                f"vocabulary size is {model.config.vocab_size}"
            )
        corpus_hash = sha256_file(corpus_path)
        calib_seqs = token_windows(tokens, calib_start, calib_end, seq_len)
        val_seqs = token_windows(tokens, val_start, val_end, seq_len)
        if adaptation_cfg is not None:
            noise_train_start = int(
                adaptation_cfg.get("train_start", calib_start)
            )
            noise_train_end = int(
                adaptation_cfg.get("train_end", calib_end)
            )
            noise_train_seqs = token_windows(
                tokens,
                noise_train_start,
                noise_train_end,
                int(adaptation_cfg.get("context_len", seq_len)),
            )
            if not noise_train_seqs:
                raise ValueError(
                    "cache-noise adaptation window produced no sequences"
                )
    else:
        rng = np.random.default_rng(seed)
        calib_seqs = [
            rng.integers(0, model.config.vocab_size, seq_len).tolist()
            for _ in range(max(1, (calib_end - calib_start) // seq_len))
        ]
        val_seqs = [
            rng.integers(0, model.config.vocab_size, seq_len).tolist()
            for _ in range(max(1, (val_end - val_start) // seq_len))
        ]

    if not calib_seqs:
        raise ValueError("calibration window produced no sequences")
    if not val_seqs:
        raise ValueError("validation window produced no sequences")

    # The reference uses the same incremental cache execution path as every
    # quantized policy, with all heads retained in the model dtype.
    baseline_val = quantized_incremental_logits(
        QuantizedCacheGPT(
            model,
            CachePolicy.all_sixteen(num_layers, num_heads),
        ),
        val_seqs,
        device,
    )

    # --- Sensitivity calibration ---
    sensitivity = compute_sensitivity_matrices(model, calib_seqs, device)

    # --- Policy allocation ---
    named_policies = build_named_policies(
        sensitivity["int8_matrix"],
        sensitivity["int4_matrix"],
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        kl_budgets=kl_budgets,
        model_dtype_bytes=model_dtype_bytes,
        model_dtype_label=model_dtype_label,
        seq_len=seq_len,
    )

    # --- Validation ---
    validation_results = []
    for pol_entry in named_policies:
        label = pol_entry["label"]
        policy = pol_entry["policy"]
        metrics = evaluate_policy(
            model,
            policy,
            val_seqs,
            baseline_val,
            device,
            time_sequences=time_sequences_n,
            policy_label=label,
        )
        # The budget applies to the additive single-head calibration proxy. The
        # gap is diagnostic and is not a held-out KL budget error.
        if pol_entry.get("predicted_kl") is not None:
            metrics["calibration_proxy_gap"] = (
                pol_entry["predicted_kl"] - metrics["mean_kl"]
            )
        validation_results.append(metrics)

    # --- Cache-noise adaptation (optional) ---
    adaptation_results: list[dict[str, Any]] | None = None
    if adaptation_cfg is not None and checkpoint_path is not None:
        noise_output_dir = Path(
            str(cfg.get("noise_output_dir", "artifacts/noise_tuned"))
        )
        # Evaluate the fixed adaptation with the first greedy policy.
        greedy_policies = [
            p for p in named_policies if p["label"].startswith("greedy_")
        ]
        adaptation_policy = (
            greedy_policies[0]["policy"]
            if greedy_policies
            else CachePolicy.all_sixteen(num_layers, num_heads)
        )
        adaptation_results = run_noise_adaptation(
            checkpoint_path=checkpoint_path,
            train_sequences=noise_train_seqs or calib_seqs,
            val_sequences=val_seqs,
            baseline_logits_list=baseline_val,
            policy=adaptation_policy,
            adaptation_cfg=adaptation_cfg,
            device=device,
            dtype=dtype,
            output_dir=noise_output_dir,
        )

    return {
        "schema_version": "kv-cache-study-v1",
        "status": "measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path) if config_path is not None else None,
        "experiment_config": cfg,
        "command": (
            f"python -m src.kv_cache_study --config {config_path}"
            if config_path is not None
            else "python -m src.kv_cache_study"
        ),
        "working_directory": cwd.name,
        "hashes": {
            "checkpoint_sha256": checkpoint_hash,
            "corpus_sha256": corpus_hash,
            "config_sha256": (
                sha256_file(config_path)
                if config_path is not None and config_path.is_file()
                else None
            ),
            "git_state_captured": "study_start",
            **source_state,
        },
        "environment": collect_environment(device),
        "model_config": model.config.to_dict(),
        "model_source": model_source,
        "splits": {
            "calib_start": calib_start,
            "calib_end": calib_end,
            "val_start": val_start,
            "val_end": val_end,
            "seq_len": seq_len,
            "num_calib_sequences": len(calib_seqs),
            "num_val_sequences": len(val_seqs),
            "overlap": False,
            "noise_train_start": (
                int(adaptation_cfg.get("train_start", calib_start))
                if adaptation_cfg is not None
                else None
            ),
            "noise_train_end": (
                int(adaptation_cfg.get("train_end", calib_end))
                if adaptation_cfg is not None
                else None
            ),
            "noise_selection_start": (
                int(adaptation_cfg["selection_start"])
                if adaptation_cfg is not None
                and "selection_start" in adaptation_cfg
                else None
            ),
            "noise_selection_end": (
                int(adaptation_cfg["selection_end"])
                if adaptation_cfg is not None
                and "selection_end" in adaptation_cfg
                else None
            ),
        },
        "sensitivity": sensitivity,
        "calibration_score_definition": (
            "sum of mean output KL values from independent single-head "
            "quantization interventions"
        ),
        "policies": [
            {
                "label": p["label"],
                "bits_per_layer": p["policy"].to_list(),
                "predicted_kl": p["predicted_kl"],
                "avg_payload_bits": p["avg_payload_bits"],
                "steps": p.get("steps", []),
                "cache_bytes": p["policy"].exact_cache_bytes(
                    batch=1,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    model_dtype_bytes=model_dtype_bytes,
                ),
                "scale_overhead_bytes_per_token": (
                    p["policy"].scale_overhead_bytes_per_token()
                ),
            }
            for p in named_policies
        ],
        "validation": {
            "window": {"start": val_start, "end": val_end, "seq_len": seq_len},
            "results": validation_results,
        },
        # Retained for compatibility with the recorded v1 artifact.
        "noise_fine_tuning": (
            {
                "config": adaptation_cfg,
                "seeds": adaptation_results,
            }
            if adaptation_results is not None
            else None
        ),
        "caveats": [
            "Persistent cache bytes differ from transient dequantization peak "
            "memory: during a decode step, model-dtype K and V are briefly "
            "reconstructed in memory even for low-bit heads.",
            "Sensitivity computation assumes per-head independence. Cross-head "
            "interactions are not modeled.",
            "Greedy marginal-KL-per-byte allocation is not globally optimal.",
            "The KL budget applies to an additive single-head calibration proxy, "
            "not a bound on held-out joint output KL. Cross-head interactions can "
            "make the proxy differ from measured KL.",
            "The fixed three-seed cache-noise adaptation did not establish a "
            "robustness gain. Per-head mixed precision is prior art.",
        ],
    }


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def write_artifact(
    payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    results_path.write_bytes(encoded)
    file_hash = sha256_bytes(encoded)
    hashes_path = output_dir / "hashes.json"
    hashes_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": [
                    {"path": str(results_path), "sha256": file_hash}
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "results": str(results_path),
        "results_sha256": file_hash,
        "hashes": str(hashes_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KV-cache quantization sensitivity, policy allocation, and validation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON config file",
    )
    parser.add_argument("--output-dir", type=Path, help="Override artifact output dir")
    parser.add_argument("--device", help="Override device (cpu / cuda / auto)")
    parser.add_argument("--checkpoint", type=Path, help="Override checkpoint path")
    parser.add_argument("--corpus", type=Path, help="Override corpus shard path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())

    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    if args.device is not None:
        cfg["device"] = args.device
    if args.checkpoint is not None:
        cfg["checkpoint"] = str(args.checkpoint)
    if args.corpus is not None:
        cfg["corpus"] = str(args.corpus)

    payload = run_study(cfg, config_path=args.config)

    output_dir = Path(str(cfg.get("output_dir", "artifacts/kv-cache-study")))
    paths = write_artifact(payload, output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "results_sha256": paths["results_sha256"],
                "num_policies": len(payload["policies"]),
                "calib_sequences": payload["splits"]["num_calib_sequences"],
                "val_sequences": payload["splits"]["num_val_sequences"],
                "environment": payload["environment"],
                "sensitivity_shape": [
                    payload["sensitivity"]["num_layers"],
                    payload["sensitivity"]["num_heads"],
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor

from src.model import GPT, GPTConfig


@dataclass(frozen=True)
class StepDeviation:
    step_index: int
    generation_position: int
    prefix_length: int
    max_abs_logit_deviation: float
    max_relative_logit_deviation: float
    uncached_next_token: int
    cached_next_token: int
    flipped: bool


@dataclass(frozen=True)
class PromptEquivalence:
    prompt_index: int
    prompt_text: str | None
    prompt_tokens: list[int]
    prompt_length: int
    generated_tokens: list[int]
    generated_new_tokens: list[int]
    first_flip_step_index: int | None
    first_flip_generation_position: int | None
    max_abs_logit_deviation: float
    max_relative_logit_deviation: float
    steps: list[StepDeviation]


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype {name}")


def dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    raise ValueError(f"unsupported dtype {dtype}")


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def max_logit_deviation(
    cached_logits: Tensor,
    uncached_logits: Tensor,
    *,
    relative_epsilon: float,
) -> tuple[float, float]:
    cached = cached_logits.detach().to(torch.float32)
    uncached = uncached_logits.detach().to(torch.float32)
    absolute = (cached - uncached).abs()
    denominator = torch.maximum(cached.abs().max(), uncached.abs().max()).clamp_min(
        relative_epsilon
    )
    max_absolute = absolute.max()
    return float(max_absolute.item()), float((max_absolute / denominator).item())


@torch.inference_mode()
def measure_prompt_equivalence(
    model: GPT,
    prompt_tokens: list[int],
    horizon: int,
    *,
    prompt_index: int = 0,
    prompt_text: str | None = None,
    relative_epsilon: float = 1e-6,
) -> PromptEquivalence:
    """Measure cached and uncached logits on one shared greedy trajectory."""
    if not prompt_tokens:
        raise ValueError("prompt_tokens cannot be empty")
    if horizon < 0:
        raise ValueError("horizon cannot be negative")
    if len(prompt_tokens) + horizon > model.config.context_len:
        raise ValueError("prompt plus horizon exceeds model context length")
    if relative_epsilon <= 0:
        raise ValueError("relative_epsilon must be positive")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    tokens = list(prompt_tokens)
    steps: list[StepDeviation] = []
    first_flip_step: int | None = None

    try:
        if horizon == 0:
            return PromptEquivalence(
                prompt_index=prompt_index,
                prompt_text=prompt_text,
                prompt_tokens=list(prompt_tokens),
                prompt_length=len(prompt_tokens),
                generated_tokens=tokens,
                generated_new_tokens=[],
                first_flip_step_index=None,
                first_flip_generation_position=None,
                max_abs_logit_deviation=0.0,
                max_relative_logit_deviation=0.0,
                steps=[],
            )

        cached_input = torch.tensor(
            [tokens], dtype=torch.long, device=device
        )
        cached_logits, cache = model(cached_input, use_cache=True)

        for step_index in range(horizon):
            uncached_input = torch.tensor(
                [tokens], dtype=torch.long, device=device
            )
            uncached_logits = model(uncached_input)[:, -1, :]
            step_cached_logits = cached_logits[:, -1, :]
            max_abs, max_relative = max_logit_deviation(
                step_cached_logits,
                uncached_logits,
                relative_epsilon=relative_epsilon,
            )
            uncached_next_token = int(
                uncached_logits.argmax(dim=-1).item()
            )
            cached_next_token = int(
                step_cached_logits.argmax(dim=-1).item()
            )
            flipped = uncached_next_token != cached_next_token
            if flipped and first_flip_step is None:
                first_flip_step = step_index
            steps.append(
                StepDeviation(
                    step_index=step_index,
                    generation_position=step_index + 1,
                    prefix_length=len(tokens),
                    max_abs_logit_deviation=max_abs,
                    max_relative_logit_deviation=max_relative,
                    uncached_next_token=uncached_next_token,
                    cached_next_token=cached_next_token,
                    flipped=flipped,
                )
            )

            tokens.append(uncached_next_token)
            if step_index + 1 < horizon:
                next_input = torch.tensor(
                    [[uncached_next_token]],
                    dtype=torch.long,
                    device=device,
                )
                cached_logits, cache = model(
                    next_input,
                    past_key_values=cache,
                    use_cache=True,
                )

        return PromptEquivalence(
            prompt_index=prompt_index,
            prompt_text=prompt_text,
            prompt_tokens=list(prompt_tokens),
            prompt_length=len(prompt_tokens),
            generated_tokens=tokens,
            generated_new_tokens=tokens[len(prompt_tokens) :],
            first_flip_step_index=first_flip_step,
            first_flip_generation_position=(
                None if first_flip_step is None else first_flip_step + 1
            ),
            max_abs_logit_deviation=max(
                step.max_abs_logit_deviation for step in steps
            ),
            max_relative_logit_deviation=max(
                step.max_relative_logit_deviation for step in steps
            ),
            steps=steps,
        )
    finally:
        model.train(was_training)


def summarize_prompt_results(
    prompt_results: list[PromptEquivalence],
) -> dict[str, int | float | None]:
    if not prompt_results:
        raise ValueError("at least one prompt result is required")
    flip_positions = [
        result.first_flip_generation_position
        for result in prompt_results
        if result.first_flip_generation_position is not None
    ]
    prompt_max_abs = [
        result.max_abs_logit_deviation for result in prompt_results
    ]
    prompt_max_relative = [
        result.max_relative_logit_deviation for result in prompt_results
    ]
    return {
        "prompt_count": len(prompt_results),
        "prompts_with_flip": len(flip_positions),
        "first_flip_generation_position_min": (
            min(flip_positions) if flip_positions else None
        ),
        "first_flip_generation_position_max": (
            max(flip_positions) if flip_positions else None
        ),
        "max_abs_logit_deviation": max(prompt_max_abs),
        "mean_prompt_max_abs_logit_deviation": sum(prompt_max_abs)
        / len(prompt_max_abs),
        "max_relative_logit_deviation": max(prompt_max_relative),
        "mean_prompt_max_relative_logit_deviation": sum(prompt_max_relative)
        / len(prompt_max_relative),
    }


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> GPT:
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = GPTConfig(**payload["config"])
    model = GPT(config)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device=device, dtype=dtype).eval()


def build_random_model(
    config_payload: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> GPT:
    torch.manual_seed(seed)
    config = GPTConfig(**config_payload)
    return GPT(config).to(device=device, dtype=dtype).eval()


def load_prompt_specs(
    config: dict[str, Any],
) -> list[tuple[str | None, list[int]]]:
    prompts: list[tuple[str | None, list[int]]] = []
    for token_prompt in config.get("prompt_tokens", []):
        prompts.append((None, [int(token) for token in token_prompt]))

    prompt_texts = config.get("prompts", [])
    if prompt_texts:
        from data.bpe_tokenizer import encode, load_tokenizer

        merges, vocab = load_tokenizer()
        for prompt_text in prompt_texts:
            prompts.append((str(prompt_text), encode(str(prompt_text), merges, vocab)))

    if not prompts:
        raise ValueError("config must define prompts or prompt_tokens")
    return prompts


def run_study(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    seed = int(config.get("seed", 1337))
    horizon = int(config["horizon"])
    relative_epsilon = float(config.get("relative_epsilon", 1e-6))
    device = select_device(str(config.get("device", "auto")))
    dtype_names = [str(name) for name in config["dtypes"]]
    prompt_specs = load_prompt_specs(config)
    checkpoint = (
        Path(str(config["checkpoint"]))
        if config.get("checkpoint") is not None
        else None
    )

    dtype_results: list[dict[str, Any]] = []
    for name in dtype_names:
        dtype = dtype_from_name(name)
        if checkpoint is None:
            model = build_random_model(
                dict(config["model_config"]),
                seed=seed,
                device=device,
                dtype=dtype,
            )
            model_state = "random-initialization"
        else:
            model = load_checkpoint_model(
                checkpoint, device=device, dtype=dtype
            )
            model_state = str(checkpoint)

        started = time.perf_counter()
        prompt_results = [
            measure_prompt_equivalence(
                model,
                prompt_tokens,
                horizon,
                prompt_index=prompt_index,
                prompt_text=prompt_text,
                relative_epsilon=relative_epsilon,
            )
            for prompt_index, (prompt_text, prompt_tokens) in enumerate(
                prompt_specs
            )
        ]
        synchronize(device)
        dtype_results.append(
            {
                "dtype": dtype_name(dtype),
                "model_state": model_state,
                "model_config": model.config.to_dict(),
                "elapsed_seconds": time.perf_counter() - started,
                "summary": summarize_prompt_results(prompt_results),
                "prompts": [
                    {
                        **asdict(result),
                        "steps": [
                            asdict(step) for step in result.steps
                        ],
                    }
                    for result in prompt_results
                ],
            }
        )
        del model

    summary = {
        result["dtype"]: result["summary"] for result in dtype_results
    }
    return {
        "status": "measured-local-development",
        "study": "kv-cache numerical equivalence under fp32 vs bf16",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path) if config_path is not None else None,
        "experiment_config": config,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "seed": seed,
        "prompt_count": len(prompt_specs),
        "horizon": horizon,
        "relative_epsilon": relative_epsilon,
        "summary": summary,
        "results": dtype_results,
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_artifacts(
    payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    hashes_path = output_dir / "hashes.json"

    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    results_path.write_bytes(encoded)
    hashes = {
        "status": "measured-local-development",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "path": str(results_path),
                "sha256": sha256_bytes(encoded),
            }
        ],
    }
    hashes_path.write_text(json.dumps(hashes, indent=2) + "\n")
    return {
        "results": str(results_path),
        "results_sha256": hashes["files"][0]["sha256"],
        "hashes": str(hashes_path),
    }


def load_json_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure KV-cache numerical equivalence"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kv_equivalence.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--horizon", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.checkpoint is not None:
        config["checkpoint"] = str(args.checkpoint)
    if args.device is not None:
        config["device"] = args.device
    if args.horizon is not None:
        config["horizon"] = args.horizon

    payload = run_study(config, config_path=args.config)
    output_dir = Path(
        str(config.get("output_dir", "artifacts/kv-equivalence-dev-local"))
    )
    artifact_paths = write_artifacts(payload, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "results_sha256": artifact_paths["results_sha256"],
                "summary": payload["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

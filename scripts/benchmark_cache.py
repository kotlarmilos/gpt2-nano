from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import platform
from pathlib import Path
import statistics
import time

import torch

from src.model import GPT, GPTConfig, generate


@dataclass(frozen=True)
class Measurement:
    use_cache: bool
    seconds: float
    tokens_per_second: float


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(
    model: GPT,
    prompt: list[int],
    new_tokens: int,
    use_cache: bool,
    repeats: int,
) -> Measurement:
    durations = []
    for repeat in range(repeats):
        synchronize(model.input_embedding.weight.device)
        started = time.perf_counter()
        generate(
            model,
            prompt,
            new_tokens,
            use_cache=use_cache,
            do_sample=False,
        )
        synchronize(model.input_embedding.weight.device)
        durations.append(time.perf_counter() - started)
    seconds = statistics.median(durations)
    return Measurement(use_cache, seconds, new_tokens / seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32])
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", default="artifacts/cache-smoke.json")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if args.checkpoint is None:
        config = GPTConfig(
            vocab_size=256,
            context_len=max(args.prompt_lengths) + args.new_tokens,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
        )
        torch.manual_seed(1337)
        model = GPT(config).to(device).eval()
        model_state = "random-initialization"
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        config = GPTConfig(**checkpoint["config"])
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model_state = str(args.checkpoint)

    results = []
    for prompt_length in args.prompt_lengths:
        if prompt_length + args.new_tokens > config.context_len:
            raise ValueError(
                f"prompt {prompt_length} plus generation {args.new_tokens} exceeds "
                f"context {config.context_len}"
            )
        torch.manual_seed(1337 + prompt_length)
        prompt = torch.randint(
            0, config.vocab_size, (prompt_length,), device="cpu"
        ).tolist()
        results.append(
            {
                "prompt_length": prompt_length,
                "full_sequence_cache_bytes": (
                    2
                    * config.num_layers
                    * (prompt_length + args.new_tokens)
                    * config.embedding_dim
                    * model.input_embedding.weight.element_size()
                ),
                "measurements": [
                    asdict(
                        measure(
                            model,
                            prompt,
                            args.new_tokens,
                            use_cache,
                            args.repeats,
                        )
                    )
                    for use_cache in (False, True)
                ],
            }
        )
    payload = {
        "status": (
            "measured-local-smoke"
            if args.checkpoint is None
            else "measured-checkpoint"
        ),
        "model_state": model_state,
        "config": config.to_dict(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

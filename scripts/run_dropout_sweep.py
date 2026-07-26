from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/publication.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/dropout")
    )
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    dropouts = (
        [args.dropout] if args.dropout is not None else config["dropout_sweep"]
    )
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    if any(dropout not in config["dropout_sweep"] for dropout in dropouts):
        raise ValueError("--dropout must be one of the configured sweep values")
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("--seed must be one of the configured seeds")
    for dropout in dropouts:
        for seed in seeds:
            run_name = f"dropout-{dropout:.2f}-seed-{seed}"
            command = [
                sys.executable,
                "-m",
                "src.gpt",
                "--context-len",
                str(config["context_len"]),
                "--embedding-dim",
                str(config["embedding_dim"]),
                "--num-layers",
                str(config["num_layers"]),
                "--num-heads",
                str(config["num_heads"]),
                "--batch-size",
                str(config["batch_size"]),
                "--num-steps",
                str(config["num_steps"]),
                "--dropout",
                str(dropout),
                "--seed",
                str(seed),
                "--checkpoint-dir",
                str(args.output_dir / run_name / "checkpoints"),
                "--metrics-output",
                str(args.output_dir / run_name / "metrics.json"),
            ]
            commands.append(command)

    plan_path = args.output_dir / "commands.json"
    plan_path.write_text(json.dumps(commands, indent=2) + "\n")
    print(f"Wrote {len(commands)} commands to {plan_path}")
    if args.dry_run:
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

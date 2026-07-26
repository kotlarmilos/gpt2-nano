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
        "--output-dir", type=Path, default=Path("artifacts/kl")
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=Path("checkpoints/final.pt"),
    )
    parser.add_argument("--beta", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    betas = [args.beta] if args.beta is not None else config["kl_beta_sweep"]
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    if any(beta not in config["kl_beta_sweep"] for beta in betas):
        raise ValueError("--beta must be one of the configured sweep values")
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("--seed must be one of the configured seeds")
    if not args.reference_checkpoint.is_file() and not args.dry_run:
        raise FileNotFoundError(args.reference_checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for beta in betas:
        for seed in seeds:
            run_name = f"beta-{beta:.2f}-seed-{seed}"
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
                str(config["kl_num_steps"]),
                "--dropout",
                str(config.get("kl_dropout", 0.0)),
                "--seed",
                str(seed),
                "--initial-checkpoint",
                str(args.reference_checkpoint),
                "--reference-checkpoint",
                str(args.reference_checkpoint),
                "--kl-beta",
                str(beta),
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

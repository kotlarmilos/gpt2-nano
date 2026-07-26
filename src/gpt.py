from __future__ import annotations

import argparse
from glob import glob
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from data.bpe_tokenizer import (
    SHARDS_PATH,
    decode,
    encode,
    load_shard,
    load_tokenizer,
)
from src.model import GPT, GPTConfig, generate
from src.objectives import categorical_kl, kl_regularized_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gpt2-nano")
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--generate-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "config": model.config.to_dict(),
        },
        output,
    )
    print(f"Checkpoint saved: {output}")


def load_checkpoint_model(path: Path, device: torch.device) -> GPT:
    payload = torch.load(path, map_location=device, weights_only=True)
    try:
        config = GPTConfig(**payload["config"])
        state_dict = payload["model_state_dict"]
    except KeyError as error:
        raise ValueError(
            f"Checkpoint {path} requires config and model_state_dict"
        ) from error
    model = GPT(config).to(device)
    model.load_state_dict(state_dict)
    return model


def structural_config(config: GPTConfig) -> tuple[int, int, int, int, int]:
    return (
        config.vocab_size,
        config.context_len,
        config.embedding_dim,
        config.num_layers,
        config.num_heads,
    )


def language_model_objective(
    model: GPT,
    inputs: torch.Tensor,
    reference: GPT | None,
    kl_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute next-token task loss with an optional frozen-reference KL term."""
    if kl_beta < 0:
        raise ValueError("kl_beta cannot be negative")
    if kl_beta > 0 and reference is None:
        raise ValueError("a reference model is required when kl_beta is positive")
    shifted_logits = model(inputs)[:, :-1]
    targets = inputs[:, 1:]
    if reference is None:
        task_loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            targets.reshape(-1),
        )
        return task_loss, task_loss, shifted_logits.new_zeros(())
    with torch.no_grad():
        reference_logits = reference(inputs)[:, :-1]
    return kl_regularized_loss(
        shifted_logits,
        targets,
        reference_logits,
        kl_beta,
    )


def main() -> None:
    args = parse_args()
    if args.kl_beta < 0:
        raise ValueError("--kl-beta cannot be negative")
    if args.kl_beta > 0 and args.reference_checkpoint is None:
        raise ValueError("--reference-checkpoint is required when --kl-beta is positive")
    torch.manual_seed(args.seed)
    device = select_device()

    merges, vocab = load_tokenizer()
    vocab_size = len(vocab)
    shard_paths = sorted(glob(os.path.join(SHARDS_PATH, "*.bin")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No token shards found under {SHARDS_PATH}. Run python -m "
            "data.bpe_tokenizer first."
        )
    tokens = np.concatenate([load_shard(path) for path in shard_paths])
    split = int(len(tokens) * 0.9)
    train_tokens, val_tokens = tokens[:split], tokens[split:]

    config = GPTConfig(
        vocab_size=vocab_size,
        context_len=args.context_len,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    if args.initial_checkpoint is not None:
        initial = load_checkpoint_model(args.initial_checkpoint, device)
        if structural_config(initial.config) != structural_config(config):
            raise ValueError("initial checkpoint architecture does not match training config")
        model.load_state_dict(initial.state_dict())
        del initial
    reference: GPT | None = None
    if args.reference_checkpoint is not None:
        reference = load_checkpoint_model(args.reference_checkpoint, device).eval()
        if structural_config(reference.config) != structural_config(config):
            raise ValueError("reference checkpoint architecture does not match training config")
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"=== Training GPT ({device}) ===")
    print(json.dumps(config.to_dict(), sort_keys=True))
    print(
        f"Batch: {args.batch_size} | Params: {parameter_count:,} | "
        f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}"
    )

    started = time.time()
    metrics: list[dict[str, float | int]] = []
    loss = torch.tensor(float("nan"))
    for step in range(args.num_steps):
        batch_tokens = args.batch_size * config.context_len
        start = (step * batch_tokens) % (len(train_tokens) - batch_tokens)
        batch = train_tokens[start : start + batch_tokens]
        inputs = torch.tensor(
            batch.reshape(args.batch_size, config.context_len),
            dtype=torch.long,
            device=device,
        )

        loss, task_loss, kl_loss = language_model_objective(
            model,
            inputs,
            reference,
            args.kl_beta,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            model.eval()
            with torch.no_grad():
                train_logits = model(inputs)
                train_eval_loss = F.cross_entropy(
                    train_logits[:, :-1].reshape(-1, vocab_size),
                    inputs[:, 1:].reshape(-1),
                )
                val_batch = val_tokens[:batch_tokens]
                val_inputs = torch.tensor(
                    val_batch.reshape(args.batch_size, config.context_len),
                    dtype=torch.long,
                    device=device,
                )
                val_logits = model(val_inputs)
                val_loss = F.cross_entropy(
                    val_logits[:, :-1].reshape(-1, vocab_size),
                    val_inputs[:, 1:].reshape(-1),
                )
                train_kl = (
                    categorical_kl(train_logits[:, :-1], reference(inputs)[:, :-1])
                    if reference is not None
                    else train_logits.new_zeros(())
                )
                val_kl = (
                    categorical_kl(val_logits[:, :-1], reference(val_inputs)[:, :-1])
                    if reference is not None
                    else val_logits.new_zeros(())
                )
            model.train()
            print(
                f"step {step:>5d}/{args.num_steps} | "
                f"loss: {train_eval_loss.item():.4f} | "
                f"val: {val_loss.item():.4f} | "
                f"kl: {val_kl.item():.4f} | "
                f"ppl: {math.exp(train_eval_loss.item()):.1f} | "
                f"val_ppl: {math.exp(val_loss.item()):.1f} | "
                f"{time.time() - started:.1f}s"
            )
            metrics.append(
                {
                    "step": step,
                    "optimization_loss": loss.item(),
                    "task_loss": task_loss.item(),
                    "optimization_kl": kl_loss.item(),
                    "train_loss": train_eval_loss.item(),
                    "validation_loss": val_loss.item(),
                    "train_kl": train_kl.item(),
                    "validation_kl": val_kl.item(),
                    "elapsed_seconds": time.time() - started,
                }
            )

        if step > 0 and step % args.generate_every == 0:
            prompt_tokens = encode("The ", merges, vocab)
            generated = generate(
                model,
                prompt_tokens,
                max_new_tokens=50,
                use_cache=True,
            )
            print(f"\n--- Generated (step {step}) ---\n{decode(generated, vocab)}\n---")

        if step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                loss.item(),
                Path(args.checkpoint_dir) / f"step_{step}.pt",
            )

    save_checkpoint(
        model,
        optimizer,
        args.num_steps,
        loss.item(),
        Path(args.checkpoint_dir) / "final.pt",
    )
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(
                {
                    "status": "measured",
                    "device": str(device),
                    "seed": args.seed,
                    "config": config.to_dict(),
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "kl_beta": args.kl_beta,
                    "initial_checkpoint": (
                        str(args.initial_checkpoint)
                        if args.initial_checkpoint is not None
                        else None
                    ),
                    "reference_checkpoint": (
                        str(args.reference_checkpoint)
                        if args.reference_checkpoint is not None
                        else None
                    ),
                    "num_steps": args.num_steps,
                    "metrics": metrics,
                    "total_seconds": time.time() - started,
                },
                indent=2,
            )
            + "\n"
        )
    print(f"\n=== Done in {time.time() - started:.1f}s ===")


if __name__ == "__main__":
    main()

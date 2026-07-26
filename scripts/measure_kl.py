from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.model import GPT, GPTConfig
from src.objectives import categorical_kl


def load_model(path: Path, device: torch.device) -> GPT:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=Path)
    parser.add_argument("reference_checkpoint", type=Path)
    parser.add_argument("token_batch", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/kl.json"))
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = load_model(args.model_checkpoint, device)
    reference = load_model(args.reference_checkpoint, device)
    token_batch = torch.tensor(
        json.loads(args.token_batch.read_text()), dtype=torch.long, device=device
    )
    with torch.inference_mode():
        model_logits = model(token_batch)
        reference_logits = reference(token_batch)
        per_token = categorical_kl(
            model_logits, reference_logits, reduction="none"
        )
    payload = {
        "status": "measured",
        "model_checkpoint": str(args.model_checkpoint),
        "reference_checkpoint": str(args.reference_checkpoint),
        "mean_token_kl": per_token.mean().item(),
        "max_token_kl": per_token.max().item(),
        "tokens": per_token.numel(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

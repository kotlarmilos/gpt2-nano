from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def categorical_kl(
    model_logits: Tensor,
    reference_logits: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    if model_logits.shape != reference_logits.shape:
        raise ValueError("model and reference logits must have identical shapes")
    model_log_probs = F.log_softmax(model_logits.float(), dim=-1)
    reference_log_probs = F.log_softmax(
        reference_logits.detach().float(), dim=-1
    )
    pointwise = model_log_probs.exp() * (model_log_probs - reference_log_probs)
    per_distribution = pointwise.sum(dim=-1)
    if reduction == "none":
        return per_distribution
    if reduction == "sum":
        return per_distribution.sum()
    if reduction == "mean":
        return per_distribution.mean()
    if reduction == "batchmean":
        return per_distribution.sum() / model_logits.size(0)
    raise ValueError(f"unsupported reduction: {reduction}")


def kl_regularized_loss(
    model_logits: Tensor,
    targets: Tensor,
    reference_logits: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if beta < 0:
        raise ValueError("beta cannot be negative")
    task_loss = F.cross_entropy(
        model_logits.float().reshape(-1, model_logits.size(-1)),
        targets.reshape(-1),
    )
    kl_loss = categorical_kl(model_logits, reference_logits)
    return task_loss + beta * kl_loss, task_loss, kl_loss

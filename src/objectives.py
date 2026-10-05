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

import torch
import torch.nn as nn

_bce = nn.BCEWithLogitsLoss()

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum((1,2,3))
    denom = probs.sum((1,2,3)) + target.sum((1,2,3)) + eps
    return 1 - ((2*inter + eps) / denom).mean()

def bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _bce(logits, target) + dice_loss(logits, target)

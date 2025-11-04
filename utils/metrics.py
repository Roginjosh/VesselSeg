import torch

@torch.no_grad()
def dice_coef(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * target).sum((1, 2, 3))
    denom = preds.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return ((2 * inter + eps) / denom).mean()

@torch.no_grad()
def iou_coef(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * target).sum((1, 2, 3))
    union = (preds + target - preds * target).sum((1, 2, 3)) + eps
    return (inter / union).mean()

# losses.py
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def dice_iou(logits, targets, thresh=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()

    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds*targets).sum(dim=(1,2,3))
    iou = (inter + eps) / (union + eps)

    dice = (2*inter + eps) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps)
    return dice.mean().item(), iou.mean().item()

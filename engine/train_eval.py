from __future__ import annotations
from typing import Tuple
from tqdm import tqdm
import torch

from utils.losses import bce_dice
from utils.metrics import dice_coef, iou_coef

def run_epoch(model, loader, optimizer, device, train: bool = True, amp: bool = True) -> Tuple[float,float,float]:
    model.train(train)
    scaler = torch.amp.GradScaler('cuda', enabled=(amp and torch.cuda.is_available()))
    total_loss = total_dice = total_iou = 0.0
    n = 0

    loop = tqdm(loader, total=len(loader), leave=False)
    loop.set_description("Train" if train else "Val")
    for imgs, masks in loop:
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.amp.autocast('cuda', enabled=(amp and torch.cuda.is_available())):
            logits = model(imgs)
            loss = bce_dice(logits, masks)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            dice = dice_coef(logits, masks).item()
            iou  = iou_coef(logits, masks).item()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_dice += dice * bs
        total_iou  += iou  * bs
        n += bs
        loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.3f}", iou=f"{iou:.3f}")

    return total_loss / n, total_dice / n, total_iou / n

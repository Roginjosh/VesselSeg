from __future__ import annotations
from typing import Tuple
from tqdm import tqdm
import torch

from utils.losses import bce_dice
from utils.metrics import dice_coef, iou_coef

def run_epoch(model, loader, optimizer, device, train: bool = True, amp: bool = True) -> Tuple[float, float, float]:
    model.train(train)

    # AMP/Scaler policy by device
    use_cuda_amp = amp and (device.type == "cuda")
    use_autocast = amp and (device.type in {"cuda", "mps"})

    # Use the CUDA amp API for scaler (only on CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    total_loss = total_dice = total_iou = 0.0
    n = 0

    loop = tqdm(loader, total=len(loader), leave=False)
    loop.set_description("Train" if train else "Val")
    for imgs, masks in loop:
        imgs, masks = imgs.to(device), masks.to(device)

        # autocast works for cuda and (limited) mps; disabled on cpu
        with torch.autocast(device_type=device.type, enabled=use_autocast):
            logits = model(imgs)
            loss = bce_dice(logits, masks)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if use_cuda_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

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

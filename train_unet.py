# train_unet.py
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from dataset import ISICKeySegDataset
from unet import UNet
from losses import DiceLoss, dice_iou

def train_one_epoch(model, loader, opt, bce, dice, device, epoch=None, epochs=None):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Train {epoch}/{epochs}" if epoch and epochs else "Train", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)

        loss = 0.5 * bce(logits, masks) + 0.5 * dice(logits, masks)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs

        # live update
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, bce, dice, device, epoch=None, epochs=None):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Val   {epoch}/{epochs}" if epoch and epochs else "Val", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)

        loss = 0.5 * bce(logits, masks) + 0.5 * dice(logits, masks)
        d, i = dice_iou(logits, masks)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_dice += d * bs
        total_iou  += i * bs
        n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.3f}", iou=f"{i:.3f}")

    return total_loss / n, total_dice / n, total_iou / n

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    ds = ISICKeySegDataset("data/imgs", "data/masks", img_size=256, augment=True)
    val_frac = 0.2
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_channels=3, out_channels=1, base=64).to(device)

    bce = torch.nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best = -1.0

    epochs = 30
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, bce, dice, device, epoch, epochs)
        va_loss, va_dice, va_iou = eval_one_epoch(model, val_loader, bce, dice, device, epoch, epochs)

        print(f"E{epoch:02d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} dice {va_dice:.4f} iou {va_iou:.4f}")

        if va_dice > best:
            best = va_dice
            torch.save(model.state_dict(), "unet_best.pt")

if __name__ == "__main__":
    main()

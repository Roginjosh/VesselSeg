from pathlib import Path

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW

from src.dataset import VesselSegDataset
from src.unet import UNet
from src.losses import DiceLoss, dice_iou


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"

SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
VAL_FRAC = 0.20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def train_one_epoch(
    model,
    loader,
    opt,
    bce,
    dice,
    device,
    epoch=None,
    epochs=None,
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(
        loader,
        desc=f"Train {epoch}/{epochs}" if epoch and epochs else "Train",
        leave=False,
    )

    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        opt.zero_grad(set_to_none=True)

        logits = model(imgs)

        loss = (
            0.5 * bce(logits, masks)
            + 0.5 * dice(logits, masks)
        )

        loss.backward()
        opt.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    bce,
    dice,
    device,
    epoch=None,
    epochs=None,
):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0

    pbar = tqdm(
        loader,
        desc=f"Val   {epoch}/{epochs}" if epoch and epochs else "Val",
        leave=False,
    )

    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)

        loss = (
            0.5 * bce(logits, masks)
            + 0.5 * dice(logits, masks)
        )

        d, i = dice_iou(logits, masks)

        bs = imgs.size(0)

        total_loss += loss.item() * bs
        total_dice += d * bs
        total_iou += i * bs
        n += bs

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{d:.3f}",
            iou=f"{i:.3f}",
        )

    return (
        total_loss / n,
        total_dice / n,
        total_iou / n,
    )


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    #
    # Create two versions of the same verified dataset.
    # Training gets augmentation; validation does not.
    #
    train_full = VesselSegDataset(
        csv_path=DATASET_CSV,
        img_size=IMG_SIZE,
        augment=True,
        project_root=PROJECT_ROOT,
    )

    val_full = VesselSegDataset(
        csv_path=DATASET_CSV,
        img_size=IMG_SIZE,
        augment=False,
        project_root=PROJECT_ROOT,
    )

    #
    # Generate one reproducible set of indices.
    #
    n_total = len(train_full)
    n_val = int(n_total * VAL_FRAC)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        n_total,
        generator=generator,
    ).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = Subset(
        train_full,
        train_indices,
    )

    val_ds = Subset(
        val_full,
        val_indices,
    )

    print(f"Total samples: {n_total}")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=64,
    ).to(device)

    bce = torch.nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    opt = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = MODEL_DIR / "unet_best.pt"

    best_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            bce,
            dice,
            device,
            epoch,
            EPOCHS,
        )

        val_loss, val_dice, val_iou = eval_one_epoch(
            model,
            val_loader,
            bce,
            dice,
            device,
            epoch,
            EPOCHS,
        )

        print(
            f"E{epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"dice {val_dice:.4f} | "
            f"iou {val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print(
                f"  Saved new best model "
                f"(Dice {best_dice:.4f})"
            )

    print()
    print("Training complete.")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Best model: {best_model_path}")


if __name__ == "__main__":
    main()
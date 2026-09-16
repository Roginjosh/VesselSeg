from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW

from src.dataset import VesselSegDataset
from src.unet import UNet
from src.losses import DiceLoss, dice_iou


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Permanent leakage-safe split file
DATASET_CSV = PROJECT_ROOT / "data" / "dataset_with_split.csv"

MODEL_DIR = PROJECT_ROOT / "models"

SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 3.0


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

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Read the permanent split assignments
    # ---------------------------------------------------------

    split_df = pd.read_csv(
        DATASET_CSV,
        dtype=str,
    ).fillna("")

    if "split" not in split_df.columns:
        raise RuntimeError(
            f"{DATASET_CSV} does not contain a 'split' column."
        )

    valid_splits = {"train", "val", "test"}

    unknown_splits = set(split_df["split"].unique()) - valid_splits

    if unknown_splits:
        raise RuntimeError(
            f"Unknown split values found: {unknown_splits}"
        )

    train_indices = split_df.index[
        split_df["split"] == "train"
    ].tolist()

    val_indices = split_df.index[
        split_df["split"] == "val"
    ].tolist()

    test_indices = split_df.index[
        split_df["split"] == "test"
    ].tolist()

    # ---------------------------------------------------------
    # Create two versions of the dataset.
    #
    # Training:
    #   augmentation ON
    #
    # Validation:
    #   augmentation OFF
    #
    # Both read the exact same CSV so their row indices match
    # the permanent split assignments.
    # ---------------------------------------------------------

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

    # Sanity check
    if len(train_full) != len(split_df):
        raise RuntimeError(
            "Dataset length does not match split CSV length."
        )

    # ---------------------------------------------------------
    # Use the permanent split assignments
    # ---------------------------------------------------------

    train_ds = Subset(
        train_full,
        train_indices,
    )

    val_ds = Subset(
        val_full,
        val_indices,
    )

    print()
    print(f"Total samples:      {len(split_df)}")
    print(f"Training samples:   {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples:       {len(test_indices)}")
    print()
    print("Test set is reserved and will NOT be used during training.")

    # Expected with your current split:
    #
    # Total: 1912
    # Train: 1534
    # Val:    187
    # Test:   191

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=64,
    ).to(device)

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [POS_WEIGHT],
            device=device,
        )
    )
    dice = DiceLoss()

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    opt = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ---------------------------------------------------------
    # Model output directory
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = MODEL_DIR / "unet_best.pt"

    best_dice = -1.0

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

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

        # -----------------------------------------------------
        # Save according to VALIDATION Dice only.
        #
        # The test set is never consulted here.
        # -----------------------------------------------------

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
    print()
    print(
        "The test set has not been evaluated. "
        "Use it only for final evaluation."
    )


if __name__ == "__main__":
    main()

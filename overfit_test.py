from pathlib import Path

import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from src.dataset import VesselSegDataset
from src.unet import UNet
from src.losses import DiceLoss, dice_iou


# --------------------------------------------------
# Settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 50

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MIN_FOREGROUND = 0.01

OUTPUT_DIR = PROJECT_ROOT / "overfit_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Helper for displaying normalized images
# --------------------------------------------------

def denormalize_image(x):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=x.device
    )[:, None, None]

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=x.device
    )[:, None, None]

    x = x * std + mean
    return x.clamp(0, 1)


# --------------------------------------------------
# Save prediction visualization
# --------------------------------------------------

def save_predictions(
    epoch,
    imgs,
    masks,
    probs,
):
    imgs = imgs.detach().cpu()
    masks = masks.detach().cpu()
    probs = probs.detach().cpu()

    preds = (probs > 0.5).float()

    n = imgs.shape[0]

    fig, axes = plt.subplots(
        n,
        4,
        figsize=(12, 3 * n),
    )

    if n == 1:
        axes = axes[None, :]

    for i in range(n):

        img = denormalize_image(
            imgs[i]
        ).permute(1, 2, 0).numpy()

        gt = masks[i, 0].numpy()
        prob = probs[i, 0].numpy()
        pred = preds[i, 0].numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")

        axes[i, 1].imshow(
            gt,
            cmap="gray"
        )
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(
            prob,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[i, 2].set_title("Probability")

        axes[i, 3].imshow(
            pred,
            cmap="gray"
        )
        axes[i, 3].set_title("Prediction > 0.5")

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()

    out_path = (
        OUTPUT_DIR
        / f"epoch_{epoch:03d}.png"
    )

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print(f"Device: {DEVICE}")

    # ----------------------------------------------
    # Load full verified V2 dataset
    # ----------------------------------------------

    full_ds = VesselSegDataset(
        csv_path=PROJECT_ROOT / "data" / "dataset.csv",
        img_size=IMG_SIZE,
        augment=False,
        project_root=PROJECT_ROOT,
    )

    # ----------------------------------------------
    # Find ISIC samples with reasonable foreground
    # ----------------------------------------------

    good_indices = []

    for i in range(len(full_ds)):

        row = full_ds.df.iloc[i]

        sample_id = str(
            row["id"]
        ).upper()

        if not sample_id.startswith("ISIC_"):
            continue

        _, mask = full_ds[i]

        foreground = mask.mean().item()

        if foreground >= MIN_FOREGROUND:
            good_indices.append(i)

        if len(good_indices) == 4:
            break

    if len(good_indices) < 4:
        raise RuntimeError(
            "Could not find 4 ISIC samples "
            f"with foreground >= {MIN_FOREGROUND}"
        )

    # ----------------------------------------------
    # Show selected samples
    # ----------------------------------------------

    print()
    print("Selected samples:")

    for i in good_indices:

        row = full_ds.df.iloc[i]
        _, mask = full_ds[i]

        foreground = mask.mean().item()

        print(
            f"{row['id']} | "
            f"foreground={foreground:.4f} "
            f"({foreground * 100:.2f}%)"
        )

    print()

    # ----------------------------------------------
    # Make 4-sample training subset
    # ----------------------------------------------

    ds = Subset(
        full_ds,
        good_indices,
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    imgs, masks = next(
        iter(loader)
    )

    imgs = imgs.to(DEVICE)
    masks = masks.to(DEVICE)

    print(
        "Images:",
        imgs.shape
    )

    print(
        "Masks:",
        masks.shape
    )

    print()

    # ----------------------------------------------
    # Model
    # ----------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=64,
    ).to(DEVICE)

    bce = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ----------------------------------------------
    # Training loop
    # ----------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad()

        logits = model(imgs)

        loss = (
            0.5 * bce(
                logits,
                masks,
            )
            +
            0.5 * dice_loss(
                logits,
                masks,
            )
        )

        loss.backward()
        optimizer.step()

        # ------------------------------------------
        # Diagnostic every 5 epochs
        # ------------------------------------------

        if epoch == 1 or epoch % 5 == 0:

            # Train-mode BatchNorm prediction
            model.train()

            with torch.no_grad():

                train_logits = model(imgs)

                train_probs = torch.sigmoid(
                    train_logits
                )

                train_dice, train_iou = dice_iou(
                    train_logits,
                    masks,
                )

            # Eval-mode BatchNorm prediction
            model.eval()

            with torch.no_grad():

                eval_logits = model(imgs)

                eval_probs = torch.sigmoid(
                    eval_logits
                )

                eval_dice, eval_iou = dice_iou(
                    eval_logits,
                    masks,
                )

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"TRAIN Dice {train_dice:.4f} "
                f"IoU {train_iou:.4f} | "
                f"EVAL Dice {eval_dice:.4f} "
                f"IoU {eval_iou:.4f}"
            )

            print(
                f"           "
                f"TRAIN prob "
                f"{train_probs.min().item():.3f}-"
                f"{train_probs.max().item():.3f} | "
                f"EVAL prob "
                f"{eval_probs.min().item():.3f}-"
                f"{eval_probs.max().item():.3f}"
            )

            save_predictions(
                epoch,
                imgs,
                masks,
                eval_probs,
            )

    print()
    print("Overfit test complete.")
    print(
        f"Predictions saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
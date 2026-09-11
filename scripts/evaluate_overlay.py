import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataset import VesselSegDataset
from src.unet import UNet


def to_display_img(x_chw: torch.Tensor) -> np.ndarray:
    """
    Undo the ImageNet normalization used by VesselSegDataset.
    """
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=x_chw.device
    )[:, None, None]

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=x_chw.device
    )[:, None, None]

    x = (x_chw * std + mean).clamp(0, 1)

    return x.permute(1, 2, 0).cpu().numpy()


def bottom_title(ax, text):
    ax.text(
        0.5,
        0.02,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps=1e-7):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    dice = (
        2 * intersection + eps
    ) / (
        pred.sum() + gt.sum() + eps
    )

    iou = (
        intersection + eps
    ) / (
        union + eps
    )

    return dice, iou


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        default="models/unet_best.pt",
        help="Path to trained V2 checkpoint",
    )

    parser.add_argument(
        "--csv",
        default="data/dataset.csv",
        help="V2 dataset CSV",
    )

    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Dataset sample index",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--out_dir",
        default="demo",
    )

    args = parser.parse_args()


    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")


    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    ds = VesselSegDataset(
        csv_path=args.csv,
        img_size=args.size,
        augment=False,
    )

    print(f"Dataset samples: {len(ds)}")

    if args.idx < 0 or args.idx >= len(ds):
        raise IndexError(
            f"Index {args.idx} out of range. "
            f"Valid range: 0-{len(ds)-1}"
        )


    # Get sample information directly from dataset dataframe
    row = ds.df.iloc[args.idx]

    sample_id = str(row["id"])

    print(f"Sample index: {args.idx}")
    print(f"Sample ID:    {sample_id}")
    print(f"Image path:   {row['image_path']}")
    print(f"Mask path:    {row['mask_path']}")


    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=64,
    ).to(device)

    print(f"Loading checkpoint: {args.ckpt}")

    checkpoint = torch.load(
        args.ckpt,
        map_location=device,
    )


    # Handle either a raw state_dict or checkpoint dictionary
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint


    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.")


    # -------------------------------------------------
    # Sample
    # -------------------------------------------------

    x, y = ds[args.idx]

    img = to_display_img(x)

    gt = (
        y[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


    # -------------------------------------------------
    # Inference
    # -------------------------------------------------

    logits = model(
        x.unsqueeze(0).to(device)
    )[0, 0]

    prob = torch.sigmoid(logits).cpu().numpy()

    print()
    print(
        f"Probability range: "
        f"{prob.min():.4f} to {prob.max():.4f}"
    )

    print(
        f"Ground truth pixels: {gt.sum()}"
    )


    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    thresholds = args.thresholds[:5]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8),
        constrained_layout=True,
    )

    axes = axes.flatten()


    # Original image
    axes[0].imshow(img)
    axes[0].set_title(
        f"{sample_id}\nOriginal Image"
    )
    axes[0].axis("off")


    # Threshold panels
    for i, threshold in enumerate(thresholds):

        pred = (
            prob >= threshold
        ).astype(np.uint8)

        tp = (pred == 1) & (gt == 1)
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)

        overlay = np.zeros(
            (*gt.shape, 3),
            dtype=np.float32,
        )

        # Green = correct vessel
        overlay[tp, 1] = 1.0

        # Red = false positive
        overlay[fp, 0] = 1.0

        # Blue = missed vessel
        overlay[fn, 2] = 1.0

        dice, iou = dice_iou(
            pred,
            gt,
        )

        ax = axes[i + 1]

        ax.imshow(img)

        ax.imshow(
            overlay,
            alpha=args.alpha,
        )

        title_text = (
            f"thr={threshold:.2f}\n"
            f"Dice={dice:.3f}  "
            f"IoU={iou:.3f}"
        )

        if i + 1 in (1, 2):
            ax.set_title(title_text)
        else:
            bottom_title(
                ax,
                title_text,
            )

        ax.axis("off")

        print(
            f"thr={threshold:.2f}  "
            f"Dice={dice:.4f}  "
            f"IoU={iou:.4f}  "
            f"Pred pixels={pred.sum()}"
        )


    # Hide unused panels
    for j in range(
        len(thresholds) + 1,
        6
    ):
        axes[j].axis("off")


    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / f"prediction_overlay_{sample_id}.png"
    )

    plt.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(f"Saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
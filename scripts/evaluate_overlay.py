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
        device=x_chw.device,
    )[:, None, None]

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=x_chw.device,
    )[:, None, None]

    x = (x_chw * std + mean).clamp(0, 1)

    return x.permute(1, 2, 0).cpu().numpy()


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps=1e-7):
    """
    Calculate Dice and IoU for binary masks.
    """
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


def make_overlay(pred: np.ndarray, gt: np.ndarray):
    """
    Create TP / FP / FN overlay.

    Green = true positive
    Red   = false positive
    Blue  = false negative
    """
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    overlay = np.zeros(
        (*gt.shape, 4),
        dtype=np.float32,
    )

    # True positive = green
    overlay[tp] = [0.0, 1.0, 0.0, 1.0]

    # False positive = red
    overlay[fp] = [1.0, 0.0, 0.0, 1.0]

    # False negative = blue
    overlay[fn] = [0.0, 0.0, 1.0, 1.0]

    return overlay


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a VesselSeg U-Net prediction."
    )

    parser.add_argument(
        "--model",
        default="models/unet_posweight3.pt",
        help="Path to trained checkpoint",
    )

    parser.add_argument(
        "--csv",
        default="data/dataset_with_split.csv",
        help="Dataset CSV containing fixed train/val/test splits",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Existing dataset split to evaluate",
    )

    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Sample index within selected split",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Image size",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used for displayed prediction",
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="Thresholds used for metric comparison",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Overlay opacity",
    )

    parser.add_argument(
        "--out_dir",
        default="demo",
        help="Directory for saved prediction images",
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

    if "split" not in ds.df.columns:
        raise ValueError(
            f"{args.csv} does not contain a 'split' column."
        )

    # Use the fixed split assignments already stored in
    # dataset_with_split.csv.
    ds.df = (
        ds.df[ds.df["split"] == args.split]
        .reset_index(drop=True)
    )

    if len(ds) == 0:
        raise ValueError(
            f"No samples found for split '{args.split}'."
        )

    print(f"Selected split: {args.split}")
    print(f"Split samples:  {len(ds)}")

    if args.idx < 0 or args.idx >= len(ds):
        raise IndexError(
            f"Index {args.idx} out of range for "
            f"'{args.split}' split. "
            f"Valid range: 0-{len(ds)-1}"
        )


    # -------------------------------------------------
    # Sample information
    # -------------------------------------------------

    row = ds.df.iloc[args.idx]

    sample_id = str(row["sample_id"])

    print()
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

    print()
    print(f"Loading checkpoint: {args.model}")

    checkpoint = torch.load(
        args.model,
        map_location=device,
    )

    # Handle either a raw state_dict or checkpoint dictionary.
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
    # Load sample
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
    # Forward pass
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
    # Threshold comparison
    # -------------------------------------------------

    print()
    print("Threshold comparison:")

    for threshold in args.thresholds:

        pred_t = (
            prob >= threshold
        ).astype(np.uint8)

        dice_t, iou_t = dice_iou(
            pred_t,
            gt,
        )

        print(
            f"thr={threshold:.2f}  "
            f"Dice={dice_t:.4f}  "
            f"IoU={iou_t:.4f}  "
            f"Pred pixels={pred_t.sum()}"
        )


    # -------------------------------------------------
    # Main displayed prediction
    # -------------------------------------------------

    pred = (
        prob >= args.threshold
    ).astype(np.uint8)

    dice, iou = dice_iou(
        pred,
        gt,
    )

    overlay = make_overlay(
        pred,
        gt,
    )

    print()
    print(
        f"Selected threshold: {args.threshold:.2f}"
    )
    print(f"Dice: {dice:.4f}")
    print(f"IoU:  {iou:.4f}")
    print(f"Predicted pixels: {pred.sum()}")


    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 5),
        constrained_layout=True,
    )


    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")


    # Ground-truth mask
    axes[1].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title("Ground Truth")
    axes[1].axis("off")


    # Predicted mask
    axes[2].imshow(
        pred,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        f"Prediction\n"
        f"Threshold = {args.threshold:.2f}"
    )

    axes[2].axis("off")


    # TP / FP / FN overlay
    axes[3].imshow(img)

    axes[3].imshow(
        overlay,
        alpha=args.alpha,
    )

    axes[3].set_title(
        "TP / FP / FN Overlay\n"
        f"Dice={dice:.3f}  IoU={iou:.3f}"
    )

    axes[3].axis("off")


    fig.suptitle(
        f"{sample_id} | "
        f"{args.split} | "
        f"{Path(args.model).name}",
        fontsize=12,
    )


    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_name = Path(args.model).stem

    out_path = (
        out_dir
        / (
            f"{args.split}_idx{args.idx}_"
            f"{sample_id}_"
            f"{model_name}_"
            f"thr{args.threshold:.2f}.png"
        )
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
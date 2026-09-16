import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import VesselSegDataset
from src.unet import UNet


def to_display_img(x_chw: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet normalization used by VesselSegDataset
    and return an RGB uint8 image.
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

    return (
        x.permute(1, 2, 0)
        .cpu()
        .numpy()
        * 255
    ).astype(np.uint8)


def make_rgba_layer(mask, color, alpha=180):
    """
    Create a transparent RGBA overlay.
    """
    h, w = mask.shape

    layer = np.zeros(
        (h, w, 4),
        dtype=np.uint8,
    )

    layer[mask, 0] = color[0]
    layer[mask, 1] = color[1]
    layer[mask, 2] = color[2]
    layer[mask, 3] = alpha

    return layer


def dice_iou(pred, gt, eps=1e-7):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(
        pred,
        gt,
    ).sum()

    union = np.logical_or(
        pred,
        gt,
    ).sum()

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

    parser = argparse.ArgumentParser(
        description=(
            "Run VesselSeg inference over an entire split "
            "and save TP/FP/FN as separate PNG layers."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model checkpoint",
    )

    parser.add_argument(
        "--csv",
        default="data/dataset_with_split.csv",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--alpha",
        type=int,
        default=180,
        help="Overlay opacity from 0-255",
    )

    parser.add_argument(
        "--out_dir",
        default="predictions",
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
            "Dataset CSV does not contain a split column."
        )

    ds.df = (
        ds.df[
            ds.df["split"] == args.split
        ]
        .reset_index(drop=True)
    )

    if len(ds) == 0:
        raise ValueError(
            f"No samples found in split '{args.split}'."
        )

    print(f"Split:   {args.split}")
    print(f"Samples: {len(ds)}")


    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=64,
    ).to(device)

    print()
    print(f"Loading model: {args.model}")

    checkpoint = torch.load(
        args.model,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print("Model loaded.")


    # -------------------------------------------------
    # Output directory
    # -------------------------------------------------

    model_name = Path(
        args.model
    ).stem

    root_out_dir = (
        Path(args.out_dir)
        / f"{args.split}_{model_name}_thr{args.threshold:.2f}"
    )

    root_out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(f"Output: {root_out_dir}")
    print()


    # -------------------------------------------------
    # Running totals
    # -------------------------------------------------

    dice_scores = []
    iou_scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0


    # -------------------------------------------------
    # Process entire split
    # -------------------------------------------------

    for idx in range(len(ds)):

        row = ds.df.iloc[idx]

        sample_id = str(
            row["sample_id"]
        )


        # ---------------------------------------------
        # Load image / ground truth
        # ---------------------------------------------

        x, y = ds[idx]

        original = to_display_img(x)

        gt = (
            y[0]
            .cpu()
            .numpy()
            .astype(bool)
        )


        # ---------------------------------------------
        # Forward pass
        # ---------------------------------------------

        logits = model(
            x.unsqueeze(0).to(device)
        )[0, 0]

        probability = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

        pred = (
            probability >= args.threshold
        )


        # ---------------------------------------------
        # TP / FP / FN
        # ---------------------------------------------

        tp = pred & gt

        fp = pred & ~gt

        fn = ~pred & gt


        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        dice, iou = dice_iou(
            pred,
            gt,
        )

        dice_scores.append(dice)
        iou_scores.append(iou)

        total_tp += int(tp.sum())
        total_fp += int(fp.sum())
        total_fn += int(fn.sum())


        # ---------------------------------------------
        # Colored layers
        # ---------------------------------------------

        fn_layer = make_rgba_layer(
            fn,
            color=(255, 0, 0),
            alpha=args.alpha,
        )

        fp_layer = make_rgba_layer(
            fp,
            color=(0, 0, 255),
            alpha=args.alpha,
        )

        tp_layer = make_rgba_layer(
            tp,
            color=(0, 255, 0),
            alpha=args.alpha,
        )


        # ---------------------------------------------
        # Sample output folder
        # ---------------------------------------------

        sample_dir = (
            root_out_dir
            / (
                f"idx{idx:03d}_"
                f"{sample_id}"
            )
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ---------------------------------------------
        # Save layers
        # ---------------------------------------------

        Image.fromarray(
            original,
            mode="RGB",
        ).save(
            sample_dir
            / "00_original.png"
        )

        Image.fromarray(
            fn_layer,
            mode="RGBA",
        ).save(
            sample_dir
            / "01_FN_red.png"
        )

        Image.fromarray(
            fp_layer,
            mode="RGBA",
        ).save(
            sample_dir
            / "02_FP_blue.png"
        )

        Image.fromarray(
            tp_layer,
            mode="RGBA",
        ).save(
            sample_dir
            / "03_TP_green.png"
        )


        # ---------------------------------------------
        # Progress
        # ---------------------------------------------

        print(
            f"[{idx + 1:03d}/{len(ds):03d}] "
            f"{sample_id}  "
            f"Dice={dice:.4f}  "
            f"IoU={iou:.4f}"
        )


    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    mean_dice = float(
        np.mean(dice_scores)
    )

    mean_iou = float(
        np.mean(iou_scores)
    )

    micro_dice = (
        2 * total_tp
        / (
            2 * total_tp
            + total_fp
            + total_fn
            + 1e-7
        )
    )

    micro_iou = (
        total_tp
        / (
            total_tp
            + total_fp
            + total_fn
            + 1e-7
        )
    )


    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)

    print(
        f"Samples:    {len(ds)}"
    )

    print(
        f"Threshold:  {args.threshold:.2f}"
    )

    print()
    print(
        f"Mean Dice:  {mean_dice:.4f}"
    )

    print(
        f"Mean IoU:   {mean_iou:.4f}"
    )

    print()
    print(
        f"Micro Dice: {micro_dice:.4f}"
    )

    print(
        f"Micro IoU:  {micro_iou:.4f}"
    )

    print()
    print(
        f"Total TP:   {total_tp}"
    )

    print(
        f"Total FP:   {total_fp}"
    )

    print(
        f"Total FN:   {total_fn}"
    )

    print()
    print(
        f"Saved to: {root_out_dir}"
    )


if __name__ == "__main__":
    main()
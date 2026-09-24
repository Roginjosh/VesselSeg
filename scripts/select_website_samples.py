from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.dataset import VesselSegDataset
from src.unet import UNet


def dice_iou(pred, gt, eps=1e-7):
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

    return float(dice), float(iou)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Rank ISIC test images by per-image segmentation performance."
    )

    parser.add_argument(
        "--model",
        default="unet_best.pt",
        help="Path to trained model checkpoint",
    )

    parser.add_argument(
        "--csv",
        default="data/dataset_with_split.csv",
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
        "--out",
        default="website_sample_candidates.csv",
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

    # Test set only
    ds.df = ds.df[
        ds.df["split"].eq("test")
    ].copy()

    # ISIC images only
    ds.df = ds.df[
        ds.df["isic_id"].notna()
    ].copy()

    # Extra safety: require ISIC-style identifier
    ds.df = ds.df[
        ds.df["isic_id"]
        .astype(str)
        .str.upper()
        .str.startswith("ISIC_")
    ].reset_index(drop=True)

    if len(ds) == 0:
        raise ValueError(
            "No ISIC samples found in the test split."
        )

    print(f"ISIC test samples: {len(ds)}")

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=64,
    ).to(device)

    checkpoint = torch.load(
        args.model,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model: {args.model}")
    print(f"Threshold:    {args.threshold:.2f}")
    print()

    # -------------------------------------------------
    # Inference
    # -------------------------------------------------

    results = []

    for idx in tqdm(
        range(len(ds)),
        desc="Scoring ISIC test images",
    ):
        row = ds.df.iloc[idx]

        x, y = ds[idx]

        gt = (
            y[0]
            .cpu()
            .numpy()
            .astype(bool)
        )

        logits = model(
            x.unsqueeze(0).to(device)
        )[0, 0]

        probability = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

        pred = probability >= args.threshold

        dice, iou = dice_iou(pred, gt)

        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())

        gt_pixels = int(gt.sum())
        pred_pixels = int(pred.sum())
        total_pixels = int(gt.size)

        results.append({
            "sample_id": row["sample_id"],
            "isic_id": row["isic_id"],
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "dice": dice,
            "iou": iou,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_pixels": gt_pixels,
            "pred_pixels": pred_pixels,
            "gt_vessel_percent":
                100.0 * gt_pixels / total_pixels,
            "pred_vessel_percent":
                100.0 * pred_pixels / total_pixels,
        })

    df = pd.DataFrame(results)

    # Highest Dice first
    df = df.sort_values(
        "dice",
        ascending=False,
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)

    # Put rank first
    cols = [
        "rank",
        "sample_id",
        "isic_id",
        "dice",
        "iou",
        "gt_vessel_percent",
        "pred_vessel_percent",
        "tp",
        "fp",
        "fn",
        "image_path",
        "mask_path",
    ]

    df = df[cols]

    df.to_csv(args.out, index=False)

    # -------------------------------------------------
    # Candidate groups
    # -------------------------------------------------

    n = len(df)
    middle = n // 2

    high = df.head(5)

    mid_start = max(0, middle - 2)
    mid_end = min(n, middle + 3)
    middle_df = df.iloc[mid_start:mid_end]

    low = df.tail(5)

    display_cols = [
        "rank",
        "isic_id",
        "dice",
        "iou",
        "gt_vessel_percent",
    ]

    print("\n" + "=" * 70)
    print("HIGH CANDIDATES")
    print("=" * 70)
    print(
        high[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print("MIDDLE CANDIDATES")
    print("=" * 70)
    print(
        middle_df[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print("LOW CANDIDATES")
    print("=" * 70)
    print(
        low[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"Full ranking saved to: {args.out}")


if __name__ == "__main__":
    main()

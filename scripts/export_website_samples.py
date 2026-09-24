from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import VesselSegDataset
from src.unet import UNet


MODEL_PATH = "unet_best.pt"
CSV_PATH = "data/dataset_with_split.csv"
OUTPUT_DIR = Path("website_samples")
THRESHOLD = 0.5
IMAGE_SIZE = 256

SAMPLES = {
    "high_1": "ISIC_0032395",
    "high_2": "ISIC_0031168_ok",
    "middle": "ISIC_0032022",
    "low": "ISIC_0030893",
}


def to_display_img(x):
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    )[:, None, None]

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    )[:, None, None]

    x = (x.cpu() * std + mean).clamp(0, 1)

    return (
        x.permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)


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


def binary_png(mask):
    return Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L",
    )


def make_error_overlay(original, pred, gt, alpha=180):
    """
    TP = green
    FP = blue
    FN = red
    """

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    base = Image.fromarray(original).convert("RGBA")

    overlay = np.zeros(
        (gt.shape[0], gt.shape[1], 4),
        dtype=np.uint8,
    )

    # TP = green
    overlay[tp] = [0, 255, 0, alpha]

    # FP = blue
    overlay[fp] = [0, 0, 255, alpha]

    # FN = red
    overlay[fn] = [255, 0, 0, alpha]

    overlay_img = Image.fromarray(
        overlay,
        mode="RGBA",
    )

    return Image.alpha_composite(
        base,
        overlay_img,
    )


@torch.no_grad()
def main():

    # -----------------------------------------
    # Device
    # -----------------------------------------

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")

    # -----------------------------------------
    # Dataset
    # -----------------------------------------

    ds = VesselSegDataset(
        csv_path=CSV_PATH,
        img_size=IMAGE_SIZE,
        augment=False,
    )

    # -----------------------------------------
    # Model
    # -----------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=64,
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
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

    print(f"Loaded: {MODEL_PATH}")

    # -----------------------------------------
    # Output
    # -----------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Process selected samples
    # -----------------------------------------

    for category, wanted_sample_id in SAMPLES.items():

        matches = ds.df.index[
            ds.df["sample_id"].astype(str)
            == wanted_sample_id
        ].tolist()

        if not matches:
            print(
                f"WARNING: sample not found: "
                f"{wanted_sample_id}"
            )
            continue

        idx = matches[0]
        row = ds.df.iloc[idx]

        isic_id = str(row["isic_id"])

        print()
        print(
            f"{category}: "
            f"{wanted_sample_id} ({isic_id})"
        )

        # -------------------------------------
        # Dataset image + GT
        # -------------------------------------

        x, y = ds[idx]

        original_256 = to_display_img(x)

        gt = (
            y[0]
            .cpu()
            .numpy()
            .astype(bool)
        )

        # -------------------------------------
        # Prediction
        # -------------------------------------

        logits = model(
            x.unsqueeze(0).to(device)
        )[0, 0]

        probability = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

        pred = probability >= THRESHOLD

        dice, iou = dice_iou(
            pred,
            gt,
        )

        print(f"  Dice: {dice:.4f}")
        print(f"  IoU:  {iou:.4f}")

        # -------------------------------------
        # Directory
        # -------------------------------------

        sample_dir = (
            OUTPUT_DIR
            / f"{category}_{isic_id}"
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -------------------------------------
        # Save model-sized original
        # -------------------------------------

        Image.fromarray(
            original_256
        ).save(
            sample_dir / "original_256.png"
        )

        # -------------------------------------
        # Copy FULL resolution original
        # -------------------------------------

        raw_path = str(row["image_path"])

        # CSV contains Windows separators.
        raw_path = raw_path.replace("\\", "/")

        raw_path = Path(raw_path)

        if raw_path.exists():
            extension = raw_path.suffix.lower()

            shutil.copy2(
                raw_path,
                sample_dir
                / f"original_full{extension}",
            )

            print(
                f"  Full image: {raw_path}"
            )

        else:
            print(
                f"  WARNING: full image not found: "
                f"{raw_path}"
            )

        # -------------------------------------
        # Save masks
        # -------------------------------------

        binary_png(gt).save(
            sample_dir / "ground_truth.png"
        )

        binary_png(pred).save(
            sample_dir / "prediction.png"
        )

        # -------------------------------------
        # Probability map
        # -------------------------------------

        prob_img = (
            np.clip(
                probability,
                0,
                1,
            ) * 255
        ).astype(np.uint8)

        Image.fromarray(
            prob_img,
            mode="L",
        ).save(
            sample_dir / "probability.png"
        )

        # -------------------------------------
        # Error overlay
        # -------------------------------------

        error_overlay = make_error_overlay(
            original_256,
            pred,
            gt,
        )

        error_overlay.save(
            sample_dir / "error_overlay.png"
        )

        # -------------------------------------
        # Metrics file
        # -------------------------------------

        with open(
            sample_dir / "metrics.txt",
            "w",
        ) as f:

            f.write(
                f"ISIC ID: {isic_id}\n"
            )

            f.write(
                f"Sample ID: "
                f"{wanted_sample_id}\n"
            )

            f.write(
                f"Threshold: {THRESHOLD}\n"
            )

            f.write(
                f"Dice: {dice:.6f}\n"
            )

            f.write(
                f"IoU: {iou:.6f}\n"
            )

    print()
    print(
        f"Finished. Samples saved to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()



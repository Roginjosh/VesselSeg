import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from dataset import ISICKeySegDataset
from unet import UNet

ISIC_RE = re.compile(r"(ISIC_\d{7})", re.IGNORECASE)

def extract_isic_key_from_pair(pair):
    # Your dataset.pairs likely stores (img_path, mask_path)
    img_path, mask_path = pair
    name = str(img_path.name)
    m = ISIC_RE.search(name)
    return m.group(1).upper() if m else img_path.stem

def dice_iou_from_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    dice = (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
    iou  = (inter + eps) / (union + eps)
    return float(dice), float(iou)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="unet_best.pt")
    ap.add_argument("--img_dir", default="data/imgs")
    ap.add_argument("--mask_dir", default="data/masks")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)  # macOS safer; set 2 on Windows if you want
    ap.add_argument("--out_dir", default="metrics_out")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    thresholds = [float(t) for t in args.thresholds]

    ds = ISICKeySegDataset(args.img_dir, args.mask_dir, img_size=args.size, augment=False)

    # DataLoader to batch inference
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = UNet(in_channels=3, out_channels=1, base=64).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # We’ll write one row per image per threshold (long format).
    rows = []

    # Optional: try to get stable IDs from dataset.pairs if present
    has_pairs = hasattr(ds, "pairs")

    processed = 0
    pbar = tqdm(loader, desc="Running inference")
    for batch_idx, (imgs, masks) in enumerate(pbar):
        imgs = imgs.to(device)
        masks = masks.cpu().numpy().astype(np.uint8)  # [B,1,H,W]

        logits = model(imgs)                           # [B,1,H,W]
        probs = torch.sigmoid(logits).cpu().numpy()    # [B,1,H,W]

        b = imgs.shape[0]
        for j in range(b):
            if args.max_images and processed >= args.max_images:
                break

            gt = masks[j, 0]
            pr = probs[j, 0]

            # Create an ID for this item
            idx = processed
            if has_pairs:
                # processed index corresponds to dataset ordering (since shuffle=False)
                # For exact mapping: compute global index in dataset:
                global_i = batch_idx * args.batch_size + j
                if global_i < len(ds.pairs):
                    sample_id = extract_isic_key_from_pair(ds.pairs[global_i])
                else:
                    sample_id = f"idx_{global_i}"
            else:
                sample_id = f"idx_{batch_idx*args.batch_size + j}"

            for thr in thresholds:
                pred = (pr >= thr).astype(np.uint8)


                tp = ((pred == 1) & (gt == 1)).sum()
                fp = ((pred == 1) & (gt == 0)).sum()
                fn = ((pred == 0) & (gt == 1)).sum()
                tn = ((pred == 0) & (gt == 0)).sum()

                dice = (2*tp) / (2*tp + fp + fn + 1e-7)
                iou  = tp / (tp + fp + fn + 1e-7)

                rows.append({
                    "sample_id": sample_id,
                    "threshold": thr,
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "dice": dice,
                    "iou": iou,
                })


            processed += 1

        if args.max_images and processed >= args.max_images:
            break

    df = pd.DataFrame(rows)

    # Save per-image-per-threshold data (long format)
    per_image_csv = out_dir / "per_image_threshold_metrics.csv"
    df.to_csv(per_image_csv, index=False)

    # Also save aggregated stats by threshold for quick look
    agg = (df.groupby("threshold")
             .agg(
                 dice_mean=("dice", "mean"),
                 dice_median=("dice", "median"),
                 dice_std=("dice", "std"),
                 iou_mean=("iou", "mean"),
                 iou_median=("iou", "median"),
                 iou_std=("iou", "std"),
                 n=("dice", "count"),
             )
             .reset_index())

    agg_csv = out_dir / "threshold_summary_stats.csv"
    agg.to_csv(agg_csv, index=False)

    print(f"\nSaved:\n- {per_image_csv}\n- {agg_csv}\n")
    print(agg)

if __name__ == "__main__":
    main()

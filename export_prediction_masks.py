import argparse
from pathlib import Path
import re
import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image

from dataset import ISICKeySegDataset
from unet import UNet

ISIC_RE = re.compile(r"(ISIC_\d{7})", re.IGNORECASE)

def extract_isic_from_path(path: Path) -> str:
    m = ISIC_RE.search(path.name)
    return m.group(1).upper() if m else path.stem

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="unet_best.pt")
    ap.add_argument("--img_dir", default="data/imgs")
    ap.add_argument("--mask_dir", default="data/masks")  # only for dataset pairing
    ap.add_argument("--out_dir", default="data/predictions")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--idx", type=int, default=None, help="If set, only export this dataset index")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset (augment=False is critical for inference)
    ds = ISICKeySegDataset(
        args.img_dir,
        args.mask_dir,
        img_size=args.size,
        augment=False
    )

    # Model
    model = UNet(in_channels=3, out_channels=1, base=64).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Single-image mode (fast sanity check)
    if args.idx is not None:
        img_path, _ = ds.pairs[args.idx]
        isic = extract_isic_from_path(img_path)

        x, _ = ds[args.idx]
        logits = model(x.unsqueeze(0).to(device))[0, 0]
        prob = torch.sigmoid(logits).cpu().numpy()

        pred = (prob >= args.threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(pred, mode="L")

        out_path = out_dir / f"{isic}_Vessel_Seg_Mask.tif"
        mask_img.save(out_path)

        print(f"[OK] Saved single prediction → {out_path}")
        return

    # Full-dataset export
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    pbar = tqdm(loader, desc="Exporting prediction masks")

    global_idx = 0
    for imgs, _ in pbar:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()  # [B,1,H,W]

        b = probs.shape[0]
        for j in range(b):
            img_path, _ = ds.pairs[global_idx]
            isic = extract_isic_from_path(img_path)

            pred = (probs[j, 0] >= args.threshold).astype(np.uint8) * 255
            mask_img = Image.fromarray(pred, mode="L")

            out_path = out_dir / f"{isic}_Vessel_Seg_Mask.tif"
            mask_img.save(out_path)

            global_idx += 1

    print(f"\n[DONE] Exported {global_idx} masks to {out_dir}")

if __name__ == "__main__":
    main()

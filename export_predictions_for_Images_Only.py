import argparse
from pathlib import Path
import re

import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as T

from unet import UNet

ISIC_RE = re.compile(r"(ISIC_\d{7})", re.IGNORECASE)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def extract_isic_from_path(path: Path) -> str:
    m = ISIC_RE.search(path.name)
    return m.group(1).upper() if m else path.stem

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

def build_preprocess(img_size: int):
    # Match your dataset.py normalization (ImageNet)
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

@torch.no_grad()
def predict_one(model, device, preprocess, img_path: Path, threshold: float) -> np.ndarray:
    im = Image.open(img_path).convert("RGB")
    x = preprocess(im).unsqueeze(0).to(device)   # [1,3,H,W]
    logits = model(x)[0, 0]                       # [H,W]
    prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob >= threshold).astype(np.uint8) * 255
    return pred

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="unet_best.pt")
    ap.add_argument("--img_dir", default="data/imgs")
    ap.add_argument("--out_dir", default="data/predictions")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--idx", type=int, default=None, help="If set, only export this image index (sorted order)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.img_dir)
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not img_paths:
        raise SystemExit(f"No images found in {img_dir}")

    preprocess = build_preprocess(args.size)

    model = UNet(in_channels=3, out_channels=1, base=64).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Single-image mode
    if args.idx is not None:
        if args.idx < 0 or args.idx >= len(img_paths):
            raise SystemExit(f"--idx out of range (0..{len(img_paths)-1})")
        img_path = img_paths[args.idx]
        isic = extract_isic_from_path(img_path)
        out_path = out_dir / f"{isic}_Vessel_Seg_Mask.tif"
        pred = predict_one(model, device, preprocess, img_path, args.threshold)
        Image.fromarray(pred, mode="L").save(out_path)
        print(f"[OK] {img_path.name} → {out_path}")
        return

    # Full export
    pbar = tqdm(img_paths, desc="Exporting prediction masks")
    saved = 0
    for img_path in pbar:
        isic = extract_isic_from_path(img_path)
        out_path = out_dir / f"{isic}_Vessel_Seg_Mask.tif"
        if out_path.exists() and not args.overwrite:
            continue

        pred = predict_one(model, device, preprocess, img_path, args.threshold)
        Image.fromarray(pred, mode="L").save(out_path)
        saved += 1
        if saved % 200 == 0:
            pbar.set_postfix(saved=saved)

    print(f"\n[DONE] Saved {saved} masks to {out_dir}")

if __name__ == "__main__":
    main()

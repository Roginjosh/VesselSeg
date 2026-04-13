from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T

from unet import UNet


# =========================
# CONFIG
# =========================
CKPT_PATH = "unet_best.pt"
IMG_DIR = "data/imgs"
MASK_DIR = "data/masks"
OUT_DIR = "data/inference_layers"

IMG_SIZE = 256
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_COMPOSITE_PREVIEW = False
# =========================


VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ISIC_RE = re.compile(r"(ISIC_\d{7})", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Inference with Paint.NET-friendly PNG layer export")
    p.add_argument("--n", type=int, default=None, help="Randomly sample n images")
    p.add_argument("--isic", type=str, default=None, help="Run a specific ISIC code, e.g. ISIC_0066680")
    return p.parse_args()


def extract_isic_key(path: Path) -> str | None:
    m = ISIC_RE.search(path.name)
    return m.group(1).upper() if m else None


def build_paired_lists(image_dir: str, mask_dir: str):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    mask_paths = [p for p in mask_dir.iterdir() if p.suffix.lower() in VALID_EXTS]

    img_map = {}
    for p in image_paths:
        k = extract_isic_key(p)
        if k:
            img_map[k] = p

    mask_map = {}
    for p in mask_paths:
        k = extract_isic_key(p)
        if k:
            mask_map[k] = p

    keys = sorted(set(img_map.keys()) & set(mask_map.keys()))
    if not keys:
        raise RuntimeError(
            "No image/mask pairs found by ISIC key. Check IMG_DIR, MASK_DIR, and filenames."
        )

    pairs = [(k, img_map[k], mask_map[k]) for k in keys]

    missing_imgs = sorted(set(mask_map.keys()) - set(img_map.keys()))
    missing_masks = sorted(set(img_map.keys()) - set(mask_map.keys()))
    if missing_imgs:
        print(f"[WARN] {len(missing_imgs)} masks have no matching image.")
    if missing_masks:
        print(f"[WARN] {len(missing_masks)} images have no matching mask.")

    return pairs


def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    img = T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR)(img)
    img_t = T.ToTensor()(img)

    # Match training normalization
    img_t = T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )(img_t)

    return img, img_t


def load_mask(path: Path):
    mask = Image.open(path).convert("L")
    mask = T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.NEAREST)(mask)
    mask_t = T.ToTensor()(mask)
    mask_t = (mask_t > 0.5).float()  # shape [1,H,W]
    return mask_t


def dice(pred, gt, eps=1e-7):
    inter = (pred * gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)


def iou(pred, gt, eps=1e-7):
    inter = (pred * gt).sum()
    union = ((pred + gt) > 0).sum()
    return (inter + eps) / (union + eps)


def make_layer(mask: np.ndarray, color: tuple[int, int, int, int]) -> Image.Image:
    h, w = mask.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[mask] = color
    return Image.fromarray(arr, "RGBA")


def make_info_layer(w: int, h: int, code: str, d: float, i: float) -> Image.Image:
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = ImageFont.load_default()

    lines = [
        code,
        f"Dice: {d:.4f}",
        f"IoU:  {i:.4f}",
        "",
        "Legend:",
        "Green = TP",
        "Red   = FP",
        "Blue  = FN",
    ]

    x0, y0 = 10, 10
    line_h = 14
    padding = 8

    box_w = 200
    box_h = padding * 2 + line_h * len(lines)

    draw.rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        fill=(0, 0, 0, 180)
    )

    y = y0 + padding
    for line in lines:
        draw.text((x0 + padding, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_h

    return layer


def save_composite_preview(
    out_path: Path,
    base: Image.Image,
    fn_layer: Image.Image,
    fp_layer: Image.Image,
    tp_layer: Image.Image,
    info_layer: Image.Image,
):
    preview = base.convert("RGBA")
    preview = Image.alpha_composite(preview, fn_layer)
    preview = Image.alpha_composite(preview, fp_layer)
    preview = Image.alpha_composite(preview, tp_layer)
    preview = Image.alpha_composite(preview, info_layer)
    preview.save(out_path)


def save_layer_stack(
    out_dir: Path,
    code: str,
    base: Image.Image,
    fn_layer: Image.Image,
    fp_layer: Image.Image,
    tp_layer: Image.Image,
    info_layer: Image.Image,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    base.save(out_dir / f"{code}_base.png")
    fn_layer.save(out_dir / f"{code}_FN.png")
    fp_layer.save(out_dir / f"{code}_FP.png")
    tp_layer.save(out_dir / f"{code}_TP.png")
    info_layer.save(out_dir / f"{code}_info.png")


def load_model():
    model = UNet(in_channels=3, out_channels=1)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()
    return model


def main():
    args = parse_args()
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = build_paired_lists(IMG_DIR, MASK_DIR)

    if args.isic:
        target = args.isic.upper()
        pairs = [triple for triple in pairs if triple[0] == target]
        if not pairs:
            print(f"No paired image/mask found for {target}")
            return
    elif args.n is not None:
        n = min(args.n, len(pairs))
        pairs = random.sample(pairs, n)

    print(f"Using device: {DEVICE}")
    print(f"Processing {len(pairs)} images...")

    model = load_model()

    for code, img_path, mask_path in pairs:
        pil_img, img_tensor = load_image(img_path)
        gt = load_mask(mask_path)  # [1,H,W]

        x = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            pred = torch.sigmoid(logits)[0, 0].cpu()

        pred_bin = pred >= THRESHOLD
        gt_bin = gt[0].bool()

        tp = (pred_bin & gt_bin)
        fp = (pred_bin & ~gt_bin)
        fn = (~pred_bin & gt_bin)

        d = dice(pred_bin.float(), gt_bin.float()).item()
        i = iou(pred_bin.float(), gt_bin.float()).item()

        base = pil_img.convert("RGBA")
        fn_layer = make_layer(fn.numpy(), (0, 0, 255, 180))     # blue
        fp_layer = make_layer(fp.numpy(), (255, 0, 0, 180))     # red
        tp_layer = make_layer(tp.numpy(), (0, 255, 0, 180))     # green
        info_layer = make_info_layer(base.width, base.height, code, d, i)

        image_out_dir = out_root / code
        save_layer_stack(
            image_out_dir,
            code,
            base,
            fn_layer,
            fp_layer,
            tp_layer,
            info_layer,
        )

        if SAVE_COMPOSITE_PREVIEW:
            preview_path = image_out_dir / f"{code}_preview.png"
            save_composite_preview(
                preview_path,
                base,
                fn_layer,
                fp_layer,
                tp_layer,
                info_layer,
            )
            print(f"{code} | Dice={d:.4f} | IoU={i:.4f} | saved PNG layer stack + preview")
        else:
            print(f"{code} | Dice={d:.4f} | IoU={i:.4f} | saved PNG layer stack")

    print("Done.")


if __name__ == "__main__":
    main()
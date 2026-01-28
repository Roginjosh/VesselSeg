# debug_overlay.py
#
# Run from repo root:
#   python .\debug_overlay.py
#   python .\debug_overlay.py --n 6
#   python .\debug_overlay.py --idx 0
#   python .\debug_overlay.py --thr 0.35
#   python .\debug_overlay.py --cfg configs/default.yaml
#
# Uses SAME pipeline as training:
# - reads cfg paths/data/model
# - builds ISICSegDataset
# - uses same random_split seed
# - applies same resizing/to_tensor/mask binarization style
#
# Mask pairing NOTE:
# Your current masks are named like: ISIC_0012294_wvs.tiff / ISIC_0013680_ok.tiff (+ sometimes "Copy of ...")
# But dataset.py expects: <image_stem>_mvig.tif[f]
# This debug script pairs masks by ISIC id (ISIC_<digits>) which matches your file listing.

import argparse
import random
import re
from pathlib import Path

import yaml
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from models.unet import UNet


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MASK_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ISIC_RE = re.compile(r"(ISIC_\d+)", re.IGNORECASE)


# ----------------------------- dataset (training-style) ----------------------
class ISICLikeDebugDataset(torch.utils.data.Dataset):
    """
    Training-style sample loader:
    - Loads RGB image + binary mask
    - Resizes image/mask to img_size (bilinear / nearest)
    - Returns img_t: (3,H,W) float [0,1]
             mask_t:(1,H,W) float {0,1}
    Pairing:
    - Extract ISIC_<digits> from image filename
    - Find mask file in mask_dir containing the same ISIC id
      (prefers non-'Copy of' masks, then alphabetical)
    """
    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.img_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        )
        if not self.img_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.resize_img  = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)

        # build mask index by ISIC id
        self.mask_index = self._build_mask_index(self.mask_dir)
        if not self.mask_index:
            raise RuntimeError(f"No masks with ISIC IDs found in {self.mask_dir}")

        # keep only images that have a matching mask
        kept = []
        for ip in self.img_paths:
            isic = self._extract_isic(ip.name)
            if isic and isic in self.mask_index:
                kept.append(ip)
        if not kept:
            raise RuntimeError("No images in data/imgs matched any masks in data/masks by ISIC id.")
        self.img_paths = kept

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        isic = self._extract_isic(img_path.name)
        mask_path = self.mask_index[isic][0]  # best candidate

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img  = self.resize_img(img)
        mask = self.resize_mask(mask)

        img_t  = TF.to_tensor(img)            # (3,H,W), [0,1]
        mask_t = TF.to_tensor(mask)           # (1,H,W), [0,1]
        mask_t = (mask_t > 0).float()

        return img_t, mask_t, img_path.name, mask_path.name

    @staticmethod
    def _extract_isic(name: str) -> str | None:
        m = ISIC_RE.search(name)
        return m.group(1).upper() if m else None

    @staticmethod
    def _build_mask_index(mask_dir: Path) -> dict[str, list[Path]]:
        masks = [p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTS]
        idx: dict[str, list[Path]] = {}
        for mp in masks:
            m = ISIC_RE.search(mp.name)
            if not m:
                continue
            isic = m.group(1).upper()
            idx.setdefault(isic, []).append(mp)

        # prefer non-"Copy of" then alphabetical
        for k in idx:
            idx[k].sort(key=lambda p: (p.name.lower().startswith("copy of "), p.name.lower()))
        return idx


# ----------------------------- visualization --------------------------------
def dice_iou(gt: np.ndarray, pred: np.ndarray):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = (gt & pred).sum()
    union = (gt | pred).sum()
    dice = (2 * inter) / (gt.sum() + pred.sum() + 1e-8)
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)

def overlay_rgb(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if img.ndim == 2:
        base = np.stack([img, img, img], axis=-1)
    else:
        base = img[..., :3]

    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    overlap = gt_b & pred_b

    color = np.zeros_like(base)
    color[gt_b] = np.array([0.0, 1.0, 0.0])      # green
    color[pred_b] = np.array([1.0, 0.0, 0.0])    # red
    color[overlap] = np.array([1.0, 1.0, 0.0])   # yellow

    return np.clip((1 - alpha) * base + alpha * color, 0, 1)

def save_triptych(img_np, gt_np, pred_np, out_path: Path, title: str, thr: float):
    d, j = dice_iou(gt_np, pred_np)
    over = overlay_rgb(img_np, gt_np, pred_np, alpha=0.45)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.axis("off")
    plt.imshow(img_np)

    plt.subplot(1, 3, 2)
    plt.title("GT (green) / Pred (red)\nOverlap = yellow")
    plt.axis("off")
    plt.imshow(over)

    plt.subplot(1, 3, 3)
    plt.title(f"Pred (thr={thr})\nDice={d:.3f} IoU={j:.3f}")
    plt.axis("off")
    plt.imshow(pred_np, cmap="gray")

    plt.suptitle(title, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return d, j


# ----------------------------- model loading --------------------------------
def load_ckpt(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        model.load_state_dict(ckpt["model"], strict=False)
        return ckpt
    # fallbacks
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return ckpt
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        model.load_state_dict(ckpt, strict=False)
        return ckpt
    raise RuntimeError(f"Unrecognized checkpoint format: {ckpt_path}")


# -------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None, help="Override checkpoint path (otherwise uses cfg paths/out_dir + best_ckpt)")
    ap.add_argument("--out", default="runs/unet/peek.png")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--idx", type=int, default=None, help="Pick a specific index from val set (otherwise random)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r"))
    paths = cfg["paths"]
    data_c = cfg["data"]
    model_c = cfg["model"]
    seed = cfg.get("seed", 42)

    # Build dataset using training-style transforms (resize, to_tensor, binarize)
    dataset = ISICLikeDebugDataset(
        paths["images_dir"],
        paths["masks_dir"],
        img_size=data_c["img_size"],
    )

    # same split logic as train.py
    val_len = max(1, int(len(dataset) * data_c["val_frac"]))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(seed),
    )

    # We only need val_set; no loader required, but we keep it simple.
    # Note: val_set returns (img_t, mask_t, img_name, mask_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_ch=model_c["in_ch"], out_ch=model_c["out_ch"], base=model_c["base"])
    model.to(device)
    model.eval()

    # determine checkpoint path like train.py
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = Path(paths["out_dir"]) / cfg["train"]["best_ckpt"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_ckpt(model, ckpt_path, device)

    out_base = Path(args.out)
    rng = random.Random(seed)

    for i in range(1, max(1, args.n) + 1):
        if args.idx is not None:
            j = args.idx
        else:
            j = rng.randrange(0, len(val_set))

        img_t, mask_t, img_name, mask_name = val_set[j]
        # add batch dimension
        x = img_t.unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            pred = (prob >= args.thr).astype(np.uint8)

        gt = mask_t[0].numpy().astype(np.uint8)

        # Convert image tensor back to numpy HxWx3 for plotting
        img_np = np.transpose(img_t.numpy(), (1, 2, 0))
        img_np = np.clip(img_np, 0, 1)

        title = f"{img_name}  |  {mask_name}  |  ckpt={ckpt_path.name}"
        out_i = out_base if args.n == 1 else out_base.with_name(f"{out_base.stem}_{i:02d}{out_base.suffix}")

        d, jacc = save_triptych(img_np, gt, pred, out_i, title=title, thr=args.thr)
        print(f"[val idx {j}] Dice={d:.4f} IoU={jacc:.4f} -> {out_i}")

if __name__ == "__main__":
    main()

from __future__ import annotations
from pathlib import Path
import re
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T

ISIC_RE = re.compile(r"(ISIC_\d{7})", re.IGNORECASE)

def extract_isic_key(path: Path) -> str | None:
    m = ISIC_RE.search(path.name)
    return m.group(1).upper() if m else None

class ISICKeySegDataset(Dataset):
    """
    Pairs images and masks by ISIC_####### key found in filenames.
    - image_dir: folder with RGB images
    - mask_dir:  folder with masks (single-channel)
    - img_size: resize target
    - augment: synced flips
    """
    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 256, augment: bool = False):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.img_size  = img_size
        self.augment   = augment

        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

        image_paths = [p for p in self.image_dir.iterdir() if p.suffix.lower() in exts]
        mask_paths  = [p for p in self.mask_dir.iterdir()  if p.suffix.lower() in exts]

        # Build key -> path maps
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

        # Intersect keys so we only keep valid pairs
        keys = sorted(set(img_map.keys()) & set(mask_map.keys()))
        if not keys:
            raise RuntimeError(
                "No image/mask pairs found by ISIC key. "
                "Check filenames contain ISIC_####### and dirs are correct."
            )

        # Store paired paths
        self.pairs = [(img_map[k], mask_map[k]) for k in keys]

        # Optionally: warn on missing
        missing_imgs = sorted(set(mask_map.keys()) - set(img_map.keys()))
        missing_masks = sorted(set(img_map.keys()) - set(mask_map.keys()))
        if missing_imgs:
            print(f"[WARN] {len(missing_imgs)} masks have no matching image (by ISIC key).")
        if missing_masks:
            print(f"[WARN] {len(missing_masks)} images have no matching mask (by ISIC key).")

        self.resize_img  = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)

        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img  = self.resize_img(img)
        mask = self.resize_mask(mask)

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img); mask = TF.hflip(mask)
            if random.random() < 0.5:
                img = TF.vflip(img); mask = TF.vflip(mask)

        img_t = TF.to_tensor(img)
        img_t = self.normalize(img_t)

        mask_t = TF.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()   # binarize (handles 0/255 too)

        return img_t, mask_t

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Optional
import re

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class ISICSegDataset(Dataset):
    """
    Loads RGB image and single-channel binary mask.
    - Resizes to img_size
    - Optional synchronized flips

    Pairing logic:
      - Extract ISIC_####### from filenames.
      - Build a mask map by ISIC id.
      - If multiple masks exist for an id: prefer _wvs over _ok, and prefer non-(1).
      - Optionally filter out images without a mask (recommended).
    """

    ISIC_RE = re.compile(r"(ISIC_\d{7})", re.IGNORECASE)

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: int = 256,
        augment: bool = False,
        filter_missing_masks: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augment = augment

        self.resize_img = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)

        # Collect image paths
        img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        all_imgs = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in img_exts])
        if len(all_imgs) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

        # Build mask map: isic_id -> best mask path
        self.mask_map: Dict[str, Path] = {}
        self._build_mask_map()

        # Filter images to only those with masks (recommended to avoid crashes)
        if filter_missing_masks:
            self.paths = [p for p in all_imgs if self._extract_isic_id(p.name) in self.mask_map]
        else:
            self.paths = all_imgs

        if len(self.paths) == 0:
            raise RuntimeError(
                f"No image/mask pairs found. "
                f"Check image_dir={self.image_dir} and mask_dir={self.mask_dir}"
            )

    def _extract_isic_id(self, name: str) -> Optional[str]:
        m = self.ISIC_RE.search(name)
        return m.group(1).upper() if m else None

    def _mask_score(self, fname_lower: str) -> int:
        # Higher score wins
        score = 0
        if "_wvs" in fname_lower:
            score += 100
        if "_ok" in fname_lower:
            score += 10
        if "(1)" in fname_lower:
            score -= 50
        if "copy of" in fname_lower:
            score -= 80
        return score

    def _build_mask_map(self) -> None:
        mask_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        candidates = [p for p in self.mask_dir.iterdir() if p.suffix.lower() in mask_exts]

        best_score: Dict[str, int] = {}

        for p in candidates:
            isic_id = self._extract_isic_id(p.name)
            if not isic_id:
                continue

            s = self._mask_score(p.name.lower())
            if (isic_id not in self.mask_map) or (s > best_score[isic_id]):
                self.mask_map[isic_id] = p
                best_score[isic_id] = s

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.paths[idx]
        isic_id = self._extract_isic_id(img_path.name)
        if not isic_id:
            raise ValueError(f"No ISIC id found in image filename: {img_path.name}")

        mpath = self.mask_map.get(isic_id)
        if mpath is None:
            # Only possible if filter_missing_masks=False
            raise FileNotFoundError(f"Mask not found for {isic_id}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mpath).convert("L")

        img = self.resize_img(img)
        mask = self.resize_mask(mask)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if torch.rand(1).item() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        img_t = TF.to_tensor(img)          # (3,H,W), [0,1]
        mask_t = TF.to_tensor(mask)        # (1,H,W), [0,1]
        mask_t = (mask_t > 0).float()
        return img_t, mask_t

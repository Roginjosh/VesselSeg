from __future__ import annotations
from pathlib import Path
from typing import Tuple
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
    Expects masks named: <image_stem>_mvig.tif[f]
    """
    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 256, augment: bool = False):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.paths = sorted([p for p in self.image_dir.iterdir()
                             if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'}])
        self.img_size = img_size
        self.augment  = augment

        self.resize_img  = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.paths[idx]
        base = img_path.stem
        mpath = self.mask_dir / f"{base}_mvig.tif"
        if not mpath.exists():
            mpath = self.mask_dir / f"{base}_mvig.tiff"
        if not mpath.exists():
            raise FileNotFoundError(f"Mask not found for {base}")

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mpath).convert('L')

        img  = self.resize_img(img)
        mask = self.resize_mask(mask)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img  = TF.hflip(img);  mask = TF.hflip(mask)
            if torch.rand(1).item() < 0.5:
                img  = TF.vflip(img);  mask = TF.vflip(mask)

        img_t  = TF.to_tensor(img)          # (3,H,W), [0,1]
        mask_t = TF.to_tensor(mask)         # (1,H,W), [0,1]
        mask_t = (mask_t > 0).float()
        return img_t, mask_t

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class VesselSegDataset(Dataset):
    """
    Segmentation dataset backed by a verified CSV.

    Expected CSV columns:
        id
        image_path
        mask_path
        match_type

    image_path and mask_path are interpreted relative to project_root
    unless they are already absolute paths.
    """

    def __init__(
        self,
        csv_path: str | Path = "data/dataset.csv",
        img_size: int = 256,
        augment: bool = False,
        project_root: str | Path | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.img_size = img_size
        self.augment = augment

        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {self.csv_path}"
            )

        self.df = pd.read_csv(self.csv_path)

        required_columns = {
            "id",
            "image_path",
            "mask_path",
        }

        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"Dataset CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if self.df.empty:
            raise RuntimeError("Dataset CSV contains no samples.")

        self.resize_img = T.Resize(
            (img_size, img_size),
            interpolation=T.InterpolationMode.BILINEAR,
        )

        self.resize_mask = T.Resize(
            (img_size, img_size),
            interpolation=T.InterpolationMode.NEAREST,
        )

        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, path_value: str) -> Path:
        normalized = str(path_value).replace("\\", "/")
        path = Path(normalized)

        if path.is_absolute():
            return path

        return self.project_root / path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self._resolve_path(row["image_path"])
        mask_path = self._resolve_path(row["mask_path"])

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found for sample {row['id']}: {img_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for sample {row['id']}: {mask_path}"
            )

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.resize_img(img)
        mask = self.resize_mask(mask)

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        img_t = TF.to_tensor(img)
        img_t = self.normalize(img_t)

        mask_t = TF.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t
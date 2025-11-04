""" IMPORTANT
DO NOT RUN THIS FILE AGAIN, IT WILL NOT WORK
I RAN THIS ONE TIME TO MAKE SURE EACH MASK HAD AN IMAGE ASSOCIATED WITH IT
"""



import os
import shutil
from pathlib import Path

# --- input folders ---
images_dir = Path('./ISIC_2019_Test_Input')
masks_dir  = Path('./masks')

# --- output folders ---
out_root   = Path('./data')
out_imgs   = out_root / 'imgs'
out_masks  = out_root / 'masks'

# make sure output dirs exist
out_imgs.mkdir(parents=True, exist_ok=True)
out_masks.mkdir(parents=True, exist_ok=True)

# --- gather file lists ---
image_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
mask_exts  = ('.tif', '.tiff')

images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_exts)]
masks  = [f for f in os.listdir(masks_dir)  if f.lower().endswith(mask_exts)]

# --- make a set of mask basenames (remove "_mvig") ---
mask_bases = set(os.path.splitext(f)[0].replace('_mvig', '') for f in masks)

# --- copy matching pairs ---
copied = 0
missing_masks = []

for fname in images:
    base = os.path.splitext(fname)[0]
    if base in mask_bases:
        src_img = images_dir / fname
        dst_img = out_imgs / fname
        shutil.copy2(src_img, dst_img)

        mask_name = f'{base}_mvig.tif'
        src_mask = masks_dir / mask_name
        # handle .tiff fallback
        if not src_mask.exists():
            src_mask = masks_dir / f'{base}_mvig.tiff'
        if src_mask.exists():
            shutil.copy2(src_mask, out_masks / src_mask.name)
            copied += 1
        else:
            missing_masks.append(base)
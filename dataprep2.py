import os
import re
from pathlib import Path
import shutil

MASK_DIR = "data/masks"
IMG_SRC_DIR = Path("C:\programming\ISIC_2019\ISIC_2019_Training_Input")  # <-- your ISIC image directory
IMG_DST_DIR = "data/imgs"

ISIC_RE = re.compile(r"(ISIC_\d{7})")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def extract_isic_ids(directory):
    ids = set()
    for fname in os.listdir(directory):
        m = ISIC_RE.search(fname)
        if m:
            ids.add(m.group(1))
    return ids


def main(dry_run=True, overwrite=False):
    os.makedirs(IMG_DST_DIR, exist_ok=True)

    mask_ids = extract_isic_ids(MASK_DIR)
    print(f"Found {len(mask_ids)} ISIC IDs with masks\n")

    copied = 0
    skipped = 0
    missing = 0

    for fname in os.listdir(IMG_SRC_DIR):
        m = ISIC_RE.search(fname)
        if not m:
            continue

        isic_id = m.group(1)
        ext = os.path.splitext(fname)[1].lower()

        if ext not in IMAGE_EXTS:
            continue

        if isic_id not in mask_ids:
            continue  # image has no mask → skip

        src = os.path.join(IMG_SRC_DIR, fname)
        dst = os.path.join(IMG_DST_DIR, fname)

        if os.path.exists(dst) and not overwrite:
            skipped += 1
            print(f"SKIP exists: {fname}")
            continue

        print(f"{'WOULD COPY' if dry_run else 'COPY'}: {fname}")
        if not dry_run:
            shutil.copy2(src, dst)
            copied += 1

    print("\nSummary")
    print(f"Copied:  {copied}")
    print(f"Skipped: {skipped}")
    print(f"Mask IDs: {len(mask_ids)}")


if __name__ == "__main__":
    # 1️⃣ Run with dry_run=True first
    # 2️⃣ Then flip to False once verified
    main(dry_run=False, overwrite=False)

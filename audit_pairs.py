import os, re
from collections import defaultdict

IMG_DIR = "data/imgs"     # or wherever your training images are
MASK_DIR = "data/masks"

ISIC_RE = re.compile(r"(ISIC_\d{7})")

def ids_in_dir(d):
    out = defaultdict(list)
    for f in os.listdir(d):
        m = ISIC_RE.search(f)
        if m:
            out[m.group(1)].append(f)
    return out

imgs = ids_in_dir(IMG_DIR)
masks = ids_in_dir(MASK_DIR)

img_ids = set(imgs.keys())
mask_ids = set(masks.keys())

print(f"Images: {len(img_ids)} IDs")
print(f"Masks : {len(mask_ids)} IDs")

missing_masks = sorted(img_ids - mask_ids)
missing_imgs  = sorted(mask_ids - img_ids)

print("\nIDs with images but NO masks:", len(missing_masks))
for i in missing_masks[:50]:
    print(" ", i, "->", imgs[i])
if len(missing_masks) > 50:
    print(" ...")

print("\nIDs with masks but NO images:", len(missing_imgs))
for i in missing_imgs[:50]:
    print(" ", i, "->", masks[i])
if len(missing_imgs) > 50:
    print(" ...")

# show the specific failing one
target = "ISIC_0024786"
print(f"\nLookup {target}:")
print("  imgs :", imgs.get(target))
print("  masks:", masks.get(target))

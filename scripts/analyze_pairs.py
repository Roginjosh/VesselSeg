from pathlib import Path
import pandas as pd
import re

img_dir = Path("data/imgs")
mask_dir = Path("data/masks")


def normalize_id(filename):
    stem = Path(filename).stem.strip()

    # DS dataset:
    # DS_0120617
    # DS_0120617mask
    match = re.match(r"^(DS_\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # ISIC dataset:
    # ISIC_0025510
    # ISIC_0025510_V_ahf_ok
    # ISIC-0025510_ok
    # Copy of ISIC_0025510_ok
    # ISIC_0012294_downsampled
    match = re.search(r"ISIC[_-](\d+)", stem, flags=re.IGNORECASE)
    if match:
        return f"ISIC_{match.group(1)}"

    # Everything else:
    # AH091208per115_CNP, etc.
    return stem

images = sorted(p for p in img_dir.iterdir() if p.is_file())
masks = sorted(p for p in mask_dir.iterdir() if p.is_file())

# Group files by normalized ID
image_groups = {}
mask_groups = {}

for p in images:
    key = normalize_id(p.name)
    image_groups.setdefault(key, []).append(p)

for p in masks:
    key = normalize_id(p.name)
    mask_groups.setdefault(key, []).append(p)


all_ids = sorted(set(image_groups) | set(mask_groups))

rows = []

for file_id in all_ids:
    img_files = image_groups.get(file_id, [])
    mask_files = mask_groups.get(file_id, [])

    # Classify the situation
    if len(img_files) == 1 and len(mask_files) == 1:
        img = img_files[0]
        mask = mask_files[0]

        if img.stem == mask.stem:
            match_type = "exact"
        else:
            match_type = "normalized"

        rows.append({
            "id": file_id,
            "image_filename": img.name,
            "mask_filename": mask.name,
            "match_type": match_type
        })

    elif len(img_files) == 0:
        for mask in mask_files:
            rows.append({
                "id": file_id,
                "image_filename": "",
                "mask_filename": mask.name,
                "match_type": "missing_image"
            })

    elif len(mask_files) == 0:
        for img in img_files:
            rows.append({
                "id": file_id,
                "image_filename": img.name,
                "mask_filename": "",
                "match_type": "missing_mask"
            })

    else:
        # Multiple images and/or masks normalize to same ID
        rows.append({
            "id": file_id,
            "image_filename": " | ".join(p.name for p in img_files),
            "mask_filename": " | ".join(p.name for p in mask_files),
            "match_type": "ambiguous"
        })


df = pd.DataFrame(rows)

# Sort useful pairs first, problems afterward
sort_order = {
    "exact": 0,
    "normalized": 1,
    "ambiguous": 2,
    "missing_mask": 3,
    "missing_image": 4
}

df["_sort"] = df["match_type"].map(sort_order)
df = df.sort_values(["_sort", "id"]).drop(columns="_sort")

# df.to_csv(
#     "normalized_filename_comparison.tsv",
#     sep="\t",
#     index=False
# )

print("Images:", len(images))
print("Masks:", len(masks))
print()

print(df["match_type"].value_counts())
print()

print("Total clean pairs:",
      len(df[df["match_type"].isin(["exact", "normalized"])]))

# print("\nSaved to normalized_filename_comparison.tsv")


# --------------------------------------------------
# Create final neural-network dataset table
# --------------------------------------------------

dataset_df = df[
    df["match_type"].isin(["exact", "normalized"])
].copy()

dataset_df["image_path"] = dataset_df["image_filename"].apply(
    lambda filename: str(img_dir / filename)
)

dataset_df["mask_path"] = dataset_df["mask_filename"].apply(
    lambda filename: str(mask_dir / filename)
)

dataset_df = dataset_df[
    ["id", "image_path", "mask_path" , "match_type"]
].reset_index(drop=True)

dataset_df.to_csv(
    "dataset.csv",
    index=False
)

print()
print("Final dataset:")
print(dataset_df.head())
print()
print(f"Total training pairs: {len(dataset_df)}")
print("Saved to dataset.csv")
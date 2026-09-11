from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT = Path("data/dataset.csv")
OUTPUT = Path("data/dataset_with_split.csv")

RANDOM_SEED = 42


def non_isic_group(sample_id: str) -> str:
    """
    Group related old non-ISIC samples together.

    Examples:
        abc123_CP
        abc123_CNP
        abc123_NCP

    all become:
        abc123
    """
    x = str(sample_id).strip()

    x = re.sub(
        r"_(CNP|CP|NCP)$",
        "",
        x,
        flags=re.IGNORECASE,
    )

    return x.lower()


def build_group_key(row: pd.Series) -> str:
    isic_id = row["isic_id"].strip()

    if isic_id:
        return f"ISIC::{isic_id}"

    return f"NONISIC::{non_isic_group(row['sample_id'])}"


def build_stratum(row: pd.Series) -> str:
    isic_id = row["isic_id"].strip()

    if isic_id:
        dx = row["diagnosis"].strip()

        if not dx:
            raise RuntimeError(
                f"ISIC row missing diagnosis: {row['sample_id']}"
            )

        return dx

    return "NON_ISIC"


def main():
    df = pd.read_csv(
        INPUT,
        dtype=str,
    ).fillna("")

    # ---------------------------------------------------------
    # Build leakage-safe group keys and strata
    # ---------------------------------------------------------

    df["group_key"] = df.apply(
        build_group_key,
        axis=1,
    )

    df["stratum"] = df.apply(
        build_stratum,
        axis=1,
    )

    # ---------------------------------------------------------
    # Verify every group has only one stratum
    # ---------------------------------------------------------

    group_strata = (
        df.groupby("group_key")["stratum"]
        .nunique()
    )

    bad_groups = group_strata[
        group_strata > 1
    ]

    if not bad_groups.empty:
        raise RuntimeError(
            "Some leakage groups contain multiple strata:\n"
            + bad_groups.to_string()
        )

    # ---------------------------------------------------------
    # Build one row per leakage group
    # ---------------------------------------------------------

    groups = (
        df.groupby("group_key")
        .agg(
            stratum=("stratum", "first"),
            row_count=("sample_id", "size"),
        )
        .reset_index()
    )

    print("Total rows:", len(df))
    print("Total groups:", len(groups))
    print()

    print("Group counts by stratum:")
    print(
        groups["stratum"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # First split:
    # 80% train, 20% temporary
    # ---------------------------------------------------------

    train_groups, temp_groups = train_test_split(
        groups,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=groups["stratum"],
    )

    # ---------------------------------------------------------
    # Second split:
    # divide temporary 20% equally -> 10% val / 10% test
    # ---------------------------------------------------------

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_groups["stratum"],
    )

    # ---------------------------------------------------------
    # Assign split names
    # ---------------------------------------------------------

    split_map = {}

    for key in train_groups["group_key"]:
        split_map[key] = "train"

    for key in val_groups["group_key"]:
        split_map[key] = "val"

    for key in test_groups["group_key"]:
        split_map[key] = "test"

    df["split"] = df["group_key"].map(
        split_map
    )

    if df["split"].isna().any():
        raise RuntimeError(
            "Some rows were not assigned a split."
        )

    # ---------------------------------------------------------
    # Leakage checks
    # ---------------------------------------------------------

    group_split_counts = (
        df.groupby("group_key")["split"]
        .nunique()
    )

    leaking_groups = group_split_counts[
        group_split_counts > 1
    ]

    if not leaking_groups.empty:
        raise RuntimeError(
            "LEAKAGE DETECTED: a group appears in multiple splits."
        )

    # ISIC leakage check
    isic = df[
        df["isic_id"].str.strip() != ""
    ]

    isic_leaks = (
        isic.groupby("isic_id")["split"]
        .nunique()
    )

    isic_leaks = isic_leaks[
        isic_leaks > 1
    ]

    if not isic_leaks.empty:
        raise RuntimeError(
            "LEAKAGE DETECTED: ISIC IDs cross splits."
        )

    # Exact image leakage check
    image_leaks = (
        df.groupby("image_path")["split"]
        .nunique()
    )

    image_leaks = image_leaks[
        image_leaks > 1
    ]

    if not image_leaks.empty:
        raise RuntimeError(
            "LEAKAGE DETECTED: image paths cross splits."
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(f"Wrote: {OUTPUT}")

    # ---------------------------------------------------------
    # Summary by ROW
    # ---------------------------------------------------------

    print()
    print("ROW COUNTS BY SPLIT")
    print(
        df["split"]
        .value_counts()
        .reindex(["train", "val", "test"])
        .to_string()
    )

    print()
    print("ROW DISTRIBUTION BY SPLIT / STRATUM")

    row_table = pd.crosstab(
        df["split"],
        df["stratum"],
    )

    row_table = row_table.reindex(
        ["train", "val", "test"]
    )

    print(row_table.to_string())

    print()
    print("ROW PERCENTAGES WITHIN EACH SPLIT")

    row_pct = (
        row_table.div(
            row_table.sum(axis=1),
            axis=0,
        )
        * 100
    )

    print(
        row_pct.round(2).to_string()
    )

    # ---------------------------------------------------------
    # Summary by GROUP
    # ---------------------------------------------------------

    group_assignment = groups.copy()

    group_assignment["split"] = (
        group_assignment["group_key"]
        .map(split_map)
    )

    print()
    print("GROUP COUNTS BY SPLIT")

    print(
        group_assignment["split"]
        .value_counts()
        .reindex(["train", "val", "test"])
        .to_string()
    )

    print()
    print("GROUP DISTRIBUTION BY SPLIT / STRATUM")

    group_table = pd.crosstab(
        group_assignment["split"],
        group_assignment["stratum"],
    )

    group_table = group_table.reindex(
        ["train", "val", "test"]
    )

    print(
        group_table.to_string()
    )

    print()
    print("Leakage checks passed:")
    print("  group_key crossing splits: 0")
    print("  ISIC IDs crossing splits:  0")
    print("  image paths crossing splits: 0")


if __name__ == "__main__":
    main()
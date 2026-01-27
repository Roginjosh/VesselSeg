import os
import re
import shutil
from collections import defaultdict

SRC_DIR = "data/raw"
DST_DIR = "data/masks"

ISIC_RE = re.compile(r"(ISIC_\d{7})")
PAREN_1_RE = re.compile(r"\s*\(\s*1\s*\)")  # matches " (1)" with flexible spacing


def has_paren_one(name: str) -> bool:
    # catches "ok (1).tiff"
    return "(1)" in name or bool(PAREN_1_RE.search(name))


def is_ok(name: str) -> bool:
    # conservative: matches "..._ok.tif" / "..._ok.tiff" and also "..._ok (1).tiff"
    return "_ok" in name.lower()


def is_wvs(name: str) -> bool:
    return "_wvs" in name.lower()


def main(dry_run: bool = True, overwrite: bool = False):
    os.makedirs(DST_DIR, exist_ok=True)

    id_to_files = defaultdict(list)

    for fname in os.listdir(SRC_DIR):
        m = ISIC_RE.search(fname)
        if m:
            id_to_files[m.group(1)].append(fname)

    to_copy = []

    for isic_id, files in sorted(id_to_files.items()):
        # Rule: if both _wvs and _ok exist, choose only _wvs (prefer non-(1) if present)
        wvs_files = [f for f in files if is_wvs(f)]
        ok_files = [f for f in files if is_ok(f)]

        chosen = []

        if wvs_files and ok_files:
            # pick best wvs: prefer not having "(1)"
            wvs_no_1 = [f for f in wvs_files if not has_paren_one(f)]
            chosen_wvs = sorted(wvs_no_1 or wvs_files)[0]
            chosen = [chosen_wvs]
        else:
            # No ok+wvs conflict. If multiple files (duplicates), drop "(1)" versions when possible.
            if len(files) == 1:
                chosen = files
            else:
                no_1 = [f for f in files if not has_paren_one(f)]
                # If everything is "(1)" (rare), fall back to all files
                chosen = no_1 if no_1 else files

        for f in chosen:
            to_copy.append((isic_id, f))

    # Execute
    copied = 0
    skipped = 0

    print(f"Source:      {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print(f"Mode:        {'DRY RUN' if dry_run else 'COPY'}")
    print()

    for isic_id, fname in to_copy:
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(DST_DIR, fname)

        if (not overwrite) and os.path.exists(dst):
            skipped += 1
            print(f"SKIP exists: {fname}")
            continue

        print(f"{'WOULD COPY' if dry_run else 'COPY'}: {fname}")
        if not dry_run:
            shutil.copy2(src, dst)
            copied += 1

    print()
    print(f"Selected: {len(to_copy)}")
    print(f"Copied:   {copied}")
    print(f"Skipped:  {skipped}")


if __name__ == "__main__":
    # 1) Run once with dry_run=True to verify what it would copy
    # 2) Then set dry_run=False to actually copy
    main(dry_run=False, overwrite=False)

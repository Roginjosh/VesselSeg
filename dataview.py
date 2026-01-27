import os
import re
from collections import Counter

directory = "data/raw"

cleaned_names = []

for fname in os.listdir(directory):
    cleaned = re.sub(r"ISIC_\d{7}_?", "", fname)
    cleaned_names.append(cleaned)

counts = Counter(cleaned_names)

for name, count in counts.most_common():
    print(f"{count:4d}  {name}")


from collections import defaultdict


id_to_files = defaultdict(list)

for fname in os.listdir(directory):
    match = re.search(r"(ISIC_\d{7})", fname)
    if match:
        id_to_files[match.group(1)].append(fname)

found_any = False

for isic_id, files in sorted(id_to_files.items()):
    if len(files) > 1:
        found_any = True
        print(f"\n{isic_id}  ({len(files)} files)")
        for f in files:
            print(f"  - {f}")

if not found_any:
    print("✅ No duplicate ISIC IDs found.")



extensions = []

for fname in os.listdir(directory):
    if "." in fname:
        ext = os.path.splitext(fname)[1].lower()
        extensions.append(ext)
    else:
        extensions.append("(no extension)")

counts = Counter(extensions)

for ext, count in counts.most_common():
    print(f"{count:4d}  {ext}")

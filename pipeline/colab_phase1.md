# Phase 1 on Colab: rebuild the data and the splits

Use a **CPU runtime** (Runtime > Change runtime type > CPU). No GPU is needed. Paste each block below into its own cell and run the cells in order. If a cell fails, stop, and send back the **last 40 lines** of its output.

Drive folders used:

- inputs: `MyDrive/WasteVision_final/data/` and `MyDrive/WasteVision_final/results/`
- outputs: `MyDrive/WasteVision_v3/data/` and `MyDrive/WasteVision_v3/audit/`

## Step 1: mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Expect: a Google sign-in pop-up the first time, then `Mounted at /content/drive`.

## Step 2: get the code

```bash
%%bash
cd /content
if [ -d WasteVision/.git ]; then
  git -C WasteVision pull --ff-only
else
  git clone https://github.com/ayushdhasmana433-rgb/WasteVision
fi
cd WasteVision/pipeline
ls
pip install -q imagehash
```

Expect: a list of files that includes `add_extra_images.py`, `rebuild_splits.py` and `make_audit_sheets.py`. If one of the three is missing, stop and report it: the code is not pushed yet.

## Step 3: check the five input files

```python
import os
base = "/content/drive/MyDrive/WasteVision_final"
names = ["data/normalized_images.tar", "data/all_images_before_dedup.csv", "results/duplicate_pairs.csv.gz",
         "data/frozen_features_effv2b0.npz", "data/splits.csv"]
for n in names:
    p = f"{base}/{n}"
    print(f"{os.path.getsize(p)/1e6:10.1f} MB  {n}" if os.path.exists(p) else f"   MISSING     {n}")
```

Expect: five lines with sizes (the tar is the big one, probably several GB). Any line that says `MISSING` means stop here and send that output.

## Step 4: extract the images to /content/data

```bash
%%bash
mkdir -p /content/data
tar -xf /content/drive/MyDrive/WasteVision_final/data/normalized_images.tar -C /content/data
ls /content/data
find /content/data -type f | wc -l
```

Expect: this takes a few minutes. `ls` lists one folder per source (for example `RealWaste`, `TrashNet`, `GD`, `TrashBox`), and the file count is in the tens of thousands. If you see `No space left on device`, send that output.

## Step 5: fetch and prepare the 755 extra images (TrashBox gloves and masks)

```bash
%%bash
cd /content/WasteVision/pipeline
mkdir -p /content/drive/MyDrive/WasteVision_v3/data
python -u add_extra_images.py \
  --raw-dir /content/raw_extra \
  --data-root /content/data \
  --out-dir /content/drive/MyDrive/WasteVision_v3/data \
  --all-csv /content/drive/MyDrive/WasteVision_final/data/all_images_before_dedup.csv
```

Expect: some `git clone` lines, then a small table like

```
Folder            found  rejected  written
medical/gloves      355         0      355   rejected reasons: none
medical/masks       400         0      400   rejected reasons: none
```

The numbers should be close to 355 and 400 (a few rejected images are fine) and the cell ends with `Wrote ... rows to .../extra_images.csv` and `Wrote ... images to .../extra_images.tar`. It is safe to run this cell again. If it stops with `ERROR: ... collide`, send the message; do not edit anything.

## Step 6: rebuild the splits

```bash
%%bash
cd /content/WasteVision/pipeline
mkdir -p /content/drive/MyDrive/WasteVision_v3/data
python -u rebuild_splits.py \
  --all-csv /content/drive/MyDrive/WasteVision_final/data/all_images_before_dedup.csv \
  --extra-csv /content/drive/MyDrive/WasteVision_v3/data/extra_images.csv \
  --features /content/drive/MyDrive/WasteVision_final/data/frozen_features_effv2b0.npz \
  --old-splits /content/drive/MyDrive/WasteVision_final/data/splits.csv \
  --images-root /content/data \
  --out-dir /content/drive/MyDrive/WasteVision_v3/data
```

Expect: many progress lines, ending with all checks passed. This can take several minutes.

One warning is normal: `WARNING: 755 of ... surviving images (...) have no row in the features file; they are not checked for feature duplicates`. The 755 extra images were never put through the feature model, so they have no saved features. (The number can be a little lower than 755 after earlier de-duplication steps, or a little different if a few images were rejected in Step 5.) It is only a warning and the script continues.

If instead the script stops with a message that says more than 5% of images have no feature row, do not change anything: send the message.

If a check fails, the script writes `splits_v3_FAILED.csv` and `report.json` instead of `splits_v3.csv`. Send the last 40 lines and `report.json`.

When it works, `MyDrive/WasteVision_v3/data/` contains `splits_v3.csv`, `count_table.md`, `count_table.csv`, `report.json` and more.

## Step 7: make the audit sheets

```bash
%%bash
cd /content/WasteVision/pipeline
mkdir -p /content/drive/MyDrive/WasteVision_v3/audit
python -u make_audit_sheets.py \
  --splits /content/drive/MyDrive/WasteVision_v3/data/splits_v3.csv \
  --images-root /content/data \
  --out-dir /content/drive/MyDrive/WasteVision_v3/audit \
  --pairs /content/drive/MyDrive/WasteVision_final/results/duplicate_pairs.csv.gz \
  --all-csv /content/drive/MyDrive/WasteVision_final/data/all_images_before_dedup.csv \
  --skip-first 20 --max-pairs 180
```

This one command makes both sets of sheets:

- the e-waste sheets and the medicine sheets (up to 300 images each, 30 per sheet);
- the duplicate-audit pairs, numbered from 21 on (the first 20 were already reviewed), at most 180 pairs, 20 per sheet.

Expect: lines such as `[e_waste] available in splits: ..., sampled: 300 ...`, `[medicines_syringes] ...`, `[pairs] considered: 180, drawn: ...`, then a `Files written:` list. A few `skipped missing/unreadable` lines are fine if the count is small. If most images are reported missing, Step 4 did not finish: send the output.

## Step 8: what to look at and send back

Send these (or tell Ayush they are on Drive):

1. `MyDrive/WasteVision_v3/data/count_table.md`
2. `MyDrive/WasteVision_v3/data/report.json`

Review these in `MyDrive/WasteVision_v3/audit/`:

- `ewaste_sheet_*.jpg` and `medicines_sheet_*.jpg`: look for batteries and power banks sitting in the wrong class (note the caption number on the sheet and the file `..._sheet_index.csv`).
- `pairs_sheet_*.jpg` with `pairs_sheet_index.csv`: for each pair fill in the last column with `y` (same photo), `n` or `similar`.

Send back the filled-in index CSV files or a list of the numbers you flagged.

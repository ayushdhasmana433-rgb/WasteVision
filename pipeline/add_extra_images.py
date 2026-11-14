#!/usr/bin/env python3
"""WasteVision phase 1, step 0: fetch and prepare the two TrashBox folders that the original run dropped.

Adds `medical/gloves` and `medical/masks` (TrashBox), both labelled `sanitary`, in exactly the way the original
notebook would have prepared them, so that the new rows can be appended to all_images_before_dedup.csv
(via rebuild_splits.py --extra-csv) and the new images dropped into the same image folder.

Every rule below is copied from the reference notebook cells in reference_notebook_cells/ (line numbers refer
to those files):

  cell_05.py l.23      IMG_EXTS (the image file extensions that count as images)
  cell_05.py l.25-26   TRASHBOX_REPO, TRASHBOX_SUBDIR
  cell_05.py l.39-41   run(): print the command, run it, stop on a non-zero exit
  cell_05.py l.148-149 git clone --depth 1 --filter=blob:none --sparse <repo> <dest>, then
                       git -C <dest> sparse-checkout set <SUBDIR>/<folder> ...  (reference sets medical and
                       e-waste; here only medical/gloves and medical/masks)
  cell_05.py l.143-144 clone destination <raw>/trashbox; images are read from <dest>/<TRASHBOX_SUBDIR>
  cell_02.py l.93-94   stable_id(*parts) = sha1("|".join(parts))[:16]
  cell_06.py l.3-4     list_images(root): sorted, recursive, regular files with suffix.lower() in IMG_EXTS
  cell_06.py l.204-210 TrashBox rows: rel = os.path.relpath(path, <SUBDIR root>); parts = rel.split(os.sep);
                       top = parts[0]; sub = parts[1] if len(parts) > 2 else None;
                       orig_class = "top/sub"; source = "TrashBox".  (label differs on purpose: the reference's
                       trashbox_label() returned None for gloves and masks, here the label is "sanitary".)
  cell_06.py l.259     id = stable_id(source, rel)
  cell_06.py l.260-262 group = source + "_" + id   (TrashBox rows have no group of their own)
  cell_08.py l.2       NORM_MAX_SIDE = 320
  cell_08.py l.4-34    normalize_one(), the branch without a box (bbox is None): im0.draft("RGB", (640, 640)); EXIF
                       transpose; transparent images are flattened on white; everything else convert("RGB");
                       reject "too_small" if min(w, h) < 24 (the min_side used for rows without a box, l.45);
                       thumbnail to at most 320 px with LANCZOS; JPEG quality 93; phash string =
                       str(imagehash.phash(im)) taken from the thumbnailed image; any exception is rejected
                       as "unreadable (<ExceptionName>)"
  cell_08.py l.39,53   norm_rel = f"{source}/{id}.jpg", written at <data root>/<norm_rel>
  cell_09.py l.171     CSV columns: id, source, orig_class, label, group, norm_rel, phash (index=False)
  cell_09.py l.177-180 (tar) every normalised file added with arcname = norm_rel

Outputs in --out-dir: extra_images.csv and extra_images.tar.  Running the script twice gives identical ids,
norm_rel paths and file names and duplicates nothing.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
import imagehash

# ---- settings copied from the reference (cell_05 l.23-26, cell_08 l.2 and l.45 and l.31) ----
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
TRASHBOX_REPO = "https://github.com/nikhilvenkatkumsetty/TrashBox"
TRASHBOX_SUBDIR = "TrashBox_train_dataset_subfolders"
NORM_MAX_SIDE = 320
MIN_SIDE_NO_BOX = 24
JPEG_QUALITY = 93
SOURCE = "TrashBox"
LABEL = "sanitary"
FOLDERS = ["medical/gloves", "medical/masks"]   # the only folders this script adds
COLS = ["id", "source", "orig_class", "label", "group", "norm_rel", "phash"]
GIT_RETRIES = 3


# ---------------------------------------------------------------- copied reference helpers

def stable_id(*parts):                                   # cell_02 l.93-94
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def list_images(root):                                   # cell_06 l.3-4
    return sorted(str(p) for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def run(cmd, **kw):                                      # cell_05 l.39-41
    print("$", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    subprocess.run(cmd, check=True, **kw)


def normalize_one(job):
    """cell_08 l.4-34 for a row without a box (bbox None): returns ((phash, w, h), None) or (None, reason)."""
    src, dst, min_side = job
    try:
        with Image.open(src) as im0:
            im0.draft("RGB", (2 * NORM_MAX_SIDE, 2 * NORM_MAX_SIDE))
            im = ImageOps.exif_transpose(im0)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.getchannel("A"))
                im = bg
            else:
                im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < min_side:
                return None, "too_small"
            im.thumbnail((NORM_MAX_SIDE, NORM_MAX_SIDE), Image.Resampling.LANCZOS)
            im.save(dst, "JPEG", quality=JPEG_QUALITY)
            return (str(imagehash.phash(im)), w, h), None
    except Exception as e:
        return None, f"unreadable ({type(e).__name__})"


# ---------------------------------------------------------------- fetching

def sparse_clone(raw_dir):
    """Sparse clone with only the wanted folders (git options as cell_05 l.148-149), 3 retries."""
    dest = os.path.join(raw_dir, "trashbox")
    root = os.path.join(dest, TRASHBOX_SUBDIR)
    if all(list_images(os.path.join(root, f)) for f in FOLDERS):
        print(f"Re-using the existing clone in {dest}")
        return root
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    last = None
    for attempt in range(GIT_RETRIES + 1):
        if attempt:
            wait = 10 * attempt
            print(f"git failed ({last}); retry {attempt}/{GIT_RETRIES} in {wait}s", flush=True)
            time.sleep(wait)
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(raw_dir, exist_ok=True)
        try:
            run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", TRASHBOX_REPO, dest], env=env)
            run(["git", "-C", dest, "sparse-checkout", "set"] + [f"{TRASHBOX_SUBDIR}/{f}" for f in FOLDERS], env=env)
            return root
        except (subprocess.CalledProcessError, OSError) as e:
            last = repr(e)
    sys.exit(f"ERROR: git clone/sparse-checkout still failing after {GIT_RETRIES} retries: {last}")


# ---------------------------------------------------------------- rows

def build_rows(root):
    """cell_06 l.204-210 and l.259-262, restricted to FOLDERS. `rel` is relative to the TrashBox subdir root."""
    rows, found = [], {f: 0 for f in FOLDERS}
    for folder in FOLDERS:
        for p in list_images(os.path.join(root, folder)):
            rel = os.path.relpath(p, root)
            parts = rel.split(os.sep)
            top, sub = parts[0], parts[1] if len(parts) > 2 else None
            orig_class = f"{top}/{sub}" if sub else top
            if orig_class != folder:   # cannot happen for files under <folder>/, kept as a guard
                sys.exit(f"ERROR: {rel} gives orig_class {orig_class!r}, expected {folder!r}")
            rid = stable_id(SOURCE, rel)
            rows.append(dict(id=rid, source=SOURCE, src_path=p, rel=rel, orig_class=orig_class, label=LABEL,
                             group=SOURCE + "_" + rid, norm_rel=f"{SOURCE}/{rid}.jpg"))
            found[folder] += 1
    return rows, found


def check_collisions(rows, all_csv):
    ids = [r["id"] for r in rows]
    rels = [r["norm_rel"] for r in rows]
    if len(set(ids)) != len(ids) or len(set(rels)) != len(rels):
        sys.exit("ERROR: duplicate id or norm_rel among the new images themselves")
    old = pd.read_csv(all_csv, dtype=str, usecols=["id", "norm_rel"])
    id_hit = sorted(set(ids) & set(old["id"]))
    rel_hit = sorted(set(rels) & set(old["norm_rel"]))
    if id_hit or rel_hit:
        sys.exit(f"ERROR: the new images collide with {all_csv}: {len(id_hit)} ids (e.g. {id_hit[:3]}) and "
                 f"{len(rel_hit)} norm_rel paths (e.g. {rel_hit[:3]}). If this CSV already contains the extra "
                 "images, give the ORIGINAL all_images_before_dedup.csv instead.")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch and prepare TrashBox medical/gloves and medical/masks (label sanitary).")
    ap.add_argument("--raw-dir", default="/content/raw_extra", help="where to clone")
    ap.add_argument("--data-root", default="/content/data", help="where normalised images live")
    ap.add_argument("--out-dir", default="/content/drive/MyDrive/WasteVision_v3/data")
    ap.add_argument("--all-csv", required=True, help="existing all_images_before_dedup.csv (collision check only)")
    ap.add_argument("--local-source", default=None,
                    help="folder that already contains medical/gloves and medical/masks (used instead of cloning)")
    args = ap.parse_args(argv)

    root = args.local_source if args.local_source else sparse_clone(args.raw_dir)
    rows, found = build_rows(root)
    for f in FOLDERS:
        if not found[f]:
            sys.exit(f"ERROR: no images found in {os.path.join(root, f)}")
    check_collisions(rows, args.all_csv)

    os.makedirs(f"{args.data_root}/{SOURCE}", exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [(r["src_path"], f"{args.data_root}/{r['norm_rel']}", MIN_SIDE_NO_BOX) for r in rows]
    with ThreadPoolExecutor(max_workers=min(16, 4 * (os.cpu_count() or 2))) as ex:
        results = list(ex.map(normalize_one, jobs, chunksize=32))

    kept, rejected = [], {f: {} for f in FOLDERS}
    for r, (ok, why) in zip(rows, results):
        if ok:
            kept.append({**{c: r[c] for c in COLS if c != "phash"}, "phash": ok[0]})
        else:
            rejected[r["orig_class"]][why] = rejected[r["orig_class"]].get(why, 0) + 1
            stale = f"{args.data_root}/{r['norm_rel']}"
            if os.path.exists(stale):          # left by an earlier run or a partly written file
                os.remove(stale)
    df = pd.DataFrame(kept, columns=COLS)

    print("\nFolder            found  rejected  written")
    for f in FOLDERS:
        n_rej = sum(rejected[f].values())
        n_ok = int((df.orig_class == f).sum()) if len(df) else 0
        print(f"{f:<16} {found[f]:>6} {n_rej:>9} {n_ok:>8}   rejected reasons: {rejected[f] or 'none'}")
        if not n_ok:
            sys.exit(f"ERROR: no image from {f} survived cleaning")

    csv_path = os.path.join(args.out_dir, "extra_images.csv")
    tar_path = os.path.join(args.out_dir, "extra_images.tar")
    df.to_csv(csv_path, index=False)
    with tarfile.open(tar_path + ".part", "w") as t:
        for rel in df.norm_rel:
            t.add(f"{args.data_root}/{rel}", arcname=rel)
    os.replace(tar_path + ".part", tar_path)
    print(f"\nWrote {len(df)} rows to {csv_path}")
    print(f"Wrote {len(df)} images to {tar_path} (paths relative to the data root {args.data_root})")
    return df


if __name__ == "__main__":
    main()

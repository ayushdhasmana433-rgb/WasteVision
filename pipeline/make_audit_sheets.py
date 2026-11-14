#!/usr/bin/env python3
"""Build manual-audit contact sheets for WasteVision.

(a) Single-image sheets for two labels (default: e_waste and medicines_syringes):
    up to 300 randomly sampled images per label, 5 columns x 6 rows (30 per sheet),
    256x256 padded thumbnails with a caption (number, source, first 10 chars of id),
    plus an index CSV with empty reviewer columns.

(b) Optional pair sheets (--pairs): side-by-side pairs, 20 per sheet, captioned with
    number, both sources, both labels and distance, plus an index CSV with an empty
    "same_photo (y/n/similar)" column.

Uses pandas and Pillow only. Missing or unreadable images are skipped with a warning.
Pair numbers are the 1-based row numbers in the full pairs CSV, so --skip-first N
continues the numbering where the earlier review stopped.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

THUMB = 256
MARGIN = 6
BG = (235, 235, 235)
TEXT = (20, 20, 20)
CAP_LINE = 16
GRID_COLS, GRID_ROWS = 5, 6
PAIR_COLS, PAIR_ROWS = 2, 10
AUDIT_LABELS = [("e_waste", "ewaste"), ("medicines_syringes", "medicines")]


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr)


def load_thumb(path):
    """Return a THUMB x THUMB padded RGB image, or None if the file cannot be read."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB, THUMB), Image.LANCZOS)
    except Exception as exc:  # missing, truncated or non-image file
        warn(f"cannot read image {path}: {exc}")
        return None
    canvas = Image.new("RGB", (THUMB, THUMB), BG)
    canvas.paste(im, ((THUMB - im.width) // 2, (THUMB - im.height) // 2))
    return canvas


def resolve(images_root, rel):
    return Path(images_root) / str(rel)


def draw_lines(draw, x, y, lines, font):
    for i, line in enumerate(lines):
        draw.text((x, y + i * CAP_LINE), line, fill=TEXT, font=font)


def write_single_sheets(items, out_dir, prefix, font):
    """items: list of dicts with number, id, source, label, img (PIL thumb)."""
    per_sheet = GRID_COLS * GRID_ROWS
    cell_w = THUMB + 2 * MARGIN
    cell_h = THUMB + CAP_LINE + 2 * MARGIN
    written = []
    for s in range(0, len(items), per_sheet):
        chunk = items[s:s + per_sheet]
        sheet = Image.new("RGB", (GRID_COLS * cell_w, GRID_ROWS * cell_h), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for k, it in enumerate(chunk):
            col, row = k % GRID_COLS, k // GRID_COLS
            x0, y0 = col * cell_w, row * cell_h
            sheet.paste(it["img"], (x0 + MARGIN, y0 + MARGIN))
            caption = f"{it['number']} {str(it['source'])[:18]} {str(it['id'])[:10]}"
            draw_lines(draw, x0 + MARGIN, y0 + MARGIN + THUMB + 2, [caption], font)
        path = Path(out_dir) / f"{prefix}_sheet_{len(written) + 1:02d}.jpg"
        sheet.save(path, "JPEG", quality=90)
        written.append(path)
    return written


def run_single_audit(splits, images_root, out_dir, seed, font):
    summary = []
    for label, prefix in AUDIT_LABELS:
        subset = splits[splits["label"] == label]
        shuffled = subset.sample(frac=1, random_state=seed)
        items, missing = [], 0
        for _, row in shuffled.iterrows():
            if len(items) >= 300:
                break
            img = load_thumb(resolve(images_root, row["norm_rel"]))
            if img is None:
                missing += 1
                continue
            items.append({
                "id": row["id"], "source": row["source"], "label": row["label"],
                "norm_rel": row["norm_rel"], "img": img,
            })
        for n, it in enumerate(items, 1):
            it["number"] = n
        sheets = write_single_sheets(items, out_dir, prefix, font)
        index = pd.DataFrame([{
            "number": it["number"], "id": it["id"], "source": it["source"],
            "label": it["label"], "norm_rel": it["norm_rel"],
            "battery_or_powerbank (y/n)": "", "note": "",
        } for it in items], columns=["number", "id", "source", "label", "norm_rel",
                                     "battery_or_powerbank (y/n)", "note"])
        index_path = Path(out_dir) / f"{prefix}_sheet_index.csv"
        index.to_csv(index_path, index=False)
        print(f"[{label}] available in splits: {len(subset)}, sampled: {len(items)}, "
              f"skipped missing/unreadable: {missing}, sheets: {len(sheets)}")
        summary.extend(sheets)
        summary.append(index_path)
    return summary


def build_lookup(splits, all_csv):
    lookup = dict(zip(splits["id"].astype(str), splits["norm_rel"].astype(str)))
    if all_csv:
        extra = pd.read_csv(all_csv, dtype=str, usecols=["id", "norm_rel"])
        for i, r in zip(extra["id"], extra["norm_rel"]):
            lookup.setdefault(str(i), str(r))
    return lookup


def run_pair_audit(pairs, lookup, images_root, out_dir, skip_first, max_pairs, font):
    pairs = pairs.reset_index(drop=True)
    pairs["number"] = pairs.index + 1  # position in the full pairs CSV
    todo = pairs.iloc[skip_first:]
    if max_pairs is not None:
        todo = todo.head(max_pairs)

    kept = []
    for _, r in todo.iterrows():
        rel_a = r["rel_a"] if "rel_a" in todo.columns and pd.notna(r.get("rel_a")) else lookup.get(str(r["id_a"]))
        rel_b = r["rel_b"] if "rel_b" in todo.columns and pd.notna(r.get("rel_b")) else lookup.get(str(r["id_b"]))
        if rel_a is None or rel_b is None:
            warn(f"pair {r['number']}: norm_rel not found for "
                 f"{'id_a' if rel_a is None else ''} {'id_b' if rel_b is None else ''}".rstrip())
            continue
        img_a = load_thumb(resolve(images_root, rel_a))
        img_b = load_thumb(resolve(images_root, rel_b))
        if img_a is None or img_b is None:
            continue  # warning already printed by load_thumb
        kept.append({"r": r, "rel_a": rel_a, "rel_b": rel_b, "img_a": img_a, "img_b": img_b})

    per_sheet = PAIR_COLS * PAIR_ROWS
    cell_w = 2 * THUMB + 3 * MARGIN
    cap_h = 3 * CAP_LINE
    cell_h = THUMB + cap_h + 2 * MARGIN
    written, index_rows = [], []
    for s in range(0, len(kept), per_sheet):
        chunk = kept[s:s + per_sheet]
        sheet_no = s // per_sheet + 1
        sheet = Image.new("RGB", (PAIR_COLS * cell_w, PAIR_ROWS * cell_h), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        sheet_name = f"pairs_sheet_{sheet_no:02d}.jpg"
        for k, item in enumerate(chunk):
            r = item["r"]
            col, row = k % PAIR_COLS, k // PAIR_COLS
            x0, y0 = col * cell_w, row * cell_h
            sheet.paste(item["img_a"], (x0 + MARGIN, y0 + MARGIN))
            sheet.paste(item["img_b"], (x0 + 2 * MARGIN + THUMB, y0 + MARGIN))
            lines = [
                f"#{r['number']}   distance {r['distance']}",
                f"A: {r['source_a']} / {r['label_a']}",
                f"B: {r['source_b']} / {r['label_b']}",
            ]
            draw_lines(draw, x0 + MARGIN, y0 + MARGIN + THUMB + 2, lines, font)
            index_rows.append({
                "number": r["number"], "id_a": r["id_a"], "id_b": r["id_b"],
                "source_a": r["source_a"], "source_b": r["source_b"],
                "label_a": r["label_a"], "label_b": r["label_b"],
                "distance": r["distance"], "rel_a": item["rel_a"], "rel_b": item["rel_b"],
                "sheet": sheet_name, "same_photo (y/n/similar)": "",
            })
        path = Path(out_dir) / sheet_name
        sheet.save(path, "JPEG", quality=90)
        written.append(path)

    index_cols = ["number", "id_a", "id_b", "source_a", "source_b", "label_a", "label_b",
                  "distance", "rel_a", "rel_b", "sheet", "same_photo (y/n/similar)"]
    index_path = Path(out_dir) / "pairs_sheet_index.csv"
    pd.DataFrame(index_rows, columns=index_cols).to_csv(index_path, index=False)
    print(f"[pairs] considered: {len(todo)}, drawn: {len(kept)}, sheets: {len(written)}")
    return written + [index_path]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", required=True, help="CSV with columns id, source, label, norm_rel, split")
    p.add_argument("--images-root", required=True, help="folder that norm_rel is relative to")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pairs", help="CSV with id_a, id_b, source_a, source_b, label_a, label_b, distance "
                                   "and optionally rel_a, rel_b")
    p.add_argument("--all-csv", help="optional CSV with id, norm_rel used to look up paths for pairs")
    p.add_argument("--skip-first", type=int, default=0, help="leave out the first N pairs")
    p.add_argument("--max-pairs", type=int, default=None, help="draw at most this many pairs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    splits = pd.read_csv(args.splits, dtype=str)
    need = {"id", "source", "label", "norm_rel", "split"}
    if not need.issubset(splits.columns):
        sys.exit(f"--splits is missing columns: {sorted(need - set(splits.columns))}")

    written = run_single_audit(splits, args.images_root, out_dir, args.seed, font)

    if args.pairs:
        pairs = pd.read_csv(args.pairs, dtype=str)
        need_p = {"id_a", "id_b", "source_a", "source_b", "label_a", "label_b", "distance"}
        if not need_p.issubset(pairs.columns):
            sys.exit(f"--pairs is missing columns: {sorted(need_p - set(pairs.columns))}")
        lookup = build_lookup(splits, args.all_csv)
        written += run_pair_audit(pairs, lookup, args.images_root, out_dir,
                                  args.skip_first, args.max_pairs, font)

    print("Files written:")
    for path in written:
        print(f"  {path}  ({Path(path).stat().st_size} bytes)")


if __name__ == "__main__":
    main()

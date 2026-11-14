#!/usr/bin/env python3
"""Build synthetic input for pipeline/rebuild_splits.py.

Writes into --out: all_images_before_dedup.csv, frozen_features_effv2b0.npz, old_splits.csv, truth.json.
The data reproduces the hard cases: 18 sources and 12 labels with the real source/class layout (from
audit/stats.json), ZeroWaste video groups with hundreds of frames, exact and near duplicate hashes across
and inside sources, GlobalWaste copies of TrashBox and GD (some with conflicting labels), GlobalWaste rows
in classes that must be dropped, tiny cells, DWSD with 192 held-out ids, and duplicate pairs whose
features have cosine near 1. truth.json lists every planted pair.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, "..", "audit", "stats.json")
SOURCE_ORDER = ["RealWaste", "TrashNet", "GD", "PSU", "Drinking", "TACO", "TIDY", "DWSD", "BePLi",
                "SGBattery", "MedPackID", "Sanitary", "TrashBox", "Mayank14", "ZeroWaste", "WastePro",
                "PharmaSynth", "GlobalWaste"]
GW_KEEP = ["sanitary", "shoes", "battery", "medicines_syringes"]
FEAT_DIM = 1280
CLASS_FLOOR = 1300       # each class keeps about this many images after the GlobalWaste filter (val >= 100 needs it)
DWSD_LABELS = {"plastic": 400, "metal": 60, "glass": 60, "paper_cardboard": 40, "clothes": 40}


def load_real_counts():
    with open(STATS) as f:
        raw = json.load(f)["mapped_counts"]
    return {tuple(k.split(":")): v for k, v in raw.items()}


def cell_counts(f, n_rows, real):
    """Cells with fewer than 300 real images stay as they are; bigger cells scale with f. Classes are
    topped up to CLASS_FLOOR images (or their real size, if smaller) so the validation check can pass."""
    cells = {}
    for (src, lab), c in real.items():
        if c < 300:
            cells[(src, lab)] = c
        else:
            dropped = src == "GlobalWaste" and lab not in GW_KEEP
            cells[(src, lab)] = max(1, round(c * f * (0.5 if dropped else 1)))
    for lab in set(l for _, l in real):
        keep = [k for k in cells if k[1] == lab and not (k[0] == "GlobalWaste" and lab not in GW_KEEP)]
        total = sum(cells[k] for k in keep)
        real_total = sum(real[k] for k in keep)
        floor = min(real_total, CLASS_FLOOR)
        big = [k for k in keep if real[k] >= 300]
        small_total = total - sum(cells[k] for k in big)
        big_total = sum(cells[k] for k in big)
        if total < floor and big_total > 0:
            for k in big:
                cells[k] = max(1, round(cells[k] * (floor - small_total) / big_total))
    return cells


def choose_scale(n_rows, real):
    """Bisect the scale factor so the total number of rows (with DWSD) is close to n_rows."""
    lo, hi = 0.001, 2.0
    dwsd = sum(DWSD_LABELS.values()) * max(1, n_rows // 20000)
    for _ in range(40):
        mid = (lo + hi) / 2
        total = sum(cell_counts(mid, n_rows, real).values()) + dwsd
        lo, hi = (mid, hi) if total < n_rows else (lo, mid)
    return (lo + hi) / 2


def flip_mask(rng, nbits):
    m = 0
    for b in rng.choice(64, size=nbits, replace=False):
        m |= 1 << int(b)
    return np.uint64(m)


def unit(v):
    return v / np.linalg.norm(v)


def build(n_rows=20000, seed=1, out_dir="."):
    rng = np.random.default_rng(seed)
    real = load_real_counts()
    f = choose_scale(n_rows, real)
    cells = cell_counts(f, n_rows, real)
    mult = max(1, n_rows // 20000)
    # ZeroWaste: a few huge video groups (hundreds of frames) in plastic and paper_cardboard
    cells[("ZeroWaste", "plastic")] = max(cells[("ZeroWaste", "plastic")], 1000 * mult)
    cells[("ZeroWaste", "paper_cardboard")] = max(cells[("ZeroWaste", "paper_cardboard")], 700 * mult)

    rows = []          # (source, label, group)
    for (src, lab), n in sorted(cells.items()):
        made, k = 0, 0
        huge = []
        if src == "ZeroWaste" and lab in ("plastic", "paper_cardboard"):
            huge = [300 * mult] * (3 if lab == "plastic" else 2)
        for h in huge:
            rows += [(src, lab, f"{src}_{lab}_video{k}")] * h
            made += h
            k += 1
        while made < n:
            size = int(rng.choice([1, 1, 1, 2, 3, 5]))
            size = min(size, n - made)
            rows += [(src, lab, f"{src}_{lab}_g{k}")] * size
            made += size
            k += 1
    # DWSD: groups of exactly 4 crops, so 48 groups are exactly the 192 held-out ids
    k = 0
    for lab, n in DWSD_LABELS.items():
        for _ in range(n * mult // 4):
            rows += [("DWSD", lab, f"DWSD_{lab}_g{k}")] * 4
            k += 1
    perm = rng.permutation(len(rows))
    rows = [rows[i] for i in perm]
    n = len(rows)
    srcs = np.array([r[0] for r in rows])
    labs = np.array([r[1] for r in rows], dtype=object)
    groups = np.array([r[2] for r in rows])
    ids = np.array([f"img{i:07d}" for i in range(n)])

    # hashes: one random parent hash per group, members differ from it by 8-12 bits (so members of a
    # group are NOT near-duplicates of each other, as with video frames at distance > 4)
    gnames, ginv = np.unique(groups, return_inverse=True)
    parent = rng.integers(0, 2 ** 64, size=len(gnames), dtype=np.uint64)
    masks = np.zeros(n, np.uint64)
    for i in range(n):
        masks[i] = flip_mask(rng, int(rng.integers(8, 13)))
    hashes = parent[ginv] ^ masks
    # features: group base vector plus noise
    base = rng.standard_normal((len(gnames), FEAT_DIM)).astype(np.float32)
    feats = 0.9 * base[ginv] + 0.45 * rng.standard_normal((n, FEAT_DIM)).astype(np.float32)
    del base
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)

    used = set()
    truth = dict(scale=f, rows=n, phash_same_label=[], phash_conflict=[], phash_same_source=[],
                 gw_dropped_class_copies=[], feature_dups=[], feature_conflicts=[], feature_midrange=[])

    def take(src, lab, k, later_than=None):
        cand = np.nonzero((srcs == src) & (labs == lab))[0]
        cand = [int(i) for i in rng.permutation(cand) if int(i) not in used][:k]
        used.update(cand)
        return cand

    def near(i, j, nbits=None, feat_noise=0.1):
        """Make j a near-copy of i: hash differs by 0, 2 or 4 bits, features nearly identical."""
        nb = int(rng.choice([0, 2, 4])) if nbits is None else nbits
        hashes[j] = hashes[i] ^ (flip_mask(rng, nb) if nb else np.uint64(0))
        feats[j] = unit(feats[i] + feat_noise * unit(rng.standard_normal(FEAT_DIM).astype(np.float32)))

    def feature_only(i, j, noise):
        feats[j] = unit(feats[i] + noise * unit(rng.standard_normal(FEAT_DIM).astype(np.float32)))

    def plant(kind, src_a, lab_a, src_b, lab_b, k, how="phash"):
        a, b = take(src_a, lab_a, k), take(src_b, lab_b, k)
        for i, j in zip(a, b):
            if how == "phash":
                near(i, j)
            elif how == "feature":
                feature_only(i, j, 0.1)
            else:
                feature_only(i, j, 0.4)
            first, second = (i, j) if SOURCE_ORDER.index(srcs[i]) <= SOURCE_ORDER.index(srcs[j]) else (j, i)
            truth[kind].append(dict(winner=ids[first], other=ids[second], label_winner=labs[first],
                                    label_other=labs[second]))
        return list(zip(a, b))

    # A, B: GlobalWaste copies (same label) of TrashBox and GD images and of Sanitary
    plant("phash_same_label", "TrashBox", "medicines_syringes", "GlobalWaste", "medicines_syringes", 40)
    plant("phash_same_label", "GD", "battery", "GlobalWaste", "battery", 25)
    plant("phash_same_label", "GD", "shoes", "GlobalWaste", "shoes", 25)
    plant("phash_same_label", "Sanitary", "sanitary", "GlobalWaste", "sanitary", 10)
    # C: conflicting labels: both images must go
    plant("phash_conflict", "TrashBox", "e_waste", "GlobalWaste", "battery", 12)
    plant("phash_conflict", "GD", "clothes", "GlobalWaste", "shoes", 12)
    plant("phash_conflict", "TrashBox", "e_waste", "GlobalWaste", "medicines_syringes", 8)
    plant("phash_conflict", "RealWaste", "glass", "WastePro", "metal", 6)
    # D: same-label duplicates between original sources: the earlier source must win
    plant("phash_same_label", "TrashNet", "glass", "GD", "glass", 10)
    plant("phash_same_label", "GD", "plastic", "PSU", "plastic", 10)
    plant("phash_same_label", "RealWaste", "biological", "WastePro", "biological", 10)
    plant("phash_same_label", "Drinking", "metal", "WastePro", "metal", 10)
    plant("phash_same_label", "TACO", "paper_cardboard", "ZeroWaste", "paper_cardboard", 10)
    # F: GlobalWaste rows in a dropped class copying an original (removed by the filter)
    for i, j in plant("gw_dropped_class_copies", "GD", "glass", "GlobalWaste", "glass", 10):
        pass
    # E: duplicates inside one source: the row that comes first in the file must win
    a = take("RealWaste", "plastic", 30)
    for i, j in zip(a[:15], a[15:]):
        near(i, j)
        first, second = (i, j) if i < j else (j, i)
        truth["phash_same_source"].append(dict(winner=ids[first], other=ids[second]))
    # G: feature-only duplicates (different hash, cosine near 1), conflicts, and mid-range candidates
    plant("feature_dups", "GD", "clothes", "WastePro", "clothes", 15, how="feature")
    plant("feature_dups", "TrashNet", "metal", "PSU", "metal", 15, how="feature")
    plant("feature_conflicts", "GD", "battery", "TrashBox", "e_waste", 8, how="feature")
    plant("feature_midrange", "RealWaste", "trash", "TrashNet", "trash", 12, how="mid")
    plant("feature_midrange", "GD", "paper_cardboard", "PSU", "paper_cardboard", 13, how="mid")

    df = pd.DataFrame(dict(id=ids, source=srcs, orig_class=[f"{l}_orig" for l in labs], label=labs,
                           group=groups,
                           norm_rel=[f"{s}/{l}/{i}.jpg" for s, l, i in zip(srcs, labs, ids)],
                           phash=[f"{int(h):016x}" for h in hashes]))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "all_images_before_dedup.csv"), index=False)
    np.savez(os.path.join(out_dir, "frozen_features_effv2b0.npz"), ids=ids, feats=feats.astype(np.float16))

    # old splits: DWSD groups of 4 -> 48 groups = 192 ids marked dwsd_test
    old_split = rng.choice(["train", "val", "test"], size=n, p=[0.7, 0.1, 0.2]).astype(object)
    dw_groups = np.unique(groups[srcs == "DWSD"])
    test_groups = set(rng.choice(dw_groups, size=48, replace=False))
    is_dw = srcs == "DWSD"
    old_split[is_dw] = "train"
    t3_mask = np.array([g in test_groups for g in groups]) & is_dw
    old_split[t3_mask] = "dwsd_test"
    pd.DataFrame(dict(id=ids, source=srcs, split=old_split)).to_csv(os.path.join(out_dir, "old_splits.csv"),
                                                                    index=False)
    truth["t3_ids"] = ids[t3_mask].tolist()
    with open(os.path.join(out_dir, "truth.json"), "w") as fh:
        json.dump(truth, fh, default=str)
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = build(a.rows, a.seed, a.out)
    print("rows", t["rows"], "scale", round(t["scale"], 3))

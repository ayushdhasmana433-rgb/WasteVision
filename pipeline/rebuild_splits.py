#!/usr/bin/env python3
"""WasteVision phase 1: rebuild the frozen splits (PLAN.md sections 5 and 6).

Steps: filter GlobalWaste -> pHash de-duplication -> optional feature-similarity
de-duplication -> per-class cap with equal source quotas -> 70/10/20 grouped split
-> hard checks -> output files.

Uses only numpy, pandas, scikit-learn and the standard library (matplotlib and
Pillow only for the optional contact sheets).
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# De-duplication order from PLAN.md section 5: originals first, GlobalWaste last.
SOURCE_ORDER = ["RealWaste", "TrashNet", "GD", "PSU", "Drinking", "TACO", "TIDY", "DWSD", "BePLi",
                "SGBattery", "MedPackID", "Sanitary", "TrashBox", "Mayank14", "ZeroWaste", "WastePro",
                "PharmaSynth", "GlobalWaste"]
CLASSES = ["biological", "paper_cardboard", "plastic", "metal", "glass", "clothes", "shoes", "trash",
           "battery", "medicines_syringes", "e_waste", "sanitary"]
GLOBALWASTE_KEEP = ["sanitary", "shoes", "battery", "medicines_syringes"]
TRAIN_ONLY_SOURCES = ["PharmaSynth"]       # capped like the others, but always split "train"
NEVER_TEST_SOURCES = ["DWSD"]              # DWSD is never in the T1 test (its test crops are T3)
MIN_CELL_FOR_SPLIT = 30                    # smaller (class, source) cells go entirely to train
CHECK_CELL_MIN = 50                        # cells at least this big must appear in validation
CHECK_CLASS_VAL_MIN = 100
N_FOLDS = 10                               # 7 train / 1 val / 2 test
EXTRA_SEEDS = 20
STAGES = ["mapped", "after_filter", "set_aside_T3", "removed_T3_group_or_near", "after_phash",
          "after_features", "after_cap"]
MAX_BAD_HASH_SHARE = 0.005                 # stop if more than this share of rows has a bad hash
MAX_MISSING_FEATURE_SHARE = 0.05           # stop if more survivors than this lack a feature row
VAL_FLOOR_WARNING = 130                    # classes below this many validation images are listed
REASON_LABEL = "label conflict"
REASON_NEAR = "near conflict-dropped image"
SPLIT_COLS = ["train", "val", "test_T1", "test_T2", "test_T3"]
OUT_COLS = ["id", "source", "orig_class", "label", "group", "norm_rel", "phash"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- pHash helpers

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x):
    """Number of set bits in each uint64 of an array."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POP8[np.ascontiguousarray(x).view(np.uint8).reshape(x.shape + (8,))].sum(-1, dtype=np.uint8)


def parse_hashes(hex_strings):
    """16-hex-digit strings -> uint64 array."""
    try:
        return np.array([int(h, 16) for h in hex_strings], dtype=np.uint64)
    except (ValueError, TypeError, OverflowError) as e:
        raise SystemExit(f"phash column has an invalid value (need 16 hex digits per row): {e}")


def hamming_pairs(H, thr, chunk=256):
    """All pairs (i < j) of one hash array with Hamming distance <= thr.
    Works in chunks of rows against the remaining rows only, so memory stays small.
    Returns three arrays: i (int32), j (int32), distance (uint8)."""
    n = len(H)
    out_i, out_j, out_d = [], [], []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = popcount64(H[s:e, None] ^ H[None, s:])          # shape (e-s, n-s)
        ii, jj = np.nonzero(d <= thr)
        keep = jj > ii                                       # column jj is image s+jj; need s+jj > s+ii
        ii, jj = ii[keep], jj[keep]
        out_i.append((ii + s).astype(np.int32))
        out_j.append((jj + s).astype(np.int32))
        out_d.append(d[ii, jj].astype(np.uint8))
    if not out_i:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.uint8)
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_d)


def within_thr_of(Hq, Hr, thr, chunk=256):
    """For each hash in Hq: is any hash in Hr within Hamming distance thr? Returns a bool array."""
    res = np.zeros(len(Hq), bool)
    if len(Hr) == 0:
        return res
    for s in range(0, len(Hq), chunk):
        d = popcount64(Hq[s:s + chunk, None] ^ Hr[None, :])
        res[s:s + chunk] = (d.min(axis=1) <= thr)
    return res


# ---------------------------------------------------------------- greedy removal

def greedy_remove(labels, pi, pj, n):
    """Greedy duplicate removal in visit order (index = visit position, pairs have pi < pj).
    Image j is kept unless an already-kept image is paired with it. If the kept partner and j
    have different labels, both are dropped (a "conflict"). Those two stay marked, so a later image
    within the threshold of either is dropped too and logged (reason "near conflict-dropped image");
    copies labelled x, y, x are therefore all dropped, the same as x, x, y. The spread stops there:
    an image dropped only for being near a marked image is not itself marked.
    Returns (kept bool array, list of (earlier_index, new_index, reason) conflict pairs)."""
    kept = np.ones(n, bool)
    conf_dropped = np.zeros(n, bool)
    conflicts = []
    if len(pi) == 0:
        return kept, conflicts
    order = np.argsort(pj, kind="stable")
    pi, pj = pi[order], pj[order]
    uniq_j, start = np.unique(pj, return_index=True)
    end = np.append(start[1:], len(pj))
    for j, a, b in zip(uniq_j, start, end):
        lows = pi[a:b]
        hits = lows[kept[lows]]
        near_bad = lows[conf_dropped[lows]]
        if len(hits) == 0 and len(near_bad) == 0:
            continue
        kept[j] = False
        bad = hits[labels[hits] != labels[j]]
        if len(bad):
            kept[bad] = False
            conf_dropped[bad] = True
            conf_dropped[j] = True
            conflicts.extend((int(h), int(j), REASON_LABEL) for h in bad)
        if len(near_bad):
            # one-step spread only: j is dropped but NOT marked, so the drop cannot run along a
            # long chain of near-duplicates (video frames); x, y, x still drops all three
            conflicts.extend((int(h), int(j), REASON_NEAR) for h in near_bad)
    return kept, conflicts


def pair_values(pi, pj, vals, conflicts, n):
    """Distance or cosine of each conflict pair (looked up by pair key, fast for millions of pairs)."""
    if not conflicts:
        return np.zeros(0)
    keys = pi.astype(np.int64) * n + pj
    order = np.argsort(keys)
    want = np.array([c[0] * n + c[1] for c in conflicts], dtype=np.int64)
    return vals[order[np.searchsorted(keys[order], want)]]


def conflict_frame(df, conflicts, stage, values, value_name):
    """Rows for label_conflicts.csv. values[k] belongs to conflicts[k]."""
    cols = ["stage", "reason", "id_a", "id_b", "source_a", "source_b", "label_a", "label_b", "distance", "cosine"]
    if not conflicts:
        return pd.DataFrame(columns=cols)
    a = np.array([c[0] for c in conflicts])
    b = np.array([c[1] for c in conflicts])
    out = pd.DataFrame(dict(stage=stage, reason=[c[2] for c in conflicts], id_a=df.id.values[a], id_b=df.id.values[b],
                            source_a=df.source.values[a], source_b=df.source.values[b],
                            label_a=df.label.values[a], label_b=df.label.values[b]))
    out["distance"] = values if value_name == "distance" else np.nan
    out["cosine"] = values if value_name == "cosine" else np.nan
    return out[cols]


# ---------------------------------------------------------------- steps 1 and 2

def load_table(path):
    df = pd.read_csv(path, dtype={"id": str, "phash": str, "group": str, "norm_rel": str,
                                  "source": str, "label": str, "orig_class": str})
    missing = [c for c in OUT_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    if df.id.duplicated().any():
        raise SystemExit("ids in the all-images CSV are not unique")
    bad_src = sorted(set(df.source) - set(SOURCE_ORDER))
    if bad_src:
        raise SystemExit(f"unknown source names: {bad_src}")
    bad_lab = sorted(set(df.label) - set(CLASSES))
    if bad_lab:
        raise SystemExit(f"unknown labels: {bad_lab}")
    return df[OUT_COLS].reset_index(drop=True)


def combine_inputs(all_csv, extra_csvs):
    """Main table plus any --extra-csv tables (same columns). Ids must not collide.
    Returns (table, number of extra rows)."""
    df = load_table(all_csv)
    n_main = len(df)
    for path in extra_csvs or []:
        extra = load_table(path)
        clash = set(extra.id) & set(df.id)
        if clash:
            raise SystemExit(f"{path}: {len(clash)} ids already exist in the other input files, e.g. "
                             f"{sorted(clash)[:5]}")
        df = pd.concat([df, extra], ignore_index=True)
    return df, len(df) - n_main


def clean_hashes(df):
    """Strip whitespace and a leading 0x, lowercase, left-pad to 16 digits. Rows whose hash is
    missing or not hexadecimal (1 to 16 digits) are split off. Returns (clean df, bad rows df)."""
    h = df.phash.astype("string").str.strip().str.lower().str.replace(r"^0x", "", regex=True)
    valid = h.str.fullmatch(r"[0-9a-f]{1,16}").fillna(False).astype(bool)
    good = df[valid].copy()
    good["phash"] = h[valid].str.zfill(16).astype(object).values
    return good.reset_index(drop=True), df[~valid].reset_index(drop=True)


def fix_missing_groups(df):
    """Rows without a group id get their own group (their id). Returns (df, number of rows fixed)."""
    missing = df.group.isna() | (df.group.astype(str).str.strip() == "")
    df = df.copy()
    df.loc[missing, "group"] = df.id[missing]
    return df, int(missing.sum())


def set_aside_t3(filtered, t3_ids, thr):
    """Take the T3 rows out. Then remove from the rest every row that shares a photo group with a T3
    row, and every row within pHash distance thr of a T3 image (so no training image can be a copy
    of a T3 image). Returns (t3, pool, removed_same_group, removed_near)."""
    is_t3 = filtered.id.isin(t3_ids)
    t3 = filtered[is_t3].reset_index(drop=True)
    rest = filtered[~is_t3]
    if len(t3) == 0:
        return t3, rest.reset_index(drop=True), rest.iloc[0:0], rest.iloc[0:0]
    same = rest.group.isin(set(t3.group))
    removed_group, rest = rest[same], rest[~same]
    near = within_thr_of(parse_hashes(rest.phash.values), parse_hashes(t3.phash.values), thr)
    removed_near, rest = rest[near], rest[~near]
    return t3, rest.reset_index(drop=True), removed_group, removed_near


def filter_globalwaste(df):
    """Step 1: GlobalWaste rows are kept only for the four chosen labels."""
    drop = (df.source == "GlobalWaste") & ~df.label.isin(GLOBALWASTE_KEEP)
    return df[~drop].reset_index(drop=True)


def visit_order(df):
    """Sort into de-duplication visit order: PLAN source order, stable within a source."""
    rank = df.source.map({s: k for k, s in enumerate(SOURCE_ORDER)}).values
    idx = np.argsort(rank, kind="stable")
    return df.iloc[idx].reset_index(drop=True)


def phash_dedup(pool, thr, chunk=256):
    """Step 2. `pool` must already be in visit order.
    Returns (kept pool, pairs DataFrame, conflicts DataFrame, overlap matrix DataFrame)."""
    H = parse_hashes(pool.phash.values)
    pi, pj, pd_ = hamming_pairs(H, thr, chunk)
    labels = pool.label.values
    kept, conflicts = greedy_remove(labels, pi, pj, len(pool))
    ids, src = pool.id.values, pool.source.values
    pairs = pd.DataFrame(dict(id_a=ids[pi], id_b=ids[pj], source_a=src[pi], source_b=src[pj],
                              label_a=labels[pi], label_b=labels[pj], distance=pd_))
    conf = conflict_frame(pool, conflicts, "phash", pair_values(pi, pj, pd_, conflicts, len(pool)), "distance")
    overlap = overlap_matrix(src[pi], src[pj])
    return pool[kept].reset_index(drop=True), pairs, conf, overlap


def overlap_matrix(src_a, src_b):
    """Source-by-source count of duplicate pairs. Symmetric: a pair between two different
    sources is counted in both cells; a pair inside one source is counted once on the diagonal."""
    m = pd.DataFrame(0, index=SOURCE_ORDER, columns=SOURCE_ORDER, dtype=np.int64)
    if len(src_a):
        counts = pd.Series(list(zip(src_a, src_b))).value_counts()
        for (a, b), c in counts.items():
            m.loc[a, b] += c
            if a != b:
                m.loc[b, a] += c
    return m


# ---------------------------------------------------------------- step 3: features

def feature_pass(pool, feat_ids, feats, cand_thr, thr, chunk_elements=50_000_000,
                 max_chunk_candidates=2_000_000):
    """Step 3. Cross-source pairs with cosine >= cand_thr are candidates; pairs >= thr are removed
    (same greedy order and conflict rule). `pool` must be in visit order.
    Memory: a chunk of rows needs about 6 bytes per matrix element (similarities, a same-source mask
    and a threshold mask), so chunk_elements=50M keeps a chunk near 300 MB. Same-source pairs and the
    threshold are applied before np.nonzero. If a chunk has more than max_chunk_candidates candidates,
    only pairs >= thr are kept for that chunk (counted in info).
    Returns (kept pool, candidates DataFrame, conflicts DataFrame, info dict)."""
    pos = pd.Index(feat_ids).get_indexer(pool.id.values)
    have = pos >= 0
    X = np.zeros((len(pool), feats.shape[1]), np.float32)
    X[have] = feats[pos[have]].astype(np.float32)
    norm = np.linalg.norm(X, axis=1)
    usable = have & (norm > 0)
    X[usable] /= norm[usable, None]
    src_code = pool.source.map({s: k for k, s in enumerate(SOURCE_ORDER)}).values
    n = len(pool)
    chunk = int(min(1024, max(1, chunk_elements // max(n, 1))))
    info = dict(images_without_features=int((~usable).sum()), chunk_rows=chunk,
                chunks_truncated_to_removal_threshold=0)
    ci, cj, cc = [], [], []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        c = e - s
        S = X[s:e] @ X[s:].T                                  # (c, n-s), float32
        S[src_code[s:e, None] == src_code[None, s:]] = -1.0   # same-source pairs out
        S[:, :c][np.tril_indices(c)] = -1.0                   # keep only j > i inside the chunk
        hit = S >= cand_thr
        if np.count_nonzero(hit) > max_chunk_candidates:
            hit = S >= max(cand_thr, thr)
            info["chunks_truncated_to_removal_threshold"] += 1
        ii, jj = np.nonzero(hit)
        del hit
        ci.append((ii + s).astype(np.int32))
        cj.append((jj + s).astype(np.int32))
        cc.append(S[ii, jj].astype(np.float32))
        del S
    ci = np.concatenate(ci) if ci else np.zeros(0, np.int32)
    cj = np.concatenate(cj) if cj else np.zeros(0, np.int32)
    cc = np.concatenate(cc) if cc else np.zeros(0, np.float32)
    ids, src, lab = pool.id.values, pool.source.values, pool.label.values
    cand = pd.DataFrame(dict(id_a=ids[ci], id_b=ids[cj], source_a=src[ci], source_b=src[cj],
                             label_a=lab[ci], label_b=lab[cj], cosine=cc))
    strong = cc >= thr
    kept, conflicts = greedy_remove(lab, ci[strong], cj[strong], n)
    si, sj, sc = ci[strong], cj[strong], cc[strong]
    conf = conflict_frame(pool, conflicts, "features", pair_values(si, sj, sc, conflicts, n), "cosine")
    info["candidate_pairs"] = int(len(cand))
    info["candidate_list_complete"] = info["chunks_truncated_to_removal_threshold"] == 0
    info["pairs_at_removal_threshold"] = int(strong.sum())
    return pool[kept].reset_index(drop=True), cand, conf, info


def write_contact_sheets(cand, pool_all, images_root, out_dir, n_pairs=100, per_sheet=20):
    """Contact sheets of candidate pairs sampled evenly across the cosine range. Never raises."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as e:                                   # matplotlib or Pillow missing
        log(f"  contact sheets skipped (matplotlib/Pillow not available: {e})")
        return 0
    try:
        if len(cand) == 0:
            return 0
        rel = dict(zip(pool_all.id, pool_all.norm_rel))
        c = cand.sort_values("cosine").reset_index(drop=True)
        idx = np.unique(np.linspace(0, len(c) - 1, min(n_pairs, len(c))).round().astype(int))
        sample = c.iloc[idx].reset_index(drop=True)
        sheet_dir = os.path.join(out_dir, "feature_pairs_sheets")
        os.makedirs(sheet_dir, exist_ok=True)
        sample.index.name = "pair_no"
        sample.to_csv(os.path.join(sheet_dir, "sampled_pairs.csv"))
        for s0 in range(0, len(sample), per_sheet):
            chunk = sample.iloc[s0:s0 + per_sheet]
            nrows = (len(chunk) + 3) // 4
            fig, axes = plt.subplots(nrows, 8, figsize=(16, 2.3 * nrows))
            axes = np.atleast_2d(axes)
            for a in axes.flat:
                a.axis("off")
            for k, (no, r) in enumerate(chunk.iterrows()):
                for side, (pid, s_, l_) in enumerate([(r.id_a, r.source_a, r.label_a),
                                                      (r.id_b, r.source_b, r.label_b)]):
                    ax = axes[k // 4, (k % 4) * 2 + side]
                    try:
                        ax.imshow(Image.open(os.path.join(images_root, rel[pid])))
                    except Exception:
                        pass                                  # image missing: leave the panel blank
                    ax.set_title(f"#{no}{'ab'[side]} {s_}:{l_} cos={r.cosine:.3f}", fontsize=6.5)
            plt.tight_layout()
            plt.savefig(os.path.join(sheet_dir, f"sheet_{s0 // per_sheet:02d}.jpg"), dpi=80)
            plt.close(fig)
        return len(sample)
    except Exception as e:
        log(f"  contact sheets stopped early ({e})")
        return 0


# ---------------------------------------------------------------- step 4: capping

def waterfill(sizes, cap, rng):
    """Equal quota per source; sources smaller than their quota give the rest to the others.
    sizes: dict source -> available images. Returns dict source -> quota (sums to cap when
    the total available exceeds cap, otherwise every source keeps all its images)."""
    if sum(sizes.values()) <= cap:
        return dict(sizes)
    quota, remaining, active = {}, cap, sorted(sizes)
    while active:
        share = remaining // len(active)
        small = [s for s in active if sizes[s] <= share]
        if small:
            for s in small:
                quota[s] = sizes[s]
                remaining -= sizes[s]
            active = [s for s in active if s not in small]
            continue
        extra = remaining - share * len(active)
        bonus = set(rng.choice(len(active), size=extra, replace=False).tolist()) if extra else set()
        for k, s in enumerate(active):
            quota[s] = share + (1 if k in bonus else 0)
        break
    return quota


def cap_pool(pool, cap, max_per_group, seed):
    """Step 4. For each class: limit every photo group to max_per_group images (random sample),
    give each source an equal quota (water-filling), and fill each quota by taking groups in
    random order. Returns (capped DataFrame, info DataFrame with one row per (label, source))."""
    rng = np.random.default_rng(seed)
    kept_parts, info = [], []
    for label in sorted(pool.label.unique()):
        cls = pool[pool.label == label]
        avail = {}
        for source in SOURCE_ORDER:
            sub = cls[cls.source == source]
            if len(sub) == 0:
                continue
            shuffled = sub.iloc[rng.permutation(len(sub))]
            limited = shuffled[shuffled.groupby("group").cumcount() < max_per_group]
            avail[source] = limited
        quota = waterfill({s: len(v) for s, v in avail.items()}, cap, rng)
        for source, limited in avail.items():
            q = quota[source]
            if q >= len(limited):
                take = limited
            else:
                by_group = limited.groupby("group").indices      # row positions; rows are in random order
                glist = np.array(list(by_group))
                glist = glist[rng.permutation(len(glist))]
                parts, left = [], q
                for g in glist:
                    if left == 0:
                        break
                    rows = by_group[g][:left]                     # only the last group can be cut short
                    parts.append(rows)
                    left -= len(rows)
                take = limited.iloc[np.concatenate(parts)]
            kept_parts.append(take)
            info.append(dict(label=label, source=source, before_group_limit=int((cls.source == source).sum()),
                             after_group_limit=len(limited), quota=int(q), taken=len(take)))
    capped = pd.concat(kept_parts) if kept_parts else pool.iloc[0:0]
    capped = visit_order(capped)
    return capped, pd.DataFrame(info)


# ---------------------------------------------------------------- step 5: split

def make_split(capped, seed):
    """Step 5 (without T3). Returns a copy of `capped` with columns split and tier.
    Cells under MIN_CELL_FOR_SPLIT images and train-only sources go to train; the rest is split
    by photo group with StratifiedGroupKFold (2 folds test, 1 fold val, 7 train)."""
    df = capped.copy().reset_index(drop=True)
    df["split"] = "train"
    cell = df.label + "|" + df.source
    cell_size = cell.map(cell.value_counts())
    eligible = (cell_size >= MIN_CELL_FOR_SPLIT) & ~df.source.isin(TRAIN_ONLY_SOURCES)
    sub = df[eligible]
    if len(sub):
        folds = np.full(len(sub), -1)
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for f, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(sub)), cell[eligible].values,
                                                             sub.group.values)):
                    folds[test_idx] = f
        except ValueError as e:
            raise SystemExit(f"cannot split by group: {e}")
        role = np.where(folds < 2, "test", np.where(folds == 2, "val", "train"))
        df.loc[eligible, "split"] = role
    # DWSD is never in the T1 test: its test-assigned images go to train.
    moved = (df.source.isin(NEVER_TEST_SOURCES) & (df.split == "test"))
    df.loc[moved, "split"] = "train"
    # A group that ended up in several splits (only possible when one group id spans cells) goes to train.
    nsplit = df.groupby("group").split.transform("nunique")
    fixed = nsplit > 1
    df.loc[fixed, "split"] = "train"
    df["tier"] = ""
    test = df.split == "test"
    df.loc[test, "tier"] = np.where(df.source[test] == "ZeroWaste", "T2", "T1")
    df.attrs["dwsd_test_moved_to_train"] = int(moved.sum())
    df.attrs["group_conflict_moved_to_train"] = int(fixed.sum())
    return df


# ---------------------------------------------------------------- step 6: checks

def run_checks(df, thr, t3=None):
    """Hard checks on a split (T3 rows are not in df; t3 is the T3 table or None). Returns (dict name -> bool, list of failure messages)."""
    fails, ok = [], {}
    val = df[df.split == "val"]
    per_class = val.label.value_counts()
    low = {c: int(per_class.get(c, 0)) for c in sorted(set(CLASSES) | set(df.label))
           if per_class.get(c, 0) < CHECK_CLASS_VAL_MIN}
    ok["every_class_has_100_val"] = not low
    if low:
        fails.append(f"classes with fewer than {CHECK_CLASS_VAL_MIN} validation images: {low}")

    cell = df.groupby(["label", "source"]).agg(n=("id", "size"), v=("split", lambda s: int((s == "val").sum())))
    need = cell[(cell.n >= CHECK_CELL_MIN) & (cell.v == 0)]
    need = need[~need.index.get_level_values("source").isin(TRAIN_ONLY_SOURCES)]
    ok["every_cell_50plus_in_val"] = need.empty
    if len(need):
        fails.append(f"(class, source) cells with >= {CHECK_CELL_MIN} images and no validation image: "
                     f"{[f'{a}|{b}' for a, b in need.index]}")

    spread = df.groupby("group").split.nunique()
    ok["no_group_in_two_splits"] = bool((spread <= 1).all())
    if not ok["no_group_in_two_splits"]:
        fails.append(f"{int((spread > 1).sum())} groups appear in more than one split")

    train = df[df.split == "train"]
    Htr = parse_hashes(train.phash.values)
    held = df[df.split != "train"]
    hit = within_thr_of(parse_hashes(held.phash.values), Htr, thr)
    ok["no_heldout_near_duplicate_of_train"] = not hit.any()
    if hit.any():
        fails.append(f"{int(hit.sum())} val/test images are within pHash distance {thr} of a training image")

    has_t3 = t3 is not None and len(t3) > 0
    bad = df.group.isin(set(t3.group)) if has_t3 else pd.Series(False, index=df.index)
    ok["no_t3_group_in_train_val_test"] = not bad.any()
    if bad.any():
        fails.append(f"{int(bad.sum())} train/val/test images share a photo group with a T3 image")
    near = (within_thr_of(parse_hashes(train.phash.values), parse_hashes(t3.phash.values), thr)
            if has_t3 else np.zeros(1, bool))
    ok["no_train_image_near_t3"] = not near.any()
    if near.any():
        fails.append(f"{int(near.sum())} training images are within pHash distance {thr} of a T3 image")
    return ok, fails


# ---------------------------------------------------------------- outputs

def cell_counts(df, name):
    return df.groupby(["source", "label"]).size().rename(name)


def build_count_table(df_all, after_filter, t3, removed, after_phash, after_features, capped, split_df):
    """One row per (source, label) with the stage counts and the split counts.
    after_filter includes the T3 rows; set_aside_T3 and removed_T3_group_or_near are taken off next,
    so after_filter - set_aside_T3 - removed_T3_group_or_near = images entering the pHash step."""
    parts = [cell_counts(df_all, "mapped"), cell_counts(after_filter, "after_filter"),
             cell_counts(t3, "set_aside_T3"), cell_counts(removed, "removed_T3_group_or_near"),
             cell_counts(after_phash, "after_phash"), cell_counts(after_features, "after_features"),
             cell_counts(capped, "after_cap")]
    tab = pd.concat(parts, axis=1)
    sp = split_df
    parts = [cell_counts(sp[sp.split == "train"], "train"), cell_counts(sp[sp.split == "val"], "val"),
             cell_counts(sp[sp.tier == "T1"], "test_T1"), cell_counts(sp[sp.tier == "T2"], "test_T2"),
             cell_counts(t3, "test_T3")]
    tab = pd.concat([tab] + parts, axis=1).fillna(0).astype(int).reset_index()
    tab["_s"] = tab.source.map({s: k for k, s in enumerate(SOURCE_ORDER)})
    tab["_l"] = tab.label.map({c: k for k, c in enumerate(CLASSES)})
    return tab.sort_values(["_l", "_s"]).drop(columns=["_s", "_l"]).reset_index(drop=True)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def write_markdown(tab, path, seed_used):
    cols = STAGES + SPLIT_COLS
    rows = []
    for c in CLASSES:
        t = tab[tab.label == c]
        rows.append([c] + [f"{int(t[k].sum())} ({int((t[k] > 0).sum())})" for k in cols])
    rows.append(["**all**"] + [f"{int(tab[k].sum())} ({int(tab.groupby('source')[k].sum().gt(0).sum())})"
                               for k in cols])
    out = ["# WasteVision count table", "",
           f"Split seed used: {seed_used}. Numbers in brackets are the number of sources with at least one image.", "",
           "## Per class totals, images (number of sources) at each stage", "",
           md_table(["class"] + cols, rows), ""]
    srcs = [s for s in SOURCE_ORDER if s in set(tab.source)]
    rows = []
    for c in CLASSES:
        t = tab[tab.label == c].set_index("source")
        rows.append([c] + [int(t.after_cap.get(s, 0)) for s in srcs] + [int(t.after_cap.sum())])
    out += ["## Class x source, images after capping", "", md_table(["class"] + srcs + ["total"], rows), ""]
    rows = []
    for c in CLASSES:
        t = tab[tab.label == c].set_index("source")
        cells = []
        for s in srcs:
            if s in t.index:
                r = t.loc[s]
                cells.append(f"{r.train}/{r.val}/{r.test_T1 + r.test_T2 + r.test_T3}")
            else:
                cells.append("0/0/0")
        rows.append([c] + cells)
    out += ["## Class x source, split counts as train/val/test (test = T1 + T2 + T3)", "",
            md_table(["class"] + srcs, rows), ""]
    with open(path, "w") as f:
        f.write("\n".join(out))


def to_py(x):
    if isinstance(x, dict):
        return {str(k): to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ---------------------------------------------------------------- main

def class_summary(split_df):
    """Per class: sources and largest source share in train, validation count."""
    train, val = split_df[split_df.split == "train"], split_df[split_df.split == "val"]
    out = {}
    for c in CLASSES:
        t = train[train.label == c].source.value_counts()
        row = dict(n_sources_in_train=int(len(t)), val_count=int((val.label == c).sum()),
                   train_images=int(t.sum()))
        if len(t):
            row.update(largest_source=t.index[0], largest_source_share_in_train=round(float(t.iloc[0] / t.sum()), 4))
        out[c] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rebuild the WasteVision splits (v3).")
    ap.add_argument("--all-csv", required=True)
    ap.add_argument("--extra-csv", action="append", default=None,
                    help="extra rows, same columns as --all-csv; may be given more than once")
    ap.add_argument("--features", default=None)
    ap.add_argument("--allow-missing-features", action="store_true",
                    help="continue even if more than 5%% of surviving images have no feature row")
    ap.add_argument("--old-splits", default=None)
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--max-per-group", type=int, default=20)
    ap.add_argument("--phash-thr", type=int, default=4)
    ap.add_argument("--feat-thr", type=float, default=0.97)
    ap.add_argument("--feat-candidate-thr", type=float, default=0.90)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)
    report = dict(thresholds=dict(phash=args.phash_thr, feature_removal=args.feat_thr,
                                  feature_candidate=args.feat_candidate_thr),
                  cap=args.cap, max_per_group=args.max_per_group, seed_requested=args.seed)

    # 0. load, hash hygiene, missing groups
    df_raw, n_extra = combine_inputs(args.all_csv, args.extra_csv)
    df_all, bad_hash = clean_hashes(df_raw)
    report["rows_in_input_files"] = len(df_raw)
    report["extra_csv_rows"] = n_extra
    report["rows_bad_hash"] = len(bad_hash)
    if len(bad_hash):
        bad_hash.to_csv(out("rows_bad_hash.csv"), index=False)
        log(f"  WARNING: {len(bad_hash)} rows have a missing or invalid phash and were dropped "
            f"(see rows_bad_hash.csv)")
        if len(bad_hash) / len(df_raw) > MAX_BAD_HASH_SHARE:
            raise SystemExit(f"{len(bad_hash)} of {len(df_raw)} rows ({len(bad_hash) / len(df_raw):.2%}) have a "
                             f"missing or invalid phash, more than {MAX_BAD_HASH_SHARE:.1%}; see rows_bad_hash.csv")
    df_all, n_missing_group = fix_missing_groups(df_all)
    report["rows_with_missing_group"] = n_missing_group
    if n_missing_group:
        log(f"  {n_missing_group} rows had no group; each got its own group (its id)")
    log(f"Loaded {len(df_all)} images, {df_all.source.nunique()} sources")

    # 1. filter
    filtered = filter_globalwaste(df_all)
    report["globalwaste_filter"] = dict(rows_before=len(df_all), rows_after=len(filtered),
                                        globalwaste_before=int((df_all.source == "GlobalWaste").sum()),
                                        globalwaste_after=int((filtered.source == "GlobalWaste").sum()))
    log(f"Step 1 filter: {len(df_all)} -> {len(filtered)} rows")
    after_filter = filtered

    # T3: held-out DWSD crops are set aside and take no part in steps 2 to 5. Rows sharing a photo group
    # with a T3 image, and rows within the pHash threshold of one, are removed from the pool.
    if args.old_splits:
        old = pd.read_csv(args.old_splits, dtype={"id": str})
        t3_ids = set(old.id[old.split == "dwsd_test"])
        t3, pool, removed_group, removed_near = set_aside_t3(filtered, t3_ids, args.phash_thr)
        report["t3"] = dict(ids_in_old_splits=len(t3_ids), found_in_all_csv=len(t3),
                            missing_from_all_csv=len(t3_ids) - len(t3))
        if len(t3_ids) - len(t3):
            log(f"  WARNING: {len(t3_ids) - len(t3)} dwsd_test ids are not in the all-images CSV")
        if (t3.source != "DWSD").any():
            raise SystemExit("some dwsd_test ids belong to a source other than DWSD")
        log(f"  T3: {len(t3)} DWSD crops set aside; removed {len(removed_group)} rows in the same photo group "
            f"and {len(removed_near)} rows within pHash distance {args.phash_thr}")
    else:
        t3, pool = filtered.iloc[0:0], filtered
        removed_group = removed_near = filtered.iloc[0:0]
        report["t3"] = dict(note="no --old-splits given: no T3 tier")
        log("  no --old-splits: no T3 tier")
    report["rows_removed_same_group_as_T3"] = len(removed_group)
    report["rows_removed_near_T3"] = len(removed_near)
    removed = pd.concat([removed_group, removed_near])
    pool = visit_order(pool)
    pool_in = pool

    # 2. pHash de-duplication
    pool, pairs, conf, overlap = phash_dedup(pool, args.phash_thr)
    pairs.to_csv(out("dup_pairs_phash.csv.gz"), index=False)
    overlap.to_csv(out("source_overlap.csv"))
    after_phash = pool
    conflict_frames = [conf]
    report["phash"] = dict(pairs=len(pairs), cross_source_pairs=int((pairs.source_a != pairs.source_b).sum()),
                           exact_pairs=int((pairs.distance == 0).sum()), label_conflict_pairs=len(conf), dropped_near_conflict=len({c[1] for c in conf if c[2] == REASON_NEAR}),
                           images_before=len(pool_in), images_after=len(after_phash))
    log(f"Step 2 pHash: {len(pool_in)} -> {len(after_phash)} images, {len(pairs)} pairs, "
        f"{len(conf)} logged conflict pairs")
    del pairs

    # 3. feature pass
    if args.features:
        z = np.load(args.features, allow_pickle=True)
        feat_ids, feats = z["ids"].astype(str), z["feats"]
        n_nofeat = int((pd.Index(feat_ids).get_indexer(pool.id.values) < 0).sum())
        share_nofeat = n_nofeat / max(len(pool), 1)
        if n_nofeat:
            log(f"  WARNING: {n_nofeat} of {len(pool)} surviving images ({share_nofeat:.2%}) have no row in the "
                f"features file; they are not checked for feature duplicates")
        if share_nofeat > MAX_MISSING_FEATURE_SHARE and not args.allow_missing_features:
            raise SystemExit(f"{n_nofeat} of {len(pool)} surviving images ({share_nofeat:.1%}) have no feature row, "
                             f"more than {MAX_MISSING_FEATURE_SHARE:.0%}. Fix the features file or pass "
                             f"--allow-missing-features.")
        pool, cand, fconf, finfo = feature_pass(pool, feat_ids, feats, args.feat_candidate_thr, args.feat_thr)
        del feats, z
        cand.to_csv(out("dup_candidates_features.csv.gz"), index=False)
        conflict_frames.append(fconf)
        report["features"] = dict(finfo, images_before=len(after_phash), images_after=len(pool),
                                  images_without_feature_row=n_nofeat, share_without_feature_row=share_nofeat,
                                  missing_features_allowed_by_flag=bool(args.allow_missing_features),
                                  label_conflict_pairs=len(fconf), dropped_near_conflict=len({c[1] for c in fconf if c[2] == REASON_NEAR}), contact_sheet_pairs=0)
        if args.images_root:
            report["features"]["contact_sheet_pairs"] = write_contact_sheets(
                cand, df_all, args.images_root, args.out_dir)
        log(f"Step 3 features: {len(after_phash)} -> {len(pool)} images, {len(cand)} candidate pairs")
        del cand
    else:
        report["features"] = dict(skipped="no --features given")
        log("Step 3 features: SKIPPED (no --features given)")
    after_features = pool
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)         # empty frame or all-NaN column in the concat
        all_conf = pd.concat(conflict_frames, ignore_index=True)
    all_conf.to_csv(out("label_conflicts.csv"), index=False)

    # 4. cap
    capped, cap_info = cap_pool(pool, args.cap, args.max_per_group, args.seed)
    log(f"Step 4 cap: {len(pool)} -> {len(capped)} images")

    def write_results(split_df, failed, seed_used, ok, fails):
        """Count table, report and split file. On failure the split file is splits_v3_FAILED.csv."""
        t3_rows = t3.copy()
        t3_rows["split"], t3_rows["tier"] = "test", "T3"
        full = pd.concat([split_df, t3_rows], ignore_index=True)
        full[OUT_COLS + ["split", "tier"]].to_csv(out("splits_v3_FAILED.csv" if failed else "splits_v3.csv"),
                                                   index=False)
        tab = build_count_table(df_all, after_filter, t3, removed, after_phash, after_features, capped, split_df)
        tab.to_csv(out("count_table.csv"), index=False)
        write_markdown(tab, out("count_table.md"), seed_used)
        gw = {}
        for name, d in [("after_filter", after_filter), ("after_phash", after_phash),
                        ("after_features", after_features), ("after_cap", capped)]:
            gw[name] = {c: int(((d.source == "GlobalWaste") & (d.label == c)).sum()) for c in GLOBALWASTE_KEEP}
        summary = class_summary(split_df)
        by_class = lambda d: d.label.value_counts().reindex(CLASSES, fill_value=0).to_dict()
        report.update(
            seed_used=None if failed else seed_used, split_seed_attempts=attempts,
            all_checks_passed=not failed,
            checks_passed=False if failed else ok,
            failing_checks=[k for k, v in ok.items() if not v] if failed else [],
            failure_messages=fails if failed else [],
            stage_totals=dict(mapped=len(df_all), after_filter=len(after_filter), t3_set_aside=len(t3),
                              set_aside_T3=len(t3), removed_T3_group_or_near=len(removed),
                              after_phash=len(after_phash), after_features=len(after_features),
                              after_cap=len(capped)),
            stage_by_class={k: by_class(d) for k, d in [("mapped", df_all), ("after_filter", after_filter),
                                                       ("after_phash", after_phash),
                                                       ("after_features", after_features), ("after_cap", capped)]},
            split_totals=dict(train=int((split_df.split == "train").sum()), val=int((split_df.split == "val").sum()),
                              test_T1=int((split_df.tier == "T1").sum()), test_T2=int((split_df.tier == "T2").sum()),
                              test_T3=len(t3)),
            split_by_class={sp: by_class(split_df[split_df.split == sp]) for sp in ["train", "val", "test"]},
            cap_info=cap_info.to_dict(orient="records"),
            dwsd_test_assigned_images_moved_to_train=split_df.attrs.get("dwsd_test_moved_to_train", 0),
            group_conflict_images_moved_to_train=split_df.attrs.get("group_conflict_moved_to_train", 0),
            globalwaste_surviving_per_class=gw,
            largest_source_share_in_train={c: dict(source=v["largest_source"],
                                                   share=v["largest_source_share_in_train"],
                                                   train_images=v["train_images"])
                                           for c, v in summary.items() if "largest_source" in v},
            per_class=summary,
            validation_counts_per_class=({c: v["val_count"] for c, v in summary.items()}),
            classes_near_validation_floor=[c for c, v in summary.items() if v["val_count"] < VAL_FLOOR_WARNING])
        with open(out("report.json"), "w") as f:
            json.dump(to_py(report), f, indent=2)

    # 5 and 6. split and hard checks, with seed retries for the split step
    seeds = [args.seed] + [args.seed + k for k in range(1, EXTRA_SEEDS + 1)]
    attempts, split_df, checks, seed_used = [], None, None, None
    for sd in seeds:
        cand_split = make_split(capped, sd)
        ok, fails = run_checks(cand_split, args.phash_thr, t3)
        attempts.append(dict(seed=sd, passed=not fails, failures=fails))
        log(f"Step 5/6 split seed {sd}: {'all checks passed' if not fails else '; '.join(fails)}")
        if not fails:
            split_df, checks, seed_used = cand_split, ok, sd
            break
    if split_df is None:
        write_results(cand_split, True, seeds[-1], ok, fails)
        raise SystemExit(f"hard checks failed for all {len(seeds)} seeds tried ({seeds[0]}..{seeds[-1]}). "
                         f"First seed: {attempts[0]['failures']}. Last seed: {attempts[-1]['failures']}. "
                         f"Diagnostics written to {args.out_dir} (report.json, count_table.*, splits_v3_FAILED.csv)")
    write_results(split_df, False, seed_used, checks, [])
    log(f"Done. Seed used {seed_used}. Outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

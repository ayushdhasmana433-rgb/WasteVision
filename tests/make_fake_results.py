#!/usr/bin/env python3
"""Build a small, structurally faithful fake results folder for testing pipeline/analyse.py.

    python make_fake_results.py --out /tmp/fake

Writes <out>/results (A1-A3, B1-B8, C1-C2, S1-S4, E1: JSON + logits npz), <out>/data (splits_v3.csv,
count_table.csv, report.json, dup_pairs_phash.csv.gz, label_conflicts.csv, source_overlap.csv) and
<out>/frozen_features.npz. The splits table is synthetic (about 3,000 rows, 12 classes, several sources,
tiers T1/T2/T3, photo groups). Files are made with run.py's own functions (prepare_job_data, compute_results,
leakage_block, pack_predictions, clip_scores, clip_summary, writers) and rebuild_splits.py's table builders,
so keys and array names are the real ones. Only the logits are invented, so that known answers exist:

  * B1 (battery held out): half of the unknown items look like another hazard class, half are ambiguous dry-looking
    items with hazard mass. The hazard-sum flag D2 clearly beats MSP (matched false-alarm rates).
  * B2 (medicines held out): CLIP mostly sees medicines as plastic, so D5 is clearly worse than D2.
  * B4 (shoes held out): harmless dry-looking items; every flagging design should flag few of them.
  * C1 (naive split) is clearly more accurate than C2 (grouped split).
  * A2 has the lowest validation loss; A1 is seed 42.
  * B5-B8: weaker margins (dataset shift); RealWaste is the only source of 'biological', so B8 has an unseen class.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, "..", "pipeline")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PIPE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RUN = _load("wv_run_fake", "run.py")
RS = _load("wv_rebuild_fake", "rebuild_splits.py")
CLASSES = RUN.CLASSES

# (source, class) -> images; scaled to about 3,000 rows
CELLS = {
    ("RealWaste", "biological"): 300,
    ("RealWaste", "paper_cardboard"): 110, ("TrashNet", "paper_cardboard"): 150, ("GD", "paper_cardboard"): 120, ("ZeroWaste", "paper_cardboard"): 80,
    ("RealWaste", "plastic"): 110, ("TrashNet", "plastic"): 130, ("GD", "plastic"): 130, ("ZeroWaste", "plastic"): 70, ("DWSD", "plastic"): 40,
    ("RealWaste", "metal"): 90, ("TrashNet", "metal"): 100, ("GD", "metal"): 90, ("ZeroWaste", "metal"): 50, ("DWSD", "metal"): 12,
    ("RealWaste", "glass"): 80, ("TrashNet", "glass"): 100, ("GD", "glass"): 130, ("DWSD", "glass"): 8,
    ("TrashBox", "clothes"): 100, ("GD", "clothes"): 110, ("DWSD", "clothes"): 8,
    ("GD", "shoes"): 140, ("TrashBox", "shoes"): 120, ("GlobalWaste", "shoes"): 70,
    ("RealWaste", "trash"): 70, ("TrashNet", "trash"): 70, ("GD", "trash"): 80, ("ZeroWaste", "trash"): 40,
    ("SGBattery", "battery"): 220, ("TrashBox", "battery"): 140, ("GD", "battery"): 90, ("GlobalWaste", "battery"): 60,
    ("MedPackID", "medicines_syringes"): 220, ("TrashBox", "medicines_syringes"): 140, ("PharmaSynth", "medicines_syringes"): 90,
    ("GlobalWaste", "medicines_syringes"): 60,
    ("TrashBox", "e_waste"): 200, ("GD", "e_waste"): 130, ("WastePro", "e_waste"): 110,
    ("Sanitary", "sanitary"): 150, ("TrashBox", "sanitary"): 120, ("GlobalWaste", "sanitary"): 70,
}


SCALE = 0.68          # about 3,000 rows in all


def make_splits(rng):
    """Synthetic splits_v3 table: groups of about three photos, split by group 70/10/20 inside each cell."""
    rows = []
    for (src, lab), n in CELLS.items():
        n = max(8, int(round(n * SCALE)))
        k = 0
        while k < n:
            size = min(int(rng.integers(1, 6)), n - k)
            gid = f"{src}:{lab}:{k}"
            for t in range(size):
                rows.append(dict(id=f"{src}_{lab}_{k + t}", source=src, orig_class=lab, label=lab, group=gid,
                                 norm_rel=f"{src}/{lab}/{k + t}.jpg",
                                 phash="%016x" % int(rng.integers(0, 2 ** 63, dtype=np.uint64))))
            k += size
    df = pd.DataFrame(rows)
    split = pd.Series("train", index=df.index)
    for (src, lab), g in df.groupby(["source", "label"]):
        gids = g.group.unique()
        rng.shuffle(gids)
        n = len(gids)
        n_test, n_val = max(1, int(round(0.2 * n))), max(1, int(round(0.1 * n)))
        role = {gid: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, gid in enumerate(gids)}
        split[g.index] = g.group.map(role)
    df["split"] = split
    df.loc[df.source == "PharmaSynth", "split"] = "train"
    df.loc[df.source == "DWSD", "split"] = "test"
    df["tier"] = ""
    t = df.split == "test"
    df.loc[t, "tier"] = np.where(df.loc[t, "source"] == "ZeroWaste", "T2", np.where(df.loc[t, "source"] == "DWSD", "T3", "T1"))
    return df


def make_all_df(splits, rng, dup_frac=0.15):
    """Pre-de-duplication table for C1/C2: the splits rows plus near-duplicate copies (same label, new photo group,
    pHash within 2 bits) of a share of the rows. Returns (all_df, pair index list)."""
    base = splits[["id", "source", "orig_class", "label", "group", "norm_rel", "phash"]].copy()
    pick = rng.choice(len(base), size=int(dup_frac * len(base)), replace=False)
    extra = []
    for i in pick:
        r = base.iloc[i].to_dict()
        h = int(r["phash"], 16)
        for b in rng.choice(64, size=int(rng.integers(0, 3)), replace=False):
            h ^= 1 << int(b)
        r.update(id=r["id"] + "_dup", group=r["group"] + "_dupgrp", phash="%016x" % h, norm_rel=r["norm_rel"] + ".dup")
        extra.append(r)
    all_df = pd.concat([base, pd.DataFrame(extra)], ignore_index=True)
    return all_df, [(int(i), len(base) + k) for k, i in enumerate(pick)]


# ---------------------------------------------------------------- logits
def logits_for(fr, space, role, jid, rng, margin, sigma, tier_drop=True):
    n, K = len(fr), len(space.names)
    names = list(space.names)
    y = fr["y"].values
    L = rng.normal(0, sigma, (n, K))
    m = np.full(n, float(margin))
    if tier_drop and role == "test":
        m -= np.where(fr["tier"].values == "T2", 1.0, 0.0) + np.where(fr["tier"].values == "T3", 1.5, 0.0)
    ok = y >= 0
    L[np.arange(n)[ok], y[ok]] += m[ok]
    job = RUN.JOBS[jid]
    if role == "unknown":
        if job.kind == "holdout_source":
            L[ok, :] = rng.normal(0, 1.3, (int(ok.sum()), K))
            L[np.arange(n)[ok], y[ok]] += margin - 1.5
            bad = ~ok
            if bad.any():
                L[bad] = rng.normal(0, 1.3, (int(bad.sum()), K))
                L[np.nonzero(bad)[0], rng.integers(0, K, int(bad.sum()))] += 2.0
        elif job.label_space == "stream":
            for i in range(n):
                L[i] = rng.normal(0, sigma, K)
                L[i, RUN.SC if rng.random() < 0.55 else RUN.DRY] += 3.2
        else:
            h = job.holdout_class
            haz = [names.index(c) for c in RUN.HAZARD_CLASSES if c in names]
            dry = [names.index(c) for c in ("plastic", "metal", "paper_cardboard", "clothes", "trash") if c in names]
            for i in range(n):
                L[i] = rng.normal(0, sigma, K)
                if h == "shoes":
                    L[i, names.index(rng.choice(["clothes", "trash", "plastic"]))] += 3.5
                elif rng.random() < 0.5:
                    L[i, haz[rng.integers(len(haz))]] += margin           # confidently another hazard class
                else:
                    L[i, dry[rng.integers(len(dry))]] += margin - 1.0     # dry-looking with hazard mass
                    L[i, haz[rng.integers(len(haz))]] += 2.4
    return L


def history(final_val_loss):
    h = [dict(phase="head", epoch=e + 1, lr=1e-3, loss=0.9 - 0.1 * e, val_loss=final_val_loss + 0.5 - 0.1 * e) for e in range(3)]
    h += [dict(phase="finetune", epoch=4 + e, lr=1e-5, loss=0.5, val_loss=final_val_loss + 0.05 * (2 - e) if e < 3 else final_val_loss + 0.1)
          for e in range(4)]
    h[-2]["val_loss"] = final_val_loss                          # best epoch of the last phase
    return h


PARAMS = {"A1": (4.0, 1.0, 0.30), "A2": (3.8, 1.0, 0.25), "A3": (4.2, 1.0, 0.28), "S1": (4.0, 1.0, 0.35)}


def make_job(jid, splits, all_df, root, splits_path, seed):
    rng = np.random.default_rng(seed)
    job = RUN.JOBS[jid]
    data = RUN.prepare_job_data(job, splits, all_df)
    margin, sigma, vl = PARAMS.get(jid, (4.0, 1.0, 0.3))
    if jid == "C1":
        margin, sigma = 4.2, 1.2
    if jid == "C2":
        margin, sigma = 3.0, 1.3
    leak = job.kind in ("leak_naive", "leak_grouped")
    val_lg = logits_for(data.val, data.space, "val", jid, rng, margin, sigma)
    test_lg = logits_for(data.test, data.space, "test", jid, rng, margin, sigma, tier_drop=not leak)
    unk_lg = logits_for(data.unknown, data.space, "unknown", jid, rng, margin, sigma) if data.unknown is not None else None
    near = None
    if leak:
        # test images with a training image within the pHash threshold get a stronger true-class logit, so accuracy
        # with a near-duplicate is clearly higher than without (known difference for the tests)
        near = RUN.has_near_neighbour(RUN.phash_array(data.test.phash.values), RUN.phash_array(data.train.phash.values),
                                      RUN.C_PHASH_THRESHOLD)
        ok = data.test["y"].values >= 0
        test_lg[np.nonzero(near & ok)[0], data.test["y"].values[near & ok]] += 2.0
    val, test = RUN._arrays(data.val, val_lg), RUN._arrays(data.test, test_lg)
    unknown = RUN._arrays(data.unknown, unk_lg) if data.unknown is not None else None
    res = RUN.compute_results(data.space, val, test, unknown, unknown_is_labelled=(job.kind == "holdout_source"))
    if leak:
        res["leakage"] = RUN.leakage_block(data.train, data.test, softmax(test_lg / res["temperature"], 1))
    parts = {"val": (data.val, val_lg), "test": (data.test, test_lg)}
    if unknown is not None:
        parts["unknown"] = (data.unknown, unk_lg)
    packed = RUN.pack_predictions(parts, res["temperature"], data.space)
    if leak:                                  # as run.py: boolean per row, True only for test rows with a near-duplicate
        flag = np.zeros(len(packed["ids"]), dtype=bool)
        flag[packed["role"] == "test"] = near
        packed["near_dup_in_train"] = flag
    RUN.write_npz_atomic(os.path.join(root, "results", f"{jid}_logits.npz"), **packed)
    result = dict(job=jid, what=job.what, kind=job.kind, seed=job.seed, label_space=data.space.kind, label_names=data.space.names,
                  hazard_threshold=RUN.SC_THRESHOLD, reject_threshold_msp=res["thresholds"]["reject_msp"],
                  held_out_class=data.meta.get("held_out_class"), held_out_source=data.meta.get("held_out_source"),
                  unseen_classes=data.meta.get("unseen_classes"), meta=data.meta, settings=RUN.SETTINGS, device="fake",
                  history=history(vl), provenance=dict(splits_sha256=None, splits_file=splits_path), finished="fake", **res)
    RUN.write_json_atomic(RUN.result_path(root, jid), result)


def make_clip(splits, root, splits_path, rng):
    """E1 file made with run.clip_scores / clip_summary and the same npz arrays as run.run_clip_job. CLIP sees medicines
    mostly as plastic (so D5 is bad on them) and the other hazard classes about 70% of the time."""
    d = 24
    ct = rng.normal(size=(12, d))
    ct /= np.linalg.norm(ct, axis=1, keepdims=True)
    hz = [CLASSES.index(c) for c in RUN.HAZARD_CLASSES]
    ok = [i for i in range(12) if i not in hz]
    h0 = ct[hz].mean(0)
    h1 = ct[ok].mean(0)
    ht = np.stack([h0 / np.linalg.norm(h0), h1 / np.linalg.norm(h1)])
    y = splits["label"].map(RUN.CLASS_TO_IDX).values
    seen = y.copy()
    for i, c in enumerate(y):
        name = CLASSES[c]
        if name == "medicines_syringes":
            seen[i] = CLASSES.index("plastic") if rng.random() < 0.85 else c
        elif rng.random() > (0.7 if name in RUN.HAZARD_CLASSES else 0.8):
            seen[i] = rng.integers(12)
    img = ct[seen] + 0.9 * rng.normal(size=(len(y), d))
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    scores = RUN.clip_scores(img, ct, ht, 12.0)
    summary = RUN.clip_summary(scores, splits)
    s = splits
    RUN.write_npz_atomic(os.path.join(root, "results", "E1_logits.npz"),
                         ids=np.asarray(s["id"].astype(str), dtype=str),
                         role=np.where(s.split == "val", "val", np.where(s.split.isin(RUN.EVAL_SPLITS), "test", "train")),
                         split=np.asarray(s["split"].astype(str), dtype=str), tier=np.asarray(s["tier"].astype(str), dtype=str),
                         source=np.asarray(s["source"].astype(str), dtype=str), group=np.asarray(s["group"].astype(str), dtype=str), label_idx=y, class_idx=y, stream_true=RUN.CLASS_STREAM_IDX[y],
                         logits=scores["class_logits"].astype(np.float32), class_probs=scores["class_probs"].astype(np.float32),
                         stream_probs=scores["stream_probs"].astype(np.float32), hazard_prob=scores["hazard_prob"].astype(np.float32),
                         label_names=np.array(CLASSES), temperature=np.float64(1.0))
    job = RUN.JOBS["E1"]
    RUN.write_json_atomic(RUN.result_path(root, "E1"),
                          dict(job="E1", what=job.what, kind=job.kind, label_space="class", label_names=CLASSES,
                               settings=dict(model="fake", logit_scale=12.0, hazard_thresholds=RUN.CLIP_HAZARD_THRESHOLDS),
                               n_images=len(s), finished="fake", provenance=dict(splits_sha256=None), **summary))


def make_data_files(splits, all_df, pairs, out, rng):
    """splits_v3.csv, count_table.csv, report.json, dup_pairs_phash.csv.gz, label_conflicts.csv, source_overlap.csv,
    made with rebuild_splits.py's own builders; and a frozen-feature file for the probes."""
    os.makedirs(out, exist_ok=True)
    cols = RS.OUT_COLS + ["split", "tier"]
    splits[cols].to_csv(os.path.join(out, "splits_v3.csv"), index=False)
    t3 = splits[splits.tier == "T3"]
    non_t3 = splits[splits.tier != "T3"]
    empty = splits.iloc[:0]
    tab = RS.build_count_table(all_df, all_df, t3, empty, non_t3, non_t3, non_t3, non_t3)
    tab.to_csv(os.path.join(out, "count_table.csv"), index=False)
    pi, pj = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    # a few cross-source pairs with other labels and the conflict rows that go with them
    extra = rng.choice(len(all_df), size=(40, 2), replace=True)
    extra = extra[extra[:, 0] != extra[:, 1]]
    pi, pj = np.r_[pi, extra[:, 0]], np.r_[pj, extra[:, 1]]
    dist = np.r_[[bin(int(all_df.phash[a], 16) ^ int(all_df.phash[b], 16)).count("1") for a, b in pairs],
                 rng.integers(0, 5, len(extra))] * 1
    ids, src, lab = all_df.id.values, all_df.source.values, all_df.label.values
    pr = pd.DataFrame(dict(id_a=ids[pi], id_b=ids[pj], source_a=src[pi], source_b=src[pj], label_a=lab[pi], label_b=lab[pj], distance=dist))
    pr.to_csv(os.path.join(out, "dup_pairs_phash.csv.gz"), index=False)
    ov = RS.overlap_matrix(src[pi], src[pj])
    ov.to_csv(os.path.join(out, "source_overlap.csv"))
    conf = [(int(a), int(b), RS.REASON_LABEL) for a, b in zip(pi, pj) if lab[a] != lab[b]]
    conf += [(int(a), int(b), RS.REASON_NEAR) for a, b in list(zip(pi, pj))[:5]]
    vals = np.array([dist[list(zip(pi, pj)).index((a, b))] for a, b, _ in conf], dtype=float)
    RS.conflict_frame(all_df, conf, "phash", vals, "distance").to_csv(os.path.join(out, "label_conflicts.csv"), index=False)
    json.dump(dict(stage_totals=dict(mapped=len(all_df), after_filter=len(all_df), after_phash=len(non_t3), after_cap=len(non_t3))),
              open(os.path.join(out, "report.json"), "w"), indent=1)
    # frozen features: class centroid + source offset + noise (16 dims)
    d = 16
    cen = {c: rng.normal(size=d) * 2 for c in CLASSES}
    off = {s: rng.normal(size=d) for s in splits.source.unique()}
    feats = np.stack([cen[l] + off[s] for l, s in zip(splits.label, splits.source)]) + rng.normal(size=(len(splits), d))
    np.savez(os.path.join(os.path.dirname(out.rstrip("/")), "frozen_features.npz"), ids=splits.id.astype(str).values,
             feats=feats.astype(np.float32))


def build(out, seed=0, scale=None):
    """Create <out>/results, <out>/data, <out>/frozen_features.npz. Returns dict of paths.
    scale: cell-size factor (default SCALE, about 3,000 rows); a bigger value makes a timing test."""
    global SCALE
    if scale is not None:
        SCALE = scale
    rng = np.random.default_rng(seed)
    root = out                       # run.py's drive root: results/ is created below it
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    splits = RUN._fill_tiers(make_splits(rng))
    all_df, pairs = make_all_df(splits, rng)
    make_data_files(splits, all_df, pairs, os.path.join(out, "data"), rng)
    sp = os.path.join(out, "data", "splits_v3.csv")
    for k, jid in enumerate(RUN.JOBS):
        if jid == "E1":
            make_clip(splits, root, sp, np.random.default_rng(seed + 99))
        else:
            make_job(jid, splits, all_df if jid in ("C1", "C2") else None, root, sp, seed + 100 + k)
    return dict(results=os.path.join(root, "results"), data=os.path.join(out, "data"), features=os.path.join(out, "frozen_features.npz"), splits=splits)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=None, help="cell-size factor; default gives about 3,000 rows")
    a = ap.parse_args()
    p = build(a.out, a.seed, a.scale)
    print({k: v for k, v in p.items() if k != "splits"}, "rows:", len(p["splits"]))

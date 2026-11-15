"""WasteVision v3 job runner (Google Colab, T4, TensorFlow/Keras, Drive mounted).

    python run.py --job A1
    python run.py --queue queue.txt

Recipe: copied from the 31 October notebook (reference cells 01, 12, 13, 14, 16-18, 13b, 22b).
Everything that does not need TensorFlow (job table, label sets, splits, metrics, thresholds,
training state machine) is plain numpy/pandas, so the tests import this file without TensorFlow.
TensorFlow, Keras, torch and open_clip are imported inside the functions that use them.

Outputs under <drive-root>:
    runs/<job>/            per-epoch checkpoints and state.json (resume point)
    results/<job>.json     finished job (its existence means "done, skip")
    results/<job>_logits.npz
    data/clip_features_vitb32.npz     (E1 feature cache)
"""
import argparse
import gc
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp, softmax

# --------------------------------------------------------------------------------------
# Settings. Values copied from reference_notebook_cells/cell_01.py unless a comment says otherwise.
# --------------------------------------------------------------------------------------
IMG_SIZE = 224                      # cell_01.py line 17
BATCH_SIZE = 32                     # cell_01.py line 18
SEED = 42                           # cell_01.py line 19; also the fixed tf.data shuffle seed, as in cell_12.py
EPOCHS_HEAD, EPOCHS_FINETUNE = 6, 8                 # cell_01.py lines 40-41
UNFREEZE_LAST_LAYERS = 30                           # cell_01.py line 42
LR_HEAD, LR_FINETUNE = 1e-3, 1e-5                   # cell_01.py line 43
PATIENCE_HEAD, PATIENCE_FINETUNE = 3, 4             # cell_14.py lines 49 and 53 (fit(..., patience, ...))
PLATEAU_FACTOR, PLATEAU_MIN_LR = 0.3, 1e-7          # cell_14.py lines 22-23
PRETRAINED_WEIGHTS = "imagenet"                     # cell_01.py line 44
COST_RATIO = 4.0                                    # cell_01.py line 48
SENSITIVITY_COST_RATIOS = [2.0, 4.0, 9.0]           # cell_01.py line 49 -> thresholds 0.33, 0.20, 0.10
TARGET_STREAM_ACC, TARGET_CLASS_ACC = 0.97, 0.95    # cell_01.py lines 50-51
REJECT_VAL_QUANTILE = 0.05                          # cell_22b.py line 174 (rejects 5% of known validation items)
SC_THRESHOLD = 1.0 / (1.0 + COST_RATIO)             # cell_02.py line 85
CAP_PER_CLASS = 2500                                # cell_01.py line 30 (leakage pool; PLAN/task: cap 2,500)
MAX_PER_GROUP = 20                                  # task text for C1/C2 (PLAN section 6.3)
C_PHASH_THRESHOLD = 4                               # task text / PLAN 6.2 (the 31 Oct notebook used 5, cell_01.py line 32)
C_TEST_FRACTION, C_VAL_FRACTION = 0.20, 0.10        # cell_01.py line 31
ECE_BINS = 15                                       # cell_16.py line 11

CLASSES = ["biological", "paper_cardboard", "plastic", "metal", "glass", "clothes", "shoes",
           "trash", "battery", "medicines_syringes", "e_waste", "sanitary"]          # cell_02.py lines 65-66
STREAMS = ["Wet", "Dry", "Special Care", "Sanitary"]                                  # cell_02.py line 73
STREAM_OF = {c: "Dry" for c in CLASSES}                                               # cell_02.py lines 74-78
STREAM_OF["biological"] = "Wet"
for _c in ["battery", "medicines_syringes", "e_waste"]:
    STREAM_OF[_c] = "Special Care"
STREAM_OF["sanitary"] = "Sanitary"
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS_STREAM_IDX = np.array([STREAMS.index(STREAM_OF[c]) for c in CLASSES])
SC = STREAMS.index("Special Care")
DRY = STREAMS.index("Dry")
HAZARD_CLASSES = ["battery", "medicines_syringes", "e_waste"]
GLOBALWASTE_KEPT = ["sanitary", "shoes", "battery", "medicines_syringes"]            # PLAN section 5

SETTINGS = dict(
    arch="efficientnetv2b0", img_size=IMG_SIZE, batch_size=BATCH_SIZE, epochs_head=EPOCHS_HEAD,
    epochs_finetune=EPOCHS_FINETUNE, unfreeze_last_layers=UNFREEZE_LAST_LAYERS, lr_head=LR_HEAD,
    lr_finetune=LR_FINETUNE, patience_head=PATIENCE_HEAD, patience_finetune=PATIENCE_FINETUNE,
    plateau_factor=PLATEAU_FACTOR, pretrained=PRETRAINED_WEIGHTS, balanced_batches=True,
    cost_ratio=COST_RATIO, sensitivity_cost_ratios=SENSITIVITY_COST_RATIOS,
    target_stream_acc=TARGET_STREAM_ACC, target_class_acc=TARGET_CLASS_ACC,
    reject_val_quantile=REJECT_VAL_QUANTILE, dataset_shuffle_seed=SEED)


# --------------------------------------------------------------------------------------
# Job table
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Job:
    id: str
    kind: str                      # full | holdout_class | holdout_source | leak_naive | leak_grouped | clip
    seed: int = 42
    label_space: str = "class"     # class (12-way) | stream (4-way)
    holdout_class: str = ""
    holdout_source: str = ""
    what: str = ""


def _build_jobs():
    jobs = {}
    for i, s in enumerate((42, 7, 123), 1):
        jobs[f"A{i}"] = Job(f"A{i}", "full", seed=s, what=f"12-class model, seed {s}")
    for i, c in enumerate(["battery", "medicines_syringes", "e_waste", "shoes"], 1):
        jobs[f"B{i}"] = Job(f"B{i}", "holdout_class", holdout_class=c, what=f"hold out class {c}")
    for i, s in enumerate(["MedPackID", "TrashBox", "GD", "RealWaste"], 5):
        jobs[f"B{i}"] = Job(f"B{i}", "holdout_source", holdout_source=s, what=f"leave out source {s}")
    jobs["C1"] = Job("C1", "leak_naive", what="leakage pool, naive random 70/10/20 split")
    jobs["C2"] = Job("C2", "leak_grouped", what="leakage pool, grouped + duplicate-cluster split")
    jobs["S1"] = Job("S1", "full", label_space="stream", what="4-stream network, full data")
    for i, c in enumerate(HAZARD_CLASSES, 2):
        jobs[f"S{i}"] = Job(f"S{i}", "holdout_class", label_space="stream", holdout_class=c,
                            what=f"4-stream network, hold out class {c}")
    jobs["E1"] = Job("E1", "clip", what="CLIP ViT-B/32 zero-shot")
    return jobs


JOBS = _build_jobs()


def read_queue(path):
    """Job ids from a queue file: one per line, blank lines and '#' comments ignored. Unknown ids raise."""
    ids = []
    with open(path) as f:
        for line in f:
            tok = line.split("#")[0].strip().upper()
            if not tok:
                continue
            if tok not in JOBS:
                raise ValueError(f"unknown job id in queue: {tok!r} (known: {', '.join(JOBS)})")
            ids.append(tok)
    return ids


def result_path(drive_root, job_id):
    return os.path.join(drive_root, "results", f"{job_id}.json")


def job_is_done(drive_root, job_id):
    """A job is finished when its result JSON exists (the JSON is written last)."""
    return os.path.exists(result_path(drive_root, job_id))


class StaleResultError(RuntimeError):
    """A result JSON exists but was made from different input files."""


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def current_hashes(job, splits_path, all_csv_path):
    """Input provenance stored in every result: SHA-256 of the splits file, and for C jobs of the all-images CSV."""
    out = dict(splits_sha256=sha256_file(splits_path) if os.path.exists(splits_path) else None)
    if job.kind in ("leak_naive", "leak_grouped"):
        out["all_csv_sha256"] = sha256_file(all_csv_path)
    return out


def result_status(drive_root, job, splits_path, all_csv_path):
    """'missing' (no result), 'done' (result exists and its stored hashes equal the current files') or 'stale'."""
    p = result_path(drive_root, job.id)
    if not os.path.exists(p):
        return "missing"
    try:
        stored = json.load(open(p)).get("provenance")
    except Exception:  # noqa: BLE001
        return "stale"
    if not isinstance(stored, dict):
        return "stale"
    cur = current_hashes(job, splits_path, all_csv_path)
    return "done" if all(stored.get(k) == v for k, v in cur.items()) else "stale"


def archive_stale(drive_root, job_id):
    """Move an out-of-date result, its logits and its run directory to results/stale/ (with a timestamp)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(drive_root, "results", "stale")
    os.makedirs(dest, exist_ok=True)
    moved = []
    for src, name in ((result_path(drive_root, job_id), f"{job_id}_{stamp}.json"),
                      (os.path.join(drive_root, "results", f"{job_id}_logits.npz"), f"{job_id}_{stamp}_logits.npz"),
                      (os.path.join(drive_root, "runs", job_id), f"{job_id}_{stamp}_runs")):
        if os.path.exists(src):
            shutil.move(src, os.path.join(dest, name))
            moved.append(name)
    return moved


def job_command(job_id, args, script=None, python=None):
    """Command line that runs one job in a fresh Python process."""
    cmd = [python or sys.executable, "-u", script or os.path.abspath(__file__), "--job", job_id,
           "--drive-root", args.drive_root, "--data-root", args.data_root, "--splits", args.splits,
           "--all-csv", args.all_csv, "--tar", args.tar, "--extra-tar", args.extra_tar]
    if args.rerun_stale:
        cmd.append("--rerun-stale")
    return cmd


def make_subprocess_runner(args, script=None, python=None):
    """Runner for run_queue: each job in its own subprocess (frees TensorFlow memory between jobs). Prints the
    exit status; a non-zero status raises so the queue reports the job as failed and goes on."""
    def runner(job_id):
        rc = subprocess.run(job_command(job_id, args, script, python)).returncode
        print(f"=== JOB {job_id} | subprocess exit status {rc} ===", flush=True)
        if rc != 0:
            raise RuntimeError(f"job {job_id} exited with status {rc}")
    return runner


def run_queue(job_ids, drive_root, runner, is_done=None):
    """Run jobs in order with runner(job_id). Finished jobs are skipped; a failing job is reported and the
    queue goes on. Returns dict(done=[...], skipped=[...], failed={id: message}).
    is_done(job_id) decides what 'finished' means (default: the result JSON exists)."""
    out = dict(done=[], skipped=[], failed={})
    is_done = is_done or (lambda j: job_is_done(drive_root, j))
    for jid in job_ids:
        if is_done(jid):
            print(f"=== JOB {jid} | SKIP (result exists) ===", flush=True)
            out["skipped"].append(jid)
            continue
        try:
            runner(jid)
            out["done"].append(jid)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - keep the queue alive, report at the end
            import traceback
            traceback.print_exc()
            print(f"=== JOB {jid} | FAILED: {e!r} ===", flush=True)
            out["failed"][jid] = repr(e)
    print(f"Queue summary: done={out['done']} skipped={out['skipped']} failed={list(out['failed'])}", flush=True)
    return out


# --------------------------------------------------------------------------------------
# Label spaces
# --------------------------------------------------------------------------------------
@dataclass
class LabelSpace:
    names: list
    stream_idx: np.ndarray         # label index -> stream index
    kind: str = "class"

    @property
    def smat(self):
        """K x 4 matrix mapping label probabilities to summed stream probabilities."""
        m = np.zeros((len(self.names), len(STREAMS)))
        m[np.arange(len(self.names)), self.stream_idx] = 1.0
        return m


def class_space(names):
    return LabelSpace(list(names), np.array([STREAMS.index(STREAM_OF[c]) for c in names]), "class")


def stream_space():
    return LabelSpace(list(STREAMS), np.arange(len(STREAMS)), "stream")


# --------------------------------------------------------------------------------------
# Tables: splits_v3.csv, all_images_before_dedup.csv
# --------------------------------------------------------------------------------------
SPLIT_COLS = ["id", "source", "orig_class", "label", "group", "norm_rel", "phash", "split"]
ALL_COLS = ["id", "source", "orig_class", "label", "group", "norm_rel", "phash"]
EVAL_SPLITS = ("test", "dwsd_test")


def _fill_tiers(df):
    """Test rows need a tier. Use the column when present; otherwise T3 for dwsd_test, T2 for ZeroWaste, else T1."""
    df = df.copy()
    if "tier" not in df.columns:
        df["tier"] = ""
    df["tier"] = df["tier"].fillna("").astype(str)
    is_test = df["split"].isin(EVAL_SPLITS)
    missing = is_test & (df["tier"] == "")
    guess = np.where(df["split"] == "dwsd_test", "T3", np.where(df["source"] == "ZeroWaste", "T2", "T1"))
    df.loc[missing, "tier"] = guess[missing.values]
    df.loc[~is_test, "tier"] = ""
    return df


def load_splits(path):
    df = pd.read_csv(path, dtype=str)           # all text: keeps pHash hex strings and group ids exactly
    miss = [c for c in SPLIT_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"{path} lacks columns {miss}")
    bad = sorted(set(df.label) - set(CLASSES))
    if bad:
        raise ValueError(f"splits file has labels outside the 12 classes: {bad}")
    return _fill_tiers(df)


def load_all_csv(path):
    df = pd.read_csv(path, dtype=str)
    miss = [c for c in ALL_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"{path} lacks columns {miss}")
    return df


def attach_paths_all(frames, data_root, exists=os.path.exists, max_missing_frac=0.005, show=20):
    """frames: {name: DataFrame or None}. Adds a 'path' column to each. The missing-file check covers the whole
    job's image list (every frame together, each file counted once): the first `show` missing norm_rel paths are
    always printed, the run stops when more than max_missing_frac of the job's images are missing, otherwise the
    missing rows are dropped from every frame."""
    root = str(data_root).rstrip("/")
    rels = set()
    for f in frames.values():
        if f is not None:
            rels.update(f["norm_rel"].astype(str))
    missing = sorted(r for r in rels if not exists(f"{root}/{r}"))
    frac = len(missing) / max(len(rels), 1)
    if missing:
        print(f"WARNING: {len(missing)} of {len(rels)} image files are missing ({frac:.3%}); first {min(show, len(missing))}:",
              flush=True)
        for r in missing[:show]:
            print(f"    {r}", flush=True)
    else:
        print(f"All {len(rels)} image files found.", flush=True)
    if frac > max_missing_frac:
        raise RuntimeError(f"{len(missing)} of {len(rels)} image files missing under {data_root} "
                           f"({frac:.2%} > {max_missing_frac:.1%}): extraction incomplete?")
    gone, out = set(missing), {}
    for k, f in frames.items():
        if f is None:
            out[k] = None
            continue
        f = f.copy()
        f["path"] = root + "/" + f["norm_rel"].astype(str)
        out[k] = f[~f["norm_rel"].astype(str).isin(gone)].reset_index(drop=True)
    return out


def add_label_columns(df, space):
    """class_idx (12-class index), stream_true (0..3) and y (label-space index, -1 when the class is not in the space)."""
    df = df.copy()
    df["class_idx"] = df["label"].map(CLASS_TO_IDX).astype(int)
    df["stream_true"] = CLASS_STREAM_IDX[df["class_idx"].values]
    if space.kind == "stream":
        df["y"] = df["stream_true"]
    else:
        pos = {n: i for i, n in enumerate(space.names)}
        df["y"] = df["label"].map(pos).fillna(-1).astype(int)
    return df


@dataclass
class JobData:
    space: LabelSpace
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    unknown: object = None         # DataFrame or None
    meta: dict = field(default_factory=dict)


def prepare_job_data(job, splits_df, all_df=None, seed=SEED):
    """Build label space and train/val/test/unknown frames for a training job (no TensorFlow, no image files)."""
    if job.kind in ("leak_naive", "leak_grouped"):
        return prepare_leakage_data(job, all_df, seed)
    df = splits_df
    train, val = df[df.split == "train"], df[df.split == "val"]
    test = df[df.split.isin(EVAL_SPLITS)]
    unknown, meta = None, dict(held_out_class=job.holdout_class or None, held_out_source=job.holdout_source or None,
                               unseen_classes=[])
    if job.kind == "holdout_class":
        h = job.holdout_class
        train, val = train[train.label != h], val[val.label != h]
        unknown, test = test[test.label == h], test[test.label != h]
        names = [c for c in CLASSES if c != h]
    elif job.kind == "holdout_source":
        s = job.holdout_source
        if s not in set(df.source):
            raise ValueError(f"source {s!r} not in splits file (sources: {sorted(set(df.source))})")
        unknown = df[df.source == s]                    # ALL splits of the left-out source
        train, val = train[train.source != s], val[val.source != s]
        test = test[test.source != s]
        names = [c for c in CLASSES if (train.label == c).any()]
        meta["unseen_classes"] = [c for c in CLASSES if c not in names]
        val = val[val.label.isin(names)]
        meta["n_test_rows_of_unseen_classes_dropped"] = int((~test.label.isin(names)).sum())
        test = test[test.label.isin(names)]
    elif job.kind == "full":
        names = list(CLASSES)
    else:
        raise ValueError(f"not a training job kind: {job.kind}")
    space = stream_space() if job.label_space == "stream" else class_space(names)
    if job.label_space == "stream":                      # label space is the 4 streams; class list kept for the record
        meta["classes_in_training"] = names
    frames = [add_label_columns(f, space).reset_index(drop=True) for f in (train, val, test)]
    if unknown is not None:
        unknown = add_label_columns(unknown, space).reset_index(drop=True)
    if len(frames[0]) == 0 or len(frames[1]) == 0:
        raise ValueError("empty train or validation set")
    meta["counts"] = dict(train=len(frames[0]), val=len(frames[1]), test=len(frames[2]),
                          unknown=0 if unknown is None else len(unknown))
    return JobData(space, frames[0], frames[1], frames[2], unknown, meta)


# --------------------------------------------------------------------------------------
# Leakage pair (C1, C2): pool, cap, splits, pHash helpers
# --------------------------------------------------------------------------------------
def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    pop8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    x = np.ascontiguousarray(x)
    return pop8[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1)


def phash_array(hex_strings):
    return np.array([int(h, 16) for h in hex_strings], dtype=np.uint64)


def near_pairs(H, thr, chunk=256):
    """All index pairs (i<j) whose 64-bit hashes differ in at most thr bits."""
    out = []
    for s in range(0, len(H), chunk):
        d = popcount64(H[s:s + chunk, None] ^ H[None, :])
        ii, jj = np.nonzero(d <= thr)
        ii = ii + s
        keep = jj > ii
        out.append(np.stack([ii[keep], jj[keep]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), np.int64)


def has_near_neighbour(H_query, H_ref, thr, chunk=256):
    """Boolean per query hash: is any reference hash within thr bits?"""
    flags = np.zeros(len(H_query), bool)
    if len(H_ref) == 0:
        return flags
    for a in range(0, len(H_query), chunk):
        flags[a:a + chunk] = popcount64(H_query[a:a + chunk, None] ^ H_ref[None, :]).min(1) <= thr
    return flags


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_pool(all_df):
    """Pre-de-duplication pool for C1/C2: drop DWSD, PharmaSynth, rows without a usable label, and GlobalWaste
    rows outside the four kept classes."""
    df = all_df[~all_df.source.isin(["DWSD", "PharmaSynth"])]
    df = df[df.label.isin(CLASSES)].copy()
    # a row without a photo group is its own group (same rule as rebuild_splits.fix_missing_groups)
    g = df["group"].astype("string").str.strip()
    missing = g.isna() | (g == "") | (g.str.lower() == "nan")
    df.loc[missing, "group"] = df.loc[missing, "id"]
    df = df[(df.source != "GlobalWaste") | df.label.isin(GLOBALWASTE_KEPT)]
    return df.reset_index(drop=True)


def _waterfill(sizes, cap):
    """Equal quota per source, unused quota shared out again. sizes: {source: available}. Returns {source: take}."""
    take = {s: 0 for s in sizes}
    remaining = cap
    open_src = [s for s in sizes if sizes[s] > 0]
    while remaining > 0 and open_src:
        share = max(remaining // len(open_src), 1)
        nxt = []
        for s in open_src:
            add = min(share, sizes[s] - take[s], remaining)
            take[s] += add
            remaining -= add
            if take[s] < sizes[s]:
                nxt.append(s)
            if remaining <= 0:
                break
        open_src = nxt
    return take


def cap_pool(df, cap=CAP_PER_CLASS, max_per_group=MAX_PER_GROUP, seed=SEED):
    """Per class: at most max_per_group images from one photo group, then at most cap images with an equal
    quota per source (unused quota shared). Random within source, seeded."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("label", sort=True):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        g = g[g.groupby("group").cumcount() < max_per_group]
        if len(g) > cap:
            sizes = g.groupby("source").size().to_dict()
            take = _waterfill(sizes, cap)
            g = pd.concat([gs.iloc[:take[s]] for s, gs in g.groupby("source")])
        parts.append(g)
    return pd.concat(parts).sort_index().reset_index(drop=True)


def duplicate_clusters(pool, thr=C_PHASH_THRESHOLD):
    """Cluster id per row: images of one photo group, and images within pHash distance <= thr, share a cluster."""
    uf = UnionFind(len(pool))
    first = {}
    for k, g in enumerate(pool.group.astype(str).values):
        if g in first:
            uf.union(first[g], k)
        else:
            first[g] = k
    for a, b in near_pairs(phash_array(pool.phash.values), thr):
        uf.union(int(a), int(b))
    return np.array([uf.find(k) for k in range(len(pool))])


def split_naive(n, seed=SEED, test_frac=C_TEST_FRACTION, val_frac=C_VAL_FRACTION):
    """Random split by image. Exact counts: round(test_frac*n) test, round(val_frac*n) val."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test, n_val = int(round(test_frac * n)), int(round(val_frac * n))
    out = np.array(["train"] * n, dtype=object)
    out[order[:n_test]] = "test"
    out[order[n_test:n_test + n_val]] = "val"
    return out


def split_grouped(clusters, labels, seed=SEED, test_frac=C_TEST_FRACTION, val_frac=C_VAL_FRACTION,
                  max_cluster_frac=0.05):
    """Whole clusters go to one split, filled per class so every class lands near 70/10/20. A cluster takes the
    class most of its images have. Per class the clusters are shuffled and filled test, then val, then train until
    round(frac * class size) images are reached (overshoot is at most one cluster). A cluster larger than
    max_cluster_frac of the pool goes to train. Returns (split array, number of oversize clusters)."""
    clusters, labels = np.asarray(clusters), np.asarray(labels)
    n = len(clusters)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(c=clusters, l=labels))
    size = df.groupby("c").size()

    def primary(s):
        vc = s.value_counts()
        return vc[vc == vc.max()].index.min()
    prim = df.groupby("c")["l"].agg(primary)
    assign, oversize = {}, 0
    for cid in size.index[size > max_cluster_frac * n]:
        assign[cid] = "train"
        oversize += 1
    for lab in sorted(prim.unique()):
        n_c = int((labels == lab).sum())
        n_test, n_val = int(round(test_frac * n_c)), int(round(val_frac * n_c))
        ids = [c for c in prim.index[prim == lab] if c not in assign]
        c_test = c_val = 0
        for k in rng.permutation(len(ids)):
            cid = ids[k]
            if c_test < n_test:
                assign[cid] = "test"; c_test += size[cid]
            elif c_val < n_val:
                assign[cid] = "val"; c_val += size[cid]
            else:
                assign[cid] = "train"
    return np.array([assign[c] for c in clusters], dtype=object), oversize


def split_report(split, labels, clusters=None, n_oversize=0):
    """Fractions overall and per class, plus cluster counts, for the result JSON."""
    split, labels = np.asarray(split), np.asarray(labels)
    per = {}
    for lab in sorted(set(labels.tolist())):
        m = labels == lab
        per[lab] = dict(n=int(m.sum()), **{s: float((split[m] == s).mean()) for s in ("train", "val", "test")},
                        n_test=int((split[m] == "test").sum()))
    rep = dict(n_images=int(len(split)), overall={s: float((split == s).mean()) for s in ("train", "val", "test")},
               per_class=per)
    if clusters is not None:
        _, counts = np.unique(np.asarray(clusters), return_counts=True)
        rep.update(n_clusters=int(len(counts)), largest_cluster_size=int(counts.max()),
                   n_oversize_clusters_sent_to_train=int(n_oversize))
    return rep


def check_grouped_split(rep, lo=0.15, hi=0.25):
    """Stop with a clear message when the grouped split is unusable."""
    t = rep["overall"]["test"]
    if not lo <= t <= hi:
        raise ValueError(f"C2 grouped split: overall test fraction {t:.3f} is outside {lo:.2f}-{hi:.2f} "
                         f"(largest cluster {rep.get('largest_cluster_size')}, {rep.get('n_clusters')} clusters)")
    empty = [c for c, v in rep["per_class"].items() if v["n_test"] == 0]
    if empty:
        raise ValueError(f"C2 grouped split: no test images for class(es) {empty}")


def prepare_leakage_data(job, all_df, seed=SEED):
    if all_df is None:
        raise ValueError("leakage jobs need all_images_before_dedup.csv")
    pool = cap_pool(build_pool(all_df), seed=seed)
    if job.kind == "leak_naive":
        pool["split"] = split_naive(len(pool), seed)
        report = split_report(pool["split"].values, pool["label"].values)
    else:
        clusters = duplicate_clusters(pool)
        pool["split"], n_over = split_grouped(clusters, pool["label"].values, seed)
        report = split_report(pool["split"].values, pool["label"].values, clusters, n_over)
        check_grouped_split(report)
    pool["tier"] = np.where(pool.split == "test", "pool_test", "")
    space = class_space(CLASSES)
    pool = add_label_columns(pool, space)
    tr, va, te = (pool[pool.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    meta = dict(held_out_class=None, held_out_source=None, unseen_classes=[],
                counts=dict(train=len(tr), val=len(va), test=len(te), unknown=0),
                pool_size=len(pool), split_report=report, test_fraction_actual=len(te) / len(pool),
                val_fraction_actual=len(va) / len(pool),
                pool_per_class=pool.label.value_counts().to_dict(),
                pool_per_source=pool.source.value_counts().to_dict())
    return JobData(space, tr, va, te, None, meta)


def leakage_block(train, test, p_test, thr=C_PHASH_THRESHOLD):
    """Share of test images with a training image within pHash distance <= thr, and class accuracy with/without."""
    near = has_near_neighbour(phash_array(test.phash.values), phash_array(train.phash.values), thr)
    ok = p_test.argmax(1) == test.y.values
    return dict(phash_threshold=thr, n_test=int(len(test)), test_with_near_duplicate_in_train=float(near.mean()),
                n_with_duplicate=int(near.sum()),
                acc_items_with_duplicate=float(ok[near].mean()) if near.any() else None,
                acc_items_without_duplicate=float(ok[~near].mean()) if (~near).any() else None)


# --------------------------------------------------------------------------------------
# Metrics (cell_16.py / cell_17.py / cell_22b.py logic)
# --------------------------------------------------------------------------------------
def fit_temperature(logits, y):
    """cell_16.py lines 6-9: temperature minimising validation NLL, searched on log T in [-3, 3]."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)

    def nll(log_t):
        return -log_softmax(logits / np.exp(log_t), axis=1)[np.arange(len(y)), y].mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def ece(probs, y, bins=ECE_BINS):
    """cell_16.py lines 11-18."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def stream_probs(p, smat):
    return np.asarray(p, dtype=np.float64) @ smat


def risk_coverage(conf, correct):
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    k = np.arange(1, len(c) + 1)
    return k / len(c), np.cumsum(c) / k, conf[order]


def pick_threshold(conf, correct, target):
    """cell_16.py lines 29-35: lowest confidence threshold whose answered items still reach the target accuracy."""
    cov, acc, th = risk_coverage(conf, correct)
    ok = np.nonzero(acc >= target)[0]
    if not len(ok):
        return float(th[max(0, int(0.8 * len(th)) - 1)]), False
    return float(th[ok.max()]), True


def selective(conf, correct, t):
    m = conf >= t
    return dict(coverage=float(m.mean()),
                accuracy_answered=float(correct[m].mean()) if m.any() else None)


def macro_f1(y_true, y_pred):
    """Macro F1 over the classes present in y_true (sklearn f1_score(average='macro', labels=unique(y), zero_division=0))."""
    if len(y_true) == 0:
        return float("nan")
    f1s = []
    for c in np.unique(y_true):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1s))


def _thr_key(t):
    return f"{t:.2f}"


def metrics(p, y, space, stream_true):
    """Class and stream metrics for probabilities p (n x K). y may hold -1 (class not in label space): those items
    count for stream metrics only. Keys follow cell_16.py / cell_22b.py."""
    p = np.asarray(p, dtype=np.float64)
    y, stream_true = np.asarray(y), np.asarray(stream_true)
    pred = p.argmax(1)
    sp = stream_probs(p, space.smat)
    s_pred = sp.argmax(1)
    s_top = space.stream_idx[pred]
    valid = y >= 0
    out = dict(n=int(len(y)), n_class_scored=int(valid.sum()))
    out["class_acc"] = float((pred[valid] == y[valid]).mean()) if valid.any() else None
    out["macro_f1"] = macro_f1(y[valid], pred[valid]) if valid.any() else None
    out["stream_acc"] = float((s_pred == stream_true).mean()) if len(y) else None
    out["stream_acc_top_class"] = float((s_top == stream_true).mean()) if len(y) else None
    for k, name in enumerate(STREAMS):
        m = stream_true == k
        out[f"recall_{name}"] = float((s_pred[m] == k).mean()) if m.any() else None
        out[f"n_{name}"] = int(m.sum())
    out["always_dry_baseline"] = float((stream_true == DRY).mean()) if len(y) else None
    is_sc = stream_true == SC
    for r in SENSITIVITY_COST_RATIOS:
        t = 1 / (1 + r)
        flag = sp[:, SC] >= t
        out[f"sc_recall@{_thr_key(t)}"] = float(flag[is_sc].mean()) if is_sc.any() else None
        out[f"sc_false_alarm@{_thr_key(t)}"] = float(flag[~is_sc].mean()) if (~is_sc).any() else None
    out["sc_recall_argmax"] = float((s_pred[is_sc] == SC).mean()) if is_sc.any() else None
    out["sc_false_alarm_argmax"] = float((s_pred[~is_sc] == SC).mean()) if (~is_sc).any() else None
    out["sc_recall_top_class"] = float((s_top[is_sc] == SC).mean()) if is_sc.any() else None
    out["sc_false_alarm_top_class"] = float((s_top[~is_sc] == SC).mean()) if (~is_sc).any() else None
    return out


def ood_scores(logits, T):
    """cell_22b.py lines 157-158. Higher = more familiar for all three."""
    logits = np.asarray(logits, dtype=np.float64)
    return dict(MSP=softmax(logits / T, 1).max(1), MaxLogit=logits.max(1), Energy=T * logsumexp(logits / T, axis=1))


def fit_thresholds(val_logits, val_y, val_stream, T, space):
    """All thresholds come from validation data only."""
    val_logits = np.asarray(val_logits, dtype=np.float64)
    p = softmax(val_logits / T, 1)
    sp = stream_probs(p, space.smat)
    ns, ok_s = pick_threshold(sp.max(1), sp.argmax(1) == val_stream, TARGET_STREAM_ACC)
    nc, ok_c = pick_threshold(p.max(1), p.argmax(1) == val_y, TARGET_CLASS_ACC)
    reject = {k: float(np.quantile(v, REJECT_VAL_QUANTILE)) for k, v in ood_scores(val_logits, T).items()}
    return dict(hazard=SC_THRESHOLD, hazard_cost_ratio=COST_RATIO, reject_msp=reject["MSP"], reject=reject,
                reject_val_quantile=REJECT_VAL_QUANTILE, not_sure_stream=ns, not_sure_class=nc,
                not_sure_targets_met_on_val=dict(stream=ok_s, cls=ok_c))


def evaluate_block(logits, y, stream_true, T, space, th):
    """Metrics for one evaluation set at temperature T, with ECE before/after scaling and reject coverage/accuracy."""
    logits = np.asarray(logits, dtype=np.float64)
    y, stream_true = np.asarray(y), np.asarray(stream_true)
    p_raw, p = softmax(logits, 1), softmax(logits / T, 1)
    out = metrics(p, y, space, stream_true)
    valid = y >= 0
    out["ece_before"] = ece(p_raw[valid], y[valid]) if valid.any() else None
    out["ece_after"] = ece(p[valid], y[valid]) if valid.any() else None
    sp = stream_probs(p, space.smat)
    out["reject_stream"] = selective(sp.max(1), sp.argmax(1) == stream_true, th["not_sure_stream"]) if len(y) else None
    out["reject_class"] = (selective(p.max(1)[valid], (p.argmax(1) == y)[valid], th["not_sure_class"])
                           if valid.any() else None)
    return out


def unknown_block(logits, T, space, th, known_logits=None):
    """What happens to items the model has no label for (cell_22b.py lines 174-193): share rejected (score below
    the validation-fitted threshold), flagged as Special Care (summed stream probability >= 0.20), and silently
    routed to each stream. AUROC / FPR95 against the known test items when those logits are given."""
    logits = np.asarray(logits, dtype=np.float64)
    p = softmax(logits / T, 1)
    sp = stream_probs(p, space.smat)
    sc = ood_scores(logits, T)
    rej = {k: sc[k] < th["reject"][k] for k in sc}
    flagged = sp[:, SC] >= th["hazard"]
    top_stream = sp.argmax(1)
    silent = ~rej["MSP"] & ~flagged
    out = dict(n=int(len(logits)), reject_threshold_msp=th["reject_msp"], hazard_threshold=th["hazard"],
               rejected=float(rej["MSP"].mean()), flagged_special_care=float(flagged.mean()),
               rejected_or_flagged=float((rej["MSP"] | flagged).mean()))
    for k in sc:
        out[f"rejected_{k}"] = float(rej[k].mean())
    for r in SENSITIVITY_COST_RATIOS:
        out[f"flagged@{_thr_key(1 / (1 + r))}"] = float((sp[:, SC] >= 1 / (1 + r)).mean())
    for k, name in enumerate(STREAMS):
        out[f"silently_routed_{name}"] = float((silent & (top_stream == k)).mean())
        out[f"top_stream_{name}"] = float((top_stream == k).mean())
    names = np.array(space.names)
    out["most_common_predicted_class"] = {str(k): int(v) for k, v in
                                          pd.Series(names[p.argmax(1)]).value_counts().head(3).items()}
    if known_logits is not None and len(known_logits):
        from sklearn.metrics import roc_auc_score
        ks = ood_scores(known_logits, T)
        out["known_rejected"] = float((ks["MSP"] < th["reject"]["MSP"]).mean())
        for k in sc:
            yb = np.r_[np.ones(len(ks[k])), np.zeros(len(sc[k]))]
            out[f"auroc_{k}"] = float(roc_auc_score(yb, np.r_[ks[k], sc[k]]))
            t95 = np.quantile(ks[k], 0.05)               # 95% of known items score at or above this
            out[f"fpr95_{k}"] = float((sc[k] >= t95).mean())
    return out


def compute_results(space, val, test, unknown=None, unknown_is_labelled=False):
    """val / test / unknown: dicts with arrays logits, y, stream_true (and tier for test).
    Returns temperature, thresholds and every metric block. Temperature and thresholds use val only."""
    T = fit_temperature(val["logits"], val["y"])
    th = fit_thresholds(val["logits"], val["y"], val["stream_true"], T, space)
    out = dict(temperature=T, thresholds=th, metrics={})
    out["metrics"]["val"] = evaluate_block(val["logits"], val["y"], val["stream_true"], T, space, th)
    tiers = np.asarray(test["tier"])
    for t in sorted(set(tiers.tolist())):
        m = tiers == t
        out["metrics"][f"test_{t}"] = evaluate_block(test["logits"][m], test["y"][m], test["stream_true"][m], T, space, th)
    if len(tiers):
        out["metrics"]["test_all"] = evaluate_block(test["logits"], test["y"], test["stream_true"], T, space, th)
    out["unknown_T1"] = out["unknown_all_tiers"] = None
    if unknown is not None and len(unknown["logits"]):
        ktier = np.asarray(test["tier"]) == "T1"
        known_t1 = test["logits"][ktier] if ktier.any() else None
        out["unknown_all_tiers"] = unknown_block(unknown["logits"], T, space, th, test["logits"])
        # default view: T1 items only, known and unknown. A left-out source keeps all its rows (known set = T1 test).
        utier = np.ones(len(unknown["logits"]), bool) if unknown_is_labelled else np.asarray(unknown["tier"]) == "T1"
        if utier.any():
            out["unknown_T1"] = unknown_block(unknown["logits"][utier], T, space, th, known_t1)
        if unknown_is_labelled:
            out["left_out_source_metrics"] = evaluate_block(unknown["logits"], unknown["y"], unknown["stream_true"],
                                                            T, space, th)
            vanished = np.asarray(unknown["y"]) < 0
            if vanished.any():
                out["unseen_class_images"] = unknown_block(unknown["logits"][vanished], T, space, th, known_t1)
    return out


# --------------------------------------------------------------------------------------
# Result files
# --------------------------------------------------------------------------------------
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def write_npz_atomic(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def pack_predictions(parts, T, space):
    """Concatenate per-role prediction sets into the arrays saved in <job>_logits.npz.
    parts: {role: (frame, logits)}. Adds MSP / MaxLogit / Energy at the fitted temperature for every image."""
    cols = {k: [] for k in ("ids", "role", "split", "tier", "source", "group", "label_idx", "class_idx", "stream_true")}
    logits = []
    for role, (fr, lg) in parts.items():
        if fr is None or len(fr) == 0:
            continue
        cols["ids"] += fr["id"].astype(str).tolist()
        cols["role"] += [role] * len(fr)
        cols["split"] += fr["split"].astype(str).tolist()
        cols["tier"] += fr["tier"].astype(str).tolist()
        cols["source"] += fr["source"].astype(str).tolist()
        cols["group"] += fr["group"].astype(str).tolist()
        cols["label_idx"] += fr["y"].tolist()
        cols["class_idx"] += fr["class_idx"].tolist()
        cols["stream_true"] += fr["stream_true"].tolist()
        logits.append(np.asarray(lg, dtype=np.float32))
    logits = np.concatenate(logits)
    sc = ood_scores(logits, T)
    arrays = {k: np.array(v) for k, v in cols.items()}
    arrays.update(logits=logits, msp=sc["MSP"].astype(np.float32), maxlogit=sc["MaxLogit"].astype(np.float32),
                  energy=sc["Energy"].astype(np.float32), temperature=np.float64(T),
                  label_names=np.array(space.names))
    return arrays


# --------------------------------------------------------------------------------------
# Training state machine (EarlyStopping + ReduceLROnPlateau + keep-best of cell_14.py, with resume)
# --------------------------------------------------------------------------------------
PHASES = [dict(name="head", epochs=EPOCHS_HEAD, patience=PATIENCE_HEAD, lr=LR_HEAD),
          dict(name="finetune", epochs=EPOCHS_FINETUNE, patience=PATIENCE_FINETUNE, lr=LR_FINETUNE)]


class PhaseTracker:
    """Per-phase callbacks of cell_14.py lines 20-26, as plain state so it can be saved after every epoch:
    EarlyStopping(val_loss, patience), ReduceLROnPlateau(factor 0.3, patience max(1, patience//2), min_lr 1e-7,
    Keras default min_delta 1e-4) and KeepBest (strictly lower val_loss)."""

    def __init__(self, patience, lr):
        self.es_patience = patience
        self.rl_patience = max(1, patience // 2)
        self.lr = lr
        self.es_best = self.rl_best = self.keep_best = math.inf
        self.es_wait = self.rl_wait = 0
        self.stop = False

    def update(self, val_loss):
        """Feed one epoch's validation loss. Returns dict(improved=True when this is the best epoch so far)."""
        v = float(val_loss)
        self.es_wait += 1
        if v < self.es_best:
            self.es_best, self.es_wait = v, 0
        if self.es_wait >= self.es_patience:
            self.stop = True
        if v < self.rl_best - 1e-4:
            self.rl_best, self.rl_wait = v, 0
        else:
            self.rl_wait += 1
            if self.rl_wait >= self.rl_patience:
                if self.lr > PLATEAU_MIN_LR + 1e-12:
                    self.lr = max(self.lr * PLATEAU_FACTOR, PLATEAU_MIN_LR)
                self.rl_wait = 0
        improved = v < self.keep_best
        if improved:
            self.keep_best = v
        return dict(improved=improved)

    def to_dict(self):
        return {k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        t = cls(d["es_patience"], d["lr"])
        for k, v in d.items():
            setattr(t, k, math.inf if v is None and k in ("es_best", "rl_best", "keep_best") else v)
        return t


def new_state():
    return dict(phase_idx=0, phase_epoch=0, epochs_done=0, ckpt_no=0, ckpts=[], best_file=None, best_entry=None,
                tracker=None, history=[], train_seconds=0.0, done=False, final_file=None, resumes=[])


def load_state(run_dir):
    p = os.path.join(run_dir, "state.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def save_state(run_dir, state):
    write_json_atomic(os.path.join(run_dir, "state.json"), state)


def _entry(state, file, with_best=False):
    """A resume point: the weights file and the counters that go with it."""
    e = dict(file=file, phase_idx=state["phase_idx"], phase_epoch=state["phase_epoch"],
             epochs_done=state["epochs_done"], tracker=json.loads(json.dumps(state["tracker"])),
             best_file=state["best_file"], history_len=len(state["history"]))
    if with_best:
        e["best_entry"] = json.loads(json.dumps(state["best_entry"]))
    return e


def _prune_checkpoints(run_dir, state):
    keep = {state.get("final_file"), state.get("best_file")}
    for e in state["ckpts"]:
        keep.update({e["file"], e["best_file"]})
    for pat in ("ckpt_*.weights.h5", "best_*.weights.h5", "tmp_*.weights.h5"):
        for f in glob.glob(os.path.join(run_dir, pat)):
            if os.path.basename(f) not in keep:
                try:
                    os.remove(f)
                except OSError:
                    pass


def _wipe_run_dir(run_dir):
    for pat in ("ckpt_*.weights.h5", "best_*.weights.h5", "tmp_*.weights.h5", "final.weights.h5", "state.json"):
        for f in glob.glob(os.path.join(run_dir, pat)):
            os.remove(f)


def _usable(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def _restore_from_state(trainer, run_dir, state, log):
    """Load the newest usable checkpoint: latest, else previous, else the best one of the current phase.
    Rewinds the counters to match the file that was loaded. Returns a label, or None when nothing works."""
    ck = state["ckpts"]
    tries = []
    if ck:
        tries.append(("latest checkpoint", ck[-1], ck, ck[-1].get("best_entry")))
    if len(ck) > 1:
        tries.append(("previous checkpoint", ck[-2], ck[:-1], ck[-2].get("best_entry")))
    if state.get("best_entry"):
        tries.append(("best checkpoint of the current phase", state["best_entry"], [], state["best_entry"]))
    for label, e, keep, best_entry in tries:
        path = os.path.join(run_dir, e["file"])
        if not _usable(path):
            log(f"  WARNING: {label} {e['file']} is missing or empty")
            continue
        try:
            trainer.load_weights(path)
        except Exception as ex:  # noqa: BLE001
            log(f"  WARNING: {label} {e['file']} could not be read ({ex!r})")
            continue
        state.update(phase_idx=e["phase_idx"], phase_epoch=e["phase_epoch"], epochs_done=e["epochs_done"],
                     tracker=e["tracker"], best_file=e["best_file"], ckpts=list(keep), best_entry=best_entry)
        del state["history"][e["history_len"]:]
        return label
    return None


def run_training(run_dir, trainer, phases=PHASES, log=print):
    """Resumable two-phase training. After every epoch: weights to ckpt_NNN.weights.h5 (and best_NNN when the
    validation loss improved), written under a temporary name, renamed and checked for non-zero size; then
    state.json (written last). A rerun continues from state.json; if the checkpoint it names is missing or
    unreadable it falls back to the previous checkpoint, then to the best checkpoint of that phase, then restarts
    the job from scratch, and says which. At the end of each phase the best epoch of that phase is restored (as
    KeepBest does in cell_14.py). Every resume is recorded in state['resumes'].
    trainer needs: build(resumed, epochs_done), load_weights(path), save_weights(path), start_phase(name, lr),
    run_epoch(phase, epoch_index, lr) -> dict with val_loss (plus loss, acc, val_acc)."""
    os.makedirs(run_dir, exist_ok=True)
    state = load_state(run_dir) or new_state()
    has_progress = bool(state["ckpts"] or state["final_file"])

    def restart_from_scratch(why):
        log(f"  {why}: restarting the job from scratch")
        resumes = state["resumes"]
        _wipe_run_dir(run_dir)
        state.clear()
        state.update(new_state())
        state["resumes"] = resumes
        trainer.build(resumed=False, epochs_done=0)

    if not has_progress:
        trainer.build(resumed=False, epochs_done=0)
    else:
        before = state["epochs_done"]
        trainer.build(resumed=True, epochs_done=before)
        if state["done"]:
            path = os.path.join(run_dir, state["final_file"] or "")
            ok = _usable(path)
            if ok:
                try:
                    trainer.load_weights(path)
                except Exception as ex:  # noqa: BLE001
                    log(f"  WARNING: final weights unreadable ({ex!r})")
                    ok = False
            if ok:
                log("  training already finished: using final weights")
                state["resumes"].append(dict(epoch=state["epochs_done"], epochs_in_state_file=before, action="final weights"))
                save_state(run_dir, state)
                return state
            restart_from_scratch("final weights missing or unreadable")
            how = "scratch"
        else:
            how = _restore_from_state(trainer, run_dir, state, log)
            if how is None:
                restart_from_scratch("no usable checkpoint")
                how = "scratch"
            else:
                log(f"  resuming from {how}: phase {state['phase_idx']}, epoch {state['epochs_done']} "
                    f"(state file said {before})")
        state["resumes"].append(dict(epoch=state["epochs_done"], epochs_in_state_file=before, action=how))
        save_state(run_dir, state)

    def write_weights(kind):
        state["ckpt_no"] += 1
        name = f"{kind}_{state['ckpt_no']:03d}.weights.h5"
        tmp = os.path.join(run_dir, "tmp_" + name)              # Keras needs the *.weights.h5 ending
        trainer.save_weights(tmp)
        if not _usable(tmp):
            raise IOError(f"checkpoint {tmp} was not written (missing or empty)")
        os.replace(tmp, os.path.join(run_dir, name))
        if not _usable(os.path.join(run_dir, name)):
            raise IOError(f"checkpoint {name} missing or empty after rename")
        return name

    for pi, ph in enumerate(phases):
        if pi < state["phase_idx"]:
            continue
        tr = PhaseTracker.from_dict(state["tracker"]) if state["tracker"] else PhaseTracker(ph["patience"], ph["lr"])
        trainer.start_phase(ph["name"], tr.lr)
        while state["phase_epoch"] < ph["epochs"] and not tr.stop:
            t0 = time.time()
            lr_used = tr.lr
            logs = trainer.run_epoch(ph["name"], state["epochs_done"], lr_used)
            upd = tr.update(logs["val_loss"])
            state["epochs_done"] += 1
            state["phase_epoch"] += 1
            state["tracker"] = tr.to_dict()
            state["history"].append(dict(phase=ph["name"], epoch=state["epochs_done"], lr=lr_used,
                                         **{k: float(v) for k, v in logs.items()}))
            if upd["improved"]:
                state["best_file"] = write_weights("best")
                state["best_entry"] = _entry(state, state["best_file"])
            name = write_weights("ckpt")
            state["ckpts"] = (state["ckpts"] + [_entry(state, name, with_best=True)])[-2:]
            state["train_seconds"] += time.time() - t0
            save_state(run_dir, state)
            _prune_checkpoints(run_dir, state)
            log(f"  [{ph['name']}] epoch {state['epochs_done']}: val_loss {float(logs['val_loss']):.4f} "
                f"lr {lr_used:g}{' (best)' if upd['improved'] else ''}{' -> early stop' if tr.stop else ''}")
        if state["best_file"]:
            trainer.load_weights(os.path.join(run_dir, state["best_file"]))   # KeepBest.on_train_end
        state.update(phase_idx=pi + 1, phase_epoch=0, tracker=None, best_file=None, best_entry=None)
        name = write_weights("ckpt")
        state["ckpts"] = (state["ckpts"] + [_entry(state, name, with_best=True)])[-2:]
        save_state(run_dir, state)
        _prune_checkpoints(run_dir, state)
    trainer.save_weights(os.path.join(run_dir, "final.weights.h5"))
    if not _usable(os.path.join(run_dir, "final.weights.h5")):
        raise IOError("final.weights.h5 was not written")
    state.update(final_file="final.weights.h5", done=True, ckpts=[], best_file=None, best_entry=None)
    save_state(run_dir, state)
    _prune_checkpoints(run_dir, state)
    return state


def shuffle_class_rows(frame, seed=SEED):
    """Shuffle one class's rows once with a fixed seed. Rows arrive sorted by source; without this, a streaming
    shuffle buffer smaller than the class under-draws the sources at the end."""
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def training_streams(train, seed=SEED):
    """One stream per 12-class class, with its rows pre-shuffled. Batches are balanced over these classes in every
    job; the label attached to a stream is the job's label (class index, or stream index for S1-S4)."""
    out = []
    for c in sorted(train["class_idx"].unique()):
        f = train[train["class_idx"] == c]
        ys = f["y"].unique()
        assert len(ys) == 1, f"class {c} has several labels"
        out.append(dict(class_idx=int(c), y=int(ys[0]), frame=shuffle_class_rows(f, seed)))
    return out


# --------------------------------------------------------------------------------------
# TensorFlow / Keras part (imported lazily)
# --------------------------------------------------------------------------------------
_TF = {}


def _tf():
    """Import TensorFlow once and define the Keras layer (cell_13.py lines 2-29)."""
    if _TF:
        return _TF
    import tensorflow as tf
    import keras
    from keras import layers

    @keras.saving.register_keras_serializable(package="WasteVision")
    class ColorJitter(layers.Layer):
        """Per-image random brightness, contrast, saturation and hue (training only)."""

        def __init__(self, brightness=0.25, contrast=0.25, saturation=0.35, hue=0.03, **kwargs):
            super().__init__(**kwargs)
            self.brightness, self.contrast, self.saturation, self.hue = brightness, contrast, saturation, hue

        def call(self, x, training=None):
            if not training:
                return x
            b = tf.shape(x)[0]
            x = tf.clip_by_value(x / 255.0, 0.0, 1.0)
            h, s, v = tf.unstack(tf.image.rgb_to_hsv(x), axis=-1)
            h = tf.math.floormod(h + tf.random.uniform([b, 1, 1], -self.hue, self.hue), 1.0)
            s = tf.clip_by_value(s * tf.random.uniform([b, 1, 1], 1 - self.saturation, 1 + self.saturation), 0.0, 1.0)
            v = tf.clip_by_value(v * tf.random.uniform([b, 1, 1], 1 - self.brightness, 1 + self.brightness), 0.0, 1.0)
            x = tf.image.hsv_to_rgb(tf.stack([h, s, v], axis=-1))
            mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
            x = (x - mean) * tf.random.uniform([b, 1, 1, 1], 1 - self.contrast, 1 + self.contrast) + mean
            return tf.clip_by_value(x, 0.0, 1.0) * 255.0

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            return {**super().get_config(), "brightness": self.brightness, "contrast": self.contrast,
                    "saturation": self.saturation, "hue": self.hue}

    _TF.update(tf=tf, keras=keras, layers=layers, ColorJitter=ColorJitter)
    return _TF


def make_augmenter():
    """cell_13.py lines 31-38."""
    t = _tf()
    keras, layers = t["keras"], t["layers"]
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="aug_flip"),
        layers.RandomRotation(0.08, fill_mode="reflect", name="aug_rotate"),
        layers.RandomZoom((-0.15, 0.15), fill_mode="reflect", name="aug_zoom"),
        layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", name="aug_translate"),
        t["ColorJitter"](name="aug_color"),
    ], name="augment")


def make_head(feature_shape, n_classes):
    """cell_13.py lines 51-58, with the output size set by the label space."""
    keras, layers = _tf()["keras"], _tf()["layers"]
    inp = keras.Input(feature_shape, name="features")
    x = layers.GlobalAveragePooling2D(name="head_gap")(inp)
    x = layers.Dropout(0.2, name="head_drop1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.3, name="head_drop2")(x)
    out = layers.Dense(n_classes, name="logits")(x)   # logits; softmax (with temperature) applied at inference
    return keras.Model(inp, out, name="head")


def build_model(n_classes, weights=None):
    """cell_13.py lines 60-89 (efficientnetv2b0 branch). Returns (training model with augmentation,
    inference model, backbone); both models share the same preprocessing, backbone and head layers."""
    keras, layers = _tf()["keras"], _tf()["layers"]
    shape = (IMG_SIZE, IMG_SIZE, 3)
    preproc = layers.Identity(name="preproc")          # EfficientNetV2 scales 0-255 input internally
    base = keras.applications.EfficientNetV2B0(input_shape=shape, include_top=False, weights=weights,
                                               include_preprocessing=True)
    backbone = keras.Model(base.input, base.output, name="backbone")
    head = make_head(backbone.output_shape[1:], n_classes)

    def forward(x):
        x = preproc(x)
        x = backbone(x, training=False)                 # batch-norm stays in inference mode
        return head(x)

    inp = keras.Input(shape, name="image")
    train_model = keras.Model(inp, forward(make_augmenter()(inp)), name="efficientnetv2b0_train")
    inp2 = keras.Input(shape, name="image")
    infer_model = keras.Model(inp2, forward(inp2), name="efficientnetv2b0")
    return train_model, infer_model, backbone


def unfreeze_top(backbone, n):
    """cell_13.py lines 91-97."""
    layers = _tf()["layers"]
    backbone.trainable = True
    for l in backbone.layers[:-n]:
        l.trainable = False
    for l in backbone.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False


def compile_model(m, lr):
    """cell_14.py lines 15-18."""
    keras = _tf()["keras"]
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")], jit_compile=False)


def load_image(path):
    """cell_12.py lines 25-29: the single preprocessing used everywhere."""
    tf = _tf()["tf"]
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE, antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def _ids_key(ids):
    return hashlib.md5("".join(map(str, ids)).encode()).hexdigest()[:10]


def cached_images(paths, ids, cache_dir):
    """Image-only dataset cached to local disk (labels are attached later, so the cache is reusable across jobs).
    The first call for a set of ids decodes every image once and writes a '.warm' marker; a cache without the
    marker (interrupted write) is deleted and rebuilt."""
    tf = _tf()["tf"]
    os.makedirs(cache_dir, exist_ok=True)
    prefix = os.path.join(cache_dir, f"img_{_ids_key(ids)}")
    ds = tf.data.Dataset.from_tensor_slices(np.asarray(paths, dtype=str)).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE).cache(prefix)
    if not os.path.exists(prefix + ".warm"):
        for f in glob.glob(prefix + "*"):
            os.remove(f)
        for _ in ds.batch(512):
            pass
        open(prefix + ".warm", "w").close()
    return ds


class KerasTrainer:
    """The TensorFlow side of run_training (same trainer interface as the tests' fake)."""

    def __init__(self, data, seed, cache_dir):
        self.data, self.seed, self.cache_dir = data, seed, cache_dir
        self.n_classes = len(data.space.names)
        self.steps = math.ceil(len(data.train) / BATCH_SIZE)            # cell_12.py line 61
        self.val_ds = None
        self.class_imgs = []

    def _prepare_data(self):
        tf = _tf()["tf"]
        for f in glob.glob(os.path.join(self.cache_dir, "*.lockfile")):
            os.remove(f)
        self.class_imgs = []
        for st in training_streams(self.data.train):           # rows shuffled once, fixed seed 42
            f = st["frame"]
            self.class_imgs.append((cached_images(f.path.values, f.id.values, self.cache_dir), len(f), st["y"]))
        v = self.data.val
        imgs = cached_images(v.path.values, v.id.values, self.cache_dir)
        labels = tf.data.Dataset.from_tensor_slices(v.y.values.astype("int32"))
        self.val_ds = (tf.data.Dataset.zip((imgs, labels)).batch(BATCH_SIZE)
                       .map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
                       .prefetch(tf.data.AUTOTUNE))

    def build(self, resumed, epochs_done=0):
        t = _tf()
        keras = t["keras"]
        keras.backend.clear_session()                                    # free memory left by an earlier model
        gc.collect()
        keras.utils.set_random_seed(self.seed + (epochs_done if resumed else 0))
        weights = None if resumed else PRETRAINED_WEIGHTS               # checkpoint weights replace ImageNet on resume
        self.train_m, self.infer_m, self.backbone = build_model(self.n_classes, weights=weights)
        self._prepare_data()

    def _train_ds(self, epoch_seed):
        """Balanced batches (cell_12.py lines 43-56): every class is its own shuffled, repeating stream and every
        class gets an equal share of each batch (also for the 4-stream jobs, whose labels are the streams). The
        shuffle buffer is the whole class (images are cached). Rebuilt each epoch with a new shuffle seed so a
        fresh iterator does not replay the same order."""
        tf = _tf()["tf"]
        streams = []
        for imgs, n, y in self.class_imgs:
            def attach(x, _y=y):
                return x, tf.constant(_y, tf.int32)
            streams.append(imgs.shuffle(n, seed=epoch_seed).repeat().map(attach))
        ds = tf.data.Dataset.sample_from_datasets(streams, weights=[1.0 / len(streams)] * len(streams), seed=epoch_seed)
        ds = ds.batch(BATCH_SIZE).map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    def start_phase(self, name, lr):
        if name == "head":
            self.backbone.trainable = False
        else:
            unfreeze_top(self.backbone, UNFREEZE_LAST_LAYERS)
        compile_model(self.train_m, lr)

    def run_epoch(self, phase, epoch_index, lr):
        self.train_m.optimizer.learning_rate = lr
        h = self.train_m.fit(self._train_ds(SEED + epoch_index), validation_data=self.val_ds,
                             epochs=epoch_index + 1, initial_epoch=epoch_index, steps_per_epoch=self.steps, verbose=2)
        return {k: float(v[-1]) for k, v in h.history.items()}

    def save_weights(self, path):
        self.train_m.save_weights(path)                 # run_training writes to a tmp name and renames

    def load_weights(self, path):
        self.train_m.load_weights(path)

    def predict(self, frame, cache=False):
        tf = _tf()["tf"]
        if cache:
            imgs = cached_images(frame.path.values, frame.id.values, self.cache_dir)
        else:
            imgs = tf.data.Dataset.from_tensor_slices(frame.path.values.astype(str)).map(
                load_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = imgs.batch(BATCH_SIZE).map(lambda x: tf.cast(x, tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
        return self.infer_m.predict(ds.prefetch(tf.data.AUTOTUNE), verbose=0).astype(np.float32)


def require_gpu():
    tf = _tf()["tf"]
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:  # noqa: BLE001
            pass
    if not gpus and not os.environ.get("WV_ALLOW_CPU"):
        raise RuntimeError("No GPU found. Runtime > Change runtime type > T4 GPU (or set WV_ALLOW_CPU=1 to override).")
    return [tf.config.experimental.get_device_details(g).get("device_name", "GPU") for g in gpus] or "none"


# --------------------------------------------------------------------------------------
# One training job
# --------------------------------------------------------------------------------------
def _arrays(fr, logits):
    return dict(logits=np.asarray(logits, dtype=np.float64), y=fr["y"].values, stream_true=fr["stream_true"].values,
                tier=fr["tier"].values)


def cache_dir_for(job, data_root):
    """Image caches are shared by A1-A3 and B1-B4 (same images); other jobs get their own, deleted afterwards."""
    group = "main" if job.kind in ("full", "holdout_class") else job.id     # class streams are the same for S jobs
    return os.path.join(os.path.dirname(os.path.abspath(str(data_root).rstrip("/"))), "wv_cache", group)


def run_training_job(job, drive_root, data_root, splits_path, all_csv_path):
    t_start = time.time()
    print(f"=== JOB {job.id} | START | {job.what} | seed {job.seed} | {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    device = require_gpu()
    splits = load_splits(splits_path) if job.kind not in ("leak_naive", "leak_grouped") else None
    all_df = load_all_csv(all_csv_path) if job.kind in ("leak_naive", "leak_grouped") else None
    data = prepare_job_data(job, splits, all_df)
    fr = attach_paths_all(dict(train=data.train, val=data.val, test=data.test, unknown=data.unknown), data_root)
    data.train, data.val, data.test, data.unknown = fr["train"], fr["val"], fr["test"], fr["unknown"]
    print(f"  label space ({data.space.kind}, {len(data.space.names)}): {data.space.names}", flush=True)
    print(f"  unseen classes: {data.meta.get('unseen_classes')} | counts: "
          f"train {len(data.train)} val {len(data.val)} test {len(data.test)} "
          f"unknown {0 if data.unknown is None else len(data.unknown)}", flush=True)

    cache_dir = cache_dir_for(job, data_root)
    run_dir = os.path.join(drive_root, "runs", job.id)
    trainer = KerasTrainer(data, job.seed, cache_dir)
    state = run_training(run_dir, trainer)
    print(f"  training finished: {state['epochs_done']} epochs, {state['train_seconds'] / 60:.1f} min", flush=True)

    val_lg = trainer.predict(data.val, cache=True)
    test_lg = trainer.predict(data.test)
    unk_lg = trainer.predict(data.unknown) if data.unknown is not None else None
    val, test = _arrays(data.val, val_lg), _arrays(data.test, test_lg)
    unknown = _arrays(data.unknown, unk_lg) if data.unknown is not None else None
    res = compute_results(data.space, val, test, unknown, unknown_is_labelled=(job.kind == "holdout_source"))
    if job.kind in ("leak_naive", "leak_grouped"):
        res["leakage"] = leakage_block(data.train, data.test, softmax(test_lg.astype(np.float64) / res["temperature"], 1))

    parts = {"val": (data.val, val_lg), "test": (data.test, test_lg)}
    if unknown is not None:
        parts["unknown"] = (data.unknown, unk_lg)
    packed = pack_predictions(parts, res["temperature"], data.space)
    if job.kind in ("leak_naive", "leak_grouped"):
        # per-image flag for the analysis: does this test image have a training image within the pHash threshold?
        near = has_near_neighbour(phash_array(data.test.phash.values), phash_array(data.train.phash.values),
                                  C_PHASH_THRESHOLD)
        flag = np.zeros(len(packed["ids"]), dtype=bool)
        flag[packed["role"] == "test"] = near
        packed["near_dup_in_train"] = flag
    write_npz_atomic(os.path.join(drive_root, "results", f"{job.id}_logits.npz"), **packed)
    hist = state["history"]
    result = dict(job=job.id, what=job.what, kind=job.kind, seed=job.seed, label_space=data.space.kind,
                  label_names=data.space.names, hazard_threshold=SC_THRESHOLD,
                  reject_threshold_msp=res["thresholds"]["reject_msp"], held_out_class=data.meta.get("held_out_class"),
                  held_out_source=data.meta.get("held_out_source"), unseen_classes=data.meta.get("unseen_classes"),
                  meta=data.meta, settings=SETTINGS, device=device,
                  epochs_run=dict(total=state["epochs_done"],
                                  **{p["name"]: sum(1 for h in hist if h["phase"] == p["name"]) for p in PHASES}),
                  training_minutes=state["train_seconds"] / 60, history=hist,
                  resumed=bool(state["resumes"]), n_resumes=len(state["resumes"]),
                  resume_epochs=[r["epoch"] for r in state["resumes"]], resume_events=state["resumes"],
                  provenance=dict(current_hashes(job, splits_path, all_csv_path), splits_file=splits_path,
                                  all_csv_file=all_csv_path if job.kind in ("leak_naive", "leak_grouped") else None), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                  **res)
    write_json_atomic(result_path(drive_root, job.id), result)          # last: marks the job done
    if os.path.basename(cache_dir) != "main":            # job-specific image cache is not reused: free the disk
        shutil.rmtree(cache_dir, ignore_errors=True)
    m = res["metrics"].get("test_T1") or res["metrics"].get("test_all") or res["metrics"]["val"]
    print(f"=== JOB {job.id} | DONE in {(time.time() - t_start) / 60:.1f} min | T={res['temperature']:.3f} | "
          f"class_acc {m['class_acc']} | stream_acc {m['stream_acc']:.4f} ===", flush=True)
    return result


# --------------------------------------------------------------------------------------
# E1: CLIP zero-shot
# --------------------------------------------------------------------------------------
CLIP_PHRASES = {
    "biological": "food or garden waste", "paper_cardboard": "paper or cardboard", "plastic": "a plastic item",
    "metal": "a metal item such as a can", "glass": "a glass bottle or glass item", "clothes": "clothes or textiles",
    "shoes": "shoes", "trash": "mixed non-recyclable rubbish", "battery": "a battery",
    "medicines_syringes": "medicine or syringes", "e_waste": "electronic waste",
    "sanitary": "a sanitary pad or diaper"}
CLIP_CLASS_PROMPTS = [f"a photo of {CLIP_PHRASES[c]}" for c in CLASSES]
CLIP_HAZARD_PROMPTS = ["a photo of a hazardous household item such as a battery, medicine or electronic waste",
                       "a photo of ordinary non-hazardous waste"]                 # index 0 = hazard
CLIP_HAZARD_THRESHOLDS = [0.5, 1 / 3, 0.2, 0.1]
CLIP_SHARD = 2048


def clip_scores(img_feats, class_text, hazard_text, logit_scale=100.0):
    """Class logits / probabilities (12), summed stream probabilities (4) and hazard-prompt probability from
    L2-normalised image and text features (CLIP: softmax of scaled cosine similarity)."""
    img = np.asarray(img_feats, dtype=np.float64)
    class_logits = logit_scale * img @ np.asarray(class_text, dtype=np.float64).T
    class_probs = softmax(class_logits, 1)
    stream = class_probs @ class_space(CLASSES).smat
    haz = softmax(logit_scale * img @ np.asarray(hazard_text, dtype=np.float64).T, 1)[:, 0]
    return dict(class_logits=class_logits, class_probs=class_probs, stream_probs=stream, hazard_prob=haz)


def clip_summary(scores, meta):
    """Metrics for the CLIP scores on each test tier, and flag rates for every hazard class (and shoes) on the
    test images, by summed-stream flag and by the hazard prompt. meta: frame with split, tier, label."""
    space = class_space(CLASSES)
    p, haz = scores["class_probs"], scores["hazard_prob"]
    split, tier, label = meta["split"].values, meta["tier"].values, meta["label"].values
    y = meta["label"].map(CLASS_TO_IDX).values
    s_true = CLASS_STREAM_IDX[y]
    out = dict(temperature=1.0, note="zero-shot: no temperature fitted, no training on WasteVision data", metrics={})
    is_test = np.isin(split, EVAL_SPLITS)
    groups = {"val": split == "val", "test_all": is_test}
    for t in sorted(set(tier[is_test].tolist())):
        groups[f"test_{t}"] = is_test & (tier == t)
    for name, m in groups.items():
        if m.any():
            blk = metrics(p[m], y[m], space, s_true[m])
            blk["ece_before"] = ece(p[m], y[m])
            is_sc = s_true[m] == SC
            for t in CLIP_HAZARD_THRESHOLDS:
                flag = haz[m] >= t
                blk[f"hazard_prompt_recall@{t:.2f}"] = float(flag[is_sc].mean()) if is_sc.any() else None
                blk[f"hazard_prompt_false_alarm@{t:.2f}"] = float(flag[~is_sc].mean()) if (~is_sc).any() else None
            out["metrics"][name] = blk
    per = {}
    sp = scores["stream_probs"]
    for c in HAZARD_CLASSES + ["shoes"]:
        m = is_test & (label == c)
        if not m.any():
            continue
        row = dict(n=int(m.sum()), top_class_routed_special_care=float((space.stream_idx[p[m].argmax(1)] == SC).mean()),
                   top_class_is_true_class=float((p[m].argmax(1) == CLASS_TO_IDX[c]).mean()))
        for r in SENSITIVITY_COST_RATIOS:
            t = 1 / (1 + r)
            row[f"stream_sum_flagged@{_thr_key(t)}"] = float((sp[m, SC] >= t).mean())
        for t in CLIP_HAZARD_THRESHOLDS:
            row[f"hazard_prompt_flagged@{t:.2f}"] = float((haz[m] >= t).mean())
        per[c] = row
    out["per_class_flag_rates_on_test"] = per
    return out


def clip_selftest_summary(scores, df):
    """Quick sanity numbers for `--job E1 --limit N` (no result file is written)."""
    y = df["label"].map(CLASS_TO_IDX).values
    pred = scores["class_probs"].argmax(1)
    names = np.array(CLASSES)
    return dict(n=int(len(df)), top1_class_acc=float((pred == y).mean()),
                stream_acc=float((scores["stream_probs"].argmax(1) == CLASS_STREAM_IDX[y]).mean()),
                hazard_prob_mean=float(scores["hazard_prob"].mean()),
                predicted_class_counts={str(k): int(v) for k, v in pd.Series(names[pred]).value_counts().items()})


def clip_image_features(df, cache_path, shard_dir, device="cuda"):
    """Image features for every row of df (in order), cached on Drive. Shards of CLIP_SHARD images are written as
    they finish, so a restart continues where the last shard ended. Returns (features float16, model, tokenizer, device)."""
    import torch
    try:
        import open_clip
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
        import open_clip
    from PIL import Image
    dev = device if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(dev).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    ids = df["id"].astype(str).values
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if len(z["ids"]) == len(ids) and (z["ids"] == ids).all():
            print("  CLIP features: reusing cache", flush=True)
            return z["feats"], model, tokenizer, dev
    os.makedirs(shard_dir, exist_ok=True)

    class DS(torch.utils.data.Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return preprocess(Image.open(self.paths[i]).convert("RGB"))

    shards = []
    for k, s in enumerate(range(0, len(df), CLIP_SHARD)):
        part = df.iloc[s:s + CLIP_SHARD]
        f = os.path.join(shard_dir, f"shard_{k:04d}_{_ids_key(part['id'].astype(str))}.npz")
        shards.append(f)
        if os.path.exists(f):
            continue
        dl = torch.utils.data.DataLoader(DS(part["path"].values), batch_size=256, num_workers=2)
        feats = []
        with torch.no_grad():
            for x in dl:
                with torch.autocast(device_type="cuda", enabled=(dev == "cuda")):
                    e = model.encode_image(x.to(dev))
                feats.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy().astype(np.float16))
        tmp = f + ".tmp.npz"
        np.savez(tmp, feats=np.concatenate(feats))
        os.replace(tmp, f)
        print(f"  CLIP shard {k + 1}/{math.ceil(len(df) / CLIP_SHARD)} done", flush=True)
    feats = np.concatenate([np.load(f)["feats"] for f in shards])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp.npz"
    np.savez(tmp, ids=ids, feats=feats)
    os.replace(tmp, cache_path)
    return feats, model, tokenizer, dev


def run_clip_job(job, drive_root, data_root, splits_path, limit=None):
    """limit=N: self-test on the first N images; prints a summary, writes only runs/E1/selftest_N.json."""
    import torch
    t_start = time.time()
    print(f"=== JOB {job.id} | START | {job.what}{f' | SELF-TEST on first {limit} images' if limit else ''} | "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    df = load_splits(splits_path)
    if limit:
        df = df.head(limit)
    df = attach_paths_all(dict(all=df), data_root)["all"]
    if limit:
        base = os.path.join(drive_root, "runs", job.id, f"selftest_{limit}")
        cache, shards = os.path.join(base, "features.npz"), os.path.join(base, "shards")
    else:
        cache, shards = os.path.join(drive_root, "data", "clip_features_vitb32.npz"), os.path.join(drive_root, "runs", job.id, "shards")
    feats, model, tokenizer, dev = clip_image_features(df, cache, shards)
    with torch.no_grad():
        txt = model.encode_text(tokenizer(CLIP_CLASS_PROMPTS + CLIP_HAZARD_PROMPTS).to(dev))
        txt = (txt / txt.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        scale = float(model.logit_scale.exp().item())
    scores = clip_scores(feats.astype(np.float32), txt[:NUM_CLASSES], txt[NUM_CLASSES:], scale)
    if limit:
        st = clip_selftest_summary(scores, df)
        write_json_atomic(os.path.join(drive_root, "runs", job.id, f"selftest_{limit}.json"), st)
        print(f"  self-test: {st}", flush=True)
        print(f"=== JOB {job.id} | SELF-TEST DONE in {(time.time() - t_start) / 60:.1f} min (no result file written) ===", flush=True)
        return st
    summary = clip_summary(scores, df)
    y = df["label"].map(CLASS_TO_IDX).values
    write_npz_atomic(os.path.join(drive_root, "results", f"{job.id}_logits.npz"),
                     ids=np.asarray(df["id"].astype(str), dtype=str), role=np.where(df.split == "val", "val",
                                                                  np.where(df.split.isin(EVAL_SPLITS), "test", "train")),
                     split=np.asarray(df["split"].astype(str), dtype=str), tier=np.asarray(df["tier"].astype(str), dtype=str),
                     source=np.asarray(df["source"].astype(str), dtype=str), group=np.asarray(df["group"].astype(str), dtype=str),
                     label_idx=y, class_idx=y, stream_true=CLASS_STREAM_IDX[y],
                     logits=scores["class_logits"].astype(np.float32), class_probs=scores["class_probs"].astype(np.float32),
                     stream_probs=scores["stream_probs"].astype(np.float32),
                     hazard_prob=scores["hazard_prob"].astype(np.float32), label_names=np.array(CLASSES),
                     temperature=np.float64(1.0))
    result = dict(job=job.id, what=job.what, kind=job.kind, label_space="class", label_names=CLASSES,
                  settings=dict(model="ViT-B-32", pretrained="openai", logit_scale=scale, class_prompts=CLIP_CLASS_PROMPTS,
                                hazard_prompts=CLIP_HAZARD_PROMPTS, hazard_thresholds=CLIP_HAZARD_THRESHOLDS,
                                preprocessing="open_clip preprocess (resize + centre crop), not the padded 224 pipeline"),
                  n_images=len(df), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                  provenance=current_hashes(job, splits_path, ""), **summary)
    write_json_atomic(result_path(drive_root, job.id), result)
    t1 = summary["metrics"].get("test_T1", summary["metrics"].get("test_all"))
    print(f"=== JOB {job.id} | DONE in {(time.time() - t_start) / 60:.1f} min | class_acc {t1['class_acc']:.4f} "
          f"| stream_acc {t1['stream_acc']:.4f} ===", flush=True)
    return result


# --------------------------------------------------------------------------------------
# Data extraction and command line
# --------------------------------------------------------------------------------------
def ensure_extracted(tar_path, data_root, extra_tar=None, run=subprocess.run):
    """Extract normalized_images.tar into data_root once (a marker file records success). If the optional second
    archive (extra images added after the original run) exists, extract it into the same root too; its marker
    stores the archive's size and time, so a replaced archive is extracted again. Returns True if anything ran."""
    did = False
    marker = os.path.join(data_root, ".extracted_ok")
    if not os.path.exists(marker):
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"tar not found: {tar_path}")
        os.makedirs(data_root, exist_ok=True)
        print(f"Extracting {tar_path} -> {data_root} (one time) ...", flush=True)
        run(["tar", "-xf", tar_path, "-C", data_root], check=True)
        with open(marker, "w") as f:
            f.write(tar_path)
        did = True
    if extra_tar and os.path.exists(extra_tar):
        st = os.stat(extra_tar)
        sig = f"{st.st_size}:{int(st.st_mtime)}"
        m2 = os.path.join(data_root, ".extracted_extra_ok")
        if not (os.path.exists(m2) and open(m2).read() == sig):
            print(f"Extracting extra images {extra_tar} -> {data_root} ...", flush=True)
            run(["tar", "-xf", extra_tar, "-C", data_root], check=True)
            with open(m2, "w") as f:
                f.write(sig)
            did = True
    elif extra_tar:
        print(f"(no extra image archive at {extra_tar}: skipping)", flush=True)
    return did


def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0]


def run_job(job_id, args):
    job = JOBS[job_id]
    if job.kind == "clip" and args.limit:                    # self-test: never marks the job done, never skipped
        ensure_extracted(args.tar, args.data_root, args.extra_tar)
        return run_clip_job(job, args.drive_root, args.data_root, args.splits, limit=args.limit)
    status = result_status(args.drive_root, job, args.splits, args.all_csv)
    if status == "done":
        print(f"=== JOB {job_id} | SKIP (result exists, input hashes match) ===", flush=True)
        return None
    if status == "stale":
        msg = (f"result for {job_id} exists but was made from different input files "
               f"(splits / all-images CSV hash differs)")
        if not args.rerun_stale:
            raise StaleResultError(msg + "; rerun with --rerun-stale to archive it under results/stale/ and rerun")
        print(f"WARNING: {msg}; archiving to results/stale/ and rerunning", flush=True)
        print("  archived:", archive_stale(args.drive_root, job_id), flush=True)
    ensure_extracted(args.tar, args.data_root, args.extra_tar)
    if job.kind == "clip":
        return run_clip_job(job, args.drive_root, args.data_root, args.splits)
    return run_training_job(job, args.drive_root, args.data_root, args.splits, args.all_csv)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="WasteVision v3 job runner")
    ap.add_argument("--job", help="one job id, e.g. A1")
    ap.add_argument("--queue", help="file with one job id per line; run them in order")
    ap.add_argument("--extract-only", action="store_true", help="only extract the image tar, then stop")
    ap.add_argument("--list", action="store_true", help="print the job table and stop")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/WasteVision_v3")
    ap.add_argument("--data-root", default="/content/data")
    ap.add_argument("--splits", default=None, help="default: <drive-root>/data/splits_v3.csv")
    ap.add_argument("--all-csv", default=None, help="default: WasteVision_final/all_images_before_dedup.csv")
    ap.add_argument("--tar", default="/content/drive/MyDrive/WasteVision_final/normalized_images.tar")
    ap.add_argument("--extra-tar", default=None, help="optional second archive; default <drive-root>/data/extra_images.tar")
    ap.add_argument("--rerun-stale", action="store_true",
                    help="rerun jobs whose stored input hash differs (old result is archived under results/stale/)")
    ap.add_argument("--limit", type=int, default=None, help="E1 only: self-test on the first N images")
    ap.add_argument("--in-process", action="store_true", help="queue: run jobs in this process (default: one subprocess each)")
    a = ap.parse_args(argv)
    if a.extra_tar is None:
        a.extra_tar = os.path.join(a.drive_root, "data", "extra_images.tar")
    if a.limit is not None and (a.limit < 1 or (a.job or "").upper() != "E1"):
        ap.error("--limit needs --job E1 and a positive number")
    if a.splits is None:
        a.splits = os.path.join(a.drive_root, "data", "splits_v3.csv")
    if a.all_csv is None:
        a.all_csv = first_existing("/content/drive/MyDrive/WasteVision_final/all_images_before_dedup.csv",
                                   os.path.join(a.drive_root, "data", "all_images_before_dedup.csv"))
    if not (a.job or a.queue or a.extract_only or a.list):
        ap.error("give --job, --queue, --extract-only or --list")
    if a.job and a.job.upper() not in JOBS:
        ap.error(f"unknown job {a.job!r}; known: {', '.join(JOBS)}")
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.list:
        for j in JOBS.values():
            print(f"{j.id:3s} {j.kind:15s} seed {j.seed:3d}  {j.what}")
        return 0
    if a.extract_only:
        print("Extracted now." if ensure_extracted(a.tar, a.data_root, a.extra_tar) else "Already extracted.")
        return 0
    if a.queue:
        ids = read_queue(a.queue)
        status = {j: result_status(a.drive_root, JOBS[j], a.splits, a.all_csv) for j in ids}
        stale = [j for j, st in status.items() if st == "stale"]
        if stale and not a.rerun_stale:
            print(f"STOP: results for {stale} were made from different input files (hash mismatch). "
                  f"Use --rerun-stale to archive them under results/stale/ and rerun.", flush=True)
            return 2
        runner = (lambda jid: run_job(jid, a)) if a.in_process else make_subprocess_runner(a)
        res = run_queue(ids, a.drive_root, runner, is_done=lambda j: status[j] == "done")
        return 1 if res["failed"] else 0
    try:
        run_job(a.job.upper(), a)
    except StaleResultError as e:
        print(f"STOP: {e}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

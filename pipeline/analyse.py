#!/usr/bin/env python3
"""WasteVision phase 3: analyses of the saved job results (PLAN.md section 8). CPU only, no training.

    python analyse.py --results-dir <drive>/results --data-dir <drive>/data --out-dir <out>
                      [--features frozen_features_effv2b0.npz] [--mix '{"Wet":0.4,"Dry":0.4,...}']

Reads results/<job>.json and results/<job>_logits.npz written by run.py, and the files written by
rebuild_splits.py. Jobs that are missing are skipped and listed. Every table is written as CSV and
every number also goes into paper_numbers.json with its interval, n and source files.

Bootstrap: grouped (photo groups resampled with replacement). Every statistic is a ratio of sums over
items, so each item's contribution is added up per group once and a (resamples x groups) matrix of
group multiplicities does the rest; no loop over resamples. Row 0 of that matrix is all ones, so row 0
of every result is the point estimate computed by exactly the same code.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.special import softmax


def _load_run():
    """Import run.py (next to this file) by path. It needs no TensorFlow to import."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
    spec = importlib.util.spec_from_file_location("wv_run", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wv_run"] = mod          # dataclasses look the module up while it executes
    spec.loader.exec_module(mod)
    return mod


RUN = _load_run()
STREAMS, SC, DRY = RUN.STREAMS, RUN.SC, RUN.DRY
CLASSES, HAZARD_CLASSES = RUN.CLASSES, RUN.HAZARD_CLASSES
A_JOBS = ["A1", "A2", "A3"]
B_CLASS_JOBS = ["B1", "B2", "B3", "B4"]
S_JOBS = ["S1", "S2", "S3", "S4"]
B_SRC_JOBS = ["B5", "B6", "B7", "B8"]
C_JOBS = ["C1", "C2"]
TIERS = ["T1", "T2", "T3"]
PI_LIST = [0.01, 0.02, 0.05]
NPZ_KEYS = ["ids", "role", "split", "tier", "source", "group", "label_idx", "class_idx", "stream_true", "logits",
            "temperature", "label_names"]
CLIP_KEYS = ["ids", "role", "split", "tier", "source", "group", "class_idx", "stream_true", "class_probs",
             "stream_probs", "hazard_prob", "label_names"]
CLIP_PROMPT_THRESHOLD = 0.5      # hazard prompt wins over the "ordinary waste" prompt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================================
# Grouped bootstrap
# ======================================================================================
class Boot:
    """Grouped bootstrap over photo groups. M[0] = ones (the data as it is); M[1:] = group multiplicities of
    n_boot resamples drawn with replacement. All statistics are ratios of weighted sums."""

    def __init__(self, groups, n_boot, seed, name=""):
        g = np.asarray(groups).astype(str)
        _, self.codes = np.unique(g, return_inverse=True)
        self.n = len(g)
        self.G = int(self.codes.max()) + 1 if self.n else 0
        self.R = int(n_boot)
        rng = np.random.default_rng([int(seed), zlib.crc32(name.encode())])
        self.M = np.ones((self.R + 1, self.G))
        if self.R > 0 and self.G > 0:
            self.M[1:] = rng.multinomial(self.G, np.full(self.G, 1.0 / self.G), size=self.R)

    def gsum(self, x):
        """Per-group sums of x (n,) or (n, d) -> (G,) or (G, d)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return np.bincount(self.codes, weights=x, minlength=self.G)
        return np.stack([np.bincount(self.codes, weights=x[:, j], minlength=self.G) for j in range(x.shape[1])], 1)

    def total(self, x):
        """Weighted sums for every resample: (R+1,) or (R+1, d)."""
        return self.M @ self.gsum(x)

    def ratio(self, num, den):
        """(values (R+1,), n) of sum(num)/sum(den); NaN where a resample has no denominator. n = den at row 0."""
        a, b = self.total(num), self.total(den)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = a / b
        v = np.where(b > 0, v, np.nan)
        return v, int(round(float(b[0])))

    def weights(self, mask):
        """(R+1, n_mask) item weights for the items where mask is True."""
        return self.M[:, self.codes[np.asarray(mask)]]


def ci(vals):
    """(point, lo, hi): row 0 is the point; 2.5 and 97.5 percentiles of the resamples (NaN resamples dropped)."""
    p = float(vals[0]) if np.isfinite(vals[0]) else None
    r = vals[1:]
    r = r[np.isfinite(r)]
    if p is None or len(r) < 2:
        return p, None, None
    lo, hi = np.percentile(r, [2.5, 97.5])
    return p, float(lo), float(hi)


def wilson(k, n, z=1.959964):
    """Wilson 95% interval for a share k/n (plain binomial, NOT grouped)."""
    if n <= 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ======================================================================================
# Loading result files
# ======================================================================================
class Res:
    def __init__(self, job, j, z, files):
        self.job, self.j, self.z, self.files = job, j, z, files


class Results:
    """Reads results/<job>.json and <job>_logits.npz. `missing`: job -> why it cannot be used (file not found or
    unreadable). `unreadable`: file name -> error, for files that exist but cannot be read (reported as errors)."""

    def __init__(self, rdir):
        self.rdir = rdir
        self.missing = {}
        self.unreadable = {}
        self.cache = {}

    def path(self, name):
        return os.path.join(self.rdir, name)

    def _miss(self, job, why):
        self.missing.setdefault(job, why)

    def _bad(self, job, fname, why):
        self.unreadable.setdefault(fname, why)
        self._miss(job, f"{fname} exists but cannot be read: {why}")

    def get(self, job, npz=True, keys=NPZ_KEYS):
        ck = (job, npz, tuple(keys))
        if ck in self.cache:
            return self.cache[ck]
        out = None
        jn, zn = f"{job}.json", f"{job}_logits.npz"
        jp, zp = self.path(jn), self.path(zn)
        if not os.path.exists(jp):
            self._miss(job, f"{jn} not found")
        else:
            j = None
            try:
                with open(jp) as f:
                    j = json.load(f)
            except Exception as e:  # noqa: BLE001
                self._bad(job, jn, repr(e))
            if j is not None and not npz:
                out = Res(job, j, None, [jp])
            elif j is not None:
                if not os.path.exists(zp):
                    self._miss(job, f"{zn} not found")
                else:
                    try:
                        zf = np.load(zp, allow_pickle=False)
                        z = {k: zf[k] for k in zf.files}
                        lack = [k for k in keys if k not in z]
                        if lack:
                            self._bad(job, zn, f"lacks arrays {lack}")
                        else:
                            out = Res(job, j, z, [jp, zp])
                    except Exception as e:  # noqa: BLE001
                        self._bad(job, zn, repr(e))
        self.cache[ck] = out
        return out


def space_for(names):
    names = [str(n) for n in names]
    return RUN.stream_space() if names == STREAMS else RUN.class_space(names)


def job_scores(z):
    """Probabilities and routing scores for every row of a training job's npz (same functions as run.py)."""
    T = float(z["temperature"])
    sp_ = space_for(z["label_names"])
    lg = z["logits"].astype(np.float64)
    p = softmax(lg / T, 1)
    sp = RUN.stream_probs(p, sp_.smat)
    ood = RUN.ood_scores(lg, T)
    return dict(space=sp_, T=T, lg=lg, p=p, p_raw=softmax(lg, 1), sp=sp, haz=sp[:, SC], s_pred=sp.argmax(1),
                top_haz=sp_.stream_idx[p.argmax(1)] == SC, msp=ood["MSP"], maxlogit=ood["MaxLogit"],
                energy=ood["Energy"])


# ======================================================================================
# Bookkeeping: report, tables, paper numbers
# ======================================================================================
def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    return v


class Report:
    def __init__(self):
        self.tables = {}      # name -> dict(rows, jobs, files, desc)
        self.skipped = []     # (item, reason)
        self.errors = []
        self.notes = []
        self.checks = []

    def skip(self, item, reason):
        self.skipped.append((item, reason))
        print(f"  SKIPPED {item}: {reason}", flush=True)


class Numbers:
    """paper_numbers.json: one entry per number, always with its source files."""

    def __init__(self):
        self.d = {}

    def add(self, key, value, lo=None, hi=None, n=None, sources=(), table="", **extra):
        if not sources:
            raise ValueError(f"paper number {key!r} has no source file")
        if key in self.d:
            raise ValueError(f"duplicate paper-number key {key!r}")
        e = dict(value=_clean(value), ci95_lo=_clean(lo), ci95_hi=_clean(hi), n=_clean(n),
                 source_files=sorted(set(sources)), table=table)
        e.update({k: _clean(v) for k, v in extra.items()})
        self.d[key] = e


class Ctx:
    """Everything an analysis needs: arguments, results, report, numbers, and values handed between analyses."""

    def __init__(self, args, L, rep, pn):
        self.args, self.L, self.rep, self.pn = args, L, rep, pn
        self.shared = {}
        self.figs = {}

    @property
    def R(self):
        return self.args.n_boot

    def boot(self, groups, name):
        return Boot(groups, self.args.n_boot, self.args.seed, name)

    def save(self, name, df, jobs, desc, id_cols=None):
        """Write <out>/<name>.csv, record it in the report and (when id_cols is given) in paper_numbers."""
        os.makedirs(self.args.out_dir, exist_ok=True)
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(self.args.out_dir, f"{name}.csv"), index=False)
        files = sorted({f for s in df.get("source_files", pd.Series([], dtype=str)) for f in str(s).split(";") if f})
        self.rep.tables[name] = dict(rows=len(df), jobs=list(jobs), files=files, desc=desc)
        if id_cols is not None and len(df):
            for _, r in df.iterrows():
                v = r.get("value")
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    continue
                key = name + "|" + "|".join(f"{c}={r[c]}" for c in id_cols)
                src = [f for f in str(r["source_files"]).split(";") if f]
                extra = {k: r[k] for k in ("sd", "target_far", "json_value") if k in df.columns}
                self.pn.add(key, v, r.get("ci_lo"), r.get("ci_hi"), r.get("n"), src, name, **extra)
        print(f"  wrote {name}.csv ({len(df)} rows)", flush=True)
        return df


def src_str(*files):
    return ";".join(os.path.normpath(f) for fl in files for f in (fl if isinstance(fl, (list, tuple)) else [fl]))


# ======================================================================================
# Metric statistics (class / stream / flag / ECE) with bootstrap
# ======================================================================================
def ece_stat(boot, p, y, valid, bins=RUN.ECE_BINS):
    """Weighted version of run.ece over the valid rows. Returns (values (R+1,), n_valid)."""
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    b = np.searchsorted(edges, conf, side="left") - 1              # bin with edges[b] < conf <= edges[b+1]
    cols = []
    for k in range(bins):
        m = (b == k) & valid
        cols += [m.astype(float), m * (pred == y), m * conf]
    S = boot.total(np.stack(cols, 1)).reshape(boot.R + 1, bins, 3)
    nv = boot.total(valid.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.abs(S[:, :, 1] - S[:, :, 2]).sum(1) / nv
    return np.where(nv > 0, v, np.nan), int(round(float(nv[0])))


def metric_stats(boot, sc, y, st, thr):
    """All class/stream/flag metrics of one evaluation set as {name: (values (R+1,), n)}.
    sc: job_scores output for these rows; y: label index (-1 = class not in label space); st: true stream.
    Names equal the keys of run.metrics / run.evaluate_block so the point row can be checked against the JSON."""
    y, st = np.asarray(y), np.asarray(st)
    space = sc["space"]
    pred = sc["p"].argmax(1)
    s_pred, s_top = sc["s_pred"], space.stream_idx[pred]
    valid = y >= 0
    ones = np.ones(len(y))
    out = {}
    out["class_acc"] = boot.ratio(valid & (pred == y), valid)
    K = len(space.names)
    oh_y = np.eye(K)[np.where(valid, y, 0)] * valid[:, None]
    oh_p = np.eye(K)[pred] * valid[:, None]
    tp, fp, fn = oh_y * oh_p, (1 - oh_y) * oh_p, oh_y * (1 - oh_p)
    TP, FP, FN = boot.total(tp), boot.total(fp), boot.total(fn)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * TP + FP + FN > 0, 2 * TP / (2 * TP + FP + FN), 0.0)
    present = (TP + FN) > 0
    nv = present.sum(1)
    mf = np.where(nv > 0, (f1 * present).sum(1) / np.maximum(nv, 1), np.nan)
    out["macro_f1"] = (mf, int(valid.sum()))
    out["stream_acc"] = boot.ratio(s_pred == st, ones)
    out["stream_acc_top_class"] = boot.ratio(s_top == st, ones)
    for k, name in enumerate(STREAMS):
        out[f"recall_{name}"] = boot.ratio((s_pred == k) & (st == k), st == k)
    out["always_dry_baseline"] = boot.ratio(st == DRY, ones)
    is_sc = st == SC
    flag = sc["haz"] >= thr
    tk = RUN._thr_key(thr)
    out[f"sc_recall@{tk}"] = boot.ratio(flag & is_sc, is_sc)
    out[f"sc_false_alarm@{tk}"] = boot.ratio(flag & ~is_sc, ~is_sc)
    out["sc_recall_top_class"] = boot.ratio((s_top == SC) & is_sc, is_sc)
    out["sc_false_alarm_top_class"] = boot.ratio((s_top == SC) & ~is_sc, ~is_sc)
    if valid.any():
        out["ece_before"] = ece_stat(boot, sc["p_raw"], y, valid)
        out["ece_after"] = ece_stat(boot, sc["p"], y, valid)
    return out


def stat_rows(stats, ident, source_files, json_block=None, only=None):
    """Long-format rows (value, interval, n) from metric_stats output; optional comparison with the run.py JSON."""
    rows = []
    for m, (vals, n) in stats.items():
        if only is not None and m not in only:
            continue
        p, lo, hi = ci(vals)
        r = dict(ident, metric=m, value=p, ci_lo=lo, ci_hi=hi, n=n)
        if json_block is not None:
            jv = json_block.get(m)
            r["json_value"] = jv
            r["abs_diff_vs_json"] = abs(p - jv) if (p is not None and jv is not None) else None
        r["source_files"] = source_files
        rows.append(r)
    return rows


def take(sc, mask):
    """Rows of a job_scores dict where mask is True."""
    n = len(sc["lg"])
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v) for k, v in sc.items()}


# ======================================================================================
# 1. Main results (A1-A3, S1)
# ======================================================================================
def main_metric_keys(thr):
    tk = RUN._thr_key(thr)
    return (["class_acc", "macro_f1", "stream_acc", "stream_acc_top_class"] + [f"recall_{s}" for s in STREAMS] +
            ["always_dry_baseline", f"sc_recall@{tk}", f"sc_false_alarm@{tk}", "sc_recall_top_class",
             "sc_false_alarm_top_class", "ece_before", "ece_after"])


def per_seed_n(blk, m):
    """Number of items behind a metric of a result-JSON block: true Special Care items for Special Care recall, the
    other items for false alarms, class-scored rows for class metrics and ECE, the stream's own count for its recall."""
    n_all, n_sc = blk.get("n"), blk.get("n_Special Care")
    if m.startswith("recall_"):
        return blk.get("n_" + m[7:])
    if m.startswith("sc_recall"):
        return n_sc
    if m.startswith("sc_false_alarm"):
        return n_all - n_sc if n_all is not None and n_sc is not None else None
    if m in ("class_acc", "macro_f1", "ece_before", "ece_after"):
        return blk.get("n_class_scored")
    return n_all


def pick_bootstrap_seed(res):
    """res: {job: Res}. Job whose kept model has the lowest validation loss (best epoch of the last phase in
    history); seed 42 when that is not recorded for every job."""
    vl, lacking = {}, []
    for jid, r in res.items():
        h = [e for e in (r.j.get("history") or []) if isinstance(e, dict) and e.get("val_loss") is not None]
        if not h:
            lacking.append(jid)
            continue
        last = h[-1].get("phase")
        vl[jid] = min(e["val_loss"] for e in h if e.get("phase") == last)
    if lacking:
        vl = None
    if vl:
        jid = min(vl, key=lambda k: vl[k])
        return jid, f"lowest validation loss of the kept model ({jid}: {vl[jid]:.4f}; all: " + \
            ", ".join(f"{k}={v:.4f}" for k, v in vl.items()) + ")"
    s42 = [jid for jid, r in res.items() if r.j.get("seed") == 42]
    return (s42[0] if s42 else sorted(res)[0]), f"seed 42 (validation loss not recorded in {lacking})"


def analyse_main(cx):
    L = cx.L
    res = {j: L.get(j, npz=False) for j in A_JOBS}
    res = {j: r for j, r in res.items() if r is not None}
    if not res:
        cx.rep.skip("main results (A1-A3)", "no A result files")
        return
    per_seed, best_note = [], ""
    for jid, r in res.items():
        thr = r.j.get("hazard_threshold", RUN.SC_THRESHOLD)
        for tier in TIERS:
            blk = (r.j.get("metrics") or {}).get(f"test_{tier}")
            if not blk:
                continue
            for m in main_metric_keys(thr):
                if blk.get(m) is None:
                    continue
                n = per_seed_n(blk, m)
                per_seed.append(dict(job=jid, seed=r.j.get("seed"), tier=tier, metric=m, value=blk[m], n=n,
                                     source_files=src_str(r.files)))
    ps = pd.DataFrame(per_seed)
    cx.save("main_per_seed", ps, list(res), "A jobs: every metric per seed on each test tier (from the result JSON)",
            ["job", "tier", "metric"])
    if len(ps):
        g = ps.groupby(["tier", "metric"], sort=False)
        ms = g.agg(value=("value", "mean"), sd=("value", lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
                   n=("seed", "nunique"), source_files=("source_files", lambda s: ";".join(sorted(set(";".join(s).split(";")))))
                   ).reset_index()
        ms["values"] = [json.dumps(list(ps[(ps.tier == t) & (ps.metric == m)].sort_values("seed").value.round(6)))
                        for t, m in zip(ms.tier, ms.metric)]
        ms["note"] = "n = number of seeds; sd uses ddof=1; T3 is a dry-only street test"
        ms["ci_lo"] = ms["ci_hi"] = np.nan
        cx.save("main_mean_sd", ms, list(res), "A jobs: mean and SD across seeds (n = number of seeds)",
                ["tier", "metric"])
    # grouped bootstrap on one seed
    full = {j: L.get(j) for j in res}
    full = {j: r for j, r in full.items() if r is not None}
    if not full:
        cx.rep.skip("main results bootstrap", "no A logits files")
        return
    bj, why = pick_bootstrap_seed(full)
    excluded = {j: L.missing.get(j, "not available") for j in A_JOBS if j not in full}
    cx.rep.notes.append(f"Main-results bootstrap uses {bj} (seed {full[bj].j.get('seed')}): {why}. Jobs considered "
                        f"(JSON and logits readable): {sorted(full)}. Excluded: {excluded or 'none'}.")
    r = full[bj]
    z, thr = r.z, r.j.get("hazard_threshold", RUN.SC_THRESHOLD)
    sc = job_scores(z)
    rows = []
    for tier in TIERS:
        m = (z["role"] == "test") & (z["tier"] == tier)
        if not m.any():
            cx.rep.notes.append(f"{bj}: no test rows on tier {tier}.")
            continue
        boot = cx.boot(z["group"][m], f"main_{tier}")
        sub = take(sc, m)
        st = metric_stats(boot, sub, z["label_idx"][m], z["stream_true"][m], thr)
        blk = (r.j.get("metrics") or {}).get(f"test_{tier}", {})
        ident = dict(job=bj, seed=r.j.get("seed"), tier=tier)
        rows += stat_rows(st, ident, src_str(r.files), blk, only=set(main_metric_keys(thr)))
        if tier == "T1":
            is_sc = z["stream_true"][m] == SC
            flag = sub["haz"] >= thr
            leak = boot.ratio(~flag & (sub["s_pred"] == DRY) & is_sc, is_sc)
            tk = RUN._thr_key(thr)
            cx.shared["A_T1"] = dict(job=bj, thr=thr, R=st[f"sc_recall@{tk}"][0], FAR=st[f"sc_false_alarm@{tk}"][0],
                                     n_sc=int(is_sc.sum()), n_non=int((~is_sc).sum()),
                                     leak_dry=leak[0], recall_stream=[st[f"recall_{s}"][0] for s in STREAMS],
                                     files=r.files)
            rows += stat_rows(dict(sc_hazard_reaches_dry_stream=leak), ident, src_str(r.files))
        del boot
    bt = pd.DataFrame(rows)
    bad = bt["abs_diff_vs_json"].dropna() if "abs_diff_vs_json" in bt else pd.Series([], dtype=float)
    if len(bad):
        cx.rep.checks.append(f"main bootstrap point values vs run.py JSON ({bj}): max abs difference {bad.max():.2e} "
                             f"over {len(bad)} numbers (logits are stored as float32)")
    cx.save("main_bootstrap", bt, [bj], f"best-seed ({bj}) metrics with grouped 95% bootstrap intervals, per tier",
            ["tier", "metric"])
    t3 = bt[(bt.tier == "T3") & (bt.metric.str.startswith("recall_")) & (bt.n > 0)]
    nz = {r_.metric: int(r_.n) for r_ in t3.itertuples()}
    cx.rep.notes.append(f"T3 (described as a dry-only street test): per-stream test counts in {bj} = {nz or 'none'}.")


def analyse_stream_model(cx):
    """S1 (full-data stream-trained network) next to the class model on T1."""
    r1 = cx.L.get("S1")
    if r1 is None:
        return
    thr = r1.j.get("hazard_threshold", RUN.SC_THRESHOLD)
    z = r1.z
    m = (z["role"] == "test") & (z["tier"] == "T1")
    if not m.any():
        cx.rep.skip("S1 table", "no T1 test rows")
        return
    sc = job_scores(z)
    sub = take(sc, m)
    boot = cx.boot(z["group"][m], "S1_T1")
    st = metric_stats(boot, sub, z["label_idx"][m], z["stream_true"][m], thr)
    keep = {"stream_acc", "always_dry_baseline", "ece_before", "ece_after"} | {k for k in st if k.startswith(("recall_", "sc_"))}
    rows = stat_rows(st, dict(job="S1", tier="T1"), src_str(r1.files), only=keep)
    cx.save("stream_model_S1", pd.DataFrame(rows), ["S1"], "S1 (4-stream network, full data) on T1", ["tier", "metric"])


# ======================================================================================
# 2. Decision designs for unseen hazards
# ======================================================================================
def sweep_to_far(sk, su, Wk, Wu, targets):
    """For each target false-alarm vector (R+1,) pick, per resample, the lowest threshold whose weighted
    false-alarm rate on the known items does not exceed the target (flag = score >= threshold, whole tie blocks
    only), then return the weighted catch rate on the unknown items. Among thresholds giving the same weighted
    false-alarm rate the highest one is used (zero-weight known items do not lower it).
    Returns a list of dicts(catch, far, thr) with (R+1,) arrays."""
    sk, su = np.asarray(sk, float), np.asarray(su, float)
    order = np.argsort(-sk, kind="stable")
    s = sk[order]
    C = np.cumsum(Wk[:, order], axis=1)
    tot = C[:, -1]
    block_end = np.r_[s[:-1] > s[1:], True]
    uo = np.argsort(-su, kind="stable")
    u = su[uo]
    CU = np.cumsum(Wu[:, uo], axis=1)
    utot = CU[:, -1]
    rows = np.arange(C.shape[0])
    outs = []
    for tgt in targets:
        tgt = np.broadcast_to(np.asarray(tgt, float), tot.shape)
        allowed = (C <= (tgt * tot)[:, None] + 1e-9) & block_end[None, :]
        anyok = allowed.any(1)
        jlast = np.where(anyok, C.shape[1] - 1 - np.argmax(allowed[:, ::-1], axis=1), 0)
        cm = C[rows, jlast]
        first = np.argmax((C >= (cm - 1e-9)[:, None]) & block_end[None, :], axis=1)
        j = np.where(anyok, first, 0)
        thr = np.where(anyok, s[j], np.inf)
        far = np.where(anyok, C[rows, j] / tot, 0.0)
        cnt = np.searchsorted(-u, -thr, side="right")                 # number of unknown items with score >= thr
        caught = np.where(cnt > 0, CU[rows, np.maximum(cnt - 1, 0)], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            catch = np.where(utot > 0, caught / utot, np.nan)
        outs.append(dict(catch=catch, far=far, thr=thr))
    return outs


def _align(z, ids):
    pos = pd.Index(z["ids"].astype(str)).get_indexer(ids)
    return pos if (pos >= 0).all() else None


BLOCK_OWN = "OWN operating point (validation-fitted where the job has one)"
BLOCK_VAL = "PRIMARY: threshold set on known validation items, reported on T1 test"
BLOCK_TEST = "SECONDARY: ROC point matched on test (comparison only, not deployable)"
VAL_OPS = [("val_match_D2", None), ("val_match_5pct", 0.05), ("val_match_10pct", 0.10)]
TEST_OPS = [("test_match_D2", None), ("test_match_5pct", 0.05), ("test_match_10pct", 0.10)]


def val_threshold(known_scores, target):
    """Threshold on a flag score (flag = score >= t) from known validation scores: the score of the k-th highest
    known item, k = floor(target * n), moved down to the end of a tie block if the k-th and (k+1)-th scores are equal.
    Returns (t, realised false-alarm rate on those validation items); t = inf and rate 0 when nothing can be flagged."""
    s = np.sort(np.asarray(known_scores, float))[::-1]
    n = len(s)
    k = int(math.floor(target * n + 1e-9))
    while 0 < k < n and s[k - 1] == s[k]:
        k -= 1
    if k <= 0:
        return math.inf, 0.0
    return float(s[k - 1]), k / n


def analyse_decisions(cx):
    L, rows = cx.L, []
    cx.shared["unseen_catch"] = {}
    cx.shared["unseen_leak"] = {}
    used = []
    s_by_class = {RUN.JOBS[j].holdout_class: j for j in S_JOBS if RUN.JOBS[j].holdout_class}
    e1 = L.get("E1", keys=CLIP_KEYS)
    for bj in B_CLASS_JOBS:
        if L.get(bj) is None:
            continue
        jrows, jused = [], []
        try:
            _decision_job(cx, bj, e1, s_by_class, jrows, jused)
        except Exception:  # noqa: BLE001 - one job's failure must not lose the others; reported, exit code 1
            tb = traceback.format_exc()
            cx.rep.errors.append(f"decision designs, job {bj}:\n{tb}")
            print(tb, flush=True)
            continue
        rows += jrows
        used += jused
    if not rows:
        cx.rep.skip("decision-design table", "no usable B1-B4 results")
        return
    df = pd.DataFrame(rows)
    cx.rep.notes.append(
        "Decision-design table, PRIMARY block (op_point val_match_*): the threshold of every sweepable design is set on the "
        "known non-hazard VALIDATION items of the same job to reach the target false-alarm rate (D2's own validation rate, "
        "5%, 10%), then fixed; catch rate on the unknown T1 items and the false-alarm rate realised on the known T1 test "
        "items are reported with grouped bootstrap intervals (thresholds fixed across resamples). SECONDARY block "
        "(op_point test_match_*): ROC point matched on the known T1 test items, comparison only, NOT deployable. "
        "op_point 'own': the design's own operating point.")
    cx.save("decision_designs", df, sorted(set(used)),
            "unseen hazards (B1-B3) and shoes control (B4): catch rate / wrongly-flagged share, false alarms, D2 minus design; "
            "primary = validation-matched thresholds",
            ["held_out_class", "design", "op_point", "measure"])
    cx.figs["decision"] = df


def _decision_job(cx, bj, e1, s_by_class, rows, used):
    """Decision-design rows for one B job (appended to rows / used)."""
    L = cx.L
    rb = L.get(bj)
    c = rb.j.get("held_out_class") or RUN.JOBS[bj].holdout_class
    z, th = rb.z, rb.j["thresholds"]
    sc = job_scores(z)
    known = (z["role"] == "test") & (z["tier"] == "T1") & (z["stream_true"] != SC)
    unk = (z["role"] == "unknown") & (z["tier"] == "T1")
    kv = (z["role"] == "val") & (z["stream_true"] != SC)          # known validation items (thresholds are set here)
    if known.sum() == 0 or unk.sum() == 0:
        cx.rep.skip(f"decision table {bj}", f"no T1 known ({int(known.sum())}) or unknown ({int(unk.sum())}) items")
        return
    if kv.sum() == 0:
        cx.rep.notes.append(f"{bj}: no known validation items; the validation-matched (primary) rows are missing.")
    used.append(bj)
    sel = np.r_[np.nonzero(known)[0], np.nonzero(unk)[0]]
    nk, nu = int(known.sum()), int(unk.sum())
    is_unk = np.r_[np.zeros(nk, bool), np.ones(nu, bool)]
    ids = z["ids"][sel].astype(str)
    thr, rej = th["hazard"], th["reject"]
    haz = sc["haz"][sel]
    files_b = src_str(rb.files)
    vn = int(kv.sum())
    # D: dict(name, score (flag score, higher = flag) or None, vscore (same score on known validation items) or None,
    #         own bool or None, sign, files, rule)
    D = [dict(name="D1 top-class routing", score=None, vscore=None, own=sc["top_haz"][sel], sign=1, files=files_b,
              rule="flag if the top class is a hazard class")]
    D.append(dict(name="D2 hazard-sum flag", score=haz, vscore=sc["haz"][kv], own=haz >= thr, sign=1, files=files_b,
                  rule=f"flag if summed hazard probability >= t (own: {thr:.2f})"))
    for nm, key in (("MSP", "msp"), ("MaxLogit", "maxlogit"), ("Energy", "energy")):
        s = sc[key][sel]
        D.append(dict(name=f"D3 {nm}", score=-s, vscore=-sc[key][kv], own=s < rej[nm], sign=-1, files=files_b,
                      rule=f"flag if {nm} <= t at a swept threshold; own point: {nm} strictly below the validation "
                           f"5% quantile {rej[nm]:.4g}"))
    D.append(dict(name="D2+reject(MSP)", score=None, vscore=None, own=(haz >= thr) | (sc["msp"][sel] < rej["MSP"]), sign=1,
                  files=files_b, rule="flag if hazard-sum >= own threshold OR MSP strictly below the validation reject threshold"))
    sj = s_by_class.get(c)
    if sj and sj != "S1":
        rs = L.get(sj)
        if rs is not None:
            pos = _align(rs.z, ids)
            if pos is None:
                cx.rep.skip(f"D4 for {c}", f"{sj} lacks some of the {len(ids)} image ids used in {bj}")
            else:
                hs = job_scores(rs.z)["haz"]
                kvs = (rs.z["role"] == "val") & (rs.z["stream_true"] != SC)
                D.append(dict(name="D4 stream-trained", score=hs[pos], vscore=hs[kvs] if kvs.any() else None,
                              own=hs[pos] >= rs.j["thresholds"]["hazard"], sign=1, files=src_str(rs.files),
                              rule=f"flag if Special Care stream probability >= t (own: {rs.j['thresholds']['hazard']:.2f})"))
                used.append(sj)
    if e1 is not None:
        pos = _align(e1.z, ids)
        if pos is None:
            cx.rep.skip(f"D5 for {c}", f"E1 lacks some of the {len(ids)} image ids used in {bj}")
        else:
            fe = src_str(e1.files)
            ssum_all, spr_all = e1.z["stream_probs"][:, SC].astype(float), e1.z["hazard_prob"].astype(float)
            # E1 holds every image: known validation items = val rows of non-hazard classes other than the held-out class
            kve = (e1.z["role"] == "val") & (e1.z["stream_true"] != SC) & (e1.z["class_idx"] != RUN.CLASS_TO_IDX[c])
            D.append(dict(name="D5 CLIP class-prompt sum", score=ssum_all[pos], vscore=ssum_all[kve] if kve.any() else None,
                          own=ssum_all[pos] >= RUN.SC_THRESHOLD, sign=1, files=fe,
                          rule=f"flag if summed hazard-class prompt probability >= t (own: {RUN.SC_THRESHOLD:.2f}, a rule, not fitted)"))
            D.append(dict(name="D5 CLIP hazard prompt", score=spr_all[pos], vscore=spr_all[kve] if kve.any() else None,
                          own=spr_all[pos] >= CLIP_PROMPT_THRESHOLD, sign=1, files=fe,
                          rule=f"flag if hazard-prompt probability >= t (own: {CLIP_PROMPT_THRESHOLD}, prompt wins; not fitted)"))
            used.append("E1")
    boot = cx.boot(z["group"][sel], f"decision_{c}")
    Wk, Wu = boot.weights(~is_unk), boot.weights(is_unk)
    meas = "catch" if c in HAZARD_CLASSES else "wrongly_flagged"
    res = {}                                   # (design, op) -> dict(catch, far, thr (R+1,), val_far, n_val)
    for d in D:
        if d["own"] is not None:
            far, _ = boot.ratio(d["own"] & ~is_unk, ~is_unk)
            cat, _ = boot.ratio(d["own"] & is_unk, is_unk)
            res[(d["name"], "own")] = dict(catch=cat, far=far, thr=None)
    by_name = {d["name"]: d for d in D}
    d1 = by_name["D2 hazard-sum flag"]
    # PRIMARY: thresholds from known validation items, fixed across resamples
    v_far_d1 = float((d1["vscore"] >= thr).mean()) if kv.any() else None
    targets = {}
    if v_far_d1 is not None:
        for op, t in VAL_OPS:
            targets[op] = v_far_d1 if t is None else t
        for d in D:
            if d["score"] is None or d["vscore"] is None:
                continue
            for op, _ in VAL_OPS:
                if d is d1 and op == "val_match_D2":
                    t_, rv, flag = thr, v_far_d1, d["own"]          # D2 stays at its own threshold
                else:
                    t_, rv = val_threshold(d["vscore"], targets[op])
                    flag = d["score"] >= t_
                cat, _ = boot.ratio(flag & is_unk, is_unk)
                far, _ = boot.ratio(flag & ~is_unk, ~is_unk)
                res[(d["name"], op)] = dict(catch=cat, far=far, thr=np.full(boot.R + 1, t_), val_far=rv, n_val=len(d["vscore"]))
    # SECONDARY: ROC point matched on the known T1 test items (re-fitted in every resample)
    d1_far = res[(d1["name"], "own")]["far"]
    for d in D:
        if d["score"] is None:
            continue
        outs = sweep_to_far(d["score"][~is_unk], d["score"][is_unk], Wk, Wu, [d1_far] + [t for _, t in TEST_OPS[1:]])
        for (op, _), o in zip(TEST_OPS, outs):
            res[(d["name"], op)] = o
    # D2 stays at its own threshold in test_match_D2; the other designs are matched to its false-alarm rate
    res[(d1["name"], "test_match_D2")] = dict(res[(d1["name"], "own")], thr=np.full_like(d1_far, thr))
    targets["test_match_D2"] = ci(d1_far)[0]
    targets["test_match_5pct"], targets["test_match_10pct"] = 0.05, 0.10
    leak = boot.ratio(~d1["own"] & (sc["s_pred"][sel] == DRY) & is_unk, is_unk)
    for (dn, op), o in res.items():
        d = by_name[dn]
        block = BLOCK_OWN if op == "own" else BLOCK_VAL if op.startswith("val_") else BLOCK_TEST
        fitted = ("own rule or validation fit" if op == "own" else "known validation items" if op.startswith("val_")
                  else "known T1 test items (re-fitted per resample; comparison only)")
        thr_pt = o["thr"][0] if o["thr"] is not None and np.isfinite(o["thr"][0]) else None
        if thr_pt is not None:
            thr_pt = d["sign"] * thr_pt
        base = dict(held_out_class=c, job=bj, design=dn, block=block, op_point=op, target_far=targets.get(op),
                    threshold=thr_pt, threshold_fitted_on=fitted, val_far_realised=o.get("val_far"),
                    n_val_known=o.get("n_val"), n_known=nk, n_unknown=nu, rule=d["rule"],
                    source_files=src_str(files_b, d["files"]))
        p, lo, hi = ci(o["catch"])
        rows.append(dict(base, measure=meas, value=p, ci_lo=lo, ci_hi=hi, n=nu))
        p, lo, hi = ci(o["far"])
        rows.append(dict(base, measure="known_false_alarm", value=p, ci_lo=lo, ci_hi=hi, n=nk))
        if dn != d1["name"]:
            diff = res[(d1["name"], op)]["catch"] - o["catch"]
            p, lo, hi = ci(diff)
            rows.append(dict(base, measure="diff_D2_minus_design", value=p, ci_lo=lo, ci_hi=hi, n=nu))
    cx.shared["unseen_leak"][c] = dict(vals=leak[0], n=nu, files=rb.files)
    cx.shared["unseen_catch"][c] = dict(vals=res[(d1["name"], "own")]["catch"], n=nu, files=rb.files, job=bj)
    if c == "shoes":
        cx.shared["shoes_flag"] = dict(vals=res[(d1["name"], "own")]["catch"], n=nu, files=rb.files)


# ======================================================================================
# 3. Unseen datasets (B5-B8)
# ======================================================================================
def analyse_unseen_datasets(cx):
    L, rows = cx.L, []
    cx.shared["far_leftout"] = {}
    ra = L.get("A1")
    used = []
    keep = ["stream_acc", "class_acc", "sc_miss_rate", "sc_false_alarm", "ece_before", "ece_after"]
    for bj in B_SRC_JOBS:
        rb = L.get(bj)
        if rb is None:
            continue
        src = rb.j.get("held_out_source") or RUN.JOBS[bj].holdout_source
        unseen = list(rb.j.get("unseen_classes") or [])
        z, thr = rb.z, rb.j["thresholds"]["hazard"]
        names = [str(n) for n in z["label_names"]]
        unk = z["role"] == "unknown"
        if not unk.any():
            cx.rep.skip(f"unseen-dataset row {bj}", "no unknown rows in the logits file")
            continue
        used.append(bj)
        sc = job_scores(z)
        sub = take(sc, unk)
        boot = cx.boot(z["group"][unk], f"unseen_{src}")
        y = z["label_idx"][unk]
        st = metric_stats(boot, sub, y, z["stream_true"][unk], thr)
        tk = RUN._thr_key(thr)
        st["sc_miss_rate"] = (1 - st[f"sc_recall@{tk}"][0], st[f"sc_recall@{tk}"][1])
        st["sc_false_alarm"] = st[f"sc_false_alarm@{tk}"]
        cx.shared["far_leftout"][src] = dict(vals=st["sc_false_alarm"][0], n=st["sc_false_alarm"][1], files=rb.files)
        common = dict(held_out_source=src, job=bj, unseen_classes=";".join(unseen) or "none")
        rows += stat_rows(st, dict(common, condition="left out of training"), src_str(rb.files), only=set(keep))
        n_unseen_rows = int((y < 0).sum())
        if n_unseen_rows:
            vm = y < 0
            b2 = cx.boot(z["group"][unk][vm], f"unseen_cls_{src}")
            flag = sub["haz"][vm] >= thr
            acc = b2.ratio(sub["s_pred"][vm] == z["stream_true"][unk][vm], np.ones(int(vm.sum())))
            fl = b2.ratio(flag, np.ones(int(vm.sum())))
            rows += stat_rows(dict(unseen_class_images_stream_acc=acc, unseen_class_images_flagged_special_care=fl),
                              dict(common, condition="left out of training (images of classes never trained)"),
                              src_str(rb.files))
        # the same source with the source in training: A1 T1 test rows
        if ra is not None:
            za = ra.z
            m = (za["role"] == "test") & (za["tier"] == "T1") & (za["source"] == src)
            if m.any():
                sca = job_scores(za)
                suba = take(sca, m)
                trained = np.isin(za["class_idx"][m], [RUN.CLASS_TO_IDX[n] for n in names])
                ya = np.where(trained, za["label_idx"][m], -1)
                ba = cx.boot(za["group"][m], f"unseen_A1_{src}")
                sta = metric_stats(ba, suba, ya, za["stream_true"][m], thr)
                sta["sc_miss_rate"] = (1 - sta[f"sc_recall@{tk}"][0], sta[f"sc_recall@{tk}"][1])
                sta["sc_false_alarm"] = sta[f"sc_false_alarm@{tk}"]
                rows += stat_rows(sta, dict(common, condition="same source in training (A1, T1 test rows)"),
                                  src_str(ra.files), only=set(keep))
                for mname in keep:
                    if mname in st and mname in sta:
                        d = st[mname][0] - sta[mname][0]      # independent resamples: different images
                        p, lo, hi = ci(d)
                        rows.append(dict(common, condition="difference (left out minus in training)", metric=mname,
                                         value=p, ci_lo=lo, ci_hi=hi, n=min(st[mname][1], sta[mname][1]),
                                         source_files=src_str(rb.files, ra.files)))
            else:
                cx.rep.notes.append(f"{bj}: A1 has no T1 test rows of source {src}; no in-training reference.")
    # frozen-feature results, if a file for them exists (format not defined by run.py: copied through unchanged)
    frozen = []
    if os.path.isdir(L.rdir):
        frozen = sorted(f for f in os.listdir(L.rdir) if f.lower().endswith(".csv") and f.lower().startswith(("f_", "frozen")))
    if frozen:
        for f in frozen:
            fr = pd.read_csv(L.path(f))
            fr["source_files"] = src_str(L.path(f))
            cx.save("frozen_" + os.path.splitext(f)[0], fr, [], f"frozen-feature results copied unchanged from {f} (no analysis)")
    else:
        cx.rep.skip("frozen-feature results (job F)", "no results/F*.csv or results/frozen*.csv found")
    if not rows:
        cx.rep.skip("unseen-dataset table", "no usable B5-B8 results")
        return
    df = pd.DataFrame(rows)
    cx.save("unseen_datasets", df, used + (["A1"] if ra is not None else []),
            "left-out source vs the same source in training (A1)", ["held_out_source", "condition", "metric"])
    cx.figs["unseen"] = df


# ======================================================================================
# 4. Leakage (C1, C2)
# ======================================================================================
def analyse_leakage(cx):
    L = cx.L
    got = {j: L.get(j) for j in C_JOBS}
    got = {j: r for j, r in got.items() if r is not None}
    if not got:
        cx.rep.skip("leakage table (C1, C2)", "no C result files")
        return
    rows, vals, boots = [], {}, {}
    names = ["class_acc", "macro_f1", "stream_acc"]
    for jid, r in got.items():
        z = r.z
        m = z["role"] == "test"
        sc = job_scores(z)
        sub = take(sc, m)
        boot = cx.boot(z["group"][m], f"leak_{jid}")
        st = metric_stats(boot, sub, z["label_idx"][m], z["stream_true"][m], r.j.get("hazard_threshold", RUN.SC_THRESHOLD))
        vals[jid] = st
        rows += stat_rows(st, dict(job=jid, what=r.j.get("what")), src_str(r.files),
                          (r.j.get("metrics") or {}).get("test_all"), only=set(names))
        lk = r.j.get("leakage")
        near_all = z.get("near_dup_in_train")
        if near_all is not None:
            near = near_all[m].astype(bool)
            ok = sub["p"].argmax(1) == z["label_idx"][m]
            f = src_str(r.files)
            base = dict(job=jid, what=r.j.get("what"), source_files=f, ci_method="grouped_bootstrap")
            share, ns = boot.ratio(near, np.ones(len(near)))
            a_with, n_with = boot.ratio(ok & near, near)
            a_wo, n_wo = boot.ratio(ok & ~near, ~near)
            for metric, v, n in (("share_test_with_train_near_duplicate", share, ns), ("acc_items_with_duplicate", a_with, n_with),
                                 ("acc_items_without_duplicate", a_wo, n_wo),
                                 ("diff_acc_with_minus_without_duplicate", a_with - a_wo, min(n_with, n_wo))):
                p, lo, hi = ci(v)
                rows.append(dict(base, metric=metric, value=p, ci_lo=lo, ci_hi=hi, n=n))
            if lk:
                for metric, key, v in (("share", "test_with_near_duplicate_in_train", share), ("accuracy with duplicate", "acc_items_with_duplicate", a_with),
                                       ("accuracy without duplicate", "acc_items_without_duplicate", a_wo)):
                    jv, pv = lk.get(key), float(v[0])
                    if (jv is None) != (not np.isfinite(pv)) or (jv is not None and abs(pv - jv) > 1e-6):
                        cx.rep.errors.append(f"{jid}: {metric} from near_dup_in_train ({pv}) differs from the JSON leakage block ({jv}) by more than 1e-6")
            else:
                cx.rep.notes.append(f"{jid}: no 'leakage' block in the JSON to check the near_dup_in_train numbers against.")
        elif lk:
            cx.rep.notes.append(f"{jid}: no near_dup_in_train array in the logits file; FALLBACK to the JSON leakage block with "
                                f"ungrouped Wilson intervals (counts rebuilt from rounded accuracies); no with-minus-without difference.")
            n, k = lk["n_test"], lk["n_with_duplicate"]
            lo, hi = wilson(k, n)
            f = src_str(r.files[0])
            base = dict(job=jid, what=r.j.get("what"), source_files=f, ci_method="wilson_ungrouped_fallback")
            rows.append(dict(base, metric="share_test_with_train_near_duplicate", value=lk["test_with_near_duplicate_in_train"],
                             ci_lo=lo, ci_hi=hi, n=n))
            for key, kk in (("acc_items_with_duplicate", k), ("acc_items_without_duplicate", n - k)):
                v = lk.get(key)
                if v is None or kk == 0:
                    continue
                lo, hi = wilson(round(v * kk), kk)
                rows.append(dict(base, metric=key, value=v, ci_lo=lo, ci_hi=hi, n=kk))
        else:
            cx.rep.notes.append(f"{jid}: neither near_dup_in_train nor a 'leakage' block; near-duplicate analysis not available.")
    if "C1" in vals and "C2" in vals:
        fl = src_str(got["C1"].files, got["C2"].files)
        for m_ in names:
            if m_ in vals["C1"] and m_ in vals["C2"]:
                d = vals["C1"][m_][0] - vals["C2"][m_][0]       # different test images: independent resamples
                p, lo, hi = ci(d)
                rows.append(dict(job="C1-C2", what="naive random split minus grouped split", metric=f"diff_{m_}", value=p,
                                 ci_lo=lo, ci_hi=hi, n=min(vals["C1"][m_][1], vals["C2"][m_][1]), source_files=fl))
    else:
        cx.rep.skip("C1 minus C2 difference", "C1 or C2 missing")
    df = pd.DataFrame(rows)
    cx.save("leakage", df, list(got), "C1 (naive split) vs C2 (grouped split) and the near-duplicate shares",
            ["job", "metric"])


# ======================================================================================
# 5. Object or dataset
# ======================================================================================
def _sample_cells(df, per_cell, rng):
    parts = []
    for _, g in df.groupby(["label", "source"]):
        parts.append(g.sample(n=min(len(g), per_cell), random_state=int(rng.integers(1 << 31))))
    return pd.concat(parts) if parts else df.iloc[:0]


def _make_probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    steps = [StandardScaler()]
    if args.probe_pca > 0:
        from sklearn.decomposition import PCA
        steps.append(PCA(n_components=args.probe_pca, random_state=args.seed))
    steps.append(LogisticRegression(max_iter=args.probe_max_iter, class_weight="balanced"))
    return make_pipeline(*steps)


def _count_convergence(fn):
    """Run fn() and return (its result, number of sklearn ConvergenceWarnings it raised)."""
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn()
    return out, sum(issubclass(x.category, ConvergenceWarning) for x in w)


def analyse_object_or_dataset(cx):
    a = cx.args
    sp_path = os.path.join(a.data_dir, "splits_v3.csv")
    if not a.features:
        cx.rep.skip("object-or-dataset probes", "no --features file given")
        return
    if not os.path.exists(a.features) or not os.path.exists(sp_path):
        cx.rep.skip("object-or-dataset probes", f"missing {a.features if not os.path.exists(a.features) else sp_path}")
        return
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    df = RUN.load_splits(sp_path)
    z = np.load(a.features, allow_pickle=True)      # ids may be an object array, as in the Drive file
    pos = pd.Index(z["ids"].astype(str)).get_indexer(df.id.astype(str))
    n_no = int((pos < 0).sum())
    df = df[pos >= 0].reset_index(drop=True)
    X = z["feats"][pos[pos >= 0]].astype(np.float32)
    if n_no:
        cx.rep.notes.append(f"Object-or-dataset: {n_no} of {len(pos)} images in splits_v3.csv have no feature row and were left out.")
    files = src_str(sp_path, a.features)
    rng = np.random.default_rng(a.seed)
    cap, minc = a.probe_cell_cap, a.probe_min_per_cell
    train = df[df.split == "train"]
    rows_a, rows_b, elig, warned = [], [], {}, []
    for c in CLASSES:
        cnt = train[train.label == c].source.value_counts()
        srcs = sorted(s for s, n in cnt.items() if n >= minc)
        if len(srcs) < 2:
            cx.rep.skip(f"object-or-dataset class {c}", f"fewer than two sources with >= {minc} training images ({ {k: int(v) for k, v in cnt.items()} })")
            continue
        elig[c] = srcs
        sub = _sample_cells(train[(train.label == c) & train.source.isin(srcs)], cap, rng)
        ys, gs = sub.source.values, sub.group.values
        ngrp = min(len(set(gs[ys == s])) for s in srcs)
        n_splits = min(5, ngrp)
        if n_splits < 2:
            cx.rep.skip(f"source probe for class {c}", "a source has a single photo group")
            continue
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=a.seed)
        oof, nwarn = _count_convergence(lambda: cross_val_predict(_make_probe(a), X[sub.index.values], ys, cv=cv, groups=gs))
        warned.append((f"source probe {c}", nwarn))
        maj = pd.Series(ys).value_counts().index[0]
        ok, is_maj = (oof == ys).astype(float), (ys == maj).astype(float)
        boot = cx.boot(gs, f"probe_src_{c}")
        acc, n = boot.ratio(ok, np.ones(len(ok)))
        diff, _ = boot.ratio(ok - is_maj, np.ones(len(ok)))
        base = dict(label=c, n_sources=len(srcs), sources=";".join(srcs), n=n, cv_folds=n_splits, majority_source=maj,
                    convergence_warnings=nwarn, source_files=files)
        for mname, v in (("source_probe_accuracy", acc), ("majority_source_baseline", is_maj.mean() * np.ones_like(acc)),
                         ("accuracy_minus_baseline", diff)):
            p, lo, hi = ci(v)
            if mname == "majority_source_baseline":
                lo = hi = None
            rows_a.append(dict(base, metric=mname, value=p, ci_lo=lo, ci_hi=hi))
    cx.save("object_or_dataset_source_probe", pd.DataFrame(rows_a), [],
            "predict the source from frozen features within a class (5-fold grouped CV, training images)",
            ["label", "metric"])
    # (b) train on source A, test on source B
    gap_by_class = {}
    for c, srcs in elig.items():
        for A in srcs:
            for B in srcs:
                if A == B:
                    continue
                pos_tr = _sample_cells(train[(train.label == c) & (train.source == A)], cap, rng)
                held = df[(df.label == c) & (df.source == A) & df.split.isin(["val", "test"])]
                cross = df[(df.label == c) & (df.source == B)]
                neg_pool = _sample_cells(train[(train.label != c) & (train.source != B)], cap, rng)
                if len(neg_pool) > a.probe_max_neg:
                    neg_pool = neg_pool.sample(n=a.probe_max_neg, random_state=int(rng.integers(1 << 31)))
                if len(held) < 10 or len(cross) < 10:
                    cx.rep.skip(f"cross-source probe {c} {A}->{B}", f"only {len(held)} held-out {A} and {len(cross)} {B} images")
                    continue
                held, cross = held.sample(n=min(len(held), cap), random_state=1), cross.sample(n=min(len(cross), cap), random_state=2)
                Xtr = np.vstack([X[pos_tr.index.values], X[neg_pool.index.values]])
                ytr = np.r_[np.ones(len(pos_tr)), np.zeros(len(neg_pool))]
                clf, nwarn = _count_convergence(lambda: _make_probe(a).fit(Xtr, ytr))
                warned.append((f"cross-source probe {c} {A}->{B}", nwarn))
                hw = clf.predict(X[held.index.values]) == 1
                hc = clf.predict(X[cross.index.values]) == 1
                bw, bc = cx.boot(held.group.values, f"pw_{c}_{A}_{B}"), cx.boot(cross.group.values, f"pc_{c}_{B}_{A}")
                rw, nw = bw.ratio(hw.astype(float), np.ones(len(hw)))
                rc, nc = bc.ratio(hc.astype(float), np.ones(len(hc)))
                gap = rw - rc                              # different images: independent resamples
                gap_by_class.setdefault(c, []).append(gap)
                base = dict(label=c, trained_on=A, tested_on=B, n_pos=len(pos_tr), n_neg=len(neg_pool),
                            convergence_warnings=nwarn, source_files=files)
                for mname, v, n in (("recall_within_source", rw, nw), ("recall_cross_source", rc, nc), ("gap_within_minus_cross", gap, min(nw, nc))):
                    p, lo, hi = ci(v)
                    rows_b.append(dict(base, metric=mname, value=p, ci_lo=lo, ci_hi=hi, n=n))
    for c, gl in gap_by_class.items():
        p, lo, hi = ci(np.mean(gl, axis=0))
        rows_b.append(dict(label=c, trained_on="(mean over pairs)", tested_on="", n_pos=None, n_neg=None, metric="gap_within_minus_cross",
                           value=p, ci_lo=lo, ci_hi=hi, n=len(gl), source_files=files))
    bad = [(nm, k) for nm, k in warned if k]
    cx.rep.notes.append(f"Probe convergence warnings (max_iter={a.probe_max_iter}): {sum(k for _, k in warned)} in {len(bad)} of "
                        f"{len(warned)} probes" + (": " + "; ".join(f"{nm} ({k})" for nm, k in bad) if bad else "") +
                        ". Per probe in the column convergence_warnings.")
    cx.save("object_or_dataset_cross_source", pd.DataFrame(rows_b), [],
            "one-vs-rest probe trained on source A, recall of the class on held-out A images and on source B images",
            ["label", "trained_on", "tested_on", "metric"])


# ======================================================================================
# 6. Material-flow scenarios
# ======================================================================================
def analyse_scenarios(cx):
    A = cx.shared.get("A_T1")
    if A is None:
        cx.rep.skip("material-flow scenarios", "no A-job T1 numbers (main results missing)")
        return
    R, files_a = A["R"], A["files"]
    R_src = {"A_T1 flag recall": (R, src_str(files_a))}
    for c, d in cx.shared.get("unseen_catch", {}).items():
        if c in HAZARD_CLASSES:
            R_src[f"unseen {c} (D2 catch)"] = (d["vals"], src_str(d["files"]))
    F_src = {"A_T1 false-alarm rate": (A["FAR"], src_str(files_a))}
    for s, d in cx.shared.get("far_leftout", {}).items():
        F_src[f"left-out source {s}"] = (d["vals"], src_str(d["files"]))
    if "shoes_flag" in cx.shared:
        F_src["shoes control (share flagged)"] = (cx.shared["shoes_flag"]["vals"], src_str(cx.shared["shoes_flag"]["files"]))
    rows = []
    for rn, (rv, rf) in R_src.items():
        for fn, (fv, ff) in F_src.items():
            for pi in PI_LIST:
                with np.errstate(invalid="ignore", divide="ignore"):
                    prec = rv * pi / (rv * pi + fv * (1 - pi))
                p, lo, hi = ci(prec)
                rows.append(dict(recall_basis=rn, false_alarm_basis=fn, hazard_prevalence=pi, metric="flag_precision",
                                 value=p, ci_lo=lo, ci_hi=hi, n=None,
                                 recall=ci(rv)[0], false_alarm_rate=ci(fv)[0], source_files=src_str(rf, ff)))
    cx.save("scenario_flag_precision", pd.DataFrame(rows).assign(a_job=A["job"]), [A["job"]],
            "flag precision = R*pi / (R*pi + FAR*(1-pi)) for hazard prevalence pi (scenario numbers, not measured prevalence)",
            ["recall_basis", "false_alarm_basis", "hazard_prevalence", "metric"])
    cx.figs["precision"] = pd.DataFrame(rows)
    rows = []
    leak_A = A["leak_dry"]
    fa = A["FAR"]
    bases = [("A_T1", R, leak_A, src_str(files_a))]
    for c, d in cx.shared.get("unseen_catch", {}).items():
        if c in HAZARD_CLASSES and c in cx.shared["unseen_leak"]:
            lk = cx.shared["unseen_leak"][c]
            bases.append((f"unseen {c}", d["vals"], lk["vals"], src_str(d["files"], lk["files"])))
    for pi in PI_LIST:
        for label, rv, leak, f in bases:
            for metric, v in (("hazards_present_per_10000", pi * 1e4 * np.ones_like(rv)),
                              ("hazards_flagged_per_10000", pi * 1e4 * rv),
                              ("hazards_not_flagged_per_10000", pi * 1e4 * (1 - rv)),
                              ("hazards_reaching_dry_stream_per_10000", pi * 1e4 * leak),
                              ("non_hazard_false_alarms_per_10000", (1 - pi) * 1e4 * fa)):
                p, lo, hi = ci(v)
                rows.append(dict(basis=label, hazard_prevalence=pi, metric=metric, value=p, ci_lo=lo, ci_hi=hi, n=None,
                                 source_files=f))
        rows.append(dict(basis="always-dry baseline", hazard_prevalence=pi, metric="hazards_reaching_dry_stream_per_10000",
                         value=pi * 1e4, ci_lo=None, ci_hi=None, n=None, source_files=src_str(files_a)))
    cx.save("scenario_per_10000", pd.DataFrame(rows).assign(a_job=A["job"]), [A["job"]],
            "scenario counts per 10,000 items; 'reaching dry' = not flagged and top stream is Dry; FAR from A T1",
            ["basis", "hazard_prevalence", "metric"])
    # reweighted stream accuracy
    if not cx.args.mix:
        cx.rep.skip("stream accuracy reweighted to a waste mix", "no --mix given (no mix is assumed)")
        return
    try:
        raw = cx.args.mix
        mix = json.load(open(raw)) if os.path.exists(raw) else json.loads(raw)
        bad = [k for k in mix if k not in STREAMS]
        if bad or abs(sum(mix.values()) - 1) > 1e-6:
            raise ValueError(f"--mix must map {STREAMS} to shares summing to 1 (got keys {list(mix)}, sum {sum(mix.values()):.4f})")
    except Exception as e:  # noqa: BLE001
        cx.rep.skip("stream accuracy reweighted to a waste mix", f"bad --mix: {e}")
        return
    v = sum(mix.get(s, 0.0) * rv_ for s, rv_ in zip(STREAMS, A["recall_stream"]))
    p, lo, hi = ci(v)
    row = dict(mix=json.dumps(mix), metric="stream_accuracy_reweighted", value=p, ci_lo=lo, ci_hi=hi, n=None,
               source_files=src_str(files_a))
    cx.save("scenario_reweighted_stream_accuracy", pd.DataFrame([row]), [A["job"]],
            f"{A['job']} T1 per-stream recall (summed-stream argmax) weighted by the given mix", ["mix", "metric"])


# ======================================================================================
# 7. Data tables
# ======================================================================================
def analyse_data_tables(cx):
    d = cx.args.data_dir
    ct_path = os.path.join(d, "count_table.csv")
    if os.path.exists(ct_path):
        ct = pd.read_csv(ct_path)
        num = [c for c in ct.columns if c not in ("source", "label")]
        f = src_str(ct_path)
        pc = ct.groupby("label", sort=False)[num].sum().reset_index()
        ps = ct.groupby("source", sort=False)[num].sum().reset_index()
        for name, t, idc in (("data_count_per_class", pc, "label"), ("data_count_per_source", ps, "source")):
            long = t.melt(id_vars=idc, var_name="stage", value_name="value")
            long["n"] = long["value"]
            long["ci_lo"] = long["ci_hi"] = np.nan
            long["source_files"] = f
            cx.save(name, t.assign(source_files=f), [], f"count table condensed by {idc} (sum over the other axis)")
            for _, r in long.iterrows():
                cx.pn.add(f"{name}|{idc}={r[idc]}|stage={r['stage']}", r["value"], None, None, r["n"], [f], name)
    else:
        cx.rep.skip("count-table summaries", f"{ct_path} not found")
    dp = os.path.join(d, "dup_pairs_phash.csv.gz")
    if os.path.exists(dp):
        pairs = pd.read_csv(dp)
        f = src_str(dp)
        pairs["cross_source"] = pairs.source_a != pairs.source_b
        by = pairs.groupby(["distance", "cross_source"]).size().unstack(fill_value=0)
        by = by.rename(columns={False: "within_source", True: "cross_source"}).reindex(columns=["within_source", "cross_source"], fill_value=0)
        by["all"] = by.sum(1)
        by = by.reset_index().assign(source_files=f)
        cx.save("data_dup_pairs_by_distance", by, [], "pHash duplicate pairs by Hamming distance")
        for _, r in by.iterrows():
            for k in ("within_source", "cross_source", "all"):
                cx.pn.add(f"data_dup_pairs_by_distance|distance={int(r['distance'])}|{k}", r[k], None, None, r[k], [f], "data_dup_pairs_by_distance")
        summ = [("pairs", len(pairs)), ("cross_source_pairs", int(pairs.cross_source.sum())),
                ("within_source_pairs", int((~pairs.cross_source).sum())), ("exact_pairs_distance_0", int((pairs.distance == 0).sum())),
                ("pairs_with_different_labels", int((pairs.label_a != pairs.label_b).sum()))]
        st = pd.DataFrame([dict(measure=k, value=v, n=len(pairs), source_files=f) for k, v in summ])
        cx.save("data_dup_summary", st, [], "duplicate pair summary", ["measure"])
        cp = pairs[pairs.cross_source].groupby(["source_a", "source_b"]).size().rename("value").reset_index().assign(source_files=f)
        cp["n"] = cp["value"]
        cx.save("data_dup_cross_source_by_source_pair", cp, [], "cross-source duplicate pairs by source pair (value = pairs)",
                ["source_a", "source_b"])
    else:
        cx.rep.skip("duplicate summary", f"{dp} not found")
    cp_ = os.path.join(d, "label_conflicts.csv")
    if os.path.exists(cp_):
        cf = pd.read_csv(cp_)
        f = src_str(cp_)
        if len(cf):
            t = cf.groupby(["stage", "reason"]).size().rename("value").reset_index()
        else:
            t = pd.DataFrame(columns=["stage", "reason", "value"])
        t["n"] = t["value"]
        t["source_files"] = f
        cx.save("data_conflicts_by_reason", t, [], "label-conflict rows by stage and reason", ["stage", "reason"])
        if len(cf):
            sp_ = cf.groupby(["source_a", "source_b"]).size().rename("value").reset_index().assign(source_files=f)
            sp_["n"] = sp_["value"]
            cx.save("data_conflicts_by_source_pair", sp_, [], "label-conflict rows by source pair (value = rows)",
                    ["source_a", "source_b"])
    else:
        cx.rep.skip("label-conflict table", f"{cp_} not found")
    op = os.path.join(d, "source_overlap.csv")
    if os.path.exists(op):
        ov = pd.read_csv(op, index_col=0)
        ov.to_csv(os.path.join(cx.args.out_dir, "data_source_overlap.csv"))
        cx.rep.tables["data_source_overlap"] = dict(rows=len(ov), jobs=[], files=[src_str(op)], desc="source-overlap matrix (duplicate pairs), copied")
        cx.figs["overlap"] = ov
    else:
        cx.rep.skip("source-overlap matrix and heatmap", f"{op} not found")
    rp = os.path.join(d, "report.json")
    if os.path.exists(rp):
        try:
            rj = json.load(open(rp))
            st = rj.get("stage_totals") or {}
            if st:
                t = pd.DataFrame([dict(stage=k, value=v, n=v, source_files=src_str(rp)) for k, v in st.items()])
                cx.save("data_stage_totals", t, [], "image counts per pipeline stage (rebuild_splits report.json)", ["stage"])
        except Exception as e:  # noqa: BLE001
            cx.rep.skip("rebuild report.json stage totals", repr(e))


# ======================================================================================
# 8. Figures
# ======================================================================================
def make_figures(cx):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        cx.rep.skip("figures", f"matplotlib not available ({e!r})")
        return
    out = cx.args.out_dir
    pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    if "decision" in cx.figs:
        df = cx.figs["decision"]
        ops = ["own", "val_match_D2", "val_match_5pct", "val_match_10pct"]
        classes = list(dict.fromkeys(df.held_out_class))
        fig, axs = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 4.2), sharey=True, squeeze=False)
        designs = list(dict.fromkeys(df.design))
        for ax, c in zip(axs[0], classes):
            d = df[(df.held_out_class == c) & df.measure.isin(["catch", "wrongly_flagged"])]
            w = 0.8 / len(designs)
            for i, dn in enumerate(designs):
                for k, op in enumerate(ops):
                    r = d[(d.design == dn) & (d.op_point == op)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    ax.bar(k + i * w, r.value, w * 0.9, color=pal[i % len(pal)], label=dn if k == 0 else None,
                           yerr=[[max(r.value - r.ci_lo, 0)], [max(r.ci_hi - r.value, 0)]] if pd.notna(r.ci_lo) else None,
                           error_kw=dict(lw=0.8))
            ax.set_xticks(np.arange(len(ops)) + 0.4)
            ax.set_xticklabels(["own point", "val: = D2 FAR", "val: 5% FAR", "val: 10% FAR"], fontsize=8)
            ax.set_title(f"{c} ({'caught' if c in HAZARD_CLASSES else 'wrongly flagged'})", fontsize=10)
            ax.set_ylim(0, 1.02)
        axs[0][0].set_ylabel("share of unknown items")
        axs[0][0].legend(fontsize=6, loc="upper right")      # first panel has every design
        fig.suptitle("Decision designs on unseen classes (T1 test). Matched points: thresholds set on known validation items",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_decision_designs.png"), dpi=150)
        plt.close(fig)
    if "unseen" in cx.figs:
        df = cx.figs["unseen"]
        d = df[df.condition.isin(["left out of training", "same source in training (A1, T1 test rows)"]) &
               df.metric.isin(["stream_acc", "sc_false_alarm", "sc_miss_rate"])]
        srcs = list(dict.fromkeys(d.held_out_source))
        mets = ["stream_acc", "sc_miss_rate", "sc_false_alarm"]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), squeeze=False)
        for ax, m in zip(axs[0], mets):
            for k, cond, col in ((0, "left out of training", pal[3]), (1, "same source in training (A1, T1 test rows)", pal[0])):
                r = d[(d.metric == m) & (d.condition == cond)].set_index("held_out_source").reindex(srcs)
                x = np.arange(len(srcs)) + k * 0.38
                ax.bar(x, r.value.fillna(0), 0.36, color=col, label=cond.split(" (")[0],
                       yerr=[(r.value - r.ci_lo).clip(lower=0).fillna(0), (r.ci_hi - r.value).clip(lower=0).fillna(0)], error_kw=dict(lw=0.8))
            ax.set_xticks(np.arange(len(srcs)) + 0.19)
            ax.set_xticklabels(srcs, fontsize=8)
            ax.set_title(m.replace("_", " "), fontsize=10)
        axs[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_unseen_datasets.png"), dpi=150)
        plt.close(fig)
    if "precision" in cx.figs:
        df = cx.figs["precision"]
        d = df[df.recall_basis == "A_T1 flag recall"]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        Rv = d.recall.iloc[0]
        pis = np.linspace(0.001, 0.10, 200)
        for i, (fn, g) in enumerate(d.groupby("false_alarm_basis", sort=False)):
            far = g.false_alarm_rate.iloc[0]
            ax.plot(pis * 100, Rv * pis / (Rv * pis + far * (1 - pis)), color=pal[i % len(pal)], label=f"{fn} (FAR {far:.3f})")
            for pi in PI_LIST:
                r = g[g.hazard_prevalence == pi].iloc[0]
                if pd.notna(r.ci_lo):
                    ax.plot([pi * 100] * 2, [r.ci_lo, r.ci_hi], color=pal[i % len(pal)], lw=2, alpha=0.5)
        ax.set_xlabel("hazard prevalence (% of items)")
        ax.set_ylabel("flag precision")
        ax.set_title(f"Flag precision, recall R = {Rv:.3f} (bars: 95% interval at 1, 2, 5%)", fontsize=9)
        ax.legend(fontsize=6)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_flag_precision.png"), dpi=150)
        plt.close(fig)
    if "overlap" in cx.figs:
        ov = cx.figs["overlap"]
        v = ov.values.astype(float)
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        from matplotlib.colors import LogNorm
        im = ax.imshow(np.where(v > 0, v, np.nan), cmap="viridis", norm=LogNorm(vmin=1, vmax=max(v.max(), 2)))
        ax.set_xticks(range(len(ov.columns)))
        ax.set_xticklabels(ov.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(ov.index)))
        ax.set_yticklabels(ov.index, fontsize=7)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if v[i, j] > 0:
                    ax.text(j, i, f"{int(v[i, j])}", ha="center", va="center", fontsize=5, color="white")
        fig.colorbar(im, label="duplicate pairs (log scale; blank = 0)")
        ax.set_title("Source overlap: pHash duplicate pairs")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_source_overlap.png"), dpi=150)
        plt.close(fig)


# ======================================================================================
# Report and main
# ======================================================================================
DEFINITIONS = """\
- Grouped bootstrap: photo groups (the `group` array) resampled with replacement, `--n-boot` resamples, percentile 95% interval; the point value is the data as it is. Differences between two designs on the same images are paired (same resample); differences between two jobs on different images use independent resamples.
- Test rows of a job: role == test. Tier from the `tier` array (T1 headline, T2 conveyor, T3 street).
- Stream accuracy: argmax of the summed (temperature-scaled) stream probabilities equals the true stream. Top-class stream accuracy: stream of the top class. Flag: summed Special Care probability >= the job's validation-fitted threshold (0.20). Class accuracy and macro-F1 use only items whose class is in the label space.
- Decision designs: unknown items = T1 test images of the held-out class; known items = T1 test images of trained non-hazard classes (true stream not Special Care). False alarm = a known item flagged. Caught = an unknown item flagged.
  D1: flagged if the top class is a hazard class (no threshold). D2: flagged if the summed temperature-scaled hazard probability >= t. D3: MSP / MaxLogit / Energy (temperature-scaled, as in run.py) are familiarity scores. At the OWN point a D3 item is flagged if its score is STRICTLY BELOW the validation 5% quantile (as in run.py, `score < threshold`); at swept (matched) thresholds it is flagged if the score is AT OR BELOW t (implemented as -score >= -t). D2+reject(MSP): D2 flag OR MSP strictly below the validation reject threshold. D4: Special Care probability of the stream-trained job >= t. D5: CLIP hazard-class prompt sum >= t, or hazard-prompt probability >= t. D5 uses all 12 class prompts, so the held-out class is not unseen to CLIP.
  Own point: D2 and D4 use the job's hazard threshold (a cost rule, 1/(1+4)); D3 uses the validation 5% quantile reject threshold; D5 has no fitted threshold (rule values 0.20 / 0.5).
  PRIMARY matched comparison (op_point val_match_D2, val_match_5pct, val_match_10pct): for each job and design the threshold is set on that job's known non-hazard VALIDATION items (role val, true stream not Special Care; for E1 also not the held-out class) to reach the target false-alarm rate (D2's own validation rate, 5%, 10%): k = floor(target * n) highest known scores are flagged (moved down to the end of a tie block), threshold = the k-th highest score, flag = score >= t. The threshold is then FIXED. Reported on the T1 test items: catch rate on the unknown items, and the false-alarm rate realised on the known T1 test items (`known_false_alarm`), each with a grouped bootstrap interval (thresholds do not change across resamples). D2 stays at its own threshold at val_match_D2 and the others are matched to D2's validation rate.
  SECONDARY block (op_point test_match_D2, test_match_5pct, test_match_10pct): "ROC point matched on test (comparison only, not deployable)". The threshold is set on the known T1 TEST items of the same job (highest rate not above the target; whole tie blocks only, so the realised rate is at most the target) and re-fitted in every bootstrap resample. It is an evaluation-time matching, not a deployable threshold.
- Material-flow scenarios are scenario arithmetic: flag precision = R*pi/(R*pi + FAR*(1-pi)); R and FAR come from the A-job T1 numbers; FAR is also taken from the left-out sources (B5-B8, at 0.20) and from the share of shoes flagged (B4).
"""


def write_report(cx, t0, argv_str):
    rep, L = cx.rep, cx.L
    lines = ["# WasteVision analysis report", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; run time {time.time() - t0:.1f} s. Command: `{argv_str}`", "",
             f"Resamples: {cx.args.n_boot}, seed {cx.args.seed}.", "", "## Tables", ""]
    lines.append("| table | rows | jobs that fed it | source files | what |")
    lines.append("|---|---|---|---|---|")
    for name, t in rep.tables.items():
        lines.append(f"| {name}.csv | {t['rows']} | {', '.join(t['jobs']) or '-'} | {', '.join(os.path.basename(f) for f in t['files']) or '-'} | {t['desc']} |")
    lines += ["", "## Result files missing or unreadable", ""]
    absent = [f"{j}{ext}" for j in RUN.JOBS for ext in (".json", "_logits.npz")
              if not os.path.exists(L.path(f"{j}{ext}"))]
    for f in absent:
        lines.append(f"- MISSING: {f}")
    for f, why in L.unreadable.items():
        lines.append(f"- UNREADABLE: {f}: {why} (see errors)")
    if not absent and not L.unreadable:
        lines.append("None.")
    lines += ["", "## Planned items missing or skipped", ""]
    planned = list(RUN.JOBS)
    got = sorted({j for t in rep.tables.values() for j in t["jobs"]})
    miss_jobs = [(j, L.missing.get(j, "not used by any analysis (no result file found)")) for j in planned if j not in got]
    if not miss_jobs and not rep.skipped:
        lines.append("None.")
    for j, why in miss_jobs:
        lines.append(f"- job {j}: {why}")
    for item, why in rep.skipped:
        lines.append(f"- {item}: {why}")
    lines += ["", "## Notes and checks", ""]
    for n in rep.notes + rep.checks:
        lines.append(f"- {n}")
    if not (rep.notes or rep.checks):
        lines.append("None.")
    if rep.errors:
        lines += ["", "## ERRORS (analysis failed; its tables are missing)", ""]
        for e in rep.errors:
            lines.append(f"```\n{e}\n```")
    lines += ["", "## Definitions", "", DEFINITIONS]
    with open(os.path.join(cx.args.out_dir, "analysis_report.md"), "w") as f:
        f.write("\n".join(lines))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="WasteVision analyses (CPU)")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data-dir", required=True, help="folder with splits_v3.csv and the rebuild_splits.py outputs")
    ap.add_argument("--features", default=None, help="frozen_features npz with keys ids, feats (object-or-dataset)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mix", default=None, help="JSON (or path to a JSON file): stream -> share, for reweighted stream accuracy")
    ap.add_argument("--probe-min-per-cell", type=int, default=50, help="min training images of a class in a source to use it in the probes")
    ap.add_argument("--probe-cell-cap", type=int, default=800, help="max images per class-source cell")
    ap.add_argument("--probe-max-neg", type=int, default=4000, help="max negatives for a one-vs-rest probe")
    ap.add_argument("--probe-max-iter", type=int, default=1000)
    ap.add_argument("--probe-pca", type=int, default=0, help="PCA components before the probe (0 = none)")
    return ap.parse_args(argv)


STEPS = [("main results", analyse_main), ("stream model S1", analyse_stream_model),
         ("decision designs", analyse_decisions), ("unseen datasets", analyse_unseen_datasets),
         ("leakage", analyse_leakage), ("object or dataset", analyse_object_or_dataset),
         ("scenarios", analyse_scenarios), ("data tables", analyse_data_tables), ("figures", make_figures)]


def main(argv=None):
    t0 = time.time()
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    cx = Ctx(args, Results(args.results_dir), Report(), Numbers())
    for name, fn in STEPS:
        t1 = time.time()
        print(f"== {name}", flush=True)
        try:
            fn(cx)
        except Exception:  # noqa: BLE001 - one failing analysis must not hide the others; it is reported and the exit code is 1
            tb = traceback.format_exc()
            cx.rep.errors.append(f"{name}:\n{tb}")
            print(tb, flush=True)
        print(f"   ({time.time() - t1:.1f} s)", flush=True)
    for fname, why in cx.L.unreadable.items():
        cx.rep.errors.append(f"result file {fname} exists but cannot be read: {why}")
    with open(os.path.join(args.out_dir, "paper_numbers.json"), "w") as f:
        json.dump(dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), n_boot=args.n_boot, seed=args.seed,
                       numbers=cx.pn.d), f, indent=1)
    write_report(cx, t0, "analyse.py " + " ".join(sys.argv[1:] if argv is None else argv))
    print(f"done in {time.time() - t0:.1f} s; {len(cx.pn.d)} paper numbers; {len(cx.rep.errors)} errors", flush=True)
    return 1 if cx.rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())

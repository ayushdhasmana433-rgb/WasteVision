#!/usr/bin/env python3
"""Extra analyses from saved files only (no training). Reuses analyse.py (Boot, ci, job_scores, metric_stats,
val_threshold, _align) and run.py. Writes extra_*.csv to an_out. Grouped bootstrap: 2000 resamples, seed 42."""
import json, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse as A

RUN, SC, DRY = A.RUN, A.SC, A.DRY
BASE = "/home/claude/wv3"
RDIR, OUT = f"{BASE}/an_in/results", f"{BASE}/an_out"
NB, SEED = 2000, 42
L = A.Results(RDIR)
HAZ = A.HAZARD_CLASSES
ALPHAS = [0.01, 0.02, 0.03]
CLIP_KEYS = A.CLIP_KEYS


def B(groups, name):
    return A.Boot(groups, NB, SEED, name)


def row(**kw):
    return kw


def get(job, keys=A.NPZ_KEYS):
    r = L.get(job, keys=keys)
    if r is None:
        raise RuntimeError(f"{job}: {L.missing.get(job)}")
    return r


def clip_sc(e1):
    """E1 scores in the dict format of analyse.job_scores (probabilities are the stored CLIP probabilities)."""
    z = e1.z
    sp = RUN.class_space(A.CLASSES)
    assert [str(x) for x in z["label_names"]] == list(A.CLASSES)
    p = z["class_probs"].astype(np.float64)
    ssp = z["stream_probs"].astype(np.float64)
    return dict(space=sp, p=p, p_raw=p, sp=ssp, haz=ssp[:, SC], s_pred=ssp.argmax(1), lg=z["logits"])


# ---------------------------------------------------------------- 1
def task1():
    e1 = get("E1", CLIP_KEYS)
    ssum = e1.z["stream_probs"][:, SC].astype(float)
    rows = []
    for bj in ["B1", "B2", "B3", "B4"]:
        rb = get(bj)
        z = rb.z
        c = rb.j.get("held_out_class") or RUN.JOBS[bj].holdout_class
        thr = rb.j["thresholds"]["hazard"]
        sc = A.job_scores(z)
        known = (z["role"] == "test") & (z["tier"] == "T1") & (z["stream_true"] != SC)
        unk = (z["role"] == "unknown") & (z["tier"] == "T1")
        sel = np.r_[np.nonzero(known)[0], np.nonzero(unk)[0]]
        nk, nu = int(known.sum()), int(unk.sum())
        is_unk = np.r_[np.zeros(nk, bool), np.ones(nu, bool)]
        ids = z["ids"][sel].astype(str)
        pos = A._align(e1.z, ids)
        assert pos is not None, f"E1 lacks ids of {bj}"
        d1 = sc["haz"][sel] >= thr
        clip = ssum[pos]
        kve = (e1.z["role"] == "val") & (e1.z["stream_true"] != SC) & (e1.z["class_idx"] != RUN.CLASS_TO_IDX[c])
        vs = ssum[kve]
        boot = B(z["group"][sel], f"decision_{c}")   # same name as analyse.py -> same resamples
        meas = "catch" if c in HAZ else "wrongly_flagged"

        def add(design, alpha, tau, vfar, flag):
            base = dict(held_out_class=c, job=bj, design=design, alpha=alpha, clip_threshold=tau,
                        clip_val_far_realised=vfar, n_val_known=len(vs), n_known=nk, n_unknown=nu, d1_threshold=thr)
            for m, num, den, n in ((meas, flag & is_unk, is_unk, nu), ("known_false_alarm", flag & ~is_unk, ~is_unk, nk)):
                v, _ = boot.ratio(num, den)
                p, lo, hi = A.ci(v)
                rows.append(dict(base, measure=m, value=p, ci_lo=lo, ci_hi=hi, n=n))
        add("D1 alone", None, None, None, d1)
        for a in ALPHAS:
            tau, vfar = A.val_threshold(vs, a)
            cf = clip >= tau
            add(f"CLIP alone (val FAR {a:.0%})", a, tau, vfar, cf)
            add(f"D1 OR CLIP (val FAR {a:.0%})", a, tau, vfar, d1 | cf)
    df = pd.DataFrame(rows)
    df["source_files"] = A.src_str([f"{RDIR}/{j}{s}" for j in ("B1", "B2", "B3", "B4", "E1") for s in (".json", "_logits.npz")])
    df.to_csv(f"{OUT}/extra_combined_flag.csv", index=False)
    return df


# ---------------------------------------------------------------- 2
def task2():
    e1, a1 = get("E1", CLIP_KEYS), get("A1")
    cs, asc = clip_sc(e1), A.job_scores(a1.z)
    rows, chk = [], []
    for name, r, sc, thr in (("E1 CLIP", e1, cs, RUN.SC_THRESHOLD), ("A1", a1, asc, a1.j["hazard_threshold"])):
        z = r.z
        y = z["label_idx"] if name == "A1" else z["class_idx"]
        for tier in ("T2", "T3"):
            m = (z["role"] == "test") & (z["tier"] == tier)
            if not m.any():
                continue
            boot = B(z["group"][m], f"extra_tier_{tier}")
            sub = {k: (v[m] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(y) else v) for k, v in sc.items()}
            st = A.metric_stats(boot, sub, y[m], z["stream_true"][m], thr)
            blk = (r.j.get("metrics") or {}).get(f"test_{tier}", {})
            tk = RUN._thr_key(thr)
            for k in ("class_acc", "stream_acc", f"sc_false_alarm@{tk}", f"sc_recall@{tk}"):
                v, n = st[k]
                p, lo, hi = A.ci(v)
                jv = blk.get(k)
                rows.append(dict(model=name, tier=tier, metric=k.split("@")[0] + "@0.20" if "@" in k else k, value=p,
                                 ci_lo=lo, ci_hi=hi, n=n, json_value=jv,
                                 abs_diff_vs_json=abs(p - jv) if jv is not None else None,
                                 n_items=int(m.sum()), n_special_care=int((z["stream_true"][m] == SC).sum())))
    df = pd.DataFrame(rows)
    df["source_files"] = A.src_str([f"{RDIR}/{j}{s}" for j in ("A1", "E1") for s in (".json", "_logits.npz")])
    df.to_csv(f"{OUT}/extra_clip_tiers.csv", index=False)
    return df


# ---------------------------------------------------------------- 3
def task3():
    rows = []
    a1 = get("A1")
    z = a1.z
    sc = A.job_scores(z)
    m = (z["role"] == "test") & (z["tier"] == "T1")
    boot = B(z["group"][m], "main_T1")
    is_sc = z["stream_true"][m] == SC
    h = sc["haz"][m]
    for r_ in (2, 4, 9):
        t = 1 / (1 + r_)
        for meas, num, den in (("sc_recall", (h >= t) & is_sc, is_sc), ("sc_false_alarm", (h >= t) & ~is_sc, ~is_sc)):
            v, n = boot.ratio(num, den)
            p, lo, hi = A.ci(v)
            rows.append(dict(job="A1", scenario="A1 on T1 test", cost_ratio=r_, threshold=t, measure=meas, value=p,
                             ci_lo=lo, ci_hi=hi, n=n))
    for bj in ("B1", "B2", "B3", "B4"):
        rb = get(bj)
        z = rb.z
        c = rb.j.get("held_out_class") or RUN.JOBS[bj].holdout_class
        sc = A.job_scores(z)
        known = (z["role"] == "test") & (z["tier"] == "T1") & (z["stream_true"] != SC)
        unk = (z["role"] == "unknown") & (z["tier"] == "T1")
        sel = np.r_[np.nonzero(known)[0], np.nonzero(unk)[0]]
        nk, nu = int(known.sum()), int(unk.sum())
        is_unk = np.r_[np.zeros(nk, bool), np.ones(nu, bool)]
        boot = B(z["group"][sel], f"decision_{c}")
        h = sc["haz"][sel]
        meas = "catch" if c in HAZ else "wrongly_flagged"
        for r_ in (2, 4, 9):
            t = 1 / (1 + r_)
            for mn, num, den, n in ((meas, (h >= t) & is_unk, is_unk, nu), ("known_false_alarm", (h >= t) & ~is_unk, ~is_unk, nk)):
                v, _ = boot.ratio(num, den)
                p, lo, hi = A.ci(v)
                rows.append(dict(job=bj, scenario=f"D1 on held-out {c}", cost_ratio=r_, threshold=t, measure=mn, value=p,
                                 ci_lo=lo, ci_hi=hi, n=n))
    df = pd.DataFrame(rows)
    df["source_files"] = A.src_str([f"{RDIR}/{j}{s}" for j in ("A1", "B1", "B2", "B3", "B4") for s in (".json", "_logits.npz")])
    df.to_csv(f"{OUT}/extra_cost_ratio.csv", index=False)
    return df


def fmt(df, cols):
    d = df.copy()
    d["val"] = [f"{v*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]" if pd.notna(v) and pd.notna(lo) else
                (f"{v*100:.1f}" if pd.notna(v) else "NA") for v, lo, hi in zip(d.value, d.ci_lo, d.ci_hi)]
    return d[cols + ["val"]].to_string(index=False)


if __name__ == "__main__":
    d1 = task1(); d2 = task2(); d3 = task3()
    print("\n== 1. combined flag (%)"); print(fmt(d1, ["held_out_class", "design", "clip_threshold", "measure", "n"]))
    print("\n== 2. CLIP tiers (%)"); print(fmt(d2, ["model", "tier", "metric", "n", "json_value"]))
    print("max |point - JSON|:", d2.abs_diff_vs_json.max())
    print("\n== 3. cost ratio (%)"); print(fmt(d3, ["job", "cost_ratio", "threshold", "measure", "n"]))

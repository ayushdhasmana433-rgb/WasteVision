"""Tests for pipeline/analyse.py on a fake results folder (tests/make_fake_results.py).

Numbers are recomputed here with plain numpy (not with analyse.py or run.py code) and compared with the CSV files.
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, "..", "pipeline")
ANALYSE = os.path.join(PIPE, "analyse.py")
N_BOOT = 400


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FAKE = _load("make_fake_results_t", os.path.join(HERE, "make_fake_results.py"))
AN = _load("analyse_t", ANALYSE)

# plain-numpy knowledge of the task, written out here on purpose
CLASS_NAMES = ["biological", "paper_cardboard", "plastic", "metal", "glass", "clothes", "shoes", "trash", "battery",
               "medicines_syringes", "e_waste", "sanitary"]
STREAM_OF_CLASS = np.array([0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3])        # Wet, Dry, Special Care, Sanitary
SC_STREAM = 2


def run_analyse(results, data, out, extra=()):
    cmd = [sys.executable, ANALYSE, "--results-dir", results, "--data-dir", data, "--out-dir", out,
           "--n-boot", str(N_BOOT), "--probe-min-per-cell", "20", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("fake"))
    paths = FAKE.build(root)
    out = os.path.join(root, "out")
    mix = json.dumps({"Wet": 0.4, "Dry": 0.4, "Special Care": 0.05, "Sanitary": 0.15})
    p = run_analyse(paths["results"], paths["data"], out, ["--features", paths["features"], "--mix", mix])
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    paths.update(root=root, out=out, stdout=p.stdout)
    return paths


def table(fake, name):
    return pd.read_csv(os.path.join(fake["out"], f"{name}.csv"))


def npz(fake, job):
    z = np.load(os.path.join(fake["results"], f"{job}_logits.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def stream_probs(z):
    """Summed stream probabilities, from the logits and temperature, plain numpy."""
    p = softmax(z["logits"].astype(np.float64) / float(z["temperature"]), axis=1)
    names = [str(n) for n in z["label_names"]]
    if names == ["Wet", "Dry", "Special Care", "Sanitary"]:
        return p
    sp = np.zeros((len(p), 4))
    for k, n in enumerate(names):
        sp[:, STREAM_OF_CLASS[CLASS_NAMES.index(n)]] += p[:, k]
    return sp


def test_outputs_exist(fake):
    out = fake["out"]
    for f in ["paper_numbers.json", "analysis_report.md", "main_per_seed.csv", "main_mean_sd.csv", "main_bootstrap.csv",
              "decision_designs.csv", "unseen_datasets.csv", "leakage.csv", "object_or_dataset_source_probe.csv",
              "object_or_dataset_cross_source.csv", "scenario_flag_precision.csv", "scenario_per_10000.csv",
              "scenario_reweighted_stream_accuracy.csv", "data_count_per_class.csv", "data_count_per_source.csv",
              "data_dup_pairs_by_distance.csv", "data_dup_summary.csv", "data_conflicts_by_reason.csv",
              "data_source_overlap.csv", "stream_model_S1.csv"]:
        assert os.path.exists(os.path.join(out, f)), f
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    for f in ["fig_decision_designs.png", "fig_unseen_datasets.png", "fig_flag_precision.png", "fig_source_overlap.png"]:
        assert os.path.getsize(os.path.join(out, f)) > 5000, f
    rep = open(os.path.join(out, "analysis_report.md")).read()
    assert "decision_designs.csv" in rep and "Planned items missing" in rep


def test_best_seed_is_lowest_validation_loss(fake):
    mb = table(fake, "main_bootstrap")
    assert set(mb.job) == {"A2"} and set(mb.seed) == {7}          # A2 has the lowest validation loss in the fake files


def test_main_t1_stream_accuracy_matches_recomputation(fake):
    # bootstrap table (seed A2) and per-seed table (A1, from the JSON)
    for job, tab, tol in (("A2", "main_bootstrap", 1e-9), ("A1", "main_per_seed", 2e-3)):
        z = npz(fake, job)
        m = (z["role"] == "test") & (z["tier"] == "T1")
        acc = float((stream_probs(z)[m].argmax(1) == z["stream_true"][m]).mean())
        t = table(fake, tab)
        row = t[(t.tier == "T1") & (t.metric == "stream_acc") & (t.job == job)]
        assert len(row) == 1
        assert abs(row.value.iloc[0] - acc) < tol, (job, row.value.iloc[0], acc)
        if tab == "main_bootstrap":
            assert int(row.n.iloc[0]) == int(m.sum())
    # mean and SD across the three seeds
    ps = table(fake, "main_per_seed")
    v = ps[(ps.tier == "T1") & (ps.metric == "stream_acc")].value.values
    ms = table(fake, "main_mean_sd")
    r = ms[(ms.tier == "T1") & (ms.metric == "stream_acc")].iloc[0]
    assert len(v) == 3 and abs(r.value - v.mean()) < 1e-12 and abs(r.sd - v.std(ddof=1)) < 1e-12 and int(r.n) == 3


def test_main_bootstrap_agrees_with_run_json(fake):
    mb = table(fake, "main_bootstrap").dropna(subset=["abs_diff_vs_json"])
    assert len(mb) > 30 and mb.abs_diff_vs_json.max() < 1e-6


def _matched_catch(known, unknown, target_far):
    """Plain recomputation: flag the k highest known scores (k = floor(target * n)), threshold = k-th highest."""
    s = np.sort(known)[::-1]
    k = int(math.floor(target_far * len(known) + 1e-9))
    assert s[k - 1] > s[k], "tie at the boundary in the fake data"
    thr = s[k - 1]
    return float((unknown >= thr).mean()), k / len(known), k


def _b_arrays(fake, job):
    z = npz(fake, job)
    known = (z["role"] == "test") & (z["tier"] == "T1") & (z["stream_true"] != SC_STREAM)
    unk = (z["role"] == "unknown") & (z["tier"] == "T1")
    sp = stream_probs(z)
    p = softmax(z["logits"].astype(np.float64) / float(z["temperature"]), axis=1)
    return z, known, unk, sp[:, SC_STREAM], p.max(1)


@pytest.mark.parametrize("target,op", [(0.05, "test_match_5pct"), (0.10, "test_match_10pct")])
def test_matched_catch_rate_and_realised_false_alarm(fake, target, op):
    d = table(fake, "decision_designs")
    z, known, unk, haz, msp = _b_arrays(fake, "B1")
    for design, known_score, unk_score in (("D2 hazard-sum flag", haz[known], haz[unk]), ("D3 MSP", -msp[known], -msp[unk])):
        catch, far, k = _matched_catch(known_score, unk_score, target)
        sel = d[(d.held_out_class == "battery") & (d.design == design) & (d.op_point == op)]
        got = sel[sel.measure == "catch"].iloc[0]
        far_row = sel[sel.measure == "known_false_alarm"].iloc[0]
        assert abs(got.value - catch) < 1e-9, (design, got.value, catch)
        assert int(got.n) == int(unk.sum())
        n_known = int(known.sum())
        assert int(far_row.n) == n_known == int(got.n_known)
        # realised false-alarm rate within one item of the target
        assert abs(far_row.value * n_known - target * n_known) <= 1.0 + 1e-9
        assert abs(far_row.value * n_known - k) < 1e-9


def test_matched_to_D2_far_reproduces_D2_own_point(fake):
    d = table(fake, "decision_designs")
    for c in ("battery", "medicines_syringes", "e_waste", "shoes"):
        s = d[(d.held_out_class == c) & (d.design == "D2 hazard-sum flag")]
        own = s[s.op_point == "own"].set_index("measure").value
        m = s[s.op_point == "test_match_D2"].set_index("measure").value
        meas = [x for x in own.index if x in ("catch", "wrongly_flagged")][0]
        assert abs(own[meas] - m[meas]) < 1e-12 and abs(own["known_false_alarm"] - m["known_false_alarm"]) < 1e-12
        # every sweepable design realises at most D2's false-alarm rate at match_D2 and at least one item less than it
        o = d[(d.held_out_class == c) & (d.op_point == "test_match_D2") & (d.measure == "known_false_alarm")]
        n_known = int(o.n.iloc[0])
        assert (o.value * n_known <= own["known_false_alarm"] * n_known + 1e-9).all()
        assert (o.value * n_known >= own["known_false_alarm"] * n_known - 3).all()     # ties can cost an item or two


def test_known_answers_in_decision_table(fake):
    d = table(fake, "decision_designs")

    def val(c, design, op, measure="catch"):
        return d[(d.held_out_class == c) & (d.design == design) & (d.op_point == op) & (d.measure == measure)].value.iloc[0]
    # B1: D2 clearly better than MSP, MaxLogit and Energy at matched false-alarm rates
    for op in ("val_match_5pct", "val_match_10pct", "test_match_5pct", "test_match_10pct"):
        for other in ("D3 MSP", "D3 MaxLogit", "D3 Energy"):
            assert val("battery", "D2 hazard-sum flag", op) - val("battery", other, op) > 0.3
    row = d[(d.held_out_class == "battery") & (d.design == "D3 MSP") & (d.op_point == "val_match_5pct") & (d.measure == "diff_D2_minus_design")].iloc[0]
    assert row.ci_lo > 0 and row.ci_lo <= row.value <= row.ci_hi
    # B2: CLIP clearly worse than D2
    for design in ("D5 CLIP class-prompt sum", "D5 CLIP hazard prompt"):
        for op in ("val_match_5pct", "test_match_5pct"):
            assert val("medicines_syringes", "D2 hazard-sum flag", op) - val("medicines_syringes", design, op) > 0.3
    # shoes (harmless control): few wrongly flagged at the matched points
    assert val("shoes", "D2 hazard-sum flag", "val_match_5pct", "wrongly_flagged") < 0.2
    # D4 and D5 rows exist for all three hazards (same image ids)
    for c in ("battery", "medicines_syringes", "e_waste"):
        names = set(d[d.held_out_class == c].design)
        assert {"D1 top-class routing", "D2 hazard-sum flag", "D3 MSP", "D4 stream-trained", "D5 CLIP hazard prompt"} <= names


def test_own_points_agree_with_run_json(fake):
    """D2 own catch = run.py's flagged_special_care on T1; MSP own = rejected; D2+reject = rejected_or_flagged."""
    d = table(fake, "decision_designs")
    for job, c in (("B1", "battery"), ("B2", "medicines_syringes"), ("B3", "e_waste")):
        u = json.load(open(os.path.join(fake["results"], f"{job}.json")))["unknown_T1"]
        own = d[(d.held_out_class == c) & (d.op_point == "own") & (d.measure == "catch")].set_index("design").value
        assert abs(own["D2 hazard-sum flag"] - u["flagged_special_care"]) < 1e-9
        assert abs(own["D3 MSP"] - u["rejected"]) < 1e-9
        assert abs(own["D3 Energy"] - u["rejected_Energy"]) < 1e-9
        assert abs(own["D3 MaxLogit"] - u["rejected_MaxLogit"]) < 1e-9
        assert abs(own["D2+reject(MSP)"] - u["rejected_or_flagged"]) < 1e-9


def test_flag_precision_formula(fake):
    t = table(fake, "scenario_flag_precision").dropna(subset=["value"])
    assert len(t) >= 15 and set(t.hazard_prevalence) == {0.01, 0.02, 0.05}
    R, F, pi = t.recall.values, t.false_alarm_rate.values, t.hazard_prevalence.values
    assert np.allclose(t.value.values, R * pi / (R * pi + F * (1 - pi)), atol=1e-12)
    # and the A-job numbers it starts from are the T1 numbers of the bootstrap seed
    mb = table(fake, "main_bootstrap")
    rec = mb[(mb.tier == "T1") & (mb.metric == "sc_recall@0.20")].value.iloc[0]
    far = mb[(mb.tier == "T1") & (mb.metric == "sc_false_alarm@0.20")].value.iloc[0]
    a = t[(t.recall_basis == "A_T1 flag recall") & (t.false_alarm_basis == "A_T1 false-alarm rate")]
    assert np.allclose(a.recall, rec) and np.allclose(a.false_alarm_rate, far)
    assert np.allclose(a.value.values, rec * a.hazard_prevalence.values / (rec * a.hazard_prevalence.values + far * (1 - a.hazard_prevalence.values)))
    # hand-computed example
    r = a[a.hazard_prevalence == 0.01].iloc[0]
    assert abs(r.value - 0.01 * rec / (0.01 * rec + 0.99 * far)) < 1e-12


def test_per_10000_and_reweighted_accuracy(fake):
    t = table(fake, "scenario_per_10000")
    a = t[(t.basis == "A_T1") & (t.hazard_prevalence == 0.02)].set_index("metric").value
    assert abs(a["hazards_present_per_10000"] - 200) < 1e-9
    assert abs(a["hazards_flagged_per_10000"] + a["hazards_not_flagged_per_10000"] - 200) < 1e-9
    base = t[(t.basis == "always-dry baseline") & (t.hazard_prevalence == 0.02)]
    assert abs(base.value.iloc[0] - 200) < 1e-9
    mb = table(fake, "main_bootstrap")
    rec = {s: mb[(mb.tier == "T1") & (mb.metric == f"recall_{s}")].value.iloc[0] for s in ("Wet", "Dry", "Special Care", "Sanitary")}
    want = 0.4 * rec["Wet"] + 0.4 * rec["Dry"] + 0.05 * rec["Special Care"] + 0.15 * rec["Sanitary"]
    assert abs(table(fake, "scenario_reweighted_stream_accuracy").value.iloc[0] - want) < 1e-12


def test_leakage_difference(fake):
    t = table(fake, "leakage").set_index(["job", "metric"])
    for metric, col in (("class_acc", None), ("stream_acc", None)):
        accs = {}
        for job in ("C1", "C2"):
            z = npz(fake, job)
            m = z["role"] == "test"
            if metric == "class_acc":
                accs[job] = float((z["logits"][m].argmax(1) == z["label_idx"][m]).mean())
            else:
                accs[job] = float((stream_probs(z)[m].argmax(1) == z["stream_true"][m]).mean())
            assert abs(t.loc[(job, metric), "value"] - accs[job]) < 1e-9
        d = t.loc[("C1-C2", f"diff_{metric}")]
        assert abs(d.value - (accs["C1"] - accs["C2"])) < 1e-9
        assert accs["C1"] - accs["C2"] > 0.05                      # the margin built into the fake files
        assert d.ci_lo > 0 and d.ci_lo <= d.value <= d.ci_hi
    # near-duplicate share against an independent count from the JSON block
    lk = json.load(open(os.path.join(fake["results"], "C1.json")))["leakage"]
    share = t.loc[("C1", "share_test_with_train_near_duplicate")]
    assert abs(share.value - lk["n_with_duplicate"] / lk["n_test"]) < 1e-12 and share.ci_lo < share.value < share.ci_hi
    assert t.loc[("C1", "acc_items_with_duplicate"), "n"] == lk["n_with_duplicate"]


def test_unseen_dataset_table(fake):
    t = table(fake, "unseen_datasets")
    assert set(t.held_out_source) == {"MedPackID", "TrashBox", "GD", "RealWaste"}
    r = t[t.held_out_source == "RealWaste"]
    assert set(r.unseen_classes) == {"biological"}                    # biological only exists in RealWaste in the fake data
    assert (t[t.held_out_source == "GD"].unseen_classes == "none").all()
    assert "left out of training (images of classes never trained)" in set(r.condition)
    assert "same source in training (A1, T1 test rows)" in set(t.condition)
    # left-out stream accuracy recomputed from the B8 file
    z = npz(fake, "B8")
    m = z["role"] == "unknown"
    acc = float((stream_probs(z)[m].argmax(1) == z["stream_true"][m]).mean())
    row = r[(r.condition == "left out of training") & (r.metric == "stream_acc")].iloc[0]
    assert abs(row.value - acc) < 1e-9 and int(row.n) == int(m.sum())


def test_object_or_dataset_tables(fake):
    a = table(fake, "object_or_dataset_source_probe")
    acc = a[a.metric == "source_probe_accuracy"]
    base = a[a.metric == "majority_source_baseline"]
    assert len(acc) >= 5 and (acc.value.values >= base.value.values - 1e-9).all()      # sources are separable in the fake features
    assert (acc.n >= 40).all()
    b = table(fake, "object_or_dataset_cross_source")
    gap = b[(b.metric == "gap_within_minus_cross") & (~b.trained_on.str.startswith("("))]
    assert len(gap) > 10 and gap.value.between(-1, 1).all()


def test_data_tables(fake):
    ct = pd.read_csv(os.path.join(fake["data"], "count_table.csv"))
    pc = table(fake, "data_count_per_class").set_index("label")
    assert int(pc.loc["battery", "after_cap"]) == int(ct[ct.label == "battery"].after_cap.sum())
    pairs = pd.read_csv(os.path.join(fake["data"], "dup_pairs_phash.csv.gz"))
    s = table(fake, "data_dup_summary").set_index("measure").value
    assert int(s["pairs"]) == len(pairs) and int(s["cross_source_pairs"]) == int((pairs.source_a != pairs.source_b).sum())
    conf = pd.read_csv(os.path.join(fake["data"], "label_conflicts.csv"))
    assert int(table(fake, "data_conflicts_by_reason").value.sum()) == len(conf)


def test_intervals_contain_point_estimate(fake):
    n_checked = 0
    for f in os.listdir(fake["out"]):
        if not f.endswith(".csv"):
            continue
        t = pd.read_csv(os.path.join(fake["out"], f))
        if not {"value", "ci_lo", "ci_hi"} <= set(t.columns):
            continue
        s = t.dropna(subset=["value", "ci_lo", "ci_hi"])
        bad = s[(s.ci_lo > s.value + 1e-9) | (s.ci_hi < s.value - 1e-9)]
        assert bad.empty, (f, bad.head())
        n_checked += len(s)
    assert n_checked > 500


def test_paper_numbers_have_sources_and_valid_intervals(fake):
    pn = json.load(open(os.path.join(fake["out"], "paper_numbers.json")))["numbers"]
    assert len(pn) > 500
    for key, e in pn.items():
        assert e["source_files"], key
        for f in e["source_files"]:
            assert os.path.exists(f), (key, f)
        assert e["value"] is not None and e["table"]
        if e["ci95_lo"] is not None:
            assert e["ci95_lo"] - 1e-9 <= e["value"] <= e["ci95_hi"] + 1e-9, key
    k = "main_bootstrap|tier=T1|metric=stream_acc"
    assert k in pn and pn[k]["n"] and pn[k]["ci95_lo"] is not None
    assert any(k.startswith("decision_designs|held_out_class=battery|design=D2") for k in pn)


def test_missing_jobs_are_reported_not_fatal(fake, tmp_path):
    res = str(tmp_path / "results")
    shutil.copytree(fake["results"], res)
    for j in ("S1", "S2", "S3", "S4", "E1"):
        os.remove(os.path.join(res, f"{j}.json"))
        os.remove(os.path.join(res, f"{j}_logits.npz"))
    out = str(tmp_path / "out")
    p = run_analyse(res, fake["data"], out)                       # also without --features
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    rep = open(os.path.join(out, "analysis_report.md")).read()
    for j in ("S1", "S2", "S3", "S4", "E1"):
        assert f"job {j}:" in rep, j
    assert "no --features" in rep
    d = pd.read_csv(os.path.join(out, "decision_designs.csv"))
    assert not d.design.str.startswith(("D4", "D5")).any() and (d.design == "D2 hazard-sum flag").any()
    assert not os.path.exists(os.path.join(out, "stream_model_S1.csv"))
    assert not os.path.exists(os.path.join(out, "object_or_dataset_source_probe.csv"))
    # numbers that exist are unchanged by the deletion (same seed, same resamples)
    a = pd.read_csv(os.path.join(out, "decision_designs.csv"))
    b = table(fake, "decision_designs")
    k = ["held_out_class", "design", "op_point", "measure"]
    m = a.merge(b, on=k, suffixes=("_a", "_b"))
    assert len(m) == len(a) and np.allclose(m.value_a.fillna(-1), m.value_b.fillna(-1))


def test_only_c_jobs_present_and_empty_folder(fake, tmp_path):
    res = str(tmp_path / "results")
    os.makedirs(res)
    for j in ("C1", "C2"):
        for ext in (".json", "_logits.npz"):
            shutil.copy(os.path.join(fake["results"], j + ext), res)
    out = str(tmp_path / "out")
    p = run_analyse(res, str(tmp_path / "no_data"), out)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert os.path.exists(os.path.join(out, "leakage.csv"))
    rep = open(os.path.join(out, "analysis_report.md")).read()
    assert "job A1:" in rep and "job B1:" in rep and "main results (A1-A3)" in rep
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    p = run_analyse(empty, str(tmp_path / "no_data"), str(tmp_path / "out2"))
    assert p.returncode == 0
    assert json.load(open(tmp_path / "out2" / "paper_numbers.json"))["numbers"] == {}


def test_bad_mix_is_reported_not_fatal(fake, tmp_path):
    out = str(tmp_path / "out")
    p = run_analyse(fake["results"], fake["data"], out, ["--mix", '{"Wet":0.9,"Dry":0.9}'])
    assert p.returncode == 0
    assert "bad --mix" in open(os.path.join(out, "analysis_report.md")).read()
    assert not os.path.exists(os.path.join(out, "scenario_reweighted_stream_accuracy.csv"))


def test_frozen_file_is_passed_through(fake, tmp_path):
    res = str(tmp_path / "results")
    shutil.copytree(fake["results"], res)
    pd.DataFrame(dict(fold=["RealWaste", "GD"], acc=[0.5, 0.6])).to_csv(os.path.join(res, "F_lodo.csv"), index=False)
    out = str(tmp_path / "out")
    assert run_analyse(res, fake["data"], out).returncode == 0
    t = pd.read_csv(os.path.join(out, "frozen_F_lodo.csv"))
    assert list(t.acc) == [0.5, 0.6] and t.source_files.str.contains("F_lodo.csv").all()


def test_corrupt_job_file_is_skipped(fake, tmp_path):
    res = str(tmp_path / "results")
    shutil.copytree(fake["results"], res)
    with open(os.path.join(res, "B2_logits.npz"), "wb") as f:
        f.write(b"not an npz")
    out = str(tmp_path / "out")
    p = run_analyse(res, fake["data"], out)
    assert p.returncode == 1                                      # exists but unreadable: an error
    rep = open(os.path.join(out, "analysis_report.md")).read()
    assert "job B2:" in rep and "UNREADABLE: B2_logits.npz" in rep and "## ERRORS" in rep
    d = pd.read_csv(os.path.join(out, "decision_designs.csv"))
    assert "medicines_syringes" not in set(d.held_out_class)


# ---------------------------------------------------------------- unit tests of the statistics
def test_boot_point_row_and_matrix():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 50, 400)
    x = rng.random(400) < 0.3
    b = AN.Boot(g, 300, 1, "t")
    v, n = b.ratio(x, np.ones(400))
    assert n == 400 and abs(v[0] - x.mean()) < 1e-12 and b.M.shape == (301, len(np.unique(g)))
    assert (b.M[1:].sum(1) == b.G).all()                           # every resample draws G groups
    lo, hi = np.percentile(v[1:], [2.5, 97.5])
    assert lo < x.mean() < hi
    p, a, c = AN.ci(v)
    assert a == lo and c == hi and p == v[0]


def test_ece_and_macro_f1_match_run():
    rng = np.random.default_rng(1)
    n, K = 500, 6
    p = softmax(rng.normal(size=(n, K)) * 2, axis=1)
    y = rng.integers(0, K, n)
    y[:30] = -1
    valid = y >= 0
    b = AN.Boot(rng.integers(0, 80, n), 10, 2, "e")
    e, nv = AN.ece_stat(b, p, y, valid)
    assert abs(e[0] - AN.RUN.ece(p[valid], y[valid])) < 1e-12 and nv == int(valid.sum())
    sc = dict(space=AN.RUN.class_space(CLASS_NAMES[:K]), p=p, p_raw=p, s_pred=np.zeros(n, int), haz=p[:, 0])
    st = AN.metric_stats(b, sc, y, np.zeros(n, int), 0.2)
    assert abs(st["macro_f1"][0][0] - AN.RUN.macro_f1(y[valid], p.argmax(1)[valid])) < 1e-12
    assert abs(st["class_acc"][0][0] - (p.argmax(1)[valid] == y[valid]).mean()) < 1e-12


def test_sweep_to_far_with_ties_and_weights():
    known = np.array([5, 4, 4, 3, 3, 3, 2, 1, 1, 0.5], float)
    unk = np.array([6, 4, 3, 2.5, 0.2], float)
    ones_k, ones_u = np.ones((1, len(known))), np.ones((1, len(unk)))
    # target 0.2 = 2 items: the first block (5) is one item, the tie block (4, 4) would make 3 -> only the 5 is flagged
    o = AN.sweep_to_far(known, unk, ones_k, ones_u, [0.2])[0]
    assert o["far"][0] == 0.1 and o["thr"][0] == 5 and abs(o["catch"][0] - 1 / 5) < 1e-12
    o = AN.sweep_to_far(known, unk, ones_k, ones_u, [0.3])[0]
    assert o["far"][0] == 0.3 and o["thr"][0] == 4 and abs(o["catch"][0] - 2 / 5) < 1e-12
    o = AN.sweep_to_far(known, unk, ones_k, ones_u, [0.0])[0]
    assert o["far"][0] == 0 and np.isinf(o["thr"][0]) and o["catch"][0] == 0
    # zero-weight known items must not lower the threshold below the highest one that gives the same weighted rate:
    # items 5 and the first 4 have weight 0, so a target of 0 is met by thr = 5 (the 4s then carry weight 1 and stay unflagged)
    w = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], float)
    o = AN.sweep_to_far(known, unk, w, ones_u, [0.0])[0]
    assert o["far"][0] == 0 and o["thr"][0] == 5 and abs(o["catch"][0] - 1 / 5) < 1e-12


# ---------------------------------------------------------------- review fixes
def _val_threshold_plain(scores, target):
    s_ = np.sort(scores)[::-1]
    k = int(math.floor(target * len(s_) + 1e-9))
    assert 0 < k < len(s_) and s_[k - 1] > s_[k], "tie at the boundary in the fake data"
    return s_[k - 1], k


def test_validation_matched_rows_recomputed(fake):
    d = table(fake, "decision_designs")
    z = npz(fake, "B1")
    sp = stream_probs(z)[:, SC_STREAM]
    p = softmax(z["logits"].astype(np.float64) / float(z["temperature"]), axis=1)
    msp = p.max(1)
    kv = (z["role"] == "val") & (z["stream_true"] != SC_STREAM)
    known = (z["role"] == "test") & (z["tier"] == "T1") & (z["stream_true"] != SC_STREAM)
    unk = (z["role"] == "unknown") & (z["tier"] == "T1")
    d1_val_far = float((sp[kv] >= 0.2).mean())
    cases = [("D2 hazard-sum flag", sp, 1), ("D3 MSP", -msp, -1)]
    for design, score, sign in cases:
        for op, target in (("val_match_5pct", 0.05), ("val_match_10pct", 0.10), ("val_match_D2", d1_val_far)):
            if design.startswith("D2") and op == "val_match_D2":
                t, k = 0.2, int((sp[kv] >= 0.2).sum())            # D2 stays at its own threshold
            else:
                t, k = _val_threshold_plain(score[kv], target)
            sel = d[(d.held_out_class == "battery") & (d.design == design) & (d.op_point == op)]
            catch = sel[sel.measure == "catch"].iloc[0]
            far = sel[sel.measure == "known_false_alarm"].iloc[0]
            assert abs(catch.value - (score[unk] >= t).mean()) < 1e-9, (design, op)
            assert abs(far.value - (score[known] >= t).mean()) < 1e-9, (design, op)
            assert abs(catch.threshold - sign * t) < 1e-9
            assert abs(catch.val_far_realised - k / kv.sum()) < 1e-12 and int(catch.n_val_known) == int(kv.sum())
            assert abs(catch.val_far_realised * kv.sum() - target * kv.sum()) <= 1.0 + 1e-9 or op == "val_match_D2"
            # realised false-alarm on test items is reported with an interval
            assert far.ci_lo <= far.value <= far.ci_hi and far.ci_hi > far.ci_lo
            assert catch.block.startswith("PRIMARY") and "validation" in catch.threshold_fitted_on
    # paired difference D2 - MSP at the validation-matched 5% point
    r = d[(d.held_out_class == "battery") & (d.op_point == "val_match_5pct")]
    c1 = r[(r.design == "D2 hazard-sum flag") & (r.measure == "catch")].value.iloc[0]
    c2 = r[(r.design == "D3 MSP") & (r.measure == "catch")].value.iloc[0]
    assert abs(r[(r.design == "D3 MSP") & (r.measure == "diff_D2_minus_design")].value.iloc[0] - (c1 - c2)) < 1e-12


def test_validation_matched_clip_uses_val_rows_without_held_out_class(fake):
    d = table(fake, "decision_designs")
    zb, ze = npz(fake, "B1"), npz(fake, "E1")
    pos = {i: k for k, i in enumerate(ze["ids"])}
    kv = (ze["role"] == "val") & (ze["stream_true"] != SC_STREAM) & (ze["class_idx"] != CLASS_NAMES.index("battery"))
    hp = ze["hazard_prob"].astype(float)
    t, k = _val_threshold_plain(hp[kv], 0.05)
    known = (zb["role"] == "test") & (zb["tier"] == "T1") & (zb["stream_true"] != SC_STREAM)
    unk = (zb["role"] == "unknown") & (zb["tier"] == "T1")
    ik = np.array([pos[i] for i in zb["ids"][known]])
    iu = np.array([pos[i] for i in zb["ids"][unk]])
    sel = d[(d.held_out_class == "battery") & (d.design == "D5 CLIP hazard prompt") & (d.op_point == "val_match_5pct")]
    assert abs(sel[sel.measure == "catch"].value.iloc[0] - (hp[iu] >= t).mean()) < 1e-9
    assert abs(sel[sel.measure == "known_false_alarm"].value.iloc[0] - (hp[ik] >= t).mean()) < 1e-9


def test_blocks_are_labelled(fake):
    d = table(fake, "decision_designs")
    assert set(d.op_point) == {"own", "val_match_D2", "val_match_5pct", "val_match_10pct", "test_match_D2",
                               "test_match_5pct", "test_match_10pct"}
    sec = d[d.op_point.str.startswith("test_match")]
    assert sec.block.str.contains("ROC point matched on test (comparison only, not deployable)", regex=False).all()
    assert d[d.op_point.str.startswith("val_match")].block.str.startswith("PRIMARY").all()
    # every sweepable design has all three validation-matched rows for every held-out class
    for c, g in d[d.op_point.str.startswith("val_match") & (d.measure != "diff_D2_minus_design")].groupby("held_out_class"):
        assert g.groupby("design").op_point.nunique().eq(3).all(), c
    rep = open(os.path.join(fake["out"], "analysis_report.md")).read()
    assert "PRIMARY matched comparison" in rep and "ROC point matched on test (comparison only, not deployable)" in rep


def test_per_seed_n_is_the_right_count(fake):
    ps = table(fake, "main_per_seed")
    for job in ("A1", "A3"):
        z = npz(fake, job)
        m = (z["role"] == "test") & (z["tier"] == "T1")
        n_sc = int((z["stream_true"][m] == SC_STREAM).sum())
        n_all = int(m.sum())
        n_cls = int((z["label_idx"][m] >= 0).sum())
        t = ps[(ps.job == job) & (ps.tier == "T1")].set_index("metric").n
        assert t["sc_recall@0.20"] == n_sc and t["sc_recall_top_class"] == n_sc and t["recall_Special Care"] == n_sc
        assert t["sc_false_alarm@0.20"] == n_all - n_sc and t["sc_false_alarm_top_class"] == n_all - n_sc
        assert t["class_acc"] == n_cls and t["macro_f1"] == n_cls and t["ece_after"] == n_cls and t["ece_before"] == n_cls
        assert t["stream_acc"] == n_all and t["always_dry_baseline"] == n_all
        assert t["recall_Dry"] == int((z["stream_true"][m] == 1).sum())


def test_missing_and_unreadable_files_are_listed(fake, tmp_path):
    res = str(tmp_path / "results")
    shutil.copytree(fake["results"], res)
    os.remove(os.path.join(res, "A3_logits.npz"))
    os.remove(os.path.join(res, "S2.json"))
    with open(os.path.join(res, "B3.json"), "w") as f:
        f.write("{not json")
    out = str(tmp_path / "out")
    p = run_analyse(res, fake["data"], out)
    assert p.returncode == 1
    rep = open(os.path.join(out, "analysis_report.md")).read()
    assert "MISSING: A3_logits.npz" in rep and "MISSING: S2.json" in rep and "UNREADABLE: B3.json" in rep and "## ERRORS" in rep
    assert "Jobs considered (JSON and logits readable): ['A1', 'A2']" in rep
    assert "Excluded: {'A3': 'A3_logits.npz not found'}" in rep
    assert os.path.exists(os.path.join(out, "decision_designs.csv"))          # everything else still analysed


def test_results_npz_with_pickled_arrays_are_rejected(fake, tmp_path):
    res = str(tmp_path / "results")
    shutil.copytree(fake["results"], res)
    z = npz(fake, "E1")
    z["ids"] = z["ids"].astype(object)                                        # the old, pickled layout
    np.savez_compressed(os.path.join(res, "E1_logits.npz"), **z)
    out = str(tmp_path / "out")
    p = run_analyse(res, fake["data"], out)
    assert p.returncode == 1
    assert "UNREADABLE: E1_logits.npz" in open(os.path.join(out, "analysis_report.md")).read()
    for f in os.listdir(fake["results"]):                                      # the real layout loads without pickle
        if f.endswith(".npz"):
            np.load(os.path.join(fake["results"], f), allow_pickle=False)["ids"]


def test_leakage_with_and_without_near_duplicate(fake):
    t = table(fake, "leakage").set_index(["job", "metric"])
    z = npz(fake, "C1")
    m = z["role"] == "test"
    near = z["near_dup_in_train"][m]
    assert z["near_dup_in_train"].dtype == bool and not z["near_dup_in_train"][~m].any()
    ok = z["logits"][m].argmax(1) == z["label_idx"][m]
    a_w, a_wo = float(ok[near].mean()), float(ok[~near].mean())
    assert a_w - a_wo > 0.02                                                   # built into the fake files
    assert abs(t.loc[("C1", "acc_items_with_duplicate"), "value"] - a_w) < 1e-12
    assert abs(t.loc[("C1", "acc_items_without_duplicate"), "value"] - a_wo) < 1e-12
    d = t.loc[("C1", "diff_acc_with_minus_without_duplicate")]
    assert abs(d.value - (a_w - a_wo)) < 1e-12 and d.ci_lo <= d.value <= d.ci_hi
    assert t.loc[("C1", "acc_items_with_duplicate"), "ci_method"] == "grouped_bootstrap"
    assert int(t.loc[("C1", "acc_items_with_duplicate"), "n"]) == int(near.sum())
    lk = json.load(open(os.path.join(fake["results"], "C1.json")))["leakage"]
    assert abs(a_w - lk["acc_items_with_duplicate"]) < 1e-6 and abs(a_wo - lk["acc_items_without_duplicate"]) < 1e-6


def test_leakage_json_mismatch_is_an_error(fake, tmp_path):
    res = str(tmp_path / "results")
    shutil.copytree(fake["results"], res)
    j = json.load(open(os.path.join(res, "C1.json")))
    j["leakage"]["acc_items_with_duplicate"] += 0.001
    json.dump(j, open(os.path.join(res, "C1.json"), "w"))
    out = str(tmp_path / "out")
    p = run_analyse(res, fake["data"], out)
    assert p.returncode == 1
    assert "differs from the JSON leakage block" in open(os.path.join(out, "analysis_report.md")).read()


def test_leakage_fallback_without_near_dup_array(fake, tmp_path):
    res = str(tmp_path / "results")
    shutil.copytree(fake["results"], res)
    z = npz(fake, "C1")
    z.pop("near_dup_in_train")
    np.savez_compressed(os.path.join(res, "C1_logits.npz"), **z)
    out = str(tmp_path / "out")
    p = run_analyse(res, fake["data"], out)
    assert p.returncode == 0
    t = pd.read_csv(os.path.join(out, "leakage.csv"))
    fb = t[(t.job == "C1") & t.ci_method.notna()]
    assert set(fb.ci_method) == {"wilson_ungrouped_fallback"}
    assert not ((t.job == "C1") & (t.metric == "diff_acc_with_minus_without_duplicate")).any()
    assert "FALLBACK" in open(os.path.join(out, "analysis_report.md")).read()


def test_scenario_tables_name_the_job_used(fake):
    for name in ("scenario_flag_precision", "scenario_per_10000"):
        assert set(table(fake, name).a_job) == {"A2"}
    rep = open(os.path.join(fake["out"], "analysis_report.md")).read()
    row = [ln for ln in rep.splitlines() if ln.startswith("| scenario_flag_precision.csv")][0]
    assert "| A2 |" in row and "A1-A3" not in row


def test_d4_wording_matches_code(fake):
    rep = open(os.path.join(fake["out"], "analysis_report.md")).read()
    assert "STRICTLY BELOW" in rep and "AT OR BELOW" in rep
    d = table(fake, "decision_designs")
    rule = d[d.design == "D3 MSP"].rule.iloc[0]
    assert "strictly below" in rule
    # own point is strict "<" as in run.py: own catch equals run.py's `rejected` (checked in test_own_points_agree_with_run_json)


def test_probe_defaults_and_convergence_warnings(fake, tmp_path):
    a = AN.parse_args(["--results-dir", "r", "--data-dir", "d", "--out-dir", "o"])
    assert a.probe_max_iter == 1000 and a.probe_pca == 0
    rep = open(os.path.join(fake["out"], "analysis_report.md")).read()
    assert "Probe convergence warnings (max_iter=1000)" in rep
    out = str(tmp_path / "out")
    p = run_analyse(fake["results"], fake["data"], out, ["--features", fake["features"], "--probe-max-iter", "1"])
    assert p.returncode == 0
    t = pd.read_csv(os.path.join(out, "object_or_dataset_source_probe.csv"))
    assert "convergence_warnings" in t.columns and t.convergence_warnings.sum() > 0
    assert pd.read_csv(os.path.join(out, "object_or_dataset_cross_source.csv")).convergence_warnings.sum() > 0
    rep = open(os.path.join(out, "analysis_report.md")).read()
    assert "Probe convergence warnings (max_iter=1)" in rep and "source probe" in rep


def test_one_failing_job_does_not_lose_the_decision_table(fake, tmp_path, monkeypatch):
    class Boom(AN.Boot):
        def __init__(self, groups, n_boot, seed, name=""):
            if name == "decision_e_waste":
                raise RuntimeError("boom")
            super().__init__(groups, n_boot, seed, name)
    monkeypatch.setattr(AN, "Boot", Boom)
    out = str(tmp_path / "out")
    rc = AN.main(["--results-dir", fake["results"], "--data-dir", fake["data"], "--out-dir", out, "--n-boot", "100"])
    assert rc == 1
    d = pd.read_csv(os.path.join(out, "decision_designs.csv"))
    assert set(d.held_out_class) == {"battery", "medicines_syringes", "shoes"}
    rep = open(os.path.join(out, "analysis_report.md")).read()
    assert "decision designs, job B3" in rep and "boom" in rep
    assert os.path.exists(os.path.join(out, "leakage.csv"))                    # later analyses still ran

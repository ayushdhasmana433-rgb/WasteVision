"""Tests for pipeline/rebuild_splits.py on synthetic data (see make_synthetic.py)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
import make_synthetic as ms          # noqa: E402
import rebuild_splits as rb          # noqa: E402

CAP = 1200
MAX_GROUP = 20
THR = 4


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    truth = ms.build(20000, 1, str(d))
    return dict(dir=str(d), truth=truth, csv=str(d / "all_images_before_dedup.csv"),
                feats=str(d / "frozen_features_effv2b0.npz"), old=str(d / "old_splits.csv"))


@pytest.fixture(scope="module")
def stages(synth):
    """The pipeline stages run in-process, to inspect what each step removed."""
    df_all, _ = rb.combine_inputs(synth["csv"], None)
    df_all, _ = rb.clean_hashes(df_all)
    df_all, _ = rb.fix_missing_groups(df_all)
    filtered = rb.filter_globalwaste(df_all)
    old = pd.read_csv(synth["old"], dtype={"id": str})
    t3_ids = set(old.id[old.split == "dwsd_test"])
    t3, pool_rows, removed_group, removed_near = rb.set_aside_t3(filtered, t3_ids, THR)
    pool0 = rb.visit_order(pool_rows)
    after_phash, pairs, conf, overlap = rb.phash_dedup(pool0, THR)
    z = np.load(synth["feats"], allow_pickle=True)
    after_feat, cand, fconf, finfo = rb.feature_pass(after_phash, z["ids"].astype(str), z["feats"], 0.90, 0.97)
    return dict(df_all=df_all, filtered=filtered, pool0=pool0, after_phash=after_phash, pairs=pairs, conf=conf,
                overlap=overlap, after_feat=after_feat, cand=cand, fconf=fconf, finfo=finfo, t3_ids=t3_ids,
                t3=t3, removed_group=removed_group, removed_near=removed_near)


@pytest.fixture(scope="module")
def run(synth, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    missing_images = str(tmp_path_factory.mktemp("noimages"))     # exists but holds no images
    rb.main(["--all-csv", synth["csv"], "--features", synth["feats"], "--old-splits", synth["old"],
             "--images-root", missing_images, "--out-dir", str(out), "--cap", str(CAP)])
    return dict(out=str(out),
                splits=pd.read_csv(out / "splits_v3.csv", dtype={"id": str}),
                table=pd.read_csv(out / "count_table.csv"),
                report=json.load(open(out / "report.json")))


# ------------------------------------------------------------------ input sanity

def test_synthetic_has_the_hard_cases(synth, stages):
    df = stages["df_all"]
    assert set(df.source) == set(rb.SOURCE_ORDER) and set(df.label) == set(rb.CLASSES)
    assert 18000 <= len(df) <= 23000
    zw = df[df.source == "ZeroWaste"].groupby("group").size()
    assert (zw >= 300).sum() >= 4                                   # huge video groups
    gw = df[df.source == "GlobalWaste"]
    assert set(gw.label) - set(rb.GLOBALWASTE_KEEP)                 # dropped classes exist
    assert len(synth["truth"]["t3_ids"]) == 192
    assert df.groupby(["label", "source"]).size().min() < 10        # tiny cells


# ------------------------------------------------------------------ step 1

def test_filter_keeps_only_four_globalwaste_labels(stages, run):
    gw_before = stages["df_all"][stages["df_all"].source == "GlobalWaste"]
    gw_after = stages["filtered"][stages["filtered"].source == "GlobalWaste"]
    assert set(gw_after.label) <= set(rb.GLOBALWASTE_KEEP)
    assert len(gw_after) == int(gw_before.label.isin(rb.GLOBALWASTE_KEEP).sum()) < len(gw_before)
    non_gw_before = stages["df_all"][stages["df_all"].source != "GlobalWaste"]
    assert len(stages["filtered"]) - len(gw_after) == len(non_gw_before)      # nothing else removed
    rep = run["report"]["globalwaste_filter"]
    assert rep["globalwaste_before"] == len(gw_before) and rep["globalwaste_after"] == len(gw_after)
    assert set(run["splits"][run["splits"].source == "GlobalWaste"].label) <= set(rb.GLOBALWASTE_KEEP)


# ------------------------------------------------------------------ step 2

def test_phash_originals_win_over_later_sources(synth, stages):
    kept = set(stages["after_phash"].id)
    same = synth["truth"]["phash_same_label"]
    assert len(same) > 100
    for p in same:
        assert p["winner"] in kept, p
        assert p["other"] not in kept, p
    src_of = stages["df_all"].set_index("id").source
    gw = [p for p in same if src_of[p["other"]] == "GlobalWaste"]
    assert len(gw) >= 80                                                # GlobalWaste copies lost every time


def test_phash_same_source_first_row_wins(synth, stages):
    kept = set(stages["after_phash"].id)
    for p in synth["truth"]["phash_same_source"]:
        assert p["winner"] in kept and p["other"] not in kept


def test_phash_conflicts_drop_both_and_are_logged(synth, stages):
    kept = set(stages["after_phash"].id)
    logged = {frozenset((r.id_a, r.id_b)) for r in stages["conf"].itertuples()}
    conflicts = synth["truth"]["phash_conflict"]
    assert len(conflicts) == 38
    for p in conflicts:
        assert p["winner"] not in kept and p["other"] not in kept
        assert frozenset((p["winner"], p["other"])) in logged
    labelled = stages["conf"][stages["conf"].reason == rb.REASON_LABEL]
    assert (labelled.label_a != labelled.label_b).all()


def test_phash_survivors_have_no_pair_within_threshold(stages):
    H = rb.parse_hashes(stages["after_phash"].phash.values)
    i, j, d = rb.hamming_pairs(H, THR)
    assert len(i) == 0


def test_hamming_pairs_matches_brute_force():
    rng = np.random.default_rng(0)
    H = rng.integers(0, 2 ** 64, size=300, dtype=np.uint64)
    H[10] = H[3] ^ np.uint64(0b11)
    H[20] = H[3]
    i, j, d = rb.hamming_pairs(H, 4, chunk=64)
    got = {(int(a), int(b), int(c)) for a, b, c in zip(i, j, d)}
    want = set()
    for a in range(300):
        for b in range(a + 1, 300):
            dist = bin(int(H[a]) ^ int(H[b])).count("1")
            if dist <= 4:
                want.add((a, b, dist))
    assert got == want and len(want) >= 3


def test_pairs_file_and_overlap_matrix(synth, stages, run):
    pairs = pd.read_csv(os.path.join(run["out"], "dup_pairs_phash.csv.gz"))
    assert list(pairs.columns) == ["id_a", "id_b", "source_a", "source_b", "label_a", "label_b", "distance"]
    assert pairs.distance.max() <= THR and len(pairs) == len(stages["pairs"])
    have = {frozenset(x) for x in zip(pairs.id_a, pairs.id_b)}
    for p in synth["truth"]["phash_same_label"] + synth["truth"]["phash_conflict"]:
        assert frozenset((p["winner"], p["other"])) in have
    ov = pd.read_csv(os.path.join(run["out"], "source_overlap.csv"), index_col=0)
    assert list(ov.index) == rb.SOURCE_ORDER == list(ov.columns)
    assert (ov.values == ov.values.T).all()
    cross = pairs[pairs.source_a != pairs.source_b]
    assert ov.values.sum() - np.trace(ov.values) == 2 * len(cross)
    assert ov.loc["GlobalWaste", "TrashBox"] >= 40 and ov.loc["GlobalWaste", "GD"] >= 50
    assert os.path.exists(os.path.join(run["out"], "label_conflicts.csv"))


# ------------------------------------------------------------------ step 3

def test_feature_pass(synth, stages):
    kept = set(stages["after_feat"].id)
    truth = synth["truth"]
    for p in truth["feature_dups"]:
        assert p["winner"] in kept and p["other"] not in kept
    for p in truth["feature_conflicts"]:
        assert p["winner"] not in kept and p["other"] not in kept
    for p in truth["feature_midrange"]:
        assert p["winner"] in kept and p["other"] in kept
    cand = stages["cand"]
    assert (cand.source_a != cand.source_b).all() and cand.cosine.min() >= 0.90
    pairs = {frozenset(x): c for x, c in zip(zip(cand.id_a, cand.id_b), cand.cosine)}
    for p in truth["feature_dups"]:
        assert pairs[frozenset((p["winner"], p["other"]))] >= 0.97
    for p in truth["feature_midrange"]:
        assert 0.90 <= pairs[frozenset((p["winner"], p["other"]))] < 0.97
    assert set(stages["fconf"].stage) <= {"features"}


def test_candidates_and_sheets_written(run):
    assert os.path.exists(os.path.join(run["out"], "dup_candidates_features.csv.gz"))
    sheets = os.path.join(run["out"], "feature_pairs_sheets")
    assert os.path.isdir(sheets) and any(f.endswith(".jpg") for f in os.listdir(sheets))   # missing images: no crash
    sample = pd.read_csv(os.path.join(sheets, "sampled_pairs.csv"))
    assert len(sample) == 100 or len(sample) == len(pd.read_csv(os.path.join(run["out"], "dup_candidates_features.csv.gz")))
    assert sample.cosine.is_monotonic_increasing


# ------------------------------------------------------------------ step 4

def test_waterfill_unit():
    rng = np.random.default_rng(0)
    assert rb.waterfill(dict(a=10, b=100, c=100), 150, rng) == dict(a=10, b=70, c=70)
    q = rb.waterfill(dict(a=5, b=7, c=1000, d=1000), 100, rng)
    assert q["a"] == 5 and q["b"] == 7 and q["c"] + q["d"] == 88 and abs(q["c"] - q["d"]) <= 1
    assert rb.waterfill(dict(a=3, b=4), 100, rng) == dict(a=3, b=4)
    q = rb.waterfill(dict(a=50, b=50, c=50), 100, rng)
    assert sorted(q.values()) == [33, 33, 34]


def test_group_limit_and_cap(stages, run):
    sp = run["splits"][run["splits"].tier != "T3"]
    assert sp.groupby(["label", "source", "group"]).size().max() <= MAX_GROUP
    per_class = sp.label.value_counts()
    avail = per_cell_available(stages["after_feat"])
    for c in rb.CLASSES:
        total = avail[avail.label == c].avail.sum()
        assert per_class[c] == min(CAP, total), c
    assert (per_class <= CAP).all()
    zw_groups = stages["after_feat"][stages["after_feat"].source == "ZeroWaste"].groupby("group").size()
    assert zw_groups.max() >= 300                                   # the huge groups are still there before capping


def per_cell_available(pool):
    """Images available per (label, source) after limiting every group to MAX_GROUP (independent of the script)."""
    g = pool.groupby(["label", "source", "group"]).size().clip(upper=MAX_GROUP).rename("n").reset_index()
    return g.groupby(["label", "source"]).n.sum().rename("avail").reset_index()


def test_quota_fairness(stages, run):
    sp = run["splits"][run["splits"].tier != "T3"]
    taken = sp.groupby(["label", "source"]).size().rename("taken")
    avail = per_cell_available(stages["after_feat"]).set_index(["label", "source"]).avail
    checked = 0
    for c in rb.CLASSES:
        a = avail[c]
        if a.sum() <= CAP:
            continue
        t = taken[c].reindex(a.index, fill_value=0)
        assert (t <= a).all()
        open_ = a.index[t < a]                                      # sources that could still have given more
        done = a.index[t == a]
        level = t[open_].max()
        assert (t[open_] >= level - 1).all(), (c, t.to_dict())      # equal quota (differ by at most the remainder)
        assert (t[done] <= level).all(), (c, t.to_dict())           # nobody exhausted holds more than the quota
        checked += 1
    assert checked >= 6
    # the old bug: a source with huge groups (ZeroWaste video frames) filling a class
    for c, n in taken.xs("ZeroWaste", level="source").items():
        assert n / CAP <= 0.15, (c, n)


def test_cap_is_deterministic(stages):
    a, _ = rb.cap_pool(stages["after_feat"], CAP, MAX_GROUP, 7)
    b, _ = rb.cap_pool(stages["after_feat"], CAP, MAX_GROUP, 7)
    c, _ = rb.cap_pool(stages["after_feat"], CAP, MAX_GROUP, 8)
    assert list(a.id) == list(b.id) and list(a.id) != list(c.id)


# ------------------------------------------------------------------ step 5 and 6

def test_split_constraints(synth, stages, run):
    sp = run["splits"]
    main = sp[sp.tier != "T3"]
    assert main.groupby("group").split.nunique().max() == 1             # no group in two splits
    assert set(main.split) == {"train", "val", "test"}
    assert (main[main.source == "PharmaSynth"].split == "train").all()
    # cells with fewer than 30 images go entirely to train
    cell = main.groupby(["label", "source"]).split.agg(["size", lambda s: (s != "train").sum()])
    cell.columns = ["n", "not_train"]
    small = cell[cell.n < 30]
    assert len(small) >= 5 and (small.not_train == 0).all()
    # every class >= 100 val, every cell >= 50 (except train-only PharmaSynth) has a validation image
    val = main[main.split == "val"]
    assert (val.label.value_counts().reindex(rb.CLASSES, fill_value=0) >= 100).all()
    cell_v = main.groupby(["label", "source"]).agg(n=("id", "size"), v=("split", lambda s: (s == "val").sum()))
    big = cell_v[(cell_v.n >= 50)].reset_index()
    big = big[big.source != "PharmaSynth"]
    assert len(big) > 20 and (big.v >= 1).all()
    # roughly 70/10/20 among images that were eligible for splitting
    elig = main[~main.source.isin(["PharmaSynth"]) & (main.groupby(["label", "source"]).id.transform("size") >= 30)]
    share = elig.split.value_counts(normalize=True)
    assert 0.07 <= share["val"] <= 0.13 and 0.12 <= share["test"] <= 0.22
    # no test/val image within the pHash threshold of a training image
    tr = rb.parse_hashes(main[main.split == "train"].phash.values)
    ho = rb.parse_hashes(main[main.split != "train"].phash.values)
    assert not rb.within_thr_of(ho, tr, THR).any()
    assert all(run["report"]["checks_passed"].values()) and len(run["report"]["checks_passed"]) == 6


def test_tiers_and_t3(synth, run):
    sp = run["splits"]
    test = sp[sp.split == "test"]
    assert set(test.tier) == {"T1", "T2", "T3"}
    assert set(test[test.tier == "T2"].source) == {"ZeroWaste"}
    assert (test[test.source == "ZeroWaste"].tier == "T2").all()
    t1 = test[test.tier == "T1"]
    assert not set(t1.source) & {"ZeroWaste", "DWSD", "PharmaSynth"}
    t3 = sp[sp.tier == "T3"]
    assert set(t3.id) == set(synth["truth"]["t3_ids"]) and len(t3) == 192 and set(t3.source) == {"DWSD"}
    dw = sp[(sp.source == "DWSD") & (sp.tier != "T3")]
    assert set(dw.split) <= {"train", "val"} and "val" in set(dw.split)
    assert run["report"]["dwsd_test_assigned_images_moved_to_train"] > 0
    assert (sp[sp.split.isin(["train", "val"])].tier.fillna("") == "").all()


# ------------------------------------------------------------------ outputs

def test_outputs_and_count_table_consistency(synth, stages, run):
    out = run["out"]
    for f in ["splits_v3.csv", "count_table.csv", "count_table.md", "report.json", "dup_pairs_phash.csv.gz",
              "label_conflicts.csv", "source_overlap.csv", "dup_candidates_features.csv.gz"]:
        assert os.path.exists(os.path.join(out, f)), f
    sp = run["splits"]
    assert list(sp.columns) == ["id", "source", "orig_class", "label", "group", "norm_rel", "phash", "split", "tier"]
    assert sp.id.is_unique
    tab = run["table"]
    cols = ["source", "label", "mapped", "after_filter", "set_aside_T3", "removed_T3_group_or_near", "after_phash", "after_features", "after_cap",
            "train", "val", "test_T1", "test_T2", "test_T3"]
    assert list(tab.columns) == cols
    assert not tab.duplicated(["source", "label"]).any()
    assert tab.mapped.sum() == len(stages["df_all"])
    gw_dropped = len(stages["df_all"]) - len(stages["filtered"])
    assert tab.after_filter.sum() == tab.mapped.sum() - gw_dropped            # after_filter includes T3
    assert tab.set_aside_T3.sum() == 192 and tab.removed_T3_group_or_near.sum() == 0
    assert (tab.after_filter - tab.set_aside_T3 - tab.removed_T3_group_or_near >= tab.after_phash).all()
    assert (tab.after_filter.sum() - tab.set_aside_T3.sum() - tab.removed_T3_group_or_near.sum()
            == run["report"]["phash"]["images_before"])
    assert (tab.after_filter >= tab.after_phash).all() and (tab.after_phash >= tab.after_features).all()
    assert (tab.after_features >= tab.after_cap).all()
    assert tab.after_phash.sum() == len(stages["after_phash"]) and tab.after_features.sum() == len(stages["after_feat"])
    assert (tab.train + tab.val + tab.test_T1 + tab.test_T2 == tab.after_cap).all()
    assert tab.test_T3.sum() == 192 and tab.after_cap.sum() == len(sp) - 192
    # table agrees with the split file
    by = sp[sp.tier != "T3"].groupby(["source", "label"]).size()
    assert (tab.set_index(["source", "label"]).after_cap[by.index] == by).all()
    md = open(os.path.join(out, "count_table.md")).read()
    assert md.count("\n| class |") == 3 and "Class x source" in md and "train/val/test" in md
    rep = run["report"]
    for key in ["thresholds", "seed_used", "checks_passed", "stage_totals", "split_totals", "stage_by_class",
                "globalwaste_surviving_per_class", "largest_source_share_in_train", "cap_info", "phash", "features"]:
        assert key in rep, key
    assert rep["seed_used"] == 42 and rep["stage_totals"]["after_cap"] == tab.after_cap.sum()
    assert set(rep["globalwaste_surviving_per_class"]["after_cap"]) == set(rb.GLOBALWASTE_KEEP)
    assert all(0 < v["share"] <= 1 for v in rep["largest_source_share_in_train"].values())


# ------------------------------------------------------------------ optional arguments and retries

def test_runs_without_features_and_old_splits(synth, tmp_path):
    rb.main(["--all-csv", synth["csv"], "--out-dir", str(tmp_path), "--cap", str(CAP)])
    rep = json.load(open(tmp_path / "report.json"))
    assert "skipped" in rep["features"]
    assert not (tmp_path / "dup_candidates_features.csv.gz").exists()
    sp = pd.read_csv(tmp_path / "splits_v3.csv")
    assert "T3" not in set(sp.tier.fillna("")) and not (sp[sp.source == "DWSD"].split == "test").any()
    tab = pd.read_csv(tmp_path / "count_table.csv")
    assert (tab.after_features == tab.after_phash).all() and tab.test_T3.sum() == 0


def test_seed_retry_and_exit(synth, tmp_path, monkeypatch):
    real = rb.run_checks
    calls = []

    def flaky(df, thr, t3=None):
        calls.append(1)
        ok, fails = real(df, thr, t3)
        if len(calls) <= 2:
            return ok, ["forced failure"]
        return ok, fails
    monkeypatch.setattr(rb, "run_checks", flaky)
    rb.main(["--all-csv", synth["csv"], "--out-dir", str(tmp_path), "--cap", str(CAP), "--seed", "10"])
    rep = json.load(open(tmp_path / "report.json"))
    assert rep["seed_used"] == 12 and [a["passed"] for a in rep["split_seed_attempts"]] == [False, False, True]

    monkeypatch.setattr(rb, "run_checks", lambda df, thr, t3=None: ({"every_class_has_100_val": False,
                                                                     "no_group_in_two_splits": True},
                                                                    ["always failing"]))
    with pytest.raises(SystemExit) as e:
        rb.main(["--all-csv", synth["csv"], "--out-dir", str(tmp_path / "b"), "--cap", str(CAP)])
    assert "always failing" in str(e.value) and "21" in str(e.value)
    # item 4: diagnostics are written before exiting; the split file is the FAILED one
    b = tmp_path / "b"
    assert (b / "count_table.csv").exists() and (b / "count_table.md").exists()
    assert (b / "splits_v3_FAILED.csv").exists() and not (b / "splits_v3.csv").exists()
    rep = json.load(open(b / "report.json"))
    assert rep["checks_passed"] is False and rep["all_checks_passed"] is False
    assert rep["failing_checks"] == ["every_class_has_100_val"] and rep["failure_messages"] == ["always failing"]
    assert set(rep["validation_counts_per_class"]) == set(rb.CLASSES)
    assert len(rep["split_seed_attempts"]) == 21 and rep["seed_used"] is None


def test_run_checks_detects_each_violation(run):
    df = run["splits"][run["splits"].tier != "T3"].copy()
    ok, fails = rb.run_checks(df, THR)
    assert not fails
    # a class with too few validation images
    bad = df.copy()
    bad.loc[(bad.label == "glass") & (bad.split == "val"), "split"] = "train"
    ok, fails = rb.run_checks(bad, THR)
    assert not ok["every_class_has_100_val"] and any("glass" in f for f in fails)
    # a big cell without validation
    bad = df.copy()
    bad.loc[(bad.label == "trash") & (bad.source == "RealWaste") & (bad.split == "val"), "split"] = "train"
    ok, fails = rb.run_checks(bad, THR)
    assert not ok["every_cell_50plus_in_val"]
    # a group in two splits
    bad = df.copy()
    g = bad[bad.split == "val"].group.iloc[0]
    first = bad.index[bad.group == g][0]
    bad.loc[first, "split"] = "train"
    bad.loc[bad.index[bad.group == g][1:], "split"] = "val"
    if (bad.group == g).sum() > 1:
        ok, fails = rb.run_checks(bad, THR)
        assert not ok["no_group_in_two_splits"]
    # a validation image that duplicates a training image
    bad = df.copy()
    v = bad.index[bad.split == "val"][0]
    t = bad.index[bad.split == "train"][0]
    bad.loc[v, "phash"] = bad.loc[t, "phash"]
    ok, fails = rb.run_checks(bad, THR)
    assert not ok["no_heldout_near_duplicate_of_train"]


# ------------------------------------------------------------------ review fixes

def test_conflict_rule_is_order_independent():
    pi = np.array([0, 0, 1], np.int32)
    pj = np.array([1, 2, 2], np.int32)
    for labels in ([0, 1, 0], [0, 0, 1], [1, 0, 0]):              # x, y, x   x, x, y   y, x, x
        kept, conflicts = rb.greedy_remove(np.array(labels), pi, pj, 3)
        assert not kept.any(), labels
        assert len(conflicts) >= 1
    kept, conflicts = rb.greedy_remove(np.array([0, 1, 0]), pi, pj, 3)
    assert (0, 1, rb.REASON_LABEL) in conflicts
    assert {c[2] for c in conflicts} == {rb.REASON_LABEL, rb.REASON_NEAR}
    # same labels, no conflict: only the first survives
    kept, conflicts = rb.greedy_remove(np.array([0, 0, 0]), pi, pj, 3)
    assert kept.tolist() == [True, False, False] and conflicts == []


def test_conflict_rule_in_pipeline_phash():
    """Three copies of one hash labelled x, y, x end up all dropped, with every pair logged."""
    rows = [dict(id=f"c{k}", source=src, orig_class="o", label=lab, group=f"g{k}", norm_rel="a.jpg",
                 phash="00000000000000ff") for k, (src, lab) in enumerate(
        [("RealWaste", "glass"), ("TrashNet", "metal"), ("GD", "glass")])]
    rows.append(dict(id="other", source="GD", orig_class="o", label="glass", group="go", norm_rel="b.jpg",
                     phash="ffff000000000000"))
    pool = rb.visit_order(pd.DataFrame(rows))
    kept, pairs, conf, _ = rb.phash_dedup(pool, 4)
    assert list(kept.id) == ["other"] and len(pairs) == 3 and len(conf) == 3
    assert set(conf.reason) == {rb.REASON_LABEL, rb.REASON_NEAR}


def test_clean_hashes_and_missing_groups():
    df = pd.DataFrame(dict(id=list("abcdefg"), group=["g", None, "", "  ", "h", "i", "j"],
                           phash=["  0xABCDEF0123456789 ", "0X00000000000000FF", "ff", "", None, "xyz",
                                  "0123456789abcdef0"]))
    clean, bad = rb.clean_hashes(df)
    assert list(clean.id) == ["a", "b", "c"] and list(bad.id) == ["d", "e", "f", "g"]
    assert list(clean.phash) == ["abcdef0123456789", "00000000000000ff", "00000000000000ff"]
    fixed, n = rb.fix_missing_groups(df)
    assert n == 3 and list(fixed.group) == ["g", "b", "c", "d", "h", "i", "j"]


def test_combine_inputs_rejects_id_collision(synth, tmp_path):
    df = pd.read_csv(synth["csv"], dtype=str).head(3)
    extra = tmp_path / "extra.csv"
    df.to_csv(extra, index=False)
    with pytest.raises(SystemExit) as e:
        rb.combine_inputs(synth["csv"], [str(extra)])
    assert "already exist" in str(e.value)


def test_feature_pass_chunk_memory_and_masking(stages):
    info = stages["finfo"]
    assert info["chunk_rows"] * len(stages["after_phash"]) * 6 <= 300e6    # default chunk stays under ~300 MB
    assert not stages["cand"].duplicated(["id_a", "id_b"]).any()           # each pair once


def test_feature_pass_small_chunks_and_truncation(synth, stages):
    z = np.load(synth["feats"], allow_pickle=True)
    pool = stages["after_phash"]
    kept, cand, fconf, info = rb.feature_pass(pool, z["ids"].astype(str), z["feats"], 0.90, 0.97,
                                              chunk_elements=len(pool) * 50, max_chunk_candidates=0)
    assert info["chunk_rows"] == 50 and info["chunks_truncated_to_removal_threshold"] > 0
    assert info["candidate_list_complete"] is False
    full = stages["cand"]
    strong = lambda c: {frozenset(x) for x in zip(c.id_a[c.cosine >= 0.97], c.id_b[c.cosine >= 0.97])}
    assert strong(cand) == strong(full)                               # removal pairs are unaffected
    assert list(kept.id) == list(stages["after_feat"].id)
    assert len(cand) < len(full)


@pytest.fixture(scope="module")
def edge(synth, tmp_path_factory):
    """Synthetic data with the review's edge cases planted, run through main."""
    d = tmp_path_factory.mktemp("edge")
    df = pd.read_csv(synth["csv"], dtype=str, keep_default_na=False)
    t3_ids = set(synth["truth"]["t3_ids"])
    t3 = df[df.id.isin(t3_ids)]
    t3_group, t3_hash = t3.group.iloc[0], t3.phash.iloc[0]
    rng = np.random.default_rng(5)
    marks = dict(same_group=[], near=[], blank_group=[], bad_hash=[], messy_hash=[])
    non_t3 = df[~df.id.isin(t3_ids)]
    pick = lambda mask, k: [int(i) for i in rng.choice(non_t3.index[mask(non_t3)], size=k, replace=False)]
    for i in pick(lambda x: x.source == "DWSD", 1) + pick(lambda x: x.source == "TIDY", 1):
        df.loc[i, "group"] = t3_group
        marks["same_group"].append(df.id[i])
    for i in pick(lambda x: x.source == "GD", 3):
        df.loc[i, "phash"] = f"{int(t3_hash, 16) ^ 0b11:016x}"
        marks["near"].append(df.id[i])
    for i in pick(lambda x: x.source == "RealWaste", 5):
        df.loc[i, "group"] = ""
        marks["blank_group"].append(df.id[i])
    for i, v in zip(pick(lambda x: x.source == "PSU", 2), ["xyz", ""]):
        df.loc[i, "phash"] = v
        marks["bad_hash"].append(df.id[i])
    for i in pick(lambda x: x.source == "Drinking", 2):
        df.loc[i, "phash"] = " 0X" + df.phash[i].upper() + " "
        marks["messy_hash"].append(df.id[i])
    main_csv = d / "main.csv"
    df.to_csv(main_csv, index=False)
    extra = pd.DataFrame(dict(id=[f"extra{k:03d}" for k in range(30)], source="TrashBox", orig_class="x",
                              label="e_waste", group=[f"extra_g{k // 2}" for k in range(30)],
                              norm_rel="x.jpg",
                              phash=[f"{int(v):016x}" for v in rng.integers(0, 2 ** 63, 30)]))
    extra_csv = d / "extra.csv"
    extra.to_csv(extra_csv, index=False)
    out = d / "out"
    rb.main(["--all-csv", str(main_csv), "--extra-csv", str(extra_csv), "--features", synth["feats"],
             "--old-splits", synth["old"], "--out-dir", str(out), "--cap", str(CAP)])
    return dict(out=out, marks=marks, df=df, t3_group=t3_group, t3_ids=t3_ids,
                splits=pd.read_csv(out / "splits_v3.csv", dtype={"id": str}),
                table=pd.read_csv(out / "count_table.csv"), report=json.load(open(out / "report.json")))


def test_missing_groups_get_own_group(edge):
    rep = edge["report"]
    assert rep["rows_with_missing_group"] == 5
    sp = edge["splits"]
    assert sp.group.notna().all() and (sp.group != "").all()
    for i in edge["marks"]["blank_group"]:
        row = sp[sp.id == i]
        assert len(row) == 0 or row.group.iloc[0] == i
    assert all(edge["report"]["checks_passed"].values())             # group check ran with these rows included


def test_bad_hash_rows_dropped_and_logged(edge):
    rep = edge["report"]
    assert rep["rows_bad_hash"] == 2
    bad = pd.read_csv(edge["out"] / "rows_bad_hash.csv", dtype=str, keep_default_na=False)
    assert set(bad.id) == set(edge["marks"]["bad_hash"])
    assert not set(edge["splits"].id) & set(bad.id)
    assert edge["splits"].phash.str.fullmatch("[0-9a-f]{16}").all()  # whitespace/0x/uppercase were cleaned
    # mapped counts every row that has a valid hash, extras included
    assert edge["table"].mapped.sum() == len(edge["df"]) + 30 - 2


def test_extra_csv_rows_are_used(edge):
    assert edge["report"]["extra_csv_rows"] == 30
    sp = edge["splits"]
    extras = sp[sp.id.str.startswith("extra")]
    assert len(extras) > 0 and set(extras.source) == {"TrashBox"}
    assert edge["table"].query("source == 'TrashBox' and label == 'e_waste'").mapped.iloc[0] >= 30
    feat = edge["report"]["features"]                                # extras have no feature rows: warned, allowed
    assert feat["images_without_feature_row"] >= 1 and feat["share_without_feature_row"] < 0.05


def test_t3_group_and_near_rows_removed(edge):
    rep, sp, tab = edge["report"], edge["splits"], edge["table"]
    assert rep["rows_removed_same_group_as_T3"] == 2 and rep["rows_removed_near_T3"] == 3
    gone = set(edge["marks"]["same_group"]) | set(edge["marks"]["near"])
    assert not gone & set(sp.id)
    nonT3 = sp[sp.tier != "T3"]
    assert not set(nonT3.group) & set(sp[sp.tier == "T3"].group)
    assert tab.removed_T3_group_or_near.sum() == 5 and tab.set_aside_T3.sum() == 192
    assert (tab.after_filter.sum() - tab.set_aside_T3.sum() - tab.removed_T3_group_or_near.sum()
            == rep["phash"]["images_before"])
    assert len(rep["checks_passed"]) == 6 and rep["checks_passed"]["no_t3_group_in_train_val_test"]
    assert rep["checks_passed"]["no_train_image_near_t3"]
    # and no remaining training image is near any T3 image
    tr = rb.parse_hashes(nonT3[nonT3.split == "train"].phash.values)
    t3h = rb.parse_hashes(sp[sp.tier == "T3"].phash.values)
    assert not rb.within_thr_of(tr, t3h, THR).any()


def test_t3_hard_checks_fire():
    base = dict(source="GD", orig_class="o", label="glass", norm_rel="x")
    df = pd.DataFrame([dict(base, id="a", group="g1", phash="0000000000000000", split="train"),
                       dict(base, id="b", group="g2", phash="ffffffffffffffff", split="val")])
    t3 = pd.DataFrame([dict(base, id="t", source="DWSD", group="g1", phash="0000000000000003")])
    ok, fails = rb.run_checks(df, THR, t3)
    assert not ok["no_t3_group_in_train_val_test"] and not ok["no_train_image_near_t3"]
    assert any("T3" in f for f in fails)


def test_report_per_class_and_validation_floor(run):
    rep, sp = run["report"], run["splits"]
    val = sp[(sp.split == "val")].label.value_counts()
    assert set(rep["per_class"]) == set(rb.CLASSES)
    for c, row in rep["per_class"].items():
        assert row["val_count"] == int(val.get(c, 0))
        assert row["n_sources_in_train"] >= 1 and 0 < row["largest_source_share_in_train"] <= 1
    assert rep["classes_near_validation_floor"] == [c for c in rb.CLASSES if int(val.get(c, 0)) < 130]


def _features_without(synth, tmp_path, share):
    z = np.load(synth["feats"], allow_pickle=True)
    keep = np.random.default_rng(3).random(len(z["ids"])) >= share
    path = tmp_path / "feats_partial.npz"
    np.savez(path, ids=z["ids"][keep], feats=z["feats"][keep])
    return str(path)


def test_missing_features_stop_unless_allowed(synth, tmp_path, capsys):
    partial = _features_without(synth, tmp_path, 0.10)
    with pytest.raises(SystemExit) as e:
        rb.main(["--all-csv", synth["csv"], "--features", partial, "--out-dir", str(tmp_path / "o1"),
                 "--cap", str(CAP)])
    assert "allow-missing-features" in str(e.value) and "WARNING" in capsys.readouterr().out
    rb.main(["--all-csv", synth["csv"], "--features", partial, "--allow-missing-features",
             "--out-dir", str(tmp_path / "o2"), "--cap", str(CAP)])
    feat = json.load(open(tmp_path / "o2" / "report.json"))["features"]
    assert feat["share_without_feature_row"] > 0.05 and feat["missing_features_allowed_by_flag"] is True


def test_small_feature_gap_only_warns(synth, tmp_path, capsys):
    partial = _features_without(synth, tmp_path, 0.02)
    rb.main(["--all-csv", synth["csv"], "--features", partial, "--out-dir", str(tmp_path / "o"),
             "--cap", str(CAP)])
    assert "WARNING" in capsys.readouterr().out
    feat = json.load(open(tmp_path / "o" / "report.json"))["features"]
    assert 0 < feat["share_without_feature_row"] < 0.05


def test_too_many_bad_hashes_stop(synth, tmp_path):
    df = pd.read_csv(synth["csv"], dtype=str, keep_default_na=False)
    df.loc[:300, "phash"] = "not-a-hash"
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(SystemExit) as e:
        rb.main(["--all-csv", str(bad), "--out-dir", str(tmp_path / "o")])
    assert "phash" in str(e.value) and (tmp_path / "o" / "rows_bad_hash.csv").exists()

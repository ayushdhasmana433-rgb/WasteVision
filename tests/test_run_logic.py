"""Tests for pipeline/run.py logic that needs no TensorFlow (and no GPU, no images)."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp, softmax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import run  # noqa: E402

HERE = os.path.dirname(__file__)


# ----------------------------------------------------------------------------- job table
def test_job_table_matches_plan_order_and_content():
    queue = run.read_queue(os.path.join(HERE, "..", "pipeline", "queue.txt"))
    assert queue == "A1 A2 A3 B1 B2 B3 B4 E1 B5 B6 B7 B8 C1 C2 S1 S2 S3 S4".split()
    assert set(queue) == set(run.JOBS)
    assert [run.JOBS[j].seed for j in ("A1", "A2", "A3")] == [42, 7, 123]
    assert [run.JOBS[j].holdout_class for j in ("B1", "B2", "B3", "B4")] == \
        ["battery", "medicines_syringes", "e_waste", "shoes"]
    assert [run.JOBS[j].holdout_source for j in ("B5", "B6", "B7", "B8")] == ["MedPackID", "TrashBox", "GD", "RealWaste"]
    assert all(run.JOBS[j].seed == 42 for j in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"))
    assert run.JOBS["C1"].kind == "leak_naive" and run.JOBS["C2"].kind == "leak_grouped"
    assert run.JOBS["S1"].label_space == "stream" and run.JOBS["S1"].kind == "full"
    assert [run.JOBS[j].holdout_class for j in ("S2", "S3", "S4")] == ["battery", "medicines_syringes", "e_waste"]
    assert all(run.JOBS[j].label_space == "stream" for j in ("S1", "S2", "S3", "S4"))
    assert run.JOBS["E1"].kind == "clip"


def test_read_queue_ignores_comments_and_rejects_unknown(tmp_path):
    q = tmp_path / "q.txt"
    q.write_text("# first\nA1\n\n  b2  # lower case ok\nE1\n")
    assert run.read_queue(str(q)) == ["A1", "B2", "E1"]
    q.write_text("A1\nZ9\n")
    with pytest.raises(ValueError):
        run.read_queue(str(q))


def test_parse_args_defaults_and_validation():
    a = run.parse_args(["--job", "a1"])
    assert a.drive_root.endswith("WasteVision_v3") and a.data_root == "/content/data"
    assert a.splits.endswith("WasteVision_v3/data/splits_v3.csv")
    with pytest.raises(SystemExit):
        run.parse_args([])
    with pytest.raises(SystemExit):
        run.parse_args(["--job", "Q1"])


# ----------------------------------------------------------------------------- stream mapping
def test_stream_mapping():
    assert run.STREAM_OF["biological"] == "Wet"
    assert run.STREAM_OF["sanitary"] == "Sanitary"
    for c in ("battery", "medicines_syringes", "e_waste"):
        assert run.STREAM_OF[c] == "Special Care"
    for c in ("paper_cardboard", "plastic", "metal", "glass", "clothes", "shoes", "trash"):
        assert run.STREAM_OF[c] == "Dry"
    sp = run.class_space(run.CLASSES)
    assert sp.smat.shape == (12, 4) and (sp.smat.sum(1) == 1).all()
    assert sp.smat[:, run.SC].sum() == 3
    assert run.SC_THRESHOLD == pytest.approx(0.2)
    s4 = run.stream_space()
    assert (s4.smat == np.eye(4)).all()


# ----------------------------------------------------------------------------- synthetic splits
def make_splits(seed=0):
    """Small synthetic splits_v3-like table: 5 sources, all 12 classes, medicines only from MedPackID/TrashBox."""
    rng = np.random.default_rng(seed)
    rows = []
    cls_src = {c: ["GD", "RealWaste", "TrashBox"] for c in run.CLASSES}
    cls_src["medicines_syringes"] = ["MedPackID", "TrashBox"]
    cls_src["trash"] = ["GD", "RealWaste"]
    cls_src["glass"] += ["ZeroWaste", "DWSD"]
    i = 0
    for c, srcs in cls_src.items():
        for s in srcs:
            for sp, n in (("train", 8), ("val", 3), ("test", 3)):
                if s == "DWSD":
                    sp = "dwsd_test" if sp == "test" else sp
                for _ in range(n):
                    tier = ""
                    if sp == "test":
                        tier = "T2" if s == "ZeroWaste" else "T1"
                    elif sp == "dwsd_test":
                        tier = "T3"
                    rows.append(dict(id=f"i{i}", source=s, orig_class="x", label=c, group=f"g{i}",
                                     norm_rel=f"{s}/{i}.jpg", phash=format(int(rng.integers(0, 2**63)), "016x"),
                                     split=sp, tier=tier))
                    i += 1
    return pd.DataFrame(rows)


@pytest.fixture
def splits():
    return run._fill_tiers(make_splits())


def test_fill_tiers_when_column_missing(splits):
    df = splits.drop(columns="tier")
    out = run._fill_tiers(df)
    assert set(out[out.split == "dwsd_test"].tier) == {"T3"}
    assert set(out[(out.split == "test") & (out.source == "ZeroWaste")].tier) == {"T2"}
    assert set(out[(out.split == "test") & (out.source == "GD")].tier) == {"T1"}
    assert set(out[out.split == "train"].tier) == {""}


def test_full_job_a1(splits):
    d = run.prepare_job_data(run.JOBS["A1"], splits)
    assert d.space.names == run.CLASSES and d.unknown is None
    assert set(d.train.split) == {"train"} and set(d.val.split) == {"val"}
    assert set(d.test.split) == {"test", "dwsd_test"}
    assert (d.train.y == d.train.class_idx).all()


@pytest.mark.parametrize("jid,held", [("B1", "battery"), ("B2", "medicines_syringes"), ("B3", "e_waste"), ("B4", "shoes")])
def test_holdout_class_label_sets(splits, jid, held):
    d = run.prepare_job_data(run.JOBS[jid], splits)
    assert held not in d.space.names and len(d.space.names) == 11
    assert d.space.smat.shape == (11, 4)
    for fr in (d.train, d.val, d.test):
        assert held not in set(fr.label)
    # unknown set = test images of the held-out class, every tier, nothing else
    expect = splits[(splits.label == held) & splits.split.isin(["test", "dwsd_test"])]
    assert set(d.unknown.id) == set(expect.id)
    assert (d.unknown.y == -1).all()
    # the held-out class is gone from train/val entirely (not just test)
    assert not ((splits.split.isin(["train", "val"])) & (splits.label == held) & splits.id.isin(d.train.id)).any()
    assert d.meta["unseen_classes"] == []
    # remaining classes are re-indexed 0..10 and still map to the right streams
    assert sorted(d.train.y.unique()) == list(range(11))
    names = np.array(d.space.names)
    assert (names[d.train.y.values] == d.train.label.values).all()


def test_holdout_class_stream_space(splits):
    d = run.prepare_job_data(run.JOBS["S3"], splits)                      # S3 holds out medicines_syringes
    assert d.space.kind == "stream" and d.space.names == run.STREAMS
    assert "medicines_syringes" not in set(d.train.label)
    assert (d.train.y == d.train.stream_true).all()
    assert (d.unknown.label == "medicines_syringes").all() and (d.unknown.y >= 0).all()
    d1 = run.prepare_job_data(run.JOBS["S1"], splits)
    assert d1.space.kind == "stream" and d1.unknown is None and set(d1.train.label) == set(run.CLASSES)


@pytest.mark.parametrize("jid,src", [("B5", "MedPackID"), ("B6", "TrashBox"), ("B7", "GD"), ("B8", "RealWaste")])
def test_holdout_source_label_sets(splits, jid, src):
    d = run.prepare_job_data(run.JOBS[jid], splits)
    for fr in (d.train, d.val, d.test):
        assert src not in set(fr.source)
    assert set(d.unknown.id) == set(splits[splits.source == src].id)          # ALL splits of that source
    assert set(d.unknown.split) >= {"train", "val", "test"}
    # a class with no training images left is dropped and recorded as unseen
    present = set(splits[(splits.split == "train") & (splits.source != src)].label)
    assert set(d.space.names) == present
    assert set(d.meta["unseen_classes"]) == set(run.CLASSES) - present
    assert d.space.smat.shape == (len(present), 4)
    # unknown rows of unseen classes get y = -1, others the label-space index
    for _, r in d.unknown.iterrows():
        if r.label in present:
            assert d.space.names[r.y] == r.label
        else:
            assert r.y == -1


def test_medpackid_leaves_medicines_to_trashbox(splits):
    d = run.prepare_job_data(run.JOBS["B5"], splits)
    assert "medicines_syringes" in d.space.names        # TrashBox still has it


def test_class_vanishes_with_its_source():
    df = make_splits()
    df = df[~((df.label == "medicines_syringes") & (df.source == "TrashBox"))]      # only MedPackID has medicines
    df = run._fill_tiers(df)
    d = run.prepare_job_data(run.JOBS["B5"], df)
    assert d.meta["unseen_classes"] == ["medicines_syringes"]
    assert len(d.space.names) == 11 and "medicines_syringes" not in d.space.names
    med = d.unknown[d.unknown.label == "medicines_syringes"]
    assert len(med) > 0 and (med.y == -1).all() and (med.stream_true == run.SC).all()


def test_unknown_source_raises(splits):
    with pytest.raises(ValueError):
        run.prepare_job_data(run.Job("Bx", "holdout_source", holdout_source="Nope"), splits)


def test_missing_files_checked_over_the_whole_job(capsys):
    df = make_splits()
    a, b = df.iloc[:100], df.iloc[100:200]
    gone = {"3.jpg", "150.jpg"}                                     # one missing file in each frame
    exists = lambda p: p.split("/")[-1] not in gone                   # noqa: E731
    # 2 of 200 = 1%: each frame alone is below 2% but the job total is above the 0.5% limit
    with pytest.raises(RuntimeError):
        run.attach_paths_all(dict(a=a, b=b), "/data", exists=exists, max_missing_frac=0.005)
    out = capsys.readouterr().out
    assert "2 of 200 image files are missing" in out and "/3.jpg" in out and "/150.jpg" in out
    res = run.attach_paths_all(dict(a=a, b=b, none=None), "/data", exists=exists, max_missing_frac=0.05)
    assert len(res["a"]) == 99 and len(res["b"]) == 99 and res["none"] is None
    assert res["a"].path.iloc[0].startswith("/data/")
    # a file shared by two frames is counted once; the printed list is capped at 20
    many = df.iloc[:300]
    capsys.readouterr()                                               # drop the output of the earlier calls
    run.attach_paths_all(dict(a=many, b=many), "/data", exists=lambda p: False, max_missing_frac=1.0)
    printed = [l for l in capsys.readouterr().out.splitlines() if l.startswith("    ")]
    assert len(printed) == 20


def test_csvs_are_read_as_text(tmp_path):
    df = make_splits().head(3).copy()
    df["phash"] = ["0000000000000001", "00ff00ff00ff00ff", "1234567890123456"]
    df["group"] = ["007", "010", "1e5"]
    df.to_csv(tmp_path / "s.csv", index=False)
    out = run.load_splits(str(tmp_path / "s.csv"))
    assert out.phash.tolist() == df.phash.tolist() and out.group.tolist() == ["007", "010", "1e5"]
    assert run.phash_array(out.phash.values)[0] == 1
    pd.DataFrame(dict(id=["a"], source=["GD"], orig_class=["x"], label=["plastic"], group=["01"], norm_rel=["a.jpg"],
                      phash=["0000000000000abc"])).to_csv(tmp_path / "a.csv", index=False)
    a = run.load_all_csv(str(tmp_path / "a.csv"))
    assert a.group.iloc[0] == "01" and a.phash.iloc[0] == "0000000000000abc"


# ----------------------------------------------------------------------------- C1 / C2
def make_pool(n_per_class=60, seed=1):
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for c in run.CLASSES:
        n_groups = n_per_class // 3
        for g in range(n_groups):
            base = int(rng.integers(0, 2**63))
            for _ in range(3):                                         # 3 crops per photo group
                h = base ^ (1 << int(rng.integers(0, 64)))             # distance 1 from the group base
                rows.append(dict(id=f"p{k}", source=["GD", "RealWaste", "TrashBox"][k % 3], orig_class="o", label=c,
                                 group=f"{c}_g{g}", norm_rel=f"p{k}.jpg", phash=format(h, "016x")))
                k += 1
    df = pd.DataFrame(rows)
    # cross-group duplicates: copy the hash of one image into an image of a different group (different split risk)
    for i in range(0, 60, 7):
        j = (i * 5 + 100) % len(df)
        df.loc[j, "phash"] = df.loc[i, "phash"]
    # a near-duplicate chain: d(a,b)=2, d(b,c)=2, d(a,c)=4 -> all linked at threshold 4
    a = 0x00FF00FF00FF00FF
    df.loc[10, "phash"] = format(a, "016x")
    df.loc[len(df) // 3, "phash"] = format(a ^ 0b11, "016x")
    df.loc[2 * len(df) // 3, "phash"] = format(a ^ 0b1111, "016x")
    return df


def test_build_pool_filters():
    rows = [
        ("DWSD", "plastic"), ("PharmaSynth", "medicines_syringes"), ("GlobalWaste", "plastic"),
        ("GlobalWaste", "e_waste"), ("GlobalWaste", "battery"), ("GlobalWaste", "sanitary"),
        ("GlobalWaste", "shoes"), ("GlobalWaste", "medicines_syringes"), ("GD", "plastic"), ("GD", None)]
    df = pd.DataFrame([dict(id=str(i), source=s, orig_class="o", label=l, group=str(i), norm_rel="x", phash="0" * 16)
                       for i, (s, l) in enumerate(rows)])
    out = run.build_pool(df)
    kept = sorted(zip(out.source, out.label))
    assert kept == sorted([("GlobalWaste", "battery"), ("GlobalWaste", "sanitary"), ("GlobalWaste", "shoes"),
                           ("GlobalWaste", "medicines_syringes"), ("GD", "plastic")])


def test_cap_pool_group_limit_class_cap_and_source_quota():
    rng = np.random.default_rng(3)
    rows = []
    # class 'plastic': source A has 1 huge group (200 images), source B 30 groups x 5, source C 10 images
    for i in range(200):
        rows.append(("A", "plastic", "A_big", i))
    for i in range(150):
        rows.append(("B", "plastic", f"B_{i // 5}", i))
    for i in range(10):
        rows.append(("C", "plastic", f"C_{i}", i))
    df = pd.DataFrame([dict(id=f"{s}{i}", source=s, orig_class="o", label=l, group=g, norm_rel="x",
                            phash=format(int(rng.integers(0, 2**63)), "016x")) for s, l, g, i in rows])
    out = run.cap_pool(df, cap=60, max_per_group=20, seed=1)
    assert out.groupby("group").size().max() <= 20
    assert len(out) == 60
    by = out.groupby("source").size().to_dict()
    assert by["C"] == 10                                   # small source fully used, its unused quota shared
    assert by["A"] == 20                                   # limited by the 20-per-group rule
    assert by["B"] == 30
    again = run.cap_pool(df, cap=60, max_per_group=20, seed=1)
    assert list(out.id) == list(again.id)                  # deterministic


def test_cap_pool_leaves_small_classes_alone():
    df = make_pool(30)
    out = run.cap_pool(df, cap=2500, max_per_group=20)
    assert len(out) == len(df)


def test_split_naive_counts_and_seed():
    s = run.split_naive(1000, seed=5)
    assert (s == "test").sum() == 200 and (s == "val").sum() == 100 and (s == "train").sum() == 700
    assert list(s) == list(run.split_naive(1000, seed=5))


def test_c2_no_group_or_duplicate_cluster_crosses_splits():
    pool = run.cap_pool(run.build_pool(make_pool(60)))
    clusters = run.duplicate_clusters(pool, thr=4)
    split, n_over = run.split_grouped(clusters, pool.label.values, seed=7)
    pool = pool.assign(split=split, cluster=clusters)
    # same photo group -> same split
    assert (pool.groupby("group").split.nunique() == 1).all()
    # same cluster -> same split
    assert (pool.groupby("cluster").split.nunique() == 1).all()
    # brute force: any two images within Hamming distance <= 4 share a split
    H = run.phash_array(pool.phash.values)
    for i, j in run.near_pairs(H, 4):
        assert split[i] == split[j]
    for i in range(len(H)):                                   # independent check of near_pairs
        d = np.array([bin(int(H[i]) ^ int(h)).count("1") for h in H])
        for j in np.nonzero(d <= 4)[0]:
            assert split[i] == split[j]
    # the three-link chain made in make_pool ended in one cluster
    hs = pool.set_index("id").phash
    assert len({clusters[k] for k in range(len(pool)) if pool.phash.iloc[k] in
                {format(0x00FF00FF00FF00FF, "016x"), format(0x00FF00FF00FF00FF ^ 3, "016x"),
                 format(0x00FF00FF00FF00FF ^ 15, "016x")}}) == 1
    # fractions close to 70/10/20 overall and for every class (the split is filled class by class)
    frac = pool.split.value_counts(normalize=True)
    assert abs(frac["test"] - 0.2) < 0.05 and abs(frac["val"] - 0.1) < 0.05
    rep = run.split_report(split, pool.label.values, clusters, n_over)
    for c, v in rep["per_class"].items():
        assert 0.1 <= v["test"] <= 0.4 and v["n_test"] > 0 and v["train"] > 0.4, (c, v)
    assert rep["n_clusters"] == len(set(clusters)) and rep["largest_cluster_size"] == max(np.bincount(np.unique(clusters, return_inverse=True)[1]))
    run.check_grouped_split(rep)                                      # no error on a good split


def test_grouped_split_oversize_cluster_and_error_checks():
    labels = np.array(["a"] * 100 + ["b"] * 100)
    clusters = np.concatenate([np.zeros(30, int), np.arange(1, 71), 100 + np.arange(100)])   # one 30-image cluster = 15% of pool
    split, n_over = run.split_grouped(clusters, labels, seed=1)
    assert n_over == 1 and set(split[:30]) == {"train"}
    rep = run.split_report(split, labels, clusters, n_over)
    assert rep["largest_cluster_size"] == 30 and rep["n_oversize_clusters_sent_to_train"] == 1
    assert rep["per_class"]["b"]["test"] == pytest.approx(0.2) and rep["per_class"]["a"]["test"] == pytest.approx(0.2)
    run.check_grouped_split(rep)
    bad = dict(overall=dict(train=0.6, val=0.1, test=0.3), per_class=dict(a=dict(n_test=5)))
    with pytest.raises(ValueError, match="outside 0.15-0.25"):
        run.check_grouped_split(bad)
    bad2 = dict(overall=dict(train=0.7, val=0.1, test=0.2), per_class=dict(a=dict(n_test=5), z=dict(n_test=0)))
    with pytest.raises(ValueError, match="no test images"):
        run.check_grouped_split(bad2)
    d2 = run.prepare_job_data(run.JOBS["C2"], None, make_pool(60))     # report is stored in the job meta
    assert d2.meta["split_report"]["n_clusters"] > 0 and "largest_cluster_size" in d2.meta["split_report"]


def test_c1_and_c2_share_the_test_fraction_and_pool():
    pool_df = make_pool(60)
    d1 = run.prepare_job_data(run.JOBS["C1"], None, pool_df)
    d2 = run.prepare_job_data(run.JOBS["C2"], None, pool_df)
    assert d1.meta["pool_size"] == d2.meta["pool_size"]
    assert abs(d1.meta["test_fraction_actual"] - d2.meta["test_fraction_actual"]) < 0.05
    assert set(d1.test.tier) == {"pool_test"} and d1.space.names == run.CLASSES
    # naive split lets duplicates and photo groups cross; grouped does not
    groups_cross = lambda d: len(set(d.train.group) & set(d.test.group))        # noqa: E731
    assert groups_cross(d1) > 0 and groups_cross(d2) == 0


def test_leakage_block_hand_case():
    # hashes: test0 equals train0; test1 is 4 bits from train1; test2 is 6 bits from everything
    tr = pd.DataFrame(dict(phash=[format(0, "016x"), format(0xFFFF, "016x")]))
    te = pd.DataFrame(dict(phash=[format(0, "016x"), format(0xFFFF ^ 0b1111, "016x"), format(0x3F << 20, "016x")],
                           y=[0, 1, 2]))
    p = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])        # correct, correct, wrong
    out = run.leakage_block(tr, te, p, thr=4)
    assert out["test_with_near_duplicate_in_train"] == pytest.approx(2 / 3)
    assert out["acc_items_with_duplicate"] == 1.0
    assert out["acc_items_without_duplicate"] == 0.0
    assert run.popcount64(np.array([0xFF, 0], dtype=np.uint64)).tolist() == [8, 0]


# ----------------------------------------------------------------------------- metrics
def probs_from(rows):
    """rows: list of {class: prob} over the 12 classes (rest zero)."""
    p = np.zeros((len(rows), 12))
    for i, r in enumerate(rows):
        for c, v in r.items():
            p[i, run.CLASS_TO_IDX[c]] = v
    return p


SPACE = run.class_space(run.CLASSES)


def test_summed_stream_vs_top_class_and_flags():
    rows = [
        {"plastic": 0.4, "battery": 0.3, "e_waste": 0.3},          # true battery: top class Dry, summed SC = 0.6
        {"battery": 0.9, "plastic": 0.1},                           # true battery: caught both ways
        {"plastic": 0.85, "battery": 0.15},                         # true battery: SC 0.15 -> flagged at 0.33? no; 0.20? no; 0.10 yes
        {"plastic": 0.7, "battery": 0.3},                           # true plastic: SC 0.3 -> false alarm at 0.20 and 0.10, not 0.33
        {"plastic": 0.95, "battery": 0.05},                         # true plastic: no alarm at any threshold
        {"biological": 1.0},                                        # true biological
    ]
    y = np.array([run.CLASS_TO_IDX[c] for c in
                  ["battery", "battery", "battery", "plastic", "plastic", "biological"]])
    s_true = run.CLASS_STREAM_IDX[y]
    m = run.metrics(probs_from(rows), y, SPACE, s_true)
    # summed-stream accuracy: item0 correct (SC 0.6), item1 correct, item2 wrong (Dry 0.85), item3 correct (Dry), 4, 5 correct
    assert m["stream_acc"] == pytest.approx(5 / 6)
    # top-class stream accuracy: item0 wrong (plastic), item1 correct, item2 wrong, 3, 4, 5 correct
    assert m["stream_acc_top_class"] == pytest.approx(4 / 6)
    # flag recall over the 3 battery items
    assert m["sc_recall@0.33"] == pytest.approx(2 / 3)                   # 0.6 and 0.9 flagged
    assert m["sc_recall@0.20"] == pytest.approx(2 / 3)
    assert m["sc_recall@0.10"] == pytest.approx(3 / 3)                   # 0.15 now flagged
    # false alarm over the 3 non-hazard items
    assert m["sc_false_alarm@0.33"] == pytest.approx(0.0)
    assert m["sc_false_alarm@0.20"] == pytest.approx(1 / 3)
    assert m["sc_false_alarm@0.10"] == pytest.approx(1 / 3)               # SC 0.3 flagged; 0.05 and 0.0 are below 0.10
    assert m["sc_recall_top_class"] == pytest.approx(1 / 3)              # only item1
    assert m["sc_recall_argmax"] == pytest.approx(2 / 3)                 # summed argmax: items 0, 1
    assert m["sc_false_alarm_top_class"] == 0.0
    assert m["always_dry_baseline"] == pytest.approx(2 / 6)
    assert m["n_Special Care"] == 3 and m["n_Dry"] == 2 and m["n_Wet"] == 1
    assert m["recall_Wet"] == 1.0 and m["recall_Dry"] == 1.0
    assert m["recall_Special Care"] == pytest.approx(2 / 3)
    assert m["class_acc"] == pytest.approx(4 / 6)                         # top class right for items 1, 3, 4, 5


def test_class_acc_hand_case_and_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    pred = np.where(rng.random(200) < 0.7, y, rng.integers(0, 5, 200))
    assert run.macro_f1(y, pred) == pytest.approx(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0))
    y2 = np.array([0, 0, 1])
    pred2 = np.array([0, 2, 1])                                           # class 2 never true: ignored like sklearn labels=unique(y)
    assert run.macro_f1(y2, pred2) == pytest.approx(f1_score(y2, pred2, average="macro", labels=np.unique(y2), zero_division=0))


def test_metrics_with_unseen_class_images():
    # 11-class space without battery; a battery image has y=-1 but counts for stream metrics
    names = [c for c in run.CLASSES if c != "battery"]
    sp = run.class_space(names)
    p = np.zeros((2, 11))
    p[0, names.index("e_waste")] = 1.0                                    # true battery (y=-1) routed to Special Care by e_waste
    p[1, names.index("plastic")] = 1.0                                    # true plastic
    y = np.array([-1, names.index("plastic")])
    s_true = np.array([run.SC, run.DRY])
    m = run.metrics(p, y, sp, s_true)
    assert m["n_class_scored"] == 1 and m["class_acc"] == 1.0
    assert m["stream_acc"] == 1.0 and m["sc_recall@0.20"] == 1.0 and m["sc_false_alarm@0.20"] == 0.0


def test_metrics_stream_label_space():
    s4 = run.stream_space()
    p = np.array([[0.1, 0.1, 0.7, 0.1], [0.6, 0.2, 0.1, 0.1]])
    y = np.array([2, 1])
    m = run.metrics(p, y, s4, y)
    assert m["class_acc"] == 0.5 and m["stream_acc"] == 0.5 and m["sc_recall@0.33"] == 1.0


def test_temperature_fit_toy_cases():
    rng = np.random.default_rng(0)
    n = 20000
    z = rng.normal(0, 1.5, (n, 4))                                        # logits of a calibrated model
    probs = softmax(z, 1)
    y = np.array([rng.choice(4, p=pr) for pr in probs])
    assert run.fit_temperature(z, y) == pytest.approx(1.0, abs=0.08)
    assert run.fit_temperature(3.0 * z, y) == pytest.approx(3.0, rel=0.1)  # overconfident -> T near 3
    assert run.fit_temperature(0.5 * z, y) == pytest.approx(0.5, rel=0.1)  # underconfident -> T below 1
    T = run.fit_temperature(3.0 * z, y)
    p_raw, p_cal = softmax(3.0 * z, 1), softmax(3.0 * z / T, 1)
    assert run.ece(p_cal, y) < run.ece(p_raw, y)


def test_ece_hand_case():
    p = np.tile([0.9, 0.1], (10, 1))
    y = np.array([0] * 5 + [1] * 5)                                       # confidence 0.9, accuracy 0.5
    assert run.ece(p, y) == pytest.approx(0.4)
    assert run.ece(np.array([[1.0, 0.0]] * 4), np.zeros(4, int)) == pytest.approx(0.0)


def test_ood_scores_msp_maxlogit_energy():
    logits = np.array([[2.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    s = run.ood_scores(logits, 1.0)
    assert s["MSP"][0] == pytest.approx(math.exp(2) / (math.exp(2) + 2))
    assert s["MSP"][1] == pytest.approx(1 / 3)
    assert s["MaxLogit"].tolist() == [2.0, 1.0]
    assert s["Energy"][0] == pytest.approx(math.log(math.exp(2) + 2))
    s2 = run.ood_scores(logits, 2.0)                                      # T scales MSP and Energy, not MaxLogit
    assert s2["MSP"][0] == pytest.approx(math.exp(1) / (math.exp(1) + 2))
    assert s2["MaxLogit"].tolist() == [2.0, 1.0]
    assert s2["Energy"][0] == pytest.approx(2 * math.log(math.exp(1) + 2))
    assert np.allclose(s["Energy"], logsumexp(logits, axis=1))


def test_pick_threshold_and_selective():
    conf = np.array([0.99, 0.95, 0.9, 0.6, 0.5])
    correct = np.array([1, 1, 1, 0, 0], bool)
    th, ok = run.pick_threshold(conf, correct, 0.97)
    assert ok and th == pytest.approx(0.9)                                # top 3 are 100% accurate; adding the 4th gives 0.75
    sel = run.selective(conf, correct, th)
    assert sel["coverage"] == pytest.approx(0.6) and sel["accuracy_answered"] == 1.0
    th2, ok2 = run.pick_threshold(conf, np.zeros(5, bool), 0.97)
    assert not ok2


def test_thresholds_and_results_use_validation_only():
    rng = np.random.default_rng(2)
    K = 12
    y_val = rng.integers(0, K, 600)
    val_lg = rng.normal(0, 1, (600, K)); val_lg[np.arange(600), y_val] += 4
    y_te = rng.integers(0, K, 400)
    te_lg = rng.normal(0, 1, (400, K)); te_lg[np.arange(400), y_te] += 4
    tier = np.array(["T1"] * 300 + ["T3"] * 100)
    val = dict(logits=val_lg, y=y_val, stream_true=run.CLASS_STREAM_IDX[y_val])
    test = dict(logits=te_lg, y=y_te, stream_true=run.CLASS_STREAM_IDX[y_te], tier=tier)
    unk_lg = rng.normal(0, 1, (80, K))
    unk_tier = np.array(["T1"] * 50 + ["T3"] * 30)
    unk = dict(logits=unk_lg, y=-np.ones(80, int), stream_true=np.full(80, run.SC), tier=unk_tier)
    r1 = run.compute_results(SPACE, val, test, unk)
    # changing the test logits must not change temperature or thresholds
    test2 = dict(test, logits=te_lg * 5)
    r2 = run.compute_results(SPACE, val, test2, unk)
    assert r1["temperature"] == r2["temperature"] and r1["thresholds"] == r2["thresholds"]
    th = r1["thresholds"]
    assert th["hazard"] == pytest.approx(0.2)
    s = run.ood_scores(val_lg, r1["temperature"])
    assert th["reject_msp"] == pytest.approx(np.quantile(s["MSP"], 0.05))
    assert (s["MSP"] < th["reject_msp"]).mean() <= 0.06
    assert set(r1["metrics"]) == {"val", "test_T1", "test_T3", "test_all"}
    blk = r1["metrics"]["test_T1"]
    for k in ("class_acc", "macro_f1", "stream_acc", "stream_acc_top_class", "recall_Wet", "always_dry_baseline",
              "sc_recall@0.33", "sc_recall@0.20", "sc_recall@0.10", "sc_false_alarm@0.20", "sc_recall_top_class",
              "ece_before", "ece_after", "reject_stream", "reject_class"):
        assert k in blk
    assert blk["n"] == 300
    u = r1["unknown_all_tiers"]
    assert u["n"] == 80 and 0 <= u["rejected"] <= 1
    assert u["rejected_or_flagged"] >= max(u["rejected"], u["flagged_special_care"]) - 1e-12
    routed = sum(u[f"silently_routed_{n}"] for n in run.STREAMS)
    assert routed + u["rejected_or_flagged"] == pytest.approx(1.0)
    for k in ("auroc_MSP", "auroc_MaxLogit", "auroc_Energy", "fpr95_MSP", "fpr95_Energy"):
        assert k in u
    assert u["auroc_MSP"] > 0.9                                           # random logits are far less peaked than known ones


def test_unknown_t1_uses_t1_known_and_t1_unknown_only():
    rng = np.random.default_rng(5)
    K = 12
    y_val = rng.integers(0, K, 300)
    val_lg = rng.normal(0, 1, (300, K)); val_lg[np.arange(300), y_val] += 4
    y_te = rng.integers(0, K, 200)
    te_lg = rng.normal(0, 1, (200, K)); te_lg[np.arange(200), y_te] += 4
    tier = np.array(["T1"] * 120 + ["T3"] * 80)
    te_lg[120:] = rng.normal(0, 0.1, (80, K))                         # T3 known items look unfamiliar
    val = dict(logits=val_lg, y=y_val, stream_true=run.CLASS_STREAM_IDX[y_val])
    test = dict(logits=te_lg, y=y_te, stream_true=run.CLASS_STREAM_IDX[y_te], tier=tier)
    unk_lg = rng.normal(0, 1, (60, K))
    unk = dict(logits=unk_lg, y=-np.ones(60, int), stream_true=np.full(60, run.SC),
               tier=np.array(["T1"] * 40 + ["T2"] * 20))
    r = run.compute_results(SPACE, val, test, unk)
    T, th = r["temperature"], r["thresholds"]
    t1, allt = r["unknown_T1"], r["unknown_all_tiers"]
    assert t1["n"] == 40 and allt["n"] == 60
    exp = run.unknown_block(unk_lg[:40], T, SPACE, th, te_lg[:120])   # T1 known items only
    assert t1 == exp
    assert allt == run.unknown_block(unk_lg, T, SPACE, th, te_lg)
    assert t1["known_rejected"] != allt["known_rejected"]
    # left-out source: all rows of the source, known comparison = T1 test rows
    unk2 = dict(unk, y=rng.integers(0, K, 60))
    r2 = run.compute_results(SPACE, val, test, unk2, unknown_is_labelled=True)
    assert r2["unknown_T1"]["n"] == 60 and r2["unknown_all_tiers"]["n"] == 60
    assert r2["unknown_T1"] == run.unknown_block(unk_lg, T, SPACE, th, te_lg[:120])
    assert r2["unknown_T1"]["known_rejected"] != r2["unknown_all_tiers"]["known_rejected"]


def test_unknown_block_hand_case():
    # 3 classes in the label space over 12-class names is awkward; use the 4-stream space
    s4 = run.stream_space()
    th = dict(hazard=0.2, reject_msp=0.5, reject=dict(MSP=0.5, MaxLogit=0.0, Energy=-1.0))
    logits = np.log(np.array([[0.05, 0.9, 0.04, 0.01],     # confident Dry, SC 0.04: silently routed Dry
                              [0.30, 0.30, 0.30, 0.10],    # MSP 0.3 < 0.5: rejected; SC 0.3 flagged
                              [0.1, 0.1, 0.7, 0.1]]))      # confident Special Care: flagged
    u = run.unknown_block(logits, 1.0, s4, th)
    assert u["rejected"] == pytest.approx(1 / 3)
    assert u["flagged_special_care"] == pytest.approx(2 / 3)
    assert u["rejected_or_flagged"] == pytest.approx(2 / 3)
    assert u["silently_routed_Dry"] == pytest.approx(1 / 3)
    assert u["silently_routed_Wet"] == 0 and u["silently_routed_Sanitary"] == 0


def test_holdout_source_results_include_unseen_class_block():
    df = make_splits()
    df = run._fill_tiers(df[~((df.label == "medicines_syringes") & (df.source == "TrashBox"))])
    d = run.prepare_job_data(run.JOBS["B5"], df)                          # medicines vanish with MedPackID
    rng = np.random.default_rng(1)

    def fake(fr):
        lg = rng.normal(0, 1, (len(fr), len(d.space.names)))
        y = fr.y.values
        lg[np.arange(len(fr))[y >= 0], y[y >= 0]] += 3
        return dict(logits=lg, y=y, stream_true=fr.stream_true.values, tier=fr.tier.values)
    res = run.compute_results(d.space, fake(d.val), fake(d.test), fake(d.unknown), unknown_is_labelled=True)
    assert res["left_out_source_metrics"]["n"] == len(d.unknown)
    assert res["unknown_T1"]["n"] == len(d.unknown)
    assert (d.unknown.y < 0).any() and res["unseen_class_images"]["n"] == int((d.unknown.y < 0).sum())


def test_pack_predictions_and_json_cleaning(tmp_path, splits):
    d = run.prepare_job_data(run.JOBS["B1"], splits)
    rng = np.random.default_rng(0)
    parts = {"val": (d.val, rng.normal(size=(len(d.val), 11))), "test": (d.test, rng.normal(size=(len(d.test), 11))),
             "unknown": (d.unknown, rng.normal(size=(len(d.unknown), 11)))}
    arr = run.pack_predictions(parts, 1.7, d.space)
    n = len(d.val) + len(d.test) + len(d.unknown)
    assert arr["logits"].shape == (n, 11) and len(arr["ids"]) == n and len(arr["msp"]) == n
    assert set(arr["role"]) == {"val", "test", "unknown"} and float(arr["temperature"]) == 1.7
    path = str(tmp_path / "out" / "B1_logits.npz")
    run.write_npz_atomic(path, **arr)
    z = np.load(path)                                                     # loads without pickle
    assert z["logits"].shape == (n, 11) and z["label_names"].tolist() == d.space.names
    run.write_json_atomic(str(tmp_path / "x.json"), dict(a=np.float32(0.5), b=float("nan"), c=np.int64(3), d=[np.nan]))
    assert json.load(open(tmp_path / "x.json")) == dict(a=0.5, b=None, c=3, d=[None])


# ----------------------------------------------------------------------------- resume / skip
def test_skip_logic_with_fake_result(tmp_path):
    root = str(tmp_path)
    assert not run.job_is_done(root, "A1")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "A1.json").write_text("{}")
    assert run.job_is_done(root, "A1") and not run.job_is_done(root, "A2")
    called = []

    def runner(jid):
        called.append(jid)
        if jid == "A3":
            raise RuntimeError("boom")
        (tmp_path / "results" / f"{jid}.json").write_text("{}")
    out = run.run_queue(["A1", "A2", "A3", "B1"], root, runner)
    assert called == ["A2", "A3", "B1"]                                   # A1 skipped
    assert out["skipped"] == ["A1"] and out["done"] == ["A2", "B1"] and list(out["failed"]) == ["A3"]
    called.clear()
    out = run.run_queue(["A1", "A2", "A3", "B1"], root, runner)           # rerun: only the failed job runs again
    assert called == ["A3"]


def test_ensure_extracted_once(tmp_path):
    tar = tmp_path / "n.tar"
    tar.write_bytes(b"x")
    calls = []
    fake_run = lambda cmd, check: calls.append(cmd)                       # noqa: E731
    root = str(tmp_path / "data")
    assert run.ensure_extracted(str(tar), root, run=fake_run) is True
    assert run.ensure_extracted(str(tar), root, run=fake_run) is False    # marker present: no second extraction
    assert len(calls) == 1 and calls[0][:2] == ["tar", "-xf"]
    with pytest.raises(FileNotFoundError):
        run.ensure_extracted(str(tmp_path / "missing.tar"), str(tmp_path / "other"), run=fake_run)


# ----------------------------------------------------------------------------- phase tracker and resumable loop
def test_phase_tracker_keras_semantics():
    t = run.PhaseTracker(patience=3, lr=1e-3)                             # plateau patience = 1
    assert t.update(1.0)["improved"] and t.lr == 1e-3
    assert t.update(0.5)["improved"] and t.lr == 1e-3
    assert not t.update(0.5)["improved"]                                  # equal is not an improvement
    assert t.lr == pytest.approx(3e-4)                                    # plateau patience 1 -> reduced
    assert not t.stop
    assert not t.update(0.6)["improved"]
    assert t.lr == pytest.approx(9e-5) and not t.stop                     # es_wait now 2 (since best at epoch 2)
    t.update(0.7)
    assert t.stop                                                         # 3 epochs without improvement
    t2 = run.PhaseTracker.from_dict(json.loads(json.dumps(t.to_dict())))
    assert t2.stop and t2.lr == t.lr and t2.keep_best == 0.5
    f = run.PhaseTracker(patience=4, lr=1e-5)
    assert f.rl_patience == 2
    f.update(1.0); f.update(1.0); assert f.lr == 1e-5
    f.update(1.0); assert f.lr == pytest.approx(3e-6)
    # a drop smaller than 1e-4 does not reset the plateau counter but is still the best weights
    g = run.PhaseTracker(patience=4, lr=1e-3)
    g.update(1.0)
    assert g.update(1.0 - 5e-5)["improved"] and g.rl_wait == 1


class FakeTrainer:
    """Same interface as KerasTrainer. 'Weights' are one number that grows each epoch; val_loss comes from a table."""

    def __init__(self, losses, crash_at=None):
        self.losses, self.crash_at, self.w, self.phases, self.log = losses, crash_at, 0, [], []

    def build(self, resumed, epochs_done=0):
        self.log.append(("build", resumed, epochs_done))

    def load_weights(self, path):
        self.w = json.load(open(path))["w"]
        self.log.append(("load", os.path.basename(path)))

    def save_weights(self, path):
        json.dump({"w": self.w}, open(path, "w"))

    def start_phase(self, name, lr):
        self.log.append(("phase", name, lr))

    def run_epoch(self, phase, epoch_index, lr):
        if self.crash_at is not None and epoch_index == self.crash_at:
            raise RuntimeError("disconnect")
        self.w = epoch_index + 1
        self.log.append(("epoch", phase, epoch_index, lr))
        return dict(loss=1.0, acc=0.5, val_loss=self.losses[epoch_index], val_acc=0.6)


# head: 6 epochs, best at epoch index 2 (loss .5); finetune: 8 epochs, best at index 8+ ...
LOSSES = [1.0, 0.8, 0.5, 0.6, 0.55, 0.7,            # head phase: epochs 0-5
          0.45, 0.40, 0.42, 0.41, 0.39, 0.395, 0.40, 0.41]    # finetune: epochs 6-13


def test_run_training_uninterrupted(tmp_path):
    ft = FakeTrainer(LOSSES)
    st = run.run_training(str(tmp_path / "r"), ft, log=lambda *_: None)
    assert st["done"] and st["epochs_done"] == 14 and st["final_file"] == "final.weights.h5"
    assert [h["phase"] for h in st["history"]].count("head") == 6
    # head phase ended after 6 epochs (patience 3: best at idx 2, then 3 non-improvements -> early stop after idx 5)
    # finetune restores the best epoch of its own phase: best loss 0.39 at epoch index 10 -> w = 11
    assert ft.w == 11
    assert json.load(open(tmp_path / "r" / "final.weights.h5"))["w"] == 11
    # no stray per-epoch checkpoints left, nothing unreferenced
    left = sorted(os.listdir(tmp_path / "r"))
    assert left == ["final.weights.h5", "state.json"]
    # the head phase hands its best (epoch index 2 -> w=3) to the finetune phase
    loads = [e for e in ft.log if e[0] == "load"]
    assert len(loads) == 2                                                # one restore per phase end


def test_run_training_early_stop_head_phase(tmp_path):
    losses = [1.0, 0.9, 0.95, 0.96, 0.97, 0.99] + [0.8] * 8
    st = run.run_training(str(tmp_path / "r"), FakeTrainer(losses), log=lambda *_: None)
    heads = [h for h in st["history"] if h["phase"] == "head"]
    assert len(heads) == 5                                                # best at idx 1, three worse epochs (idx 2-4) -> stop
    assert st["epochs_done"] == len(st["history"])


@pytest.mark.parametrize("crash_at", [0, 2, 5, 6, 9, 11, 13])
def test_run_training_resume_matches_uninterrupted(tmp_path, crash_at):
    ref = run.run_training(str(tmp_path / "ref"), FakeTrainer(LOSSES), log=lambda *_: None)
    d = str(tmp_path / "x")
    with pytest.raises(RuntimeError):
        run.run_training(d, FakeTrainer(LOSSES, crash_at=crash_at), log=lambda *_: None)
    state_mid = run.load_state(d)
    assert state_mid is not None or crash_at == 0
    tr = FakeTrainer(LOSSES)
    st = run.run_training(d, tr, log=lambda *_: None)                     # "run the same cell again"
    assert st["done"] and st["epochs_done"] == ref["epochs_done"]
    assert [(h["phase"], h["epoch"], h["val_loss"], h["lr"]) for h in st["history"]] == \
           [(h["phase"], h["epoch"], h["val_loss"], h["lr"]) for h in ref["history"]]
    assert json.load(open(os.path.join(d, "final.weights.h5"))) == json.load(open(tmp_path / "ref" / "final.weights.h5"))
    if crash_at > 0:
        assert ("build", True, crash_at) in tr.log                        # resumed, not restarted
        assert not any(e[0] == "epoch" and e[2] < crash_at for e in tr.log)   # finished epochs are not repeated


def test_run_training_finished_state_is_reused(tmp_path):
    d = str(tmp_path / "r")
    run.run_training(d, FakeTrainer(LOSSES), log=lambda *_: None)
    tr = FakeTrainer(LOSSES)
    st = run.run_training(d, tr, log=lambda *_: None)
    assert st["done"] and not any(e[0] == "epoch" for e in tr.log) and tr.w == 11


# ----------------------------------------------------------------------------- E1 pure parts
def test_clip_prompts_and_scores():
    assert len(run.CLIP_CLASS_PROMPTS) == 12 and all(p.startswith("a photo of ") for p in run.CLIP_CLASS_PROMPTS)
    assert "a photo of a battery" in run.CLIP_CLASS_PROMPTS
    assert "a photo of medicine or syringes" in run.CLIP_CLASS_PROMPTS
    assert "a photo of electronic waste" in run.CLIP_CLASS_PROMPTS
    assert "a photo of a sanitary pad or diaper" in run.CLIP_CLASS_PROMPTS
    assert "a photo of food or garden waste" in run.CLIP_CLASS_PROMPTS
    assert run.CLIP_HAZARD_PROMPTS[0] == "a photo of a hazardous household item such as a battery, medicine or electronic waste"
    assert run.CLIP_HAZARD_PROMPTS[1] == "a photo of ordinary non-hazardous waste"
    txt = np.eye(14)                                                      # 12 class prompts + 2 hazard prompts, orthonormal
    img = np.zeros((2, 14)); img[0, run.CLASS_TO_IDX["battery"]] = 1; img[1, run.CLASS_TO_IDX["plastic"]] = 1
    img[0, 12] = 1                                                        # image 0 also matches the hazard prompt
    sc = run.clip_scores(img, txt[:12], txt[12:], logit_scale=100.0)
    assert sc["class_probs"].shape == (2, 12) and sc["stream_probs"].shape == (2, 4)
    assert sc["class_probs"][0].argmax() == run.CLASS_TO_IDX["battery"]
    assert sc["stream_probs"][0, run.SC] > 0.99 and sc["stream_probs"][1, run.DRY] > 0.99
    assert sc["hazard_prob"][0] > 0.99 and sc["hazard_prob"][1] == pytest.approx(0.5)


def test_clip_summary_structure(splits):
    n = len(splits)
    rng = np.random.default_rng(0)
    img = rng.normal(size=(n, 16)); img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt = rng.normal(size=(14, 16)); txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    sc = run.clip_scores(img, txt[:12], txt[12:])
    out = run.clip_summary(sc, splits)
    assert {"val", "test_all", "test_T1", "test_T2", "test_T3"} <= set(out["metrics"])
    assert "hazard_prompt_recall@0.50" in out["metrics"]["test_T1"]
    assert {"battery", "medicines_syringes", "e_waste", "shoes"} <= set(out["per_class_flag_rates_on_test"])
    row = out["per_class_flag_rates_on_test"]["battery"]
    assert "stream_sum_flagged@0.20" in row and "hazard_prompt_flagged@0.20" in row


# ----------------------------------------------------------------------------- whole job with a fake trainer
class FakeKeras(FakeTrainer):
    """Stands in for KerasTrainer so run_training_job can be driven end to end without TensorFlow."""
    instances = []

    def __init__(self, data, seed, cache_dir):
        super().__init__([1.0 - 0.01 * i for i in range(14)])      # improves every epoch -> all 14 epochs run
        self.data, self.k = data, len(data.space.names)
        FakeKeras.instances.append(self)

    def predict(self, frame, cache=False):
        rng = np.random.default_rng(len(frame))
        lg = rng.normal(0, 1, (len(frame), self.k))
        y = frame["y"].values
        lg[np.arange(len(frame))[y >= 0], y[y >= 0]] += 3.0
        return lg.astype(np.float32)


@pytest.fixture
def fake_env(tmp_path, monkeypatch, splits):
    sp = tmp_path / "splits_v3.csv"
    splits.to_csv(sp, index=False)
    allc = tmp_path / "all.csv"
    make_pool(60).to_csv(allc, index=False)
    monkeypatch.setattr(run, "KerasTrainer", FakeKeras)
    monkeypatch.setattr(run, "require_gpu", lambda: "fake-gpu")
    monkeypatch.setattr(run, "ensure_extracted", lambda tar, root, extra=None, **kw: False)
    monkeypatch.setattr(run, "attach_paths_all", lambda frames, root, **kw: {
        k: (None if f is None else f.assign(path=root + "/" + f["norm_rel"])) for k, f in frames.items()})
    FakeKeras.instances.clear()
    drive = tmp_path / "drive"
    base = ["--drive-root", str(drive), "--data-root", "/nodata", "--splits", str(sp), "--all-csv", str(allc),
            "--tar", "/nonexistent.tar"]
    return drive, base


def test_end_to_end_b1_writes_result_and_logits_then_skips(fake_env):
    drive, base = fake_env
    assert run.main(["--job", "B1"] + base) == 0
    res = json.load(open(drive / "results" / "B1.json"))
    z = np.load(drive / "results" / "B1_logits.npz")
    assert res["job"] == "B1" and res["seed"] == 42 and res["held_out_class"] == "battery"
    assert "battery" not in res["label_names"] and len(res["label_names"]) == 11
    assert res["epochs_run"]["total"] == 14 and res["epochs_run"]["head"] == 6 and res["epochs_run"]["finetune"] == 8
    assert res["hazard_threshold"] == pytest.approx(0.2) and res["thresholds"]["reject_msp"] > 0
    assert res["temperature"] > 0 and res["training_minutes"] >= 0
    assert {"val", "test_T1", "test_T2", "test_T3", "test_all"} <= set(res["metrics"])
    assert res["unknown_all_tiers"]["n"] == int((z["role"] == "unknown").sum())
    n_t1 = int(((z["role"] == "unknown") & (z["tier"] == "T1")).sum())
    assert res["unknown_T1"]["n"] == n_t1 and 0 < n_t1 <= res["unknown_all_tiers"]["n"]
    assert set(z["role"]) == {"val", "test", "unknown"}
    assert (z["label_idx"][z["role"] == "unknown"] == -1).all()
    # logits saved for every val and test image
    d = run.prepare_job_data(run.JOBS["B1"], run._fill_tiers(make_splits()))
    assert (z["role"] == "val").sum() == len(d.val) and (z["role"] == "test").sum() == len(d.test)
    assert z["msp"].shape == z["maxlogit"].shape == z["energy"].shape == (len(z["ids"]),)
    n_inst = len(FakeKeras.instances)
    assert run.main(["--job", "B1"] + base) == 0                         # finished job is skipped
    assert len(FakeKeras.instances) == n_inst


def test_end_to_end_holdout_source_leakage_and_stream_jobs(fake_env):
    drive, base = fake_env
    for jid in ("B5", "C1", "C2", "S3"):
        assert run.main(["--job", jid] + base) == 0
    b5 = json.load(open(drive / "results" / "B5.json"))
    assert b5["held_out_source"] == "MedPackID" and "left_out_source_metrics" in b5 and b5["unknown_T1"]["n"] > 0
    c1, c2 = (json.load(open(drive / "results" / f"{j}.json")) for j in ("C1", "C2"))
    assert c1["leakage"]["test_with_near_duplicate_in_train"] > c2["leakage"]["test_with_near_duplicate_in_train"]
    assert set(c1["metrics"]) >= {"val", "test_pool_test", "test_all"}
    s3 = json.load(open(drive / "results" / "S3.json"))
    assert s3["label_space"] == "stream" and s3["label_names"] == run.STREAMS and s3["unknown_T1"]["n"] > 0


def test_interrupted_job_resumes_on_rerun(fake_env, monkeypatch):
    drive, base = fake_env
    orig = FakeKeras.run_epoch

    def crash(self, phase, epoch_index, lr):
        if epoch_index == 7 and not getattr(FakeKeras, "crashed", False):
            FakeKeras.crashed = True
            raise RuntimeError("runtime disconnected")
        return orig(self, phase, epoch_index, lr)
    monkeypatch.setattr(FakeKeras, "run_epoch", crash)
    monkeypatch.setattr(FakeKeras, "crashed", False, raising=False)
    q = drive.parent / "q.txt"
    q.write_text("A1\n")
    base = base + ["--in-process"]
    assert run.main(["--queue", str(q)] + base) == 1                     # failure reported through the exit code
    assert not (drive / "results" / "A1.json").exists()
    assert run.load_state(str(drive / "runs" / "A1"))["epochs_done"] == 7
    assert run.main(["--queue", str(q)] + base) == 0                     # same command again
    assert json.load(open(drive / "results" / "A1.json"))["epochs_run"]["total"] == 14
    assert ("build", True, 7) in FakeKeras.instances[-1].log


# ----------------------------------------------------------------------------- review fixes
# item 1 and 8: balanced streams
def test_training_streams_shuffle_once_and_balance_over_12_classes():
    df = run._fill_tiers(make_splits())
    d = run.prepare_job_data(run.JOBS["A1"], df)
    tr = d.train.sort_values(["label", "source"]).reset_index(drop=True)          # rows sorted by source, as in the CSV
    streams = run.training_streams(tr)
    assert [s["class_idx"] for s in streams] == list(range(12)) and [s["y"] for s in streams] == list(range(12))
    again = run.training_streams(tr)
    for a, b in zip(streams, again):
        assert list(a["frame"].id) == list(b["frame"].id)                          # fixed seed: same order every time
    for s in streams:
        f = s["frame"]
        assert len(f) == int((tr.class_idx == s["class_idx"]).sum())
        sources = sorted(set(f.source))
        if len(sources) > 1:                                                       # sources are mixed along the stream
            half = len(f) // 2
            for src in sources:
                share_all = (f.source == src).mean()
                assert abs((f.source.iloc[:half] == src).mean() - share_all) < 0.25
        assert f.id.tolist() != sorted(f.id.tolist()) or len(f) < 3


def test_stream_jobs_balance_over_classes_but_label_with_streams():
    df = run._fill_tiers(make_splits())
    s1 = run.training_streams(run.prepare_job_data(run.JOBS["S1"], df).train)
    assert len(s1) == 12 and [s["class_idx"] for s in s1] == list(range(12))       # same 12 streams as the class model
    assert [s["y"] for s in s1] == [int(run.CLASS_STREAM_IDX[c]) for c in range(12)]
    s2 = run.training_streams(run.prepare_job_data(run.JOBS["S2"], df).train)       # battery held out
    assert len(s2) == 11 and run.CLASS_TO_IDX["battery"] not in [s["class_idx"] for s in s2]
    assert set(s["y"] for s in s2) == {0, 1, 2, 3}
    b1 = run.training_streams(run.prepare_job_data(run.JOBS["B1"], df).train)
    assert [s["class_idx"] for s in s2] == [c for c in range(12) if c != run.CLASS_TO_IDX["battery"]]
    assert [len(s["frame"]) for s in s2] == [len(s["frame"]) for s in b1]           # identical exposure to the B1 class model
    assert run.cache_dir_for(run.JOBS["S1"], "/content/data").endswith("wv_cache/main")
    assert run.cache_dir_for(run.JOBS["B5"], "/content/data").endswith("wv_cache/B5")


def _quartile_draws(n, buffer, order, epochs=14, frac=0.75, seed0=42):
    import tensorflow as tf
    counts = np.zeros(4)
    for e in range(epochs):
        ds = tf.data.Dataset.range(n).shuffle(buffer, seed=seed0 + e).repeat().take(int(frac * n))
        for v in ds.as_numpy_iterator():
            counts[order[v] * 4 // n] += 1
    return counts


def test_sampling_is_flat_over_the_rows_of_a_class():
    pytest.importorskip("tensorflow")
    n = 4000
    f = pd.DataFrame(dict(orig=np.arange(n), id=[f"x{i}" for i in range(n)], class_idx=0, y=0))
    shuffled_orig = run.shuffle_class_rows(f)["orig"].values                      # cache position -> original (source-sorted) row
    flat = _quartile_draws(n, buffer=n, order=shuffled_orig)                      # the fix: shuffle once, buffer = class size
    assert flat.sum() == 14 * int(0.75 * n)
    assert (np.abs(flat / flat.mean() - 1) < 0.10).all(), flat
    old = _quartile_draws(n, buffer=1024, order=np.arange(n))                     # old: sorted rows, 1,024 buffer
    assert old.max() / old.min() > 1.5 and (np.abs(old / old.mean() - 1) > 0.10).any(), old


# item 2: resume robustness
def _crashed_dir(tmp_path, crash_at=9):
    d = str(tmp_path / "x")
    with pytest.raises(RuntimeError):
        run.run_training(d, FakeTrainer(LOSSES, crash_at=crash_at), log=lambda *_: None)
    return d


def _finish(d):
    lines = []
    tr = FakeTrainer(LOSSES)
    st = run.run_training(d, tr, log=lines.append)
    return st, tr, "\n".join(lines)


def _same_as_reference(tmp_path, st, d):
    ref = run.run_training(str(tmp_path / "ref"), FakeTrainer(LOSSES), log=lambda *_: None)
    assert [(h["phase"], h["epoch"], h["val_loss"], h["lr"]) for h in st["history"]] == \
           [(h["phase"], h["epoch"], h["val_loss"], h["lr"]) for h in ref["history"]]
    assert json.load(open(os.path.join(d, "final.weights.h5"))) == json.load(open(tmp_path / "ref" / "final.weights.h5"))


def test_resume_falls_back_to_previous_checkpoint(tmp_path):
    d = _crashed_dir(tmp_path)
    s = run.load_state(d)
    assert s["epochs_done"] == 9 and len(s["ckpts"]) == 2
    os.remove(os.path.join(d, s["ckpts"][-1]["file"]))                                # latest checkpoint lost
    st, tr, log = _finish(d)
    assert "latest checkpoint" in log and "missing or empty" in log and "resuming from previous checkpoint" in log
    assert st["resumes"][0]["action"] == "previous checkpoint" and st["resumes"][0]["epoch"] == 8
    assert st["resumes"][0]["epochs_in_state_file"] == 9
    _same_as_reference(tmp_path, st, d)


def test_resume_falls_back_to_best_checkpoint_of_the_phase(tmp_path):
    d = _crashed_dir(tmp_path)
    s = run.load_state(d)
    for e in s["ckpts"]:
        os.remove(os.path.join(d, e["file"]))
    st, tr, log = _finish(d)
    assert "resuming from best checkpoint of the current phase" in log
    assert st["resumes"][0]["action"] == "best checkpoint of the current phase" and st["resumes"][0]["epoch"] == 8
    _same_as_reference(tmp_path, st, d)


def test_resume_unreadable_file_and_scratch_restart(tmp_path):
    d = _crashed_dir(tmp_path, crash_at=9)
    s = run.load_state(d)
    open(os.path.join(d, s["ckpts"][-1]["file"]), "w").close()                        # zero-byte file
    open(os.path.join(d, s["ckpts"][-2]["file"]), "w").write("not json")              # unreadable file
    st, tr, log = _finish(d)
    assert "could not be read" in log and "resuming from best checkpoint" in log
    _same_as_reference(tmp_path, st, d)
    # nothing usable at all: restart from scratch and say so
    d2 = _crashed_dir(tmp_path / "again", crash_at=3)
    for f in os.listdir(d2):
        if f.endswith(".weights.h5"):
            os.remove(os.path.join(d2, f))
    st2, tr2, log2 = _finish(d2)
    assert "no usable checkpoint: restarting the job from scratch" in log2
    assert ("build", False, 0) in tr2.log and st2["resumes"][-1]["action"] == "scratch" and st2["resumes"][-1]["epoch"] == 0
    assert st2["done"] and st2["epochs_done"] == 14
    ref = run.run_training(str(tmp_path / "ref2"), FakeTrainer(LOSSES), log=lambda *_: None)
    assert [h["val_loss"] for h in st2["history"]] == [h["val_loss"] for h in ref["history"]]


def test_missing_final_weights_restart(tmp_path):
    d = str(tmp_path / "r")
    run.run_training(d, FakeTrainer(LOSSES), log=lambda *_: None)
    os.remove(os.path.join(d, "final.weights.h5"))
    st, tr, log = _finish(d)
    assert "final weights missing or unreadable" in log and st["done"] and st["epochs_done"] == 14


def test_checkpoint_write_is_verified_and_renamed(tmp_path):
    class Empty(FakeTrainer):
        def save_weights(self, path):
            open(path, "w").close()                                                   # writes nothing
    with pytest.raises(IOError):
        run.run_training(str(tmp_path / "r"), Empty(LOSSES), log=lambda *_: None)
    seen = []

    class Spy(FakeTrainer):
        def save_weights(self, path):
            seen.append(os.path.basename(path))
            super().save_weights(path)
    run.run_training(str(tmp_path / "s"), Spy(LOSSES), log=lambda *_: None)
    assert all(n.startswith("tmp_") or n == "final.weights.h5" for n in seen)          # temp name first, then rename
    assert not [f for f in os.listdir(tmp_path / "s") if f.startswith("tmp_")]


def test_result_records_resumes(fake_env, monkeypatch):
    drive, base = fake_env
    assert run.main(["--job", "B2"] + base) == 0
    res = json.load(open(drive / "results" / "B2.json"))
    assert res["resumed"] is False and res["n_resumes"] == 0 and res["resume_epochs"] == []
    orig = FakeKeras.run_epoch

    def crash(self, phase, epoch_index, lr):
        if epoch_index == 4 and not FakeKeras.__dict__.get("crashed_once"):
            FakeKeras.crashed_once = True
            raise RuntimeError("disconnect")
        return orig(self, phase, epoch_index, lr)
    monkeypatch.setattr(FakeKeras, "run_epoch", crash)
    with pytest.raises(RuntimeError):
        run.run_job("B3", run.parse_args(["--job", "B3"] + base))
    assert run.main(["--job", "B3"] + base) == 0
    res = json.load(open(drive / "results" / "B3.json"))
    assert res["resumed"] is True and res["n_resumes"] == 1 and res["resume_epochs"] == [4]
    assert res["resume_events"][0]["action"] == "latest checkpoint"


# item 5: subprocess queue
def test_job_command_and_subprocess_queue(tmp_path):
    a = run.parse_args(["--queue", "q.txt", "--drive-root", str(tmp_path / "d"), "--rerun-stale"])
    cmd = run.job_command("B3", a)
    assert cmd[0] == sys.executable and cmd[1] == "-u" and cmd[2].endswith("run.py")
    assert cmd[cmd.index("--job") + 1] == "B3" and "--rerun-stale" in cmd and cmd[cmd.index("--drive-root") + 1] == str(tmp_path / "d")
    stub = tmp_path / "stub.py"                                                       # stands in for run.py in a real subprocess
    stub.write_text(
        "import sys, os, json\nav = sys.argv\njob = av[av.index('--job') + 1]\nroot = av[av.index('--drive-root') + 1]\n"
        "if job == 'A2':\n    sys.exit(3)\nos.makedirs(root + '/results', exist_ok=True)\n"
        "open(root + '/results/' + job + '.json', 'w').write('{}')\nprint('stub ran', job, 'pid', os.getpid())\n")
    a2 = run.parse_args(["--queue", "q.txt", "--drive-root", str(tmp_path / "d")])
    runner = run.make_subprocess_runner(a2, script=str(stub))
    out = run.run_queue(["A1", "A2", "B1"], str(tmp_path / "d"), runner)
    assert out["done"] == ["A1", "B1"] and list(out["failed"]) == ["A2"]               # A2 failed, the queue went on
    out2 = run.run_queue(["A1", "A2", "B1"], str(tmp_path / "d"), runner)
    assert out2["skipped"] == ["A1", "B1"] and list(out2["failed"]) == ["A2"]


def test_subprocess_prints_exit_status(tmp_path, capfd):
    stub = tmp_path / "ok.py"
    stub.write_text("print('hello')\n")
    a = run.parse_args(["--queue", "q.txt", "--drive-root", str(tmp_path)])
    run.make_subprocess_runner(a, script=str(stub))("A1")
    assert "=== JOB A1 | subprocess exit status 0 ===" in capfd.readouterr().out


# item 7: provenance and stale results
def test_provenance_hash_and_stale_handling(fake_env, tmp_path):
    drive, base = fake_env
    sp = base[base.index("--splits") + 1]
    assert run.main(["--job", "B4"] + base) == 0
    res = json.load(open(drive / "results" / "B4.json"))
    assert res["provenance"]["splits_sha256"] == run.sha256_file(sp)
    a = run.parse_args(["--job", "B4"] + base)
    assert run.result_status(a.drive_root, run.JOBS["B4"], a.splits, a.all_csv) == "done"
    n = len(FakeKeras.instances)
    assert run.main(["--job", "B4"] + base) == 0 and len(FakeKeras.instances) == n          # same hash: skipped
    with open(sp, "a") as f:                                                                # the splits file changes
        f.write("\n")
    assert run.result_status(a.drive_root, run.JOBS["B4"], a.splits, a.all_csv) == "stale"
    assert run.main(["--job", "B4"] + base) == 2                                           # stops, nothing rerun
    assert len(FakeKeras.instances) == n and (drive / "results" / "B4.json").exists()
    q = tmp_path / "q.txt"; q.write_text("B4\n")
    assert run.main(["--queue", str(q), "--in-process"] + base) == 2                       # the queue stops as well
    assert run.main(["--job", "B4", "--rerun-stale"] + base) == 0
    stale = os.listdir(drive / "results" / "stale")
    assert any(x.startswith("B4_") and x.endswith(".json") for x in stale) and any(x.endswith("_logits.npz") for x in stale)
    assert any(x.endswith("_runs") for x in stale)                                         # old run dir archived: no reuse of old weights
    res2 = json.load(open(drive / "results" / "B4.json"))
    assert res2["provenance"]["splits_sha256"] == run.sha256_file(sp) != res["provenance"]["splits_sha256"]
    assert res2["n_resumes"] == 0 and len(FakeKeras.instances) == n + 1


def test_result_without_provenance_is_stale_and_c_jobs_hash_both_files(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "A1.json").write_text("{}")
    s, c = tmp_path / "s.csv", tmp_path / "c.csv"
    s.write_text("a\n"); c.write_text("b\n")
    assert run.result_status(str(tmp_path), run.JOBS["A1"], str(s), str(c)) == "stale"
    h = run.current_hashes(run.JOBS["C1"], str(s), str(c))
    assert set(h) == {"splits_sha256", "all_csv_sha256"} and h["all_csv_sha256"] == run.sha256_file(str(c))
    assert set(run.current_hashes(run.JOBS["A1"], str(s), str(c))) == {"splits_sha256"}
    (tmp_path / "results" / "C1.json").write_text(json.dumps(dict(provenance=h)))
    assert run.result_status(str(tmp_path), run.JOBS["C1"], str(s), str(c)) == "done"
    c.write_text("changed\n")
    assert run.result_status(str(tmp_path), run.JOBS["C1"], str(s), str(c)) == "stale"


# item 9: E1 self-test
def test_limit_option_and_selftest_summary(splits):
    a = run.parse_args(["--job", "E1", "--limit", "200"])
    assert a.limit == 200
    with pytest.raises(SystemExit):
        run.parse_args(["--job", "A1", "--limit", "200"])
    with pytest.raises(SystemExit):
        run.parse_args(["--job", "E1", "--limit", "0"])
    n = 50
    rng = np.random.default_rng(0)
    img = rng.normal(size=(n, 16)); img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt = rng.normal(size=(14, 16)); txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    sc = run.clip_scores(img, txt[:12], txt[12:])
    st = run.clip_selftest_summary(sc, splits.head(n))
    assert st["n"] == n and 0 <= st["top1_class_acc"] <= 1 and sum(st["predicted_class_counts"].values()) == n


# optional second archive
def test_extra_tar_extracted_into_same_root(tmp_path):
    tar, extra = tmp_path / "n.tar", tmp_path / "extra.tar"
    tar.write_bytes(b"x")
    calls = []
    fake_run = lambda cmd, check: calls.append(cmd)                                    # noqa: E731
    root = str(tmp_path / "data")
    assert run.ensure_extracted(str(tar), root, str(extra), run=fake_run) is True      # extra does not exist: only the main tar
    assert len(calls) == 1
    extra.write_bytes(b"yy")
    assert run.ensure_extracted(str(tar), root, str(extra), run=fake_run) is True
    assert len(calls) == 2 and calls[1] == ["tar", "-xf", str(extra), "-C", root]
    assert run.ensure_extracted(str(tar), root, str(extra), run=fake_run) is False     # both markers present
    assert len(calls) == 2
    extra.write_bytes(b"replaced with a bigger archive")                                # archive replaced: extracted again
    assert run.ensure_extracted(str(tar), root, str(extra), run=fake_run) is True and len(calls) == 3
    assert run.parse_args(["--job", "A1", "--drive-root", "/d"]).extra_tar == "/d/data/extra_images.tar"
    assert run.job_command("A1", run.parse_args(["--job", "A1", "--extra-tar", "/e.tar"]))[-1] == "/e.tar"

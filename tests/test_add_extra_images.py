"""Tests for pipeline/add_extra_images.py using local synthetic images (no network, no git)."""
import hashlib
import os
import random
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "pipeline" / "add_extra_images.py"
COLS = ["id", "source", "orig_class", "label", "group", "norm_rel", "phash"]

# (folder, file name, size, mode, format)  -- the corrupt and tiny files are added separately
GOOD = []
for i in range(16):
    GOOD.append(("gloves", f"g{i:02d}", [(800, 500), (1000, 300), (320, 240), (150, 200), (640, 640), (321, 100),
                                         (90, 60), (400, 900)][i % 8], "RGB", ["JPEG", "PNG"][i % 2]))
for i in range(13):
    GOOD.append(("masks", f"m{i:02d}", [(512, 384), (2000, 1500), (60, 60), (300, 300), (700, 200)][i % 5],
                 ["RGB", "RGBA", "L", "P"][i % 4], ["JPEG", "PNG"][i % 2]))
N_GOOD_G = sum(1 for g in GOOD if g[0] == "gloves")
N_GOOD_M = sum(1 for g in GOOD if g[0] == "masks")


def make_image(size, mode, seed):
    # smooth random colour blobs (a hard checkerboard has a degenerate perceptual hash)
    rnd = random.Random(seed)
    tiny = Image.new("RGB", (6, 6))
    tiny.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(36)])
    base = tiny.resize(size, Image.Resampling.BICUBIC)
    if mode == "RGBA":
        im = base.convert("RGBA")
        im.putalpha(Image.new("L", size, 128))
        return im
    if mode == "L":
        return base.convert("L")
    if mode == "P":
        return base.convert("P")
    return base


def build_source(root):
    for k, (folder, name, size, mode, fmt) in enumerate(GOOD):
        d = Path(root) / "medical" / folder
        d.mkdir(parents=True, exist_ok=True)
        im = make_image(size, mode, k)
        ext = "jpg" if fmt == "JPEG" else "png"
        if fmt == "JPEG" and mode in ("RGBA", "P"):
            im = im.convert("RGB")
        im.save(d / f"{name}.{ext}", fmt)
    # one corrupt file and one tiny image (below the reference's 24 px minimum side)
    (Path(root) / "medical" / "gloves" / "corrupt.jpg").write_bytes(b"this is not a jpeg" * 20)
    make_image((20, 20), "RGB", 99).save(Path(root) / "medical" / "masks" / "tiny.png")
    # files that are not images are ignored by the reference's extension filter
    (Path(root) / "medical" / "masks" / "notes.txt").write_text("ignore me")


def run_script(src, tmp, tag, all_csv, check=True):
    data_root = tmp / f"data_{tag}"
    out_dir = tmp / f"out_{tag}"
    cmd = [sys.executable, str(SCRIPT), "--local-source", str(src), "--data-root", str(data_root),
           "--out-dir", str(out_dir), "--raw-dir", str(tmp / "raw"), "--all-csv", str(all_csv)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if check:
        assert res.returncode == 0, res.stdout + res.stderr
    return res, data_root, out_dir


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("extra")
    src = tmp / "src"
    build_source(src)
    all_csv = tmp / "all_images_before_dedup.csv"
    pd.DataFrame([dict(id="0123456789abcdef", source="GD", orig_class="plastic", label="plastic",
                       group="GD_x", norm_rel="GD/0123456789abcdef.jpg", phash="0" * 16)]).to_csv(all_csv, index=False)
    r1, d1, o1 = run_script(src, tmp, "a", all_csv)
    r2, d2, o2 = run_script(src, tmp, "b", all_csv)
    return dict(tmp=tmp, src=src, all_csv=all_csv, r1=r1, d1=d1, o1=o1, r2=r2, d2=d2, o2=o2,
                df1=pd.read_csv(o1 / "extra_images.csv", dtype=str), df2=pd.read_csv(o2 / "extra_images.csv", dtype=str))


def test_columns_and_values(env):
    df = env["df1"]
    assert list(df.columns) == COLS
    assert (df.source == "TrashBox").all()
    assert (df.label == "sanitary").all()
    assert set(df.orig_class) == {"medical/gloves", "medical/masks"}
    assert (df.orig_class == "medical/gloves").sum() == N_GOOD_G
    assert (df.orig_class == "medical/masks").sum() == N_GOOD_M
    assert len(df) == N_GOOD_G + N_GOOD_M
    assert df.id.is_unique and df.norm_rel.is_unique
    for r in df.itertuples():
        assert re.fullmatch(r"[0-9a-f]{16}", r.id)
        assert r.norm_rel == f"TrashBox/{r.id}.jpg"
        assert r.group == f"TrashBox_{r.id}"
        assert not r.norm_rel.startswith("/")


def test_ids_follow_the_reference_formula(env):
    df = env["df1"]
    # the reference: stable_id(source, rel) with rel relative to the TrashBox subfolder, e.g. medical/gloves/g00.jpg
    expected = {hashlib.sha1(f"TrashBox|medical/{f}/{n}.{'jpg' if fmt == 'JPEG' else 'png'}".encode()).hexdigest()[:16]
                for f, n, _, _, fmt in GOOD}
    assert set(df.id) == expected


def test_ids_stable_across_two_runs(env):
    assert env["df1"].equals(env["df2"])
    t1 = sorted(tarfile.open(env["o1"] / "extra_images.tar").getnames())
    t2 = sorted(tarfile.open(env["o2"] / "extra_images.tar").getnames())
    assert t1 == t2


def test_rerun_in_same_folders_does_not_duplicate(env):
    tmp = env["tmp"]
    res, d, o = run_script(env["src"], tmp, "a", env["all_csv"])   # same data root and out dir as run "a"
    df = pd.read_csv(o / "extra_images.csv", dtype=str)
    assert df.equals(env["df1"])
    assert len(list((d / "TrashBox").glob("*.jpg"))) == len(df)
    assert len(tarfile.open(o / "extra_images.tar").getnames()) == len(df)


def test_images_are_rgb_jpeg_max_320(env):
    df = env["df1"]
    for rel in df.norm_rel:
        with Image.open(env["d1"] / rel) as im:
            assert im.format == "JPEG"
            assert im.mode == "RGB"
            assert max(im.size) <= 320
    # a large image really was shrunk and a small one kept at its size
    sizes = {rel: Image.open(env["d1"] / rel).size for rel in df.norm_rel}
    assert max(max(s) for s in sizes.values()) == 320
    assert (60, 60) in sizes.values()


def test_hash_strings_are_16_hex(env):
    import imagehash
    df = env["df1"]
    for r in df.itertuples():
        assert re.fullmatch(r"[0-9a-f]{16}", r.phash), r.phash
        with Image.open(env["d1"] / r.norm_rel) as im:   # hash of the saved image matches to within JPEG re-encoding
            h = imagehash.phash(im.convert("RGB"))
        assert int((imagehash.hex_to_hash(r.phash).hash != h.hash).sum()) <= 8


def test_tar_extracts_to_same_paths(env, tmp_path):
    df = env["df1"]
    with tarfile.open(env["o1"] / "extra_images.tar") as t:
        names = t.getnames()
        assert sorted(names) == sorted(df.norm_rel)
        t.extractall(tmp_path, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(tmp_path)
    for rel in df.norm_rel:
        assert (tmp_path / rel).read_bytes() == (env["d1"] / rel).read_bytes()


def test_corrupt_and_tiny_are_rejected_not_fatal(env):
    out = env["r1"].stdout
    gl = [l for l in out.splitlines() if l.startswith("medical/gloves")][0]
    ma = [l for l in out.splitlines() if l.startswith("medical/masks")][0]
    nums = lambda l: [int(x) for x in re.findall(r"\s(\d+)\s", l + " ")[:3]]
    assert nums(gl) == [N_GOOD_G + 1, 1, N_GOOD_G]
    assert nums(ma) == [N_GOOD_M + 1, 1, N_GOOD_M]
    assert "unreadable" in gl and "too_small" in ma
    # nothing written for them
    df = env["df1"]
    assert len(list((env["d1"] / "TrashBox").glob("*.jpg"))) == len(df)


def test_collision_with_existing_csv_stops(env):
    tmp = env["tmp"]
    df = env["df1"]
    bad_id = tmp / "bad_id.csv"
    row = df.iloc[0].to_dict()
    row["norm_rel"] = "Other/zzzz.jpg"
    pd.DataFrame([row], columns=COLS).to_csv(bad_id, index=False)
    res, d, o = run_script(env["src"], tmp, "c1", bad_id, check=False)
    assert res.returncode != 0 and "collide" in (res.stdout + res.stderr)
    assert not (o / "extra_images.csv").exists()

    bad_rel = tmp / "bad_rel.csv"
    row = df.iloc[0].to_dict()
    row["id"] = "ffffffffffffffff"
    pd.DataFrame([row], columns=COLS).to_csv(bad_rel, index=False)
    res, d, o = run_script(env["src"], tmp, "c2", bad_rel, check=False)
    assert res.returncode != 0 and "collide" in (res.stdout + res.stderr)

    # running with the CSV that already contains the extra rows must also stop
    res, d, o = run_script(env["src"], tmp, "c3", env["o1"] / "extra_images.csv", check=False)
    assert res.returncode != 0


def test_empty_folder_stops(tmp_path):
    src = tmp_path / "src"
    (src / "medical" / "gloves").mkdir(parents=True)
    (src / "medical" / "masks").mkdir(parents=True)
    make_image((100, 100), "RGB", 1).save(src / "medical" / "masks" / "a.jpg")
    all_csv = tmp_path / "all.csv"
    pd.DataFrame(columns=COLS).to_csv(all_csv, index=False)
    res, d, o = run_script(src, tmp_path, "e", all_csv, check=False)
    assert res.returncode != 0 and "no images found" in (res.stdout + res.stderr)

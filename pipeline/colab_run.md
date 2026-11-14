# Running the WasteVision v3 jobs on Colab

Use a **T4 GPU** runtime (Runtime > Change runtime type > T4 GPU). Paste each block below into its own cell and run the cells in this order:

1. Mount Drive
2. Clone / pull the code
3. Check the GPU and the input files
4. Extract the images
5. Run **A1 alone** and check it
6. E1 self-test (`--job E1 --limit 200`), once
7. Run the whole queue

Files that must already be on Drive before you start:

- `MyDrive/WasteVision_v3/data/splits_v3.csv` (written by `rebuild_splits.py`)
- `MyDrive/WasteVision_final/normalized_images.tar`
- `MyDrive/WasteVision_final/all_images_before_dedup.csv` (only the leakage jobs C1 and C2 read it)
- Optional: `MyDrive/WasteVision_v3/data/extra_images.tar` (images added after the original run). If this file exists it is extracted into the same folder as the main archive; if it does not exist, nothing happens.

## Cell 1: mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Expect: a Google sign-in prompt the first time, then `Mounted at /content/drive`.

## Cell 2: get the code

```bash
%%bash
cd /content
if [ -d WasteVision/.git ]; then
  git -C WasteVision pull --ff-only
else
  git clone https://github.com/ayushdhasmana433-rgb/WasteVision
fi
ls WasteVision/pipeline
```

Expect: `Already up to date.` or a list of updated files, then a listing that includes `run.py`, `queue.txt` and `colab_run.md`.

## Cell 3: check the GPU and the input files

```python
import os, tensorflow as tf
print("TensorFlow", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU:", gpus)
assert gpus, "No GPU: Runtime > Change runtime type > T4 GPU, then run the cells again"
for p in ["/content/drive/MyDrive/WasteVision_v3/data/splits_v3.csv",
          "/content/drive/MyDrive/WasteVision_final/normalized_images.tar",
          "/content/drive/MyDrive/WasteVision_final/all_images_before_dedup.csv",
          "/content/drive/MyDrive/WasteVision_v3/data/extra_images.tar"]:
    print("OK     " if os.path.exists(p) else "MISSING", p)
```

Expect: a TensorFlow version, `GPU: [PhysicalDevice(... 'GPU' ...)]`, and `OK` for the first three files. `MISSING` for `extra_images.tar` is fine if you have no extra images.

## Cell 4: extract the images (once per Colab session)

```bash
%%bash
cd /content/WasteVision/pipeline
python -u run.py --extract-only 2>&1 | tee /content/last_run.log
```

Expect: `Extracting ... normalized_images.tar -> /content/data (one time) ...`, then, if the extra archive exists, `Extracting extra images ...`, then `Extracted now.` If the images are already there you see `Already extracted.` (`run.py` also does this by itself before the first job.)

## Cell 5: run job A1 alone first, and check it

```bash
%%bash
cd /content/WasteVision/pipeline
set -o pipefail
python -u run.py --job A1 2>&1 | tee /content/last_run.log
```

Expect, in this order:

- `=== JOB A1 | START | 12-class model, seed 42 | <date time> ===`
- `label space (class, 12): [...]` and `unseen classes: [] | counts: train ... val ... test ... unknown 0`
- `All <N> image files found.` (a `WARNING: ... image files are missing` line followed by up to 20 file names means part of the images did not extract; more than 0.5% missing stops the job)
- Keras epoch lines with `acc`, `loss`, `val_acc`, `val_loss` and the time for the epoch, each followed by `  [head] epoch 1: val_loss ...`
- at the end `=== JOB A1 | DONE in ... min | T=... | class_acc ... | stream_acc ... ===`

What to check before starting the queue:

1. **First epoch time.** The first epoch is slower than the others because every image is decoded once and cached to local disk. Note the time of the **second** epoch: the whole job is about 14 epochs of that length or fewer (training stops early when the validation loss stops improving).
2. **Validation accuracy.** `val_acc` after the first epoch should be far above chance (chance for 12 classes is 0.08) and should rise over the head-phase epochs. If it stays near 0.08, or `val_loss` becomes `nan`, stop and send the output (see the end of this page).
3. The line `DONE` appears and `MyDrive/WasteVision_v3/results/A1.json` exists.

## Cell 6: E1 self-test, once

```bash
%%bash
cd /content/WasteVision/pipeline
set -o pipefail
python -u run.py --job E1 --limit 200 2>&1 | tee /content/last_run.log
```

Expect: `=== JOB E1 | START | ... | SELF-TEST on first 200 images ...`, a line `self-test: {...}` with `top1_class_acc`, `stream_acc` and `predicted_class_counts`, and `SELF-TEST DONE ... (no result file written)`. It installs `open_clip_torch` the first time. The first 200 rows of the splits file may all come from a few classes, so only check that the numbers are not all zero and the job finishes. This does not mark E1 as done; the full E1 runs later as part of the queue.

## Cell 7: run the whole queue

```bash
%%bash
cd /content/WasteVision/pipeline
set -o pipefail
python -u run.py --queue queue.txt 2>&1 | tee /content/last_run.log
```

The queue runs A1 A2 A3 B1 B2 B3 B4 E1 B5 B6 B7 B8 C1 C2 S1 S2 S3 S4 in that order. **Each job runs in its own Python process** (this frees GPU and memory between jobs). For every job you see `=== JOB <id> | START ...`, the job's output, `=== JOB <id> | DONE ...` and then `=== JOB <id> | subprocess exit status 0 ===`. Finished jobs print `=== JOB <id> | SKIP (result exists) ===`. A failed job prints a non-zero exit status, and the queue goes on to the next job. At the end you get `Queue summary: done=[...] skipped=[...] failed=[...]`.

To run a single job: `python -u run.py --job B3`. To list the jobs: `python run.py --list`.

## If Colab disconnects or you stop the cell

Run the same cells again (Cell 1 to Cell 4, then Cell 7; no edits needed).

- Each training job saves its weights and its epoch counter to `MyDrive/WasteVision_v3/runs/<job>/` after every epoch. On the rerun it picks up from the last finished epoch (you lose at most one epoch). If the newest checkpoint file is damaged, it uses the previous one, then the best one of that phase, and otherwise restarts that job; the output says which. The result file records how many times the job was resumed and at which epochs.
- A job whose `results/<job>.json` exists and was made from the same `splits_v3.csv` (and, for C1 and C2, the same `all_images_before_dedup.csv`) is skipped.
- If you rebuild `splits_v3.csv`, old results no longer match it. The queue then prints `STOP: results for [...] were made from different input files` and does not run. To archive the old results under `results/stale/` and run those jobs again, add `--rerun-stale` to the command in Cell 7.
- If one job fails, the queue prints the error, goes on to the next job and lists the failed jobs at the end. Running Cell 7 again retries only the failed ones.
- The local copy of the images (`/content/data`) and the image cache (`/content/wv_cache`) vanish when the runtime is recycled. They are rebuilt automatically, which costs a few minutes at the start of the first job.

## If something fails: what to send back

Send these two things:

1. The **job id** (for example `B3`).
2. The **last 40 lines of the output**. Every cell above saves its output to `/content/last_run.log`. Run this cell and copy what it prints:

```bash
%%bash
tail -n 40 /content/last_run.log
```

If the cell died before printing anything, say which cell it was.

## Where the results go (all under `MyDrive/WasteVision_v3/`)

| Path | Content |
|---|---|
| `results/<job>.json` | settings, label set, seed, epochs run, resumes, training minutes, temperature, thresholds, input file hashes, metrics per tier |
| `results/<job>_logits.npz` | logits and labels for every validation and test image, plus the unknown / left-out images |
| `results/stale/` | results moved aside by `--rerun-stale` |
| `data/clip_features_vitb32.npz` | CLIP image features for every image in `splits_v3.csv` (job E1) |
| `runs/<job>/` | checkpoints and `state.json` (safe to delete after the job is finished) |

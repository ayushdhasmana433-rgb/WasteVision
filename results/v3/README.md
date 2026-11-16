# results/v3: files behind the journal manuscript

"Unseen items and unseen datasets in photo-based routing of household waste to India's four segregation streams" (A. Dhasmana, P. Gupta). These files replace the October numbers in `results/final/` for the manuscript. No images are stored here; the images stay with their public sources.

| Folder | Contents |
|---|---|
| `data/` | `splits_v3.csv` (one row per image: source, class, photo group, perceptual hash, split and test tier), `count_table.csv` (images per source and class at each stage), `dup_pairs_phash.csv.gz` (near-duplicate pairs), `label_conflicts.csv`, `source_overlap.csv`, `report.json` (run report of `pipeline/rebuild_splits.py`) |
| `runs/` | One `<job>.json` and `<job>_logits.npz` per job in `pipeline/queue.txt` (A1–A3, B1–B8, C1–C2, S1–S4, E1): settings, metrics, thresholds and the logits of every test image |
| `seeds/` | The hazard hold-outs repeated with seeds 7 and 123 (`B1R7` … `S4R123`) and the three-seed summaries |
| `analysis/` | Output of `pipeline/analyse.py` and `pipeline/extra_analyses.py`: every table and figure value in the manuscript (`paper_numbers.json`, `decision_designs.csv`, `unseen_datasets.csv`, `leakage.csv`, feature probes, scenarios) |

To reproduce the analysis from these files: `python pipeline/analyse.py --results-dir results/v3/runs --data-dir results/v3/data --out-dir out`.

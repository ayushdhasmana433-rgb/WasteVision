# WasteVision analysis report

Generated 2026-11-15 15:24:49; run time 17.9 s. Command: `analyse.py --results-dir ../an_in/results --data-dir ../an_in/data --out-dir ../an_out`

Resamples: 2000, seed 42.

## Tables

| table | rows | jobs that fed it | source files | what |
|---|---|---|---|---|
| main_per_seed.csv | 105 | A1, A2, A3 | A1.json, A2.json, A3.json | A jobs: every metric per seed on each test tier (from the result JSON) |
| main_mean_sd.csv | 35 | A1, A2, A3 | A1.json, A2.json, A3.json | A jobs: mean and SD across seeds (n = number of seeds) |
| main_bootstrap.csv | 46 | A1 | A1.json, A1_logits.npz | best-seed (A1) metrics with grouped 95% bootstrap intervals, per tier |
| stream_model_S1.csv | 12 | S1 | S1.json, S1_logits.npz | S1 (4-stream network, full data) on T1 |
| decision_designs.csv | 563 | B1, B2, B3, B4, E1, S2, S3, S4 | B1.json, B1_logits.npz, B2.json, B2_logits.npz, B3.json, B3_logits.npz, B4.json, B4_logits.npz, E1.json, E1_logits.npz, S2.json, S2_logits.npz, S3.json, S3_logits.npz, S4.json, S4_logits.npz | unseen hazards (B1-B3) and shoes control (B4): catch rate / wrongly-flagged share, false alarms, D1 minus design; primary = validation-matched thresholds |
| unseen_datasets.csv | 72 | B5, B6, B7, B8, A1 | A1.json, A1_logits.npz, B5.json, B5_logits.npz, B6.json, B6_logits.npz, B7.json, B7_logits.npz, B8.json, B8_logits.npz | left-out source vs the same source in training (A1) |
| leakage.csv | 17 | C1, C2 | C1.json, C1_logits.npz, C2.json, C2_logits.npz | C1 (naive split) vs C2 (grouped split) and the near-duplicate shares |
| scenario_flag_precision.csv | 72 | A1 | A1.json, A1_logits.npz, B1.json, B1_logits.npz, B2.json, B2_logits.npz, B3.json, B3_logits.npz, B4.json, B4_logits.npz, B5.json, B5_logits.npz, B6.json, B6_logits.npz, B7.json, B7_logits.npz, B8.json, B8_logits.npz | flag precision = R*pi / (R*pi + FAR*(1-pi)) for hazard prevalence pi (scenario numbers, not measured prevalence) |
| scenario_per_10000.csv | 63 | A1 | A1.json, A1_logits.npz, B1.json, B1_logits.npz, B2.json, B2_logits.npz, B3.json, B3_logits.npz | scenario counts per 10,000 items; 'reaching dry' = not flagged and top stream is Dry; FAR from A T1 |
| data_count_per_class.csv | 12 | - | count_table.csv | count table condensed by label (sum over the other axis) |
| data_count_per_source.csv | 18 | - | count_table.csv | count table condensed by source (sum over the other axis) |
| data_dup_pairs_by_distance.csv | 3 | - | dup_pairs_phash.csv.gz | pHash duplicate pairs by Hamming distance |
| data_dup_summary.csv | 5 | - | dup_pairs_phash.csv.gz | duplicate pair summary |
| data_dup_cross_source_by_source_pair.csv | 49 | - | dup_pairs_phash.csv.gz | cross-source duplicate pairs by source pair (value = pairs) |
| data_conflicts_by_reason.csv | 3 | - | label_conflicts.csv | label-conflict rows by stage and reason |
| data_conflicts_by_source_pair.csv | 49 | - | label_conflicts.csv | label-conflict rows by source pair (value = rows) |
| data_source_overlap.csv | 18 | - | source_overlap.csv | source-overlap matrix (duplicate pairs), copied |
| data_stage_totals.csv | 8 | - | report.json | image counts per pipeline stage (rebuild_splits report.json) |

## Result files missing or unreadable

None.

## Planned items missing or skipped

- frozen-feature results (job F): no results/F*.csv or results/frozen*.csv found
- object-or-dataset probes: no --features file given
- stream accuracy reweighted to a waste mix: no --mix given (no mix is assumed)

## Notes and checks

- Main-results bootstrap uses A1 (seed 42): lowest validation loss of the kept model (A1: 0.4255; all: A1=0.4255, A2=0.4295, A3=0.4322). Jobs considered (JSON and logits readable): ['A1', 'A2', 'A3']. Excluded: none.
- T3 (described as a dry-only street test): per-stream test counts in A1 = {'recall_Dry': 192}.
- Decision-design table, PRIMARY block (op_point val_match_*): the threshold of every sweepable design is set on the known non-hazard VALIDATION items of the same job to reach the target false-alarm rate (D1's own validation rate, 5%, 10%), then fixed; catch rate on the unknown T1 items and the false-alarm rate realised on the known T1 test items are reported with grouped bootstrap intervals (thresholds fixed across resamples). SECONDARY block (op_point test_match_*): ROC point matched on the known T1 test items, comparison only, NOT deployable. op_point 'own': the design's own operating point.
- main bootstrap point values vs run.py JSON (A1): max abs difference 1.63e-16 over 35 numbers (logits are stored as float32)

## Definitions

- Grouped bootstrap: photo groups (the `group` array) resampled with replacement, `--n-boot` resamples, percentile 95% interval; the point value is the data as it is. Differences between two designs on the same images are paired (same resample); differences between two jobs on different images use independent resamples.
- Test rows of a job: role == test. Tier from the `tier` array (T1 headline, T2 conveyor, T3 street).
- Stream accuracy: argmax of the summed (temperature-scaled) stream probabilities equals the true stream. Top-class stream accuracy: stream of the top class. Flag: summed Special Care probability >= the job's validation-fitted threshold (0.20). Class accuracy and macro-F1 use only items whose class is in the label space.
- Decision designs: unknown items = T1 test images of the held-out class; known items = T1 test images of trained non-hazard classes (true stream not Special Care). False alarm = a known item flagged. Caught = an unknown item flagged.
  D0: flagged if the top class is a hazard class (no threshold). D1: flagged if the summed temperature-scaled hazard probability >= t. D4: MSP / MaxLogit / Energy (temperature-scaled, as in run.py) are familiarity scores. At the OWN point a D4 item is flagged if its score is STRICTLY BELOW the validation 5% quantile (as in run.py, `score < threshold`); at swept (matched) thresholds it is flagged if the score is AT OR BELOW t (implemented as -score >= -t). D1+reject(MSP): D1 flag OR MSP strictly below the validation reject threshold. D2: Special Care probability of the stream-trained job >= t. D6: CLIP hazard-class prompt sum >= t, or hazard-prompt probability >= t. D6 uses all 12 class prompts, so the held-out class is not unseen to CLIP.
  Own point: D1 and D2 use the job's hazard threshold (a cost rule, 1/(1+4)); D4 uses the validation 5% quantile reject threshold; D6 has no fitted threshold (rule values 0.20 / 0.5).
  PRIMARY matched comparison (op_point val_match_D1, val_match_5pct, val_match_10pct): for each job and design the threshold is set on that job's known non-hazard VALIDATION items (role val, true stream not Special Care; for E1 also not the held-out class) to reach the target false-alarm rate (D1's own validation rate, 5%, 10%): k = floor(target * n) highest known scores are flagged (moved down to the end of a tie block), threshold = the k-th highest score, flag = score >= t. The threshold is then FIXED. Reported on the T1 test items: catch rate on the unknown items, and the false-alarm rate realised on the known T1 test items (`known_false_alarm`), each with a grouped bootstrap interval (thresholds do not change across resamples). D1 stays at its own threshold at val_match_D1 and the others are matched to D1's validation rate.
  SECONDARY block (op_point test_match_D1, test_match_5pct, test_match_10pct): "ROC point matched on test (comparison only, not deployable)". The threshold is set on the known T1 TEST items of the same job (highest rate not above the target; whole tie blocks only, so the realised rate is at most the target) and re-fitted in every bootstrap resample. It is an evaluation-time matching, not a deployable threshold.
- Material-flow scenarios are scenario arithmetic: flag precision = R*pi/(R*pi + FAR*(1-pi)); R and FAR come from the A-job T1 numbers; FAR is also taken from the left-out sources (B5-B8, at 0.20) and from the share of shoes flagged (B4).

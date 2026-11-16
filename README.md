# WasteVision

Which bin does this go in? WasteVision classifies a photo of one household waste item into 12 material classes and routes it to
one of the four streams that India's **Solid Waste Management Rules, 2026** require at source: **Wet, Dry, Sanitary and Special Care**.
It flags hazards with a cost-derived threshold, says "not sure" when it should, and shows where it looked (Grad-CAM).

**Try it:** https://ayushdhasmana433-rgb.github.io/WasteVision/ (runs in your browser; English and Hindi; works with a phone camera).
The browser app runs the model of the October 2026 course run (`results/final/`), not the models of the manuscript below.

The project also asks a research question: does a good score on a shuffled test split mean the bin decision is safe?
We test the decision on datasets the model never saw, on hazard types it never saw, and with near-duplicate photos removed.

## Results (journal manuscript, November 2026)

Manuscript: A. Dhasmana and P. Gupta, "Unseen items and unseen datasets in photo-based routing of household waste to India's
four segregation streams". All files behind these numbers are in [`results/v3/`](results/v3/README.md).

**Data.** 99,497 images from 18 public sources were mapped to the 12 classes. De-duplication removed 15,627 near-duplicates
(20.9%); 64.8% of the 18,962 near-duplicate pairs joined two different datasets, and 1,628 pairs carried two different labels.
After a per-class cap, 27,592 images were split by photo group: 19,440 training, 2,704 validation and three test tiers
(T1 main 5,267; T2 conveyor 181; T3 street 192).

**In-source test images (T1), EfficientNetV2-B0, three seeds.**

| Measure | Value |
|---|---|
| Class accuracy | 86.8 ± 0.4% |
| Stream accuracy | 94.6 ± 0.1% (always-dry rule: 56.2%) |
| Hazards caught by the special-care flag (4:1 cost) | 95.9 ± 0.3% |
| False alarms on harmless items | 6.0 ± 0.1% |

**Hazard types the model never saw.** One hazardous class was removed from training and its test images were scored as unknown
items. Catch rate at the same false-alarm rate (thresholds fixed on validation data), seed 42:

| Design | Battery | Medicines | E-waste |
|---|---|---|---|
| D1 top-class routing | 51.9% | 20.6% | 30.2% |
| D2 hazard-sum flag | 74.5% | 29.1% | 41.6% |
| D3 familiarity scores (best of MSP, MaxLogit, Energy) | 12.3% | 18.2% | 32.2% |
| D4 network trained on the four streams | 80.9% | 30.3% | 37.6% |
| D5 CLIP zero-shot, class prompts | 96.9% | 55.9% | 61.0% |
| D2 or CLIP check (CLIP alone flags 2% of harmless items) | 94.9% | 63.3% | 67.8% |

Over three training seeds, D2 caught 75.0 ± 0.5%, 30.7 ± 2.6% and 42.9 ± 1.1%, and D4 caught 82.6 ± 2.2%, 33.9 ± 3.2% and
40.9 ± 2.9%. CLIP was given the names of all classes, including the one removed from the trained network.

**Datasets the model never saw.** Stream accuracy when the dataset was part of training, and when it was left out:

| Left-out dataset | In training | Left out | Always-dry rule | Harmless items flagged (left out) |
|---|---|---|---|---|
| MedPackID | 84.3% | 28.8% | 0% | – |
| TrashBox | 90.9% | 48.4% | 0% | 45.3% |
| GD | 97.3% | 93.7% | 76.6% | 7.1% |
| RealWaste | 96.1% | 51.8% | 71.6% | 55.8% |

**Dataset signature.** A linear probe on generic image features named the source dataset of an image correctly 84.3–98.4% of
the time within each class. A one-class detector trained on one dataset had a median recall of 89.6% on that dataset and 6.6%
on another.

**Leakage.** Under a naive random split, 17.0% of test images had a near-duplicate in training. Class accuracy was 86.3%
against 85.9% under a grouped split (0.4 points).

## Data sources

RealWaste, TrashNet, Garbage Dataset (GD), PSU, Drinking waste, TACO, TIDY, DWSD, BePLi,
SGBattery, MedPackID, a sanitary-pad set, TrashBox, Mayank14, ZeroWaste, WastePro, PharmaSynth (training only) and four classes
of GlobalWasteData. Images per source at each stage are in `results/v3/data/count_table.csv`.

Images are not redistributed here; every source is downloaded from its original host and keeps its own licence.
The split list with perceptual hashes, the duplicate pairs and the label conflicts are in `results/v3/data/`.

## Repository

- `pipeline/`: the code of the manuscript. `rebuild_splits.py` (de-duplication, cap, grouped split), `run.py` (training and CLIP
  jobs; `queue.txt` and `queue_seeds.txt` list them), `analyse.py` and `extra_analyses.py` (every number in the manuscript).
- `tests/`: unit tests of the pipeline on synthetic data.
- `results/v3/`: split lists, run outputs, seed repeats and analysis tables of the manuscript.
- `results/final/`, `results/v2/`, `results/v1/` and `notebooks/`: the earlier course-project runs (October 2026), kept for the record.
- `app/` and `docs/`: the local Gradio app and the browser app (ONNX Runtime Web, served by GitHub Pages), with the October model.
- `tools/export_web_model.py`: exports a trained model for the browser app.

## Cite

A. Dhasmana and P. Gupta, "Unseen items and unseen datasets in photo-based routing of household waste to India's four
segregation streams," Manipal University Jaipur, 2026 (manuscript).

Licence: MIT for the code. Datasets keep their own licences.

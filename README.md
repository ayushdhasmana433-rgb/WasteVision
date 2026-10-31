# WasteVision

Which bin does this go in? WasteVision classifies a photo of one household waste item into 12 material classes and routes it to
one of the four streams that India's **Solid Waste Management Rules, 2026** require at source: **Wet, Dry, Sanitary and Special Care**.
It flags hazards with a cost-derived threshold, says "not sure" when it should, and shows where it looked (Grad-CAM).

**Try it:** https://ayushdhasmana433-rgb.github.io/WasteVision/ (runs in your browser; English and Hindi; works with a phone camera).

The project also asks a research question: does a good score on a shuffled test split mean the bin decision is safe?
We test the decision on datasets the model never saw, on hazard types it never saw, and with near-duplicate photos removed.

## Results (final run)

Data: 29,717 images from 18 public sources after cleaning and cross-source de-duplication
(24,672 near-duplicates removed; 27,918 duplicate pairs crossed two datasets).

| Model | Class acc. | Stream acc. | Special Care recall (4:1) | False alarm |
|---|---|---|---|---|
| EfficientNetV2-B0 (seed 42) | 85.1% | 96.2% | 97.4% | 4.0% |
| MobileNetV2 | 83.3% | 94.6% | 97.7% | 6.4% |
| EfficientNetV2-B0 (seed 7) | 85.3% | 95.9% | 97.6% | 4.9% |
| EfficientNetV2-B0 (seed 123) | 85.3% | 95.9% | 97.4% | 3.9% |
| Scratch CNN | 52.8% | 78.5% | 87.6% | 22.5% |

EfficientNetV2-B0 over three seeds: stream accuracy 96.0 ± 0.1%, class accuracy 85.2 ± 0.1%,
Special Care recall 97.4 ± 0.1%. Answering "Dry" for everything would score 69.6% stream accuracy.

**On a dataset the model never saw** (leave one dataset out, frozen features): stream accuracy 27.6% to 98.7%
(median 81.6%, against 94.9% when the dataset was seen).

| Held-out dataset | Images | Stream acc. | Seen in training | Always-Dry | Special Care miss |
|---|---|---|---|---|---|
| MedPackID | 689 | 27.6% | 94.6% | 0.0% | 53.8% |
| Sanitary | 176 | 39.8% | 81.2% | 0.0% | – |
| Drinking | 641 | 60.4% | 94.9% | 100.0% | – |
| WastePro | 2,011 | 67.8% | 95.4% | 50.2% | 72.5% |
| Mayank14 | 560 | 67.9% | 83.8% | 53.6% | 30.0% |
| GlobalWaste | 10,202 | 71.6% | 92.8% | 39.5% | 19.0% |
| RealWaste | 2,186 | 81.4% | 95.1% | 69.4% | – |
| ZeroWaste | 4,897 | 81.8% | 99.8% | 100.0% | – |
| TACO | 760 | 88.8% | 94.0% | 98.7% | 0.0% |
| TrashNet | 543 | 93.2% | 94.5% | 100.0% | – |
| SGBattery | 314 | 93.3% | 100.0% | 0.0% | 1.3% |
| DWSD | 927 | 94.1% | – | 100.0% | – |
| GD | 4,477 | 95.7% | 97.6% | 69.6% | 0.0% |
| PSU | 716 | 98.7% | 100.0% | 100.0% | – |

**Batteries the model never saw:** 20.2% rejected as "not sure", 76.2% flagged as Special Care,
19.5% sent to the dry bin without a warning. The model called most unseen batteries e-waste, metal or
medicines; because the Special Care flag sums all hazardous classes, that mistake still raised the warning.

**False alarms on new datasets:** the 4:1 hazard threshold, tuned on known datasets, raised false hazard flags on a median of
18% of items from an unseen dataset (up to 54% for drink containers).

**Leakage:** a naive random split scores 87.3%; keeping duplicates together scores 86.7%
(0.7 points of difference). The frozen-feature head cannot memorise photos, so read this as a lower bound.
Whole datasets re-host others: 99% of TrashBox was already inside GlobalWasteData, and 78% of WastePro repeats RealWaste,
TrashNet and GlobalWasteData photos.

Full numbers: `results/final/`.

## Data sources

| Source | Images used |
|---|---|
| GlobalWaste | 10,202 |
| ZeroWaste | 4,897 |
| GD | 4,477 |
| RealWaste | 2,186 |
| WastePro | 2,011 |
| DWSD | 927 |
| TACO | 760 |
| PSU | 716 |
| MedPackID | 689 |
| Drinking | 641 |
| Mayank14 | 560 |
| TrashNet | 543 |
| PharmaSynth | 488 |
| SGBattery | 314 |
| Sanitary | 176 |
| TIDY | 95 |
| TrashBox | 27 |
| BePLi | 8 |

Images are not redistributed here; the notebook downloads every source from its original host. Licences differ by source
(see the paper). The split manifest, perceptual hashes and the duplicate pair list are in `results/final/`.

## Repository

- `notebooks/WasteVision_final_training.ipynb`: the whole pipeline in one notebook (download, clean, de-duplicate, split,
  train, evaluate, research analyses, export). Run the data steps on a CPU runtime, then switch to a T4 GPU and Run all.
- `app/`: local Gradio app (`run_app.cmd`), the same decision logic as the paper.
- `docs/`: the browser app (ONNX Runtime Web), served by GitHub Pages.
- `tools/export_web_model.py`: exports the trained model for the browser app.
- `results/final/`: metrics, figures and analysis tables of the final run.

## Cite

A. Dhasmana and P. Gupta, "Hazard-Aware Waste-Stream Routing from Photographs under Source Shift," Manipal University Jaipur, 2026.

Licence: MIT for the code. Datasets keep their own licences.

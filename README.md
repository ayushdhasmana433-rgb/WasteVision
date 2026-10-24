# WasteVision

WasteVision classifies a photograph of a single household waste item into one of 11 material classes and maps the result to the three source-segregation streams of India's Solid Waste Management (SWM) Rules, 2026: **Wet**, **Dry** and **Special Care**. The app also reports how confident the model is, refuses to answer when confidence is low, and shows a Grad-CAM heat map of the image regions that drove the decision.

**Live demo:** https://ayushdhasmana433-rgb.github.io/WasteVision/ (runs entirely in the browser, on phones too; the photo never leaves the device)

By Ayush Dhasmana, 2026.

## Results (v2, held-out test set, 4,069 images)

| Model | Params (M) | Class accuracy | Macro-F1 | Stream accuracy | Special Care recall at 0.20 | False alarm at 0.20 |
|---|---|---|---|---|---|---|
| Scratch CNN | 1.91 | 0.540 | 0.535 | 0.864 | 0.914 | 0.170 |
| MobileNetV2 | 2.42 | 0.842 | 0.847 | 0.968 | 0.979 | 0.042 |
| **EfficientNetV2-B0** (deployed) | 6.08 | **0.866** | **0.867** | **0.971** | 0.978 | **0.039** |

On 192 street crops from Kolkata (DWSD photographs held out from training) the deployed model reaches 0.755 class accuracy (95% CI 0.678 to 0.828) and 0.979 stream accuracy (0.953 to 1.000). Accuracy on real-life photographs in the test set is 0.802, against 0.925 on web and studio images; that gap is the main limitation.

The training and validation curves stay within one percentage point of each other for the deployed model (train 0.881, validation 0.872 at the kept epoch; validation to test drop 0.6 points), so the model shows no sign of memorising the training set. See `results/v2/overfitting_check.csv` and `results/v2/figures/overfitting_curves.png`.

## Classes and streams

| Stream | Classes |
|---|---|
| Wet | food and garden organics |
| Dry | paper and cardboard, plastic (including thermocol), metal, glass, clothes, shoes, other dry trash |
| Special Care | batteries, medicines and syringes, e-waste |

The stream probability is the sum of the calibrated class probabilities in that stream. If the Special Care probability reaches 0.20 the answer is Special Care, a threshold derived from a 4:1 cost of a missed hazard against a false alarm. Sanitary waste is a separate stream under the Rules but is not covered, because no suitable public image set exists.

## Repository layout

| Path | Contents |
|---|---|
| `app/` | Gradio app (English and Hindi), the trained v2 model in `app/export/`, run scripts |
| `notebooks/WasteVision_v2_training.ipynb` | Final pipeline: data download, cleaning, deduplication, grouped splits, training of three models, calibration, evaluation, export |
| `notebooks/WasteVision_v1_training.ipynb` | First version, trained on two web datasets |
| `results/v2/` | Metrics, figures, per-epoch training logs and the overfitting check for the final models |
| `results/v1/` | Summary of the first version |
| `docs/` | The live web version served by GitHub Pages: the same model exported to ONNX and run with ONNX Runtime Web, the classification head and Grad-CAM computed in JavaScript |
| `tools/export_web_model.py` | Converts the Keras model into the files the web version loads |

## Web version

`docs/` is a static site, so GitHub Pages can host it without a server. The EfficientNetV2-B0 backbone was exported to ONNX (`docs/model/backbone.onnx`) and runs in ONNX Runtime Web. The head (global average pooling, Dense 128 with ReLU, Dense 11) is computed in `docs/app.js` from the exported weights (`docs/model/head.bin`). Grad-CAM is computed analytically from the same weights. On the 35 example and gallery images, the browser version gives the same top class and the same stream decision as the Keras model in `app/`. The median probability difference is 0.5 percentage points, caused by image resampling in the browser. To run it locally, serve the folder with any static server, for example `python -m http.server -d docs`.

## Run the app locally

```
cd app
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS or Linux: . .venv/bin/activate
pip install -r requirements.txt
python app.py                   # add --lan to open it from a phone on the same Wi-Fi
```

On Windows, double-clicking `app/run_app.cmd` does the same. On a phone, the app shows **Take a photo** and **Choose from gallery** buttons above the photo area.

## Reproduce the training

Open `notebooks/WasteVision_v2_training.ipynb` in Google Colab with a T4 GPU and run all cells. The notebook downloads the public datasets itself (TrashNet, TACO, RealWaste, TrashBox, Garbage Dataset, BePLi, Drinking Waste, TIDY and DWSD), so no data is stored in this repository. Each dataset keeps its own licence. Training the three models took about 55 minutes on a T4; downloading, cleaning and evaluation add to that.

## Licence

Code: MIT (see `LICENSE`). Fonts in `app/assets/fonts/`: SIL Open Font License. Example images in `app/export/` come from the public datasets listed above and remain under their original licences.

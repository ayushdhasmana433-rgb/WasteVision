# WasteVision app

Shows which SWM Rules 2026 stream an item belongs in (Wet, Dry, Special Care or Sanitary), how sure the model is, and which part of the photo the answer is based on. English and हिंदी.

## Start it
Double-click **`run_app.cmd`** (Windows). The browser opens at `http://127.0.0.1:7860` once the model has loaded (about 10 seconds). Keep the black window open while you use the app.

On macOS or Linux, from this folder:

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python app.py
```

- **Phone on the same Wi-Fi:** run `run_app_phone.cmd` (or `python app.py --lan`). The window prints an address such as `http://192.168.1.5:7860`; open it on the phone. Phones get two large buttons above the photo area: **Take a photo** opens the rear camera, **Choose from gallery** opens the photo library. The first time, Windows asks whether Python may use the network: allow **Private networks**.
- **Laptop webcam:** click the camera icon under the photo area.
- **No internet needed.** The model runs inside the app on this computer. Nothing is sent anywhere.

## What's in this folder
| Item | What it is |
|---|---|
| `app.py` | The interface (Gradio 5.50) |
| `wv_core.py` | Loads the model once, prepares photos exactly as in training, predicts, draws Grad-CAM, saves feedback |
| `wv_text.py` | Every visible word in English and Hindi |
| `assets/` | Styles, logo, icons and the Anek Latin, Anek Devanagari and Noto Sans Devanagari fonts (SIL Open Font License, bundled so the app works offline) |
| `export/` | The trained model (`wastevision.keras`, EfficientNetV2-B0), its settings (`config.json`), example photos and the Grad-CAM gallery |
| `feedback/` | Created when you confirm or correct an answer: `labels.csv` and `images/` |
| `.venv/` | Private Python environment (TensorFlow, Keras, Gradio). Delete it to free about 1.2 GB; `run_app.cmd` rebuilds it |

## How it decides
- **Stream** = the sum of the calibrated probabilities of the materials in that stream (temperature set in `config.json`).
- **Safety rule:** if the chance of a battery, medicine or e-waste reaches the Special Care threshold in `config.json` (20% at a 4:1 cost ratio), the answer is Special Care (a missed hazard is costed at 4 times a false alarm).
- **Not sure:** below the stream confidence threshold in `config.json`, the app asks for a better photo instead of guessing. Below the material confidence threshold, the wording changes to "Might be" instead of "Looks like".
- These thresholds come from `export/config.json`, which the training notebook wrote from the validation set.

## Retraining with feedback
Each confirmation or correction adds a row to `feedback/labels.csv` in the format the notebook's feedback cell reads. After about 50, copy the `feedback` folder to the feedback folder the notebook reads on Google Drive, set `RUN_FEEDBACK_ROUND = True` in that cell and run it. The new model replaces the current one only if the frozen test scores do not drop; then download the new `export` folder here.

## If something goes wrong
- *Window closes immediately or shows an error:* read the last error line in the window and check that the `export` folder and `.venv` are present.
- *Port already in use:* the app picks the next free port and prints it.
- *Wrong or odd answers:* look at the "Why this answer" image. Retake with one item on a plain surface in good light.

"""Export the trained Keras model for the browser version in docs/.

    python tools/export_web_model.py app/export/wastevision.keras docs/model

Writes, into the output folder:
  backbone.onnx  EfficientNetV2-B0 feature extractor, input 1x224x224x3 in 0..255, output 1x7x7x1280.
  head.bin       float32 little-endian: W1 1280x128, b1 128, W2 128xN, b2 N  (N = number of classes, 12).
  site.json      what the page reads: the model's config.json (classes, the four streams, thresholds, metrics)
                 plus params_m, the example and gallery lists, and the English and Hindi interface text
                 (app/wv_text.py and the few page-only strings below).

config.json, examples.json and gallery.json are read from the folder that holds the .keras file.
To rebuild only site.json (for example after editing app/wv_text.py), add --site-only; the model is
not loaded then, and params_m and the model download size are taken from the existing files in the output folder.

Requires tensorflow, keras and tf2onnx (not needed with --site-only).
"""
import argparse
import importlib.util
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT, HID, SIZE = 1280, 128, 224

# Interface strings that exist only on the web page (the Gradio app has no use for them).
WEB_TEXT = {
    "en": {
        "web_private": "The model runs in your browser. Your photo never leaves this device.",
        "loading": "Loading the model ({mb} MB, first visit only)…",
        "analysing": "Checking the photo…",
        "drop": "Drop a photo here, paste one, or",
        "choose_file": "choose a file",
        "use_camera": "Use camera",
        "capture": "Capture",
        "cancel": "Cancel",
        "error_camera": "The camera could not be opened. Allow camera access, or choose a photo instead.",
        "error_load": "The model could not be loaded. Check the connection and reload the page.",
        "code": "Code and notebooks on GitHub",
        "another": "Try another photo",
        "camera_app": "Open the camera app instead",
    },
    "hi": {
        "web_private": "मॉडल आपके ब्राउज़र में चलता है। आपकी फ़ोटो इस डिवाइस से बाहर नहीं जाती।",
        "loading": "मॉडल लोड हो रहा है ({mb} MB, सिर्फ़ पहली बार)…",
        "analysing": "फ़ोटो जाँची जा रही है…",
        "drop": "फ़ोटो यहाँ छोड़ें, पेस्ट करें, या",
        "choose_file": "फ़ाइल चुनें",
        "use_camera": "कैमरा चालू करें",
        "capture": "फ़ोटो लें",
        "cancel": "रद्द करें",
        "error_camera": "कैमरा नहीं खुल सका। कैमरे की अनुमति दें या कोई फ़ोटो चुनें।",
        "error_load": "मॉडल लोड नहीं हो सका। कनेक्शन जाँचें और पेज दोबारा खोलें।",
        "code": "कोड और नोटबुक GitHub पर",
        "another": "दूसरी फ़ोटो आज़माएँ",
        "camera_app": "इसके बजाय कैमरा ऐप खोलें",
    },
}


def clean(x):
    """JSON for the browser has no NaN: metrics that do not apply become null."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def load_wv_text():
    spec = importlib.util.spec_from_file_location("wv_text", os.path.join(ROOT, "app", "wv_text.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_site(src_dir, out, params_m):
    with open(os.path.join(src_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    classes, streams, stream_of = cfg["classes"], cfg["streams"], cfg["stream_of"]
    for cls in classes:
        assert stream_of.get(cls) in streams, f"class {cls!r} has no valid stream in config.json"
    wv = load_wv_text()
    # every class and stream in the config needs its text in both languages: fail here, not in the browser
    key = lambda name: name.split()[0].lower()
    for lg in wv.LANGS.values():
        for cls in classes:
            assert cls in wv.CLASS_NAMES[lg] and cls in wv.TIPS[lg], f"wv_text.py: missing text for class {cls!r} ({lg})"
        for s in streams:
            assert key(s) in wv.BINS[lg] and key(s) in wv.LEGEND[lg], f"wv_text.py: missing text for stream {s!r} ({lg})"
    mb = round(os.path.getsize(os.path.join(out, "backbone.onnx")) / 1e6)
    text = {"LANGS": wv.LANGS, "CLASS_NAMES": wv.CLASS_NAMES, "BINS": wv.BINS, "LEGEND": wv.LEGEND, "TIPS": wv.TIPS, "T": {}}
    for lg in wv.LANGS.values():
        extra = {k: v.format(mb=mb) for k, v in WEB_TEXT[lg].items()}
        text["T"][lg] = {**json.loads(json.dumps(wv.T[lg])), **extra}
    with open(os.path.join(src_dir, "examples.json"), encoding="utf-8") as f:
        examples = json.load(f)
    with open(os.path.join(src_dir, "gallery.json"), encoding="utf-8") as f:
        gallery = json.load(f)
    site = {"config": clean({**cfg, "params_m": round(params_m, 6), "examples": examples, "gallery": gallery}), "text": text}
    with open(os.path.join(out, "site.json"), "w", encoding="utf-8") as f:
        json.dump(site, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return cfg


def export_model(src, out):
    import numpy as np
    import tensorflow as tf
    import keras
    import tf2onnx
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    model = keras.saving.load_model(src, compile=False)
    backbone, head = model.get_layer("backbone"), model.get_layer("head")
    spec = (tf.TensorSpec((None, SIZE, SIZE, 3), tf.float32, name="image"),)

    @tf.function(input_signature=spec)
    def features(x):
        return {"features": backbone(x, training=False)}

    frozen = convert_variables_to_constants_v2(features.get_concrete_function())   # bake normalisation constants in
    tf2onnx.convert.from_graph_def(frozen.graph.as_graph_def(), input_names=[i.name for i in frozen.inputs],
                                   output_names=[o.name for o in frozen.outputs], opset=17,
                                   output_path=os.path.join(out, "backbone.onnx"))
    W1, b1 = head.get_layer("head_dense").get_weights()
    W2, b2 = head.get_layer("logits").get_weights()
    n = b2.shape[0]
    assert W1.shape == (FEAT, HID) and W2.shape == (HID, n), f"unexpected head shapes {W1.shape}, {W2.shape}"
    np.concatenate([W1.ravel(), b1, W2.ravel(), b2]).astype("<f4").tofile(os.path.join(out, "head.bin"))
    params_m = sum(int(w.shape.num_elements()) for w in model.weights) / 1e6
    return n, params_m


def main():
    ap = argparse.ArgumentParser(description="Export the WasteVision model for the browser")
    ap.add_argument("model", help="path to the .keras file (config.json, examples.json and gallery.json sit beside it)")
    ap.add_argument("out", help="output folder, normally docs/model")
    ap.add_argument("--site-only", action="store_true", help="rebuild site.json only; do not load or convert the model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src_dir = os.path.dirname(os.path.abspath(a.model))

    if a.site_only:
        with open(os.path.join(a.out, "site.json"), encoding="utf-8") as f:
            params_m = json.load(f)["config"]["params_m"]
        n_out = None
    else:
        n_out, params_m = export_model(a.model, a.out)
    cfg = build_site(src_dir, a.out, params_m)

    n_cfg = len(cfg["classes"])
    if n_out is not None:
        assert n_out == n_cfg, f"the model has {n_out} outputs but config.json lists {n_cfg} classes"
        expected = (FEAT * HID + HID + HID * n_out + n_out) * 4
        assert os.path.getsize(os.path.join(a.out, "head.bin")) == expected, "head.bin has an unexpected size"
    print(f"wrote {sorted(os.listdir(a.out))}: {n_cfg} classes, streams {cfg['streams']}")


if __name__ == "__main__":
    main()

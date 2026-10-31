"""WasteVision model engine: loading, preprocessing, prediction, Grad-CAM and feedback storage.

The model runs in this same Python process as the app, so there is no network hop between
the interface and the model. Preprocessing mirrors the training notebook exactly.
"""
import csv
import datetime
import hashlib
import io
import json
import math
import os
import threading
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import logging
import warnings

import numpy as np
from PIL import Image, ImageOps

logging.getLogger("tensorflow").setLevel(logging.ERROR)   # hides the "no GPU on native Windows" notice
warnings.filterwarnings("ignore", module="tensorflow")
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="WasteVision")
class ColorJitter(keras.layers.Layer):
    """Training-time augmentation layer from the notebook. Registered so any saved model loads;
    it is inactive at inference."""

    def __init__(self, brightness=0.25, contrast=0.25, saturation=0.35, hue=0.03, **kwargs):
        super().__init__(**kwargs)
        self.brightness, self.contrast, self.saturation, self.hue = brightness, contrast, saturation, hue

    def call(self, x, training=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "brightness": self.brightness, "contrast": self.contrast,
                "saturation": self.saturation, "hue": self.hue}


FEEDBACK_FIELDS = ["timestamp", "file", "predicted_class", "predicted_stream", "confidence",
                   "human_label", "model_version"]


def stream_key(name):
    """CSS / text key of a stream, derived from its name in config.json:
    "Wet" -> "wet", "Dry" -> "dry", "Special Care" -> "special", "Sanitary" -> "sanitary"."""
    return name.split()[0].lower()


class WasteModel:
    def __init__(self, export_dir):
        self.export_dir = os.path.abspath(export_dir)
        with open(os.path.join(self.export_dir, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        c = self.cfg
        self.classes = c["classes"]
        self.streams = c["streams"]
        self.stream_of = c["stream_of"]
        self.size = int(c["img_size"])
        self.T = float(c["temperature"])
        self.sc_threshold = float(c["special_care_threshold"])
        self.t_stream = float(c["not_sure_stream_threshold"])
        self.t_class = float(c["not_sure_class_threshold"])
        # Only recorded in the feedback file (never shown): the config's own id, else its build timestamp.
        self.version = str(c.get("version") or c.get("created") or "")
        self.arch = c.get("architecture", "")
        missing = [cls for cls in self.classes if self.stream_of.get(cls) not in self.streams]
        if missing:
            raise ValueError(f"config.json: classes without a valid stream in 'stream_of': {missing}")
        self.stream_matrix = np.zeros((len(self.classes), len(self.streams)), np.float32)
        for i, cls in enumerate(self.classes):
            self.stream_matrix[i, self.streams.index(self.stream_of[cls])] = 1.0
        self.sc_index = self.streams.index("Special Care")
        self.sc_classes = [i for i, cls in enumerate(self.classes) if self.stream_of[cls] == "Special Care"]

        t0 = time.time()
        self.model = keras.models.load_model(os.path.join(self.export_dir, c.get("model_file", "wastevision.keras")),
                                             compile=False)
        self._pre = self.model.get_layer("preproc")
        self._backbone = self.model.get_layer("backbone")
        self._head = self.model.get_layer("head")
        self._lock = threading.Lock()
        self._step = tf.function(self._forward_with_cam, reduce_retracing=True)
        n_out = int(self._step(tf.zeros((1, self.size, self.size, 3), tf.float32))[0].shape[-1])  # warm-up: first real photo is fast
        if n_out != len(self.classes):
            raise ValueError(f"The model has {n_out} outputs but config.json lists {len(self.classes)} classes. "
                             "Use the config.json that was exported together with this model.")
        self.load_seconds = time.time() - t0

    # ---------- preprocessing (identical to training) ----------
    @staticmethod
    def clean_image(img):
        """Phone-photo hygiene: respect EXIF rotation, flatten transparency onto white, force RGB."""
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.getchannel("A"))
            return bg
        return img.convert("RGB")

    def model_input(self, img):
        """Training images were stored as RGB JPEG with max side 320, then resize_with_pad to 224."""
        small = img.copy()
        small.thumbnail((320, 320), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=93)
        arr = tf.io.decode_jpeg(buf.getvalue(), channels=3)
        arr = tf.image.resize_with_pad(arr, self.size, self.size, antialias=True)
        arr = tf.clip_by_value(tf.round(arr), 0, 255)
        return tf.cast(arr, tf.float32)[None]

    # ---------- model ----------
    def _forward_with_cam(self, x):
        with tf.GradientTape() as tape:
            fmap = self._backbone(self._pre(x), training=False)
            tape.watch(fmap)
            logits = self._head(fmap, training=False)
            score = tf.reduce_max(logits, axis=1)
        grads = tape.gradient(score, fmap)
        cam = tf.nn.relu(tf.reduce_sum(tf.reduce_mean(grads, axis=(1, 2), keepdims=True) * fmap, axis=-1))
        cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
        return logits, cam

    def predict(self, pil_image):
        img = self.clean_image(pil_image)
        x = self.model_input(img)
        with self._lock:
            logits, cam = self._step(x)
        z = logits.numpy()[0] / self.T
        p = np.exp(z - z.max())
        p = p / p.sum()
        sp = p @ self.stream_matrix
        top = int(p.argmax())
        s_idx = int(sp.argmax())
        hazard = max(self.sc_classes, key=lambda i: p[i])
        if sp[self.sc_index] >= self.sc_threshold and s_idx != self.sc_index:
            mode, verdict = "precaution", "Special Care"
        elif sp[s_idx] < self.t_stream:
            mode, verdict = "unsure", self.streams[s_idx]
        else:
            mode, verdict = "sure", self.streams[s_idx]
        order = np.argsort(-p)
        return dict(
            mode=mode, verdict=verdict, verdict_key=stream_key(verdict),
            top_class=self.classes[top], top_prob=float(p[top]),
            class_sure=bool(p[top] >= self.t_class),
            stream_probs={s: float(v) for s, v in zip(self.streams, sp)},
            stream_prob=float(sp[s_idx]),
            hazard_class=self.classes[hazard], hazard_prob=float(sp[self.sc_index]),
            ranked=[(self.classes[i], float(p[i])) for i in order],
            cam=cam.numpy()[0], image=img,
        )

    # ---------- explanation ----------
    @staticmethod
    def spotlight(img, cam, size=448):
        """Grad-CAM as a spotlight: regions the model relied on stay bright, the rest dims.
        The photo is padded to a square exactly as the model saw it, so the map lines up."""
        sq = tf.image.resize_with_pad(np.asarray(img), size, size, antialias=True).numpy() / 255.0
        heat = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((size, size), Image.BICUBIC)) / 255.0
        heat = np.clip(heat, 0, 1)[..., None]
        warm = np.array([1.0, 0.72, 0.28])
        out = sq * (0.28 + 0.72 * heat) + warm * (heat ** 2) * 0.22
        return np.uint8(np.clip(out, 0, 1) * 255)


class FeedbackStore:
    """Writes human-confirmed labels in the exact format the notebook's retraining cell reads:
    <dir>/labels.csv and <dir>/images/<file>."""

    def __init__(self, folder):
        self.folder = os.path.abspath(folder)
        self.images = os.path.join(self.folder, "images")
        os.makedirs(self.images, exist_ok=True)
        self.csv_path = os.path.join(self.folder, "labels.csv")
        self._lock = threading.Lock()

    def count(self):
        if not os.path.exists(self.csv_path):
            return 0
        with open(self.csv_path, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)

    def save(self, img, result, human_label, version):
        now = datetime.datetime.now()
        buf = io.BytesIO()
        keep = img.copy()
        keep.thumbnail((640, 640), Image.Resampling.LANCZOS)
        keep.save(buf, "JPEG", quality=92)
        data = buf.getvalue()
        name = f"{now:%Y%m%d_%H%M%S}_{hashlib.sha1(data).hexdigest()[:8]}.jpg"
        with self._lock:
            with open(os.path.join(self.images, name), "wb") as f:
                f.write(data)
            new = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(dict(timestamp=now.isoformat(timespec="seconds"), file=name,
                                predicted_class=result["top_class"], predicted_stream=result["verdict"],
                                confidence=round(result["top_prob"], 4), human_label=human_label,
                                model_version=version))
        return self.count()

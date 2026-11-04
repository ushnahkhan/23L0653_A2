import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_NAME = "23L0653_A2_Ops_VisionGuard"
ALIAS = "champion"
import config
STORE = Path(config.MODEL_DIR)                                     # local copy of the championMETA_FILE = STORE / "model_meta.json"
META_FILE = STORE / "model_meta.json"


def preprocess(raw_bytes: bytes, img_size: int) -> tf.Tensor:
    """Same steps as pipeline.load_image(): decode -> RGB -> resize_with_pad -> 0-255.
    Pixel scaling ([-1, 1]) is INSIDE the model, so it cannot be forgotten here."""
    try:
        img = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    except (tf.errors.InvalidArgumentError, ValueError) as e:
        raise ValueError("File is not a readable image (JPEG, PNG, GIF or BMP)") from e
    img = tf.image.resize_with_pad(img, img_size, img_size)                 # keep aspect ratio
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.float32)      # same as uint8 cache
    return img[tf.newaxis]                                                  # batch of 1

def fetch_champion():
    import mlflow
    from dotenv import load_dotenv
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        uri = f"models:/{MODEL_NAME}@{ALIAS}"
    except MlflowException:                                   # registry without alias support
        mv = client.search_model_versions(f"name = '{MODEL_NAME}' and tags.alias = '{ALIAS}'")[0]
        uri = f"models:/{MODEL_NAME}/{mv.version}"

    STORE.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(STORE))
    meta = {"model_name": MODEL_NAME, "alias": ALIAS, "version": mv.version, "run_id": mv.run_id,
            "img_size": int(mv.tags["img_size"]), "threshold": float(mv.tags["threshold"]),
            "source_run": mv.tags.get("source_run", "")}
    META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"[inference] downloaded {uri} (version {mv.version}) -> {STORE}")


class VisionGuard:
    def __init__(self):
        t0 = time.perf_counter()
        if config.MODEL_PATH:                                  # explicit file from env
            model_file = Path(config.MODEL_PATH)
            meta_path = model_file.parent / "model_meta.json"
            self.meta = (json.loads(meta_path.read_text()) if meta_path.exists()
                            else {"version": "local", "source_run": model_file.name})
        else:                                                  # champion from the registry
            if not META_FILE.exists() or not list(STORE.rglob("*.keras")):
                fetch_champion()
            model_file = next(STORE.rglob("*.keras"))
            self.meta = json.loads(META_FILE.read_text())

        self.img_size = config.IMG_SIZE or self.meta.get("img_size")
        self.threshold = config.THRESHOLD if config.THRESHOLD is not None else self.meta.get("threshold")
        if self.img_size is None or self.threshold is None:
            raise RuntimeError("Set VISIONGUARD_IMG_SIZE and VISIONGUARD_THRESHOLD (no model metadata found)")
        self.model_path = str(model_file)
        self.model = tf.keras.models.load_model(model_file)
        self.model.predict(np.zeros((1, self.img_size, self.img_size, 3), np.float32),
                            verbose=0)                     # warm-up: builds the graph once
        self.load_seconds = time.perf_counter() - t0
        print(f"[inference] model v{self.meta.get('version')} ({self.meta['source_run']}) loaded in "
                f"{self.load_seconds:.2f} s | input {self.img_size}px | threshold {self.threshold}")

    def predict(self, raw_bytes: bytes) -> dict:
        t0 = time.perf_counter()
        x = preprocess(raw_bytes, self.img_size)
        p_ai = float(self.model(x, training=False)[0, 0])        # direct call: fast for 1 image
        is_ai = p_ai >= self.threshold
        return {"label": "AI-generated" if is_ai else "Real",
                "confidence": p_ai if is_ai else 1.0 - p_ai,       # confidence in the shown label
                "p_ai": p_ai,
                "threshold": self.threshold,
                "model_version": self.meta["version"],
                "latency_ms": (time.perf_counter() - t0) * 1000}


_MODEL = None


def get_model() -> VisionGuard:
    """Return the single shared model; load it only on the first call."""
    global _MODEL
    if _MODEL is None:
        _MODEL = VisionGuard()
    return _MODEL

if __name__ == "__main__":
    import sys

    import pandas as pd

    import pipeline

    vg = get_model()
    test = pd.read_csv("output/splits/test.csv")
    images = sys.argv[1:] or [test[test["label"] == 1].iloc[0]["filepath"],   # one known AI
                                test[test["label"] == 0].iloc[0]["filepath"]]   # one known real
    truth = dict(zip(test["filepath"], test["label"].map({1: "AI-generated", 0: "Real"})))

    # proof that serving preprocessing == training preprocessing
    pipeline.IMG_SIZE = vg.img_size
    train_img, _ = pipeline.load_image(images[0], 0.0)
    serve_img = preprocess(open(images[0], "rb").read(), vg.img_size)[0]
    diff = float(tf.reduce_max(tf.abs(tf.cast(train_img, tf.float32) - serve_img)))
    print(f"\nPreprocessing check (training vs serving), max pixel difference: {diff}")

    for path in images:
        r = vg.predict(open(path, "rb").read())
        print(f"\n{path}\n  true: {truth.get(path, '?'):13s} predicted: {r['label']:13s} "
                f"confidence {r['confidence']:.1%} | P(AI) {r['p_ai']:.3f} | "
                f"{r['latency_ms']:.0f} ms")

    # call get_model() again -> same object, no reload
    t0 = time.perf_counter()
    same = get_model()
    print(f"\nSecond get_model() call: same object = {same is vg}, "
          f"took {(time.perf_counter() - t0) * 1000:.3f} ms (vs {vg.load_seconds:.2f} s first load)")
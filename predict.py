import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME, ALIAS = "23L0653_A2_Ops_VisionGuard", "champion"

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

# ---------- 1. resolve alias -> version ----------
try:
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
except MlflowException:                                   # fallback: version tagged alias=champion
    mv = client.search_model_versions(f"name = '{MODEL_NAME}' and tags.alias = '{ALIAS}'")[0]
    model_uri = f"models:/{MODEL_NAME}/{mv.version}"

img_size, threshold = int(mv.tags["img_size"]), float(mv.tags["threshold"])
print(f"Alias '{ALIAS}' -> version {mv.version} (run {mv.run_id[:8]}, from '{mv.tags.get('source_run')}')")
print(f"Model URI: {model_uri} | input {img_size}x{img_size} | threshold {threshold}")

# ---------- 2. download + load the Keras model ----------
local_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
keras_file = next(local_dir.rglob("*.keras"))
model = tf.keras.models.load_model(keras_file)
print(f"Loaded {keras_file.name} ({model.count_params():,} params)")

# ---------- 3. pick the image ----------
true_label = None
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    row = pd.read_csv("output/splits/test.csv").iloc[0]
    image_path, true_label = row["filepath"], "AI" if row["label"] == 1 else "Real"

# ---------- 4. same preprocessing as training ----------
img = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
img = tf.image.resize_with_pad(img, img_size, img_size)          # keep aspect ratio
img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.float32)[tf.newaxis]

# ---------- 5. predict ----------
p_ai = float(model.predict(img, verbose=0)[0, 0])
label = "AI-generated" if p_ai >= threshold else "Real"
print(f"\nImage     : {image_path}")
print(f"P(AI)     : {p_ai:.4f}  (threshold {threshold})")
print(f"Prediction: {label}" + (f"   | true label: {true_label}" if true_label else ""))
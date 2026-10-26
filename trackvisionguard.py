import os
import random
import subprocess
import sys

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
import numpy as np
import tensorflow as tf
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import pandas as pd
import yaml
from dotenv import load_dotenv

from pipeline import BATCH_SIZE, IMG_SIZE, make_dataset
from transferlearning import get_metrics

MODE = sys.argv[1] if len(sys.argv) > 1 else "manual"
assert MODE in ("manual", "autolog"), "Usage: python trackvisionguard.py manual | autolog"

HP = {"model": "MobileNetV2", "learning_rate": 3e-4, "dropout": 0.5, "unfrozen_layers": 20,
        "epochs": 12, "batch_size": BATCH_SIZE, "img_size": IMG_SIZE, "optimizer": "adam",
        "loss": "binary_crossentropy", "early_stopping_patience": 4,
        "reduce_lr_patience": 2, "reduce_lr_factor": 0.5,
        "augmentation": "flip_h,rot0.05,zoom0.1,shift0.05,contrast0.1", "seed": SEED}
CKPT = Path(f"output/models/visionguard_{MODE}.keras")
CKPT.parent.mkdir(parents=True, exist_ok=True)

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("23L0653_A2_Ops_visionguard")


def git_info():
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    return {"git_commit": run("rev-parse", "HEAD"),
            "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": str(bool(run("status", "--porcelain")))}


def dvc_info():
    out = yaml.safe_load(open("data.dvc"))["outs"][0]
    return {"dvc_data_md5": out["md5"], "dvc_data_nfiles": str(out.get("nfiles")),
            "dvc_data_size_bytes": str(out.get("size"))}


TAGS = {"student": "23L0653", "task": "3.3", "logging_mode": MODE,
        "seed": str(SEED), **git_info(), **dvc_info()}


def build_model():
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                            include_top=False, weights="imagenet")
    base.trainable = True
    n = HP["unfrozen_layers"]
    for layer in base.layers[:-n]:
        layer.trainable = False
    for layer in base.layers[-n:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(HP["dropout"])(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs, name="visionguard_mobilenetv2")
    model.compile(optimizer=tf.keras.optimizers.Adam(HP["learning_rate"]),
                    loss=HP["loss"], metrics=get_metrics())
    return model


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Manual approach: log every Keras metric + the current learning rate each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        metrics = {k: float(v) for k, v in (logs or {}).items()}
        metrics["lr"] = float(np.array(self.model.optimizer.learning_rate))
        mlflow.log_metrics(metrics, step=epoch + 1)


def base_callbacks():
    return [tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=HP["reduce_lr_factor"],
                                                patience=HP["reduce_lr_patience"], min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=HP["early_stopping_patience"],
                                            restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(CKPT, monitor="val_loss", save_best_only=True)]

train_ds = make_dataset("train", training=True)
val_ds = make_dataset("val", training=False)
test_ds = make_dataset("test", training=False)
sizes = {f"n_{s}": len(pd.read_csv(f"output/splits/{s}.csv")) for s in ("train", "val", "test")}

with mlflow.start_run(run_name=f"visionguard_{MODE}") as run:
    mlflow.set_tags(TAGS)                      # added by hand in BOTH modes (autolog doesn't)

    if MODE == "autolog":
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False)
        callbacks = base_callbacks()
    else:
        mlflow.log_params({**HP, **sizes})
        callbacks = base_callbacks() + [MLflowEpochLogger()]

    model = build_model()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=HP["epochs"],
                    callbacks=callbacks, verbose=2)

    if MODE == "autolog":
        mlflow.tensorflow.autolog(disable=True)

    best = tf.keras.models.load_model(CKPT)
    test_scores = best.evaluate(test_ds, return_dict=True, verbose=0)
    mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_scores.items()})

    if MODE == "manual":
        h = pd.DataFrame(hist.history)
        best_epoch = int(h["val_loss"].idxmin()) + 1
        mlflow.log_metric("best_epoch", best_epoch)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, m in zip(axes, ["loss", "accuracy", "auc"]):
            ax.plot(range(1, len(h) + 1), h[m], "o-", label="train")
            ax.plot(range(1, len(h) + 1), h[f"val_{m}"], "o-", label="validation")
            ax.axvline(best_epoch, color="red", ls="--", label=f"best epoch {best_epoch}")
            ax.set_title(m)
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        mlflow.log_figure(fig, "training_curves.png")
        plt.close(fig)

        mlflow.log_artifact(str(CKPT), artifact_path="model")       # best model file
        mlflow.log_artifacts("output/splits", artifact_path="splits")  # exact split lists

    print(f"\nRun ID: {run.info.run_id}")
    print("Test:", {k: round(float(v), 4) for k, v in test_scores.items()})
    print("Tags:", TAGS)
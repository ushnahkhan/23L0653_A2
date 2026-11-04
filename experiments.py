import os
import random
import subprocess
import sys
import time

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
import numpy as np
import tensorflow as tf

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score

from pipeline import augment as augment_layer
from transferlearning import get_metrics

EXPERIMENT = "23L0653_A2_Ops_visionguard_comparison"
BATCH_SIZE, AUTOTUNE = 32, tf.data.AUTOTUNE
BASE = dict(backbone="MobileNetV2", learning_rate=3e-4, dropout=0.5, unfrozen_layers=20,
            img_size=224, augment=True, epochs=12)
CONFIGS = {
    "baseline":       dict(change="baseline (tuned config)"),
    "lr_1e-4":        dict(learning_rate=1e-4, change="learning rate 3e-4 -> 1e-4"),
    "no_aug":         dict(augment=False, change="augmentation off"),
    "frozen":         dict(unfrozen_layers=0, change="backbone fully frozen"),
    "img_160":        dict(img_size=160, change="image size 224 -> 160"),
    "efficientnetb0": dict(backbone="EfficientNetB0", change="backbone -> EfficientNetB0"),
}
MODEL_DIR = Path("output/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT)


# ---------------- tags ----------------
def repro_tags():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    dvc = yaml.safe_load(open("data.dvc"))["outs"][0]
    return {"student": "23L0653", "task": "3.5", "seed": str(SEED),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": str(bool(git("status", "--porcelain"))),
            "dvc_data_md5": dvc["md5"]}


# ---------------- data (image size + augmentation configurable) ----------------
def make_ds(split, img_size, training=False, augment=True):
    df = pd.read_csv(f"output/splits/{split}.csv")
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values,
                                                df["label"].values.astype("float32")))

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize_with_pad(img, img_size, img_size)
        return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8), label

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).map(lambda x, y: (tf.cast(x, tf.float32), y),
                                    num_parallel_calls=AUTOTUNE)
    if training and augment:
        ds = ds.map(lambda x, y: (augment_layer(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ---------------- model (backbone + frozen layers configurable) ----------------
def build_model(cfg):
    s = cfg["img_size"]
    if cfg["backbone"] == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(input_shape=(s, s, 3), include_top=False,
                                                    weights="imagenet")
        prep = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")  # -> [-1, 1]
    else:
        base = tf.keras.applications.EfficientNetB0(input_shape=(s, s, 3), include_top=False,
                                                    weights="imagenet")
        prep = tf.keras.layers.Identity(name="preprocess")  # EfficientNet rescales 0-255 internally

    n = cfg["unfrozen_layers"]
    base.trainable = n > 0
    if n:
        for layer in base.layers[:-n]:
            layer.trainable = False
        for layer in base.layers[-n:]:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    inputs = tf.keras.layers.Input(shape=(s, s, 3), name="image")
    x = base(prep(inputs), training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(cfg["dropout"])(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
                    loss="binary_crossentropy", metrics=get_metrics())
    return model


class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        m = {k: float(v) for k, v in (logs or {}).items()}
        m["lr"] = float(np.array(self.model.optimizer.learning_rate))
        m["epoch_time_s"] = time.time() - self.t0
        mlflow.log_metrics(m, step=epoch + 1)


# ---------------- one experiment ----------------
def run_experiment(name):
    cfg = {**BASE, **CONFIGS[name]}
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    ckpt = MODEL_DIR / f"exp_{name}.keras"

    train_ds = make_ds("train", cfg["img_size"], training=True, augment=cfg["augment"])
    val_ds = make_ds("val", cfg["img_size"])

    with mlflow.start_run(run_name=name):
        mlflow.set_tags({**repro_tags(), "change": cfg["change"]})
        mlflow.log_params({k: v for k, v in cfg.items() if k != "change"})

        model = build_model(cfg)
        mlflow.log_metric("total_params", model.count_params())
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                                    min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                                restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
            EpochLogger()]
        t0 = time.time()
        hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"],
                            callbacks=callbacks, verbose=2)
        train_minutes = (time.time() - t0) / 60

        # selection metrics on VALIDATION (best checkpoint)
        best = tf.keras.models.load_model(ckpt)
        y = np.concatenate([yb.numpy() for _, yb in val_ds]).astype(int)
        p = best.predict(val_ds, verbose=0).ravel()
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
        f05 = [fbeta_score(y, (p >= t).astype(int), beta=0.5, zero_division=0) for t in grid]
        t = float(grid[int(np.argmax(f05))])
        tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
        sel = {"sel_val_auc": roc_auc_score(y, p),
                "sel_val_loss": float(best.evaluate(val_ds, return_dict=True, verbose=0)["loss"]),
                "sel_val_f05": float(max(f05)), "sel_threshold": t,
                "sel_val_precision": tp / max(tp + fp, 1), "sel_val_recall": tp / max(tp + fn, 1),
                "sel_val_fpr": fp / max(fp + tn, 1),
                "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
                "train_minutes": train_minutes, "model_size_mb": ckpt.stat().st_size / 1e6}
        mlflow.log_metrics(sel)
        mlflow.log_artifact(str(ckpt), artifact_path="model")
        print(f"\n{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in sel.items()))


# ---------------- comparison ----------------
def compare():
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    runs = mlflow.search_runs([exp.experiment_id], filter_string="tags.task = '3.5'",
                                order_by=["start_time DESC"])
    runs = runs.drop_duplicates("tags.mlflow.runName")          # latest run per name
    cols = {"tags.mlflow.runName": "run", "tags.change": "change",
            "metrics.sel_val_auc": "val_auc", "metrics.sel_val_loss": "val_loss",
            "metrics.sel_val_f05": "val_F0.5", "metrics.sel_val_precision": "val_precision",
            "metrics.sel_val_recall": "val_recall", "metrics.sel_val_fpr": "val_FPR",
            "metrics.sel_threshold": "threshold", "metrics.best_epoch": "best_epoch",
            "metrics.train_minutes": "train_min", "metrics.model_size_mb": "size_MB"}
    df = runs[list(cols)].rename(columns=cols).sort_values("val_auc", ascending=False)
    df.to_csv("output/experiment_comparison.csv", index=False)
    print(df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, m in zip(axes, ["val_auc", "val_F0.5", "val_FPR"]):
        d = df.sort_values(m, ascending=(m == "val_FPR"))
        bars = ax.barh(d["run"], d[m], color=["tab:green" if r == df.iloc[0]["run"] else "tab:blue"
                                                for r in d["run"]])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_title(m + (" (lower is better)" if m == "val_FPR" else " (higher is better)"))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("VisionGuard experiment comparison (validation set)", fontsize=13)
    fig.tight_layout()
    fig.savefig("output/figures/experiment_comparison.png", dpi=150)
    print("\nSaved output/experiment_comparison.csv and output/figures/experiment_comparison.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["compare"]:
        compare()
    else:
        for name in (args or list(CONFIGS)):
            run_experiment(name)
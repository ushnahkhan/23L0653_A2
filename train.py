import os
import random
import sys

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)          # Python + NumPy + TensorFlow
tf.config.experimental.enable_op_determinism()  # same ops -> same results

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline import make_dataset

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "cnn"
OUT_DIR, FIG_DIR, MODEL_DIR = Path("output"), Path("output/figures"), Path("output/models")
for d in (FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
CKPT_PATH = MODEL_DIR / f"{MODEL_NAME}_best.keras"


# ---------- 2. Callbacks ----------
class LRLogger(tf.keras.callbacks.Callback):
    """Record the learning rate used in each epoch (to show LR reductions)."""
    def on_epoch_end(self, epoch, logs=None):
        logs["lr"] = float(np.array(self.model.optimizer.learning_rate))


def make_callbacks(checkpoint, min_lr):
    return [
        LRLogger(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                            min_lr=min_lr, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                        restore_best_weights=True, verbose=1),
        checkpoint,
    ]


def to_df(history, start=0):
    df = pd.DataFrame(history.history)
    df.insert(0, "epoch", range(start + 1, start + len(df) + 1))
    return df


train_ds = make_dataset("train", training=True)
val_ds = make_dataset("val", training=False)
checkpoint = tf.keras.callbacks.ModelCheckpoint(CKPT_PATH, monitor="val_loss",
                                                save_best_only=True, verbose=1)
boundary = None

if MODEL_NAME == "cnn":
    from baselinecnn import build_cnn
    model = build_cnn()                                        # Adam 1e-3, compiled
    h = model.fit(train_ds, validation_data=val_ds, epochs=40,
                    callbacks=make_callbacks(checkpoint, min_lr=1e-6))
    hist = to_df(h)

elif MODEL_NAME == "mobilenet":
    from transferlearning import build_model, get_metrics, LAST_BLOCK
    model, base = build_model()

    # Stage 1 - frozen backbone
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss="binary_crossentropy", metrics=get_metrics())
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=15,
                    callbacks=make_callbacks(checkpoint, min_lr=1e-5))
    hist1 = to_df(h1)

    # Stage 2 - unfreeze last block only, 100x smaller LR
    base.trainable = True
    for layer in base.layers:
        layer.trainable = (layer.name.startswith(LAST_BLOCK)
                        and not isinstance(layer, tf.keras.layers.BatchNormalization))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                    loss="binary_crossentropy", metrics=get_metrics())
    start = len(hist1)
    h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=start,
                    epochs=start + 15, callbacks=make_callbacks(checkpoint, min_lr=1e-7))
    hist = pd.concat([hist1, to_df(h2, start)], ignore_index=True)
    boundary = start + 0.5
else:
    sys.exit("Usage: python train.py cnn | mobilenet")

hist.to_csv(OUT_DIR / f"history_{MODEL_NAME}.csv", index=False)


# ---------- 4. Best epoch + curves ----------
best = hist.loc[hist["val_loss"].idxmin()]
best_epoch = int(best["epoch"])
lr_drops = [int(e) for e, prev, cur in zip(hist["epoch"][1:], hist["lr"][:-1], hist["lr"][1:])
            if cur < prev]

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
for ax, m, title in zip(axes[:2], ["loss", "accuracy"], ["Loss", "Accuracy"]):
    ax.plot(hist["epoch"], hist[m], "o-", label="train")
    ax.plot(hist["epoch"], hist[f"val_{m}"], "o-", label="validation")
    ax.axvline(best_epoch, color="red", ls="--", label=f"best epoch = {best_epoch}")
    ax.scatter(best_epoch, best[f"val_{m}"], s=220, marker="*", color="red", zorder=5)
    ax.annotate(f"{best[f'val_{m}']:.3f}", (best_epoch, best[f"val_{m}"]),
                textcoords="offset points", xytext=(8, 8), color="red")
    ax.set_title(title)
axes[2].plot(hist["epoch"], hist["lr"], "o-", color="purple")
axes[2].set_yscale("log")
axes[2].set_title("Learning rate (ReduceLROnPlateau)")
for ax in axes:
    if boundary:
        ax.axvline(boundary, color="gray", ls=":", label="start fine-tuning")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle(f"{MODEL_NAME}: training curves (best epoch {best_epoch}, "
            f"val_loss {best['val_loss']:.3f})", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / f"curves_{MODEL_NAME}_best.png", dpi=150)
plt.close(fig)


# ---------- 5. Reload best checkpoint and confirm ----------
best_model = tf.keras.models.load_model(CKPT_PATH)
val_scores = best_model.evaluate(val_ds, return_dict=True, verbose=0)

summary = {"model": MODEL_NAME, "seed": SEED, "epochs_run": len(hist),
            "best_epoch": best_epoch, "lr_reduced_after_epochs": lr_drops,
            "best_checkpoint": str(CKPT_PATH),
            "val_scores_of_best_checkpoint": {k: round(float(v), 4) for k, v in val_scores.items()}}
(OUT_DIR / f"summary_{MODEL_NAME}.json").write_text(json.dumps(summary, indent=2))
print("\n" + json.dumps(summary, indent=2))
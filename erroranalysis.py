import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from pipeline import make_dataset

Image.MAX_IMAGE_PIXELS = None
MODEL_NAME, MODEL_PATH = "MobileNetV2", "output/models/mobilenet_best.keras"
THRESHOLD = json.load(open("output/thresholds.json"))[MODEL_NAME]
OUT_DIR, FIG_DIR = Path("output"), Path("output/figures")
N_EACH = 5


def sharpness(img):
    """Variance of the Laplacian on a 512-px greyscale copy: low = blurry / smooth."""
    g = img.convert("L")
    g.thumbnail((512, 512))
    g = np.asarray(g, dtype=np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def saturation(img):
    """Mean colour saturation 0-255: low = greyscale / muted, high = vivid."""
    small = img.convert("RGB")
    small.thumbnail((256, 256))
    return float(np.asarray(small.convert("HSV"))[..., 1].mean())


model = tf.keras.models.load_model(MODEL_PATH)
test = pd.read_csv("output/splits/test.csv")
ds = make_dataset("test", training=False)
test["prob_ai"] = model.predict(ds, verbose=0).ravel()
test["pred"] = (test["prob_ai"] >= THRESHOLD).astype(int)

meta = pd.read_csv("output/clean_index.csv")
meta["filepath"] = "data/" + meta["path"]
test = test.merge(meta[["filepath", "width", "height", "size", "format", "mode"]],
                    on="filepath", how="left")

sharp, sat = [], []
for fp in test["filepath"]:
    with Image.open(fp) as im:
        sharp.append(sharpness(im))
        sat.append(saturation(im))
test["sharpness"], test["saturation"] = sharp, sat
test["aspect"] = (test["width"] / test["height"]).round(2)

fp = test[(test["label"] == 0) & (test["pred"] == 1)].sort_values("prob_ai", ascending=False)
fn = test[(test["label"] == 1) & (test["pred"] == 0)].sort_values("prob_ai")
wrong = pd.concat([fp.head(N_EACH).assign(error="FP: real -> AI"),
                    fn.head(N_EACH).assign(error="FN: AI -> real")], ignore_index=True)
wrong.insert(0, "id", range(1, len(wrong) + 1))

cols = ["id", "error", "prob_ai", "width", "height", "aspect", "size", "format",
        "mode", "sharpness", "saturation", "filepath"]
wrong[cols].to_csv(OUT_DIR / "misclassified.csv", index=False)

print(f"Threshold {THRESHOLD} | total errors: FP = {len(fp)}, FN = {len(fn)}\n")
print("Test-set medians per class (for comparison):")
print(test.groupby("class")[["width", "height", "size", "sharpness", "saturation"]]
        .median().round(1).to_string(), "\n")
print(wrong[cols[:-1]].round(2).to_string(index=False))

fig, axes = plt.subplots(2, N_EACH, figsize=(4 * N_EACH, 9.5))
for ax, (_, r) in zip(axes.flat, wrong.iterrows()):
    with Image.open(r["filepath"]) as im:
        im = im.convert("RGB")
        im.thumbnail((500, 500))
        ax.imshow(im)
    true_lbl = "Real" if r["label"] == 0 else "AI"
    pred_lbl = "AI" if r["pred"] == 1 else "Real"
    ax.set_title(f"#{r['id']}  True: {true_lbl} -> Pred: {pred_lbl}\n"
                    f"P(AI) = {r['prob_ai']:.2f} | {int(r['width'])}x{int(r['height'])} "
                    f"{r['format']} {r['size']:.0f} KB", fontsize=9)
    ax.axis("off")
axes[0, 0].text(-0.08, 0.5, "False positives\n(real called AI)", transform=axes[0, 0].transAxes,
                rotation=90, ha="right", va="center", fontsize=12, fontweight="bold")
axes[1, 0].text(-0.08, 0.5, "False negatives\n(AI called real)", transform=axes[1, 0].transAxes,
                rotation=90, ha="right", va="center", fontsize=12, fontweight="bold")
fig.suptitle(f"{MODEL_NAME}: 10 most confident test-set errors (threshold {THRESHOLD})",
                fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "misclassified_grid.png", dpi=150)
plt.close(fig)
print(f"\nSaved {FIG_DIR / 'misclassified_grid.png'} and {OUT_DIR / 'misclassified.csv'}")
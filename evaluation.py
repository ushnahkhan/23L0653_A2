import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                                f1_score, fbeta_score, precision_score, recall_score,
                                roc_auc_score, roc_curve)

from pipeline import make_dataset

tf.keras.utils.set_random_seed(42)
MODELS = {"Baseline CNN": "output/models/cnn_best.keras",
            "MobileNetV2": "output/models/mobilenet_best.keras"}
BETA = 0.5          # <1 favours precision: a false "AI" accusation costs more than a miss
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
OUT_DIR, FIG_DIR = Path("output"), Path("output/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def predict(model, split):
    ds = make_dataset(split, training=False)                  # no augmentation, fixed order
    y = np.concatenate([yb.numpy() for _, yb in ds]).astype(int)
    p = model.predict(ds, verbose=0).ravel()                  # P(image is AI)
    return y, p


def threshold_sweep(y, p):
    rows = []
    for t in THRESHOLDS:
        pred = (p >= t).astype(int)
        rows.append({"threshold": t,
                        "precision": precision_score(y, pred, zero_division=0),
                        "recall": recall_score(y, pred, zero_division=0),
                        "f1": f1_score(y, pred, zero_division=0),
                        f"f{BETA}": fbeta_score(y, pred, beta=BETA, zero_division=0)})
    return pd.DataFrame(rows)


def scores(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": t,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p),
            "false_positive_rate": fp / (fp + tn),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


results, curves, rows = {}, {}, []
for name, path in MODELS.items():
    model = tf.keras.models.load_model(path)

    y_val, p_val = predict(model, "val")
    sweep = threshold_sweep(y_val, p_val)
    t_best = float(sweep.loc[sweep[f"f{BETA}"].idxmax(), "threshold"])
    curves[name] = sweep

    y_test, p_test = predict(model, "test")
    s_chosen = scores(y_test, p_test, t_best)
    s_default = scores(y_test, p_test, 0.5)
    results[name] = {"y": y_test, "p": p_test, "t": t_best, "s": s_chosen}

    rows.append({"model": name, "threshold_rule": f"max F{BETA} on val", **s_chosen})
    rows.append({"model": name, "threshold_rule": "default 0.5 (reference)", **s_default})
    print(f"{name}: threshold chosen on validation = {t_best}")

table = pd.DataFrame(rows)
table.to_csv(OUT_DIR / "test_metrics.csv", index=False)
print("\n" + table.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (name, sw) in zip(axes, curves.items()):
    for col in ["precision", "recall", "f1", f"f{BETA}"]:
        ax.plot(sw["threshold"], sw[col], label=col, lw=2 if col == f"f{BETA}" else 1.2)
    t = results[name]["t"]
    ax.axvline(t, color="red", ls="--", label=f"chosen t = {t:.2f}")
    ax.axvline(0.5, color="gray", ls=":", label="default 0.5")
    ax.set_title(f"{name} - validation set")
    ax.set_xlabel("Decision threshold")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "threshold_selection_val.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(r["y"], r["p"])
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {r['s']['roc_auc']:.3f})")
    ax.scatter(r["s"]["false_positive_rate"], r["s"]["recall"], s=90, zorder=5,
                label=f"{name} @ t = {r['t']:.2f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guess")
ax.set_xlabel("False positive rate (real called AI)")
ax.set_ylabel("True positive rate / recall (AI caught)")
ax.set_title("ROC curves - test set")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "roc_test.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(r["y"], (r["p"] >= r["t"]).astype(int), labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Real", "AI"]).plot(ax=ax, cmap="Blues",
                                                                    colorbar=False)
    ax.set_title(f"{name} - test (t = {r['t']:.2f})")
fig.tight_layout()
fig.savefig(FIG_DIR / "confusion_matrices_test.png", dpi=150)
plt.close(fig)

json.dump({n: r["t"] for n, r in results.items()},
            open(OUT_DIR / "thresholds.json", "w"), indent=2)
print("\nSaved: output/test_metrics.csv, output/thresholds.json,")
print("       output/figures/threshold_selection_val.png, roc_test.png, confusion_matrices_test.png")
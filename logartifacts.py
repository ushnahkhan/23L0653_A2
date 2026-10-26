import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, fbeta_score,
                                roc_auc_score, roc_curve)

from pipeline import make_dataset

RUN_NAME = "visionguard_manual"
MODEL_PATH = "output/models/visionguard_manual.keras"
EXPERIMENT = "23L0653_A2_Ops_visionguard"

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

exp = mlflow.get_experiment_by_name(EXPERIMENT)
runs = mlflow.search_runs([exp.experiment_id],
                            filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
                            order_by=["start_time DESC"])
if runs.empty:
    sys.exit(f"No run named {RUN_NAME} found")
run_id = runs.iloc[0]["run_id"]
print("Attaching artifacts to run", run_id)


def predict(model, split):
    ds = make_dataset(split, training=False)
    y = np.concatenate([yb.numpy() for _, yb in ds]).astype(int)
    return y, model.predict(ds, verbose=0).ravel()


model = tf.keras.models.load_model(MODEL_PATH)

y_val, p_val = predict(model, "val")
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
thr = float(grid[np.argmax([fbeta_score(y_val, (p_val >= t).astype(int), beta=0.5,
                                        zero_division=0) for t in grid])])
y_test, p_test = predict(model, "test")

with mlflow.start_run(run_id=run_id), tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    cm = confusion_matrix(y_test, (p_test >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Real", "AI"]).plot(ax=ax, cmap="Blues",
                                                                colorbar=False)
    ax.set_title(f"Test confusion matrix (threshold {thr:.2f})")
    fig.tight_layout()
    mlflow.log_figure(fig, "evaluation/confusion_matrix.png")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc = roc_auc_score(y_test, p_test)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"MobileNetV2 (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title("ROC curve - test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    mlflow.log_figure(fig, "evaluation/roc_curve.png")
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    (tmp / "threshold.json").write_text(json.dumps(
        {"threshold": thr, "rule": "max F0.5 on validation", "test_auc": auc,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}, indent=2))
    mlflow.log_artifact(str(tmp / "threshold.json"), artifact_path="evaluation")

    mlflow.log_artifacts("output/splits", artifact_path="splits")

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    (tmp / "requirements-lock.txt").write_text(freeze)
    mlflow.log_artifact(str(tmp / "requirements-lock.txt"), artifact_path="environment")
    mlflow.log_artifact("requirements.txt", artifact_path="environment")

    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    try:
        mlflow.tensorflow.log_model(model, name="mlflow_model")
    except Exception as e:
        print(f"MLflow model format not supported by server ({type(e).__name__}) - .keras file logged")

print(f"Done. Threshold {thr:.2f} | test AUC {auc:.3f} | TN={tn} FP={fp} FN={fn} TP={tp}")
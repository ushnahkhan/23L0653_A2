from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from pipeline import make_dataset, IMG_SIZE

SEED = 42
tf.keras.utils.set_random_seed(SEED)

OUT_DIR = Path("output")
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
for d in (FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

STAGE1_EPOCHS, STAGE1_LR = 10, 1e-3
STAGE2_EPOCHS, STAGE2_LR = 10, 1e-5            # 100x smaller
LAST_BLOCK = ("block_16_", "Conv_1", "out_relu")  # final block of MobileNetV2


def get_metrics():
    return ["accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")]


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,          # drop the 1000-class ImageNet classifier
        weights="imagenet")
    base.trainable = False          # Stage 1: freeze every backbone layer

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
    x = base(x, training=False)     # keep BatchNorm in inference mode, even when fine-tuning
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return models.Model(inputs, outputs, name="mobilenetv2_transfer"), base


def check_preprocessing():
    """Prove the Rescaling layer equals Keras' official preprocess_input."""
    sample = np.random.uniform(0, 255, (2, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    official = tf.keras.applications.mobilenet_v2.preprocess_input(sample.copy())
    ours = layers.Rescaling(1.0 / 127.5, offset=-1.0)(sample).numpy()
    print(f"Preprocessing check - max difference vs preprocess_input: "
        f"{np.abs(official - ours).max():.2e} | range [{ours.min():.2f}, {ours.max():.2f}]")


def trainable_report(model, base, stage, path):
    head = [l for l in model.layers if l is not base and l.trainable_weights]
    backbone = [l for l in base.layers if l.trainable_weights]
    n_train = sum(int(tf.size(w)) for w in model.trainable_weights)
    n_frozen = sum(int(tf.size(w)) for w in model.non_trainable_weights)

    lines = [f"===== {stage} =====",
            f"Trainable params: {n_train:,} | Non-trainable params: {n_frozen:,}",
            f"Trainable backbone layers ({len(backbone)}):"]
    lines += [f"   {l.name:28s} {l.__class__.__name__:22s} {l.count_params():>9,}" for l in backbone] or ["   (none - backbone frozen)"]
    lines += [f"Trainable head layers ({len(head)}):"]
    lines += [f"   {l.name:28s} {l.__class__.__name__:22s} {l.count_params():>9,}" for l in head]
    text = "\n".join(lines)
    print(text + "\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n\n")


def plot_history(hist, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, m in zip(axes, ["loss", "accuracy", "auc"]):
        ax.plot(hist["epoch"], hist[m], "o-", label="train")
        ax.plot(hist["epoch"], hist[f"val_{m}"], "o-", label="validation")
        ax.set_title(m.upper() if m == "auc" else m.capitalize())
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_df(history, start_epoch=0):
    df = pd.DataFrame(history.history)
    df.insert(0, "epoch", range(start_epoch + 1, start_epoch + len(df) + 1))
    return df

if __name__ == "__main__":
    check_preprocessing()
    train_ds = make_dataset("train", training=True)
    val_ds = make_dataset("val", training=False)

    model, base = build_model()
    report_path = OUT_DIR / "trainable_layers.txt"
    report_path.write_text("", encoding="utf-8")
    model.compile(optimizer=tf.keras.optimizers.Adam(STAGE1_LR),
                loss="binary_crossentropy", metrics=get_metrics())
    model.summary(line_length=100)
    trainable_report(model, base, "STAGE 1 - frozen backbone", report_path)

    early1 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                            restore_best_weights=True)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=STAGE1_EPOCHS, callbacks=[early1])
    hist1 = to_df(h1)
    hist1.to_csv(OUT_DIR / "history_stage1.csv", index=False)
    plot_history(hist1, f"Stage 1 - frozen MobileNetV2, head only (lr={STAGE1_LR})",
                FIG_DIR / "curves_stage1.png")
    print("Stage 1 validation:", model.evaluate(val_ds, return_dict=True, verbose=0))

    base.trainable = True
    for layer in base.layers:
        in_last_block = layer.name.startswith(LAST_BLOCK)
        layer.trainable = in_last_block and not isinstance(layer, layers.BatchNormalization)

    model.compile(optimizer=tf.keras.optimizers.Adam(STAGE2_LR),   # must recompile after changing trainable
                loss="binary_crossentropy", metrics=get_metrics())
    trainable_report(model, base, "STAGE 2 - last block unfrozen", report_path)

    early2 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                            restore_best_weights=True)
    start = len(hist1)
    h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=start,
                epochs=start + STAGE2_EPOCHS, callbacks=[early2])
    hist2 = to_df(h2, start_epoch=start)
    hist2.to_csv(OUT_DIR / "history_stage2.csv", index=False)
    plot_history(hist2, f"Stage 2 - last block fine-tuned (lr={STAGE2_LR})",
                FIG_DIR / "curves_stage2.png")
    print("Stage 2 validation:", model.evaluate(val_ds, return_dict=True, verbose=0))

    model.save(MODEL_DIR / "mobilenetv2_finetuned.keras")
    print(f"\nSaved model -> {MODEL_DIR / 'mobilenetv2_finetuned.keras'}")
    print(f"Saved curves -> {FIG_DIR / 'curves_stage1.png'}, {FIG_DIR / 'curves_stage2.png'}")
    print(f"Saved trainable layer lists -> {report_path}")
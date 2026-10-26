import sys
from pathlib import Path

import keras_tuner as kt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
OUT_DIR, FIG_DIR, TUNER_DIR = Path("output"), Path("output/figures"), Path("output/tuner")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def results_table(tuner, hp_names, extra_metrics=()):
    rows = []
    for t in tuner.oracle.get_best_trials(num_trials=1000):       # sorted best -> worst
        row = {"trial": t.trial_id}
        row.update({h: t.hyperparameters.get(h) for h in hp_names})
        row["score"] = t.score
        for m in extra_metrics:
            row[m] = t.metrics.get_best_value(m)
        rows.append(row)
    return pd.DataFrame(rows)


def run_toy():
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    x_tr, x_te = x_tr / 255.0, x_te / 255.0

    def build(hp):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=10, seed=SEED,
                            directory=TUNER_DIR, project_name="fashion_mnist_toy")
    tuner.search_space_summary()
    tuner.search(x_tr, y_tr, epochs=5, validation_split=0.2, verbose=2)

    df = results_table(tuner, ["units", "learning_rate"]).rename(columns={"score": "val_accuracy"})
    df.to_csv(OUT_DIR / "tuner_toy_results.csv", index=False)
    print("\n=== Toy results (best first) ===\n" + df.round(4).to_string(index=False))

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]
    _, test_acc = best_model.evaluate(x_te, y_te, verbose=0)
    print(f"\nBest settings: units = {best_hp.get('units')}, "
            f"learning_rate = {best_hp.get('learning_rate')}")
    print(f"Best val_accuracy = {df['val_accuracy'].iloc[0]:.4f} | test accuracy = {test_acc:.4f}")


def run_visionguard():
    from pipeline import IMG_SIZE, make_dataset
    from transferlearning import get_metrics

    train_ds = make_dataset("train", training=True)
    val_ds = make_dataset("val", training=False)

    def build(hp):
        lr = hp.Choice("learning_rate", [1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
        dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
        n_unfrozen = hp.Choice("unfrozen_layers", [0, 10, 20, 40])   # last N backbone layers

        base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                    include_top=False, weights="imagenet")
        base.trainable = n_unfrozen > 0
        if n_unfrozen:
            for layer in base.layers[:-n_unfrozen]:
                layer.trainable = False
            for layer in base.layers[-n_unfrozen:]:          # BatchNorm stays frozen
                layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

        inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                        loss="binary_crossentropy", metrics=get_metrics())
        return model

    tuner = kt.RandomSearch(build, objective=kt.Objective("val_auc", direction="max"),
                            max_trials=8, seed=SEED,
                            directory=TUNER_DIR, project_name="visionguard_mobilenetv2")
    tuner.search_space_summary()
    early = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2,
                                                restore_best_weights=True)
    tuner.search(train_ds, validation_data=val_ds, epochs=6, callbacks=[early], verbose=2)

    hps = ["learning_rate", "dropout", "unfrozen_layers"]
    df = results_table(tuner, hps, extra_metrics=["val_loss", "val_accuracy"])
    df = df.rename(columns={"score": "val_auc"})
    df["dropout"] = df["dropout"].round(1)
    df.to_csv(OUT_DIR / "tuner_visionguard_results.csv", index=False)
    print("\n=== VisionGuard tuner results (best first) ===\n" + df.round(4).to_string(index=False))

    # which hyperparameter mattered most: spread of mean val_auc across its values
    print("\nEffect of each hyperparameter (max - min of mean val_auc per value):")
    for h in hps:
        means = df.groupby(h)["val_auc"].mean()
        print(f"  {h:16s} spread = {means.max() - means.min():.4f} | "
                + ", ".join(f"{k}: {v:.3f}" for k, v in means.items()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, h in zip(axes, hps):
        ax.scatter(df[h], df["val_auc"], s=70)
        if h == "learning_rate":
            ax.set_xscale("log")
        ax.set_xlabel(h)
        ax.set_ylabel("best val AUC")
        ax.grid(alpha=0.3)
    fig.suptitle("Keras Tuner - VisionGuard MobileNetV2 (8 trials)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tuner_visionguard.png", dpi=150)
    plt.close(fig)

    best = tuner.get_best_hyperparameters(1)[0]
    print(f"\nBest: learning_rate = {best.get('learning_rate')}, dropout = "
            f"{best.get('dropout'):.1f}, unfrozen_layers = {best.get('unfrozen_layers')} "
            f"| val_auc = {df['val_auc'].iloc[0]:.4f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "toy"
    run_toy() if mode == "toy" else run_visionguard()
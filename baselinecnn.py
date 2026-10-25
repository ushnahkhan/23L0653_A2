from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
LEARNING_RATE = 1e-3
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)


def conv_block(x, filters, name):
    """Conv -> BatchNorm -> ReLU -> MaxPool"""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    return x


def build_cnn():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)      # 0-255 -> 0-1

    x = conv_block(x, 32, "block1")     # 224 -> 112
    x = conv_block(x, 64, "block2")     # 112 -> 56
    x = conv_block(x, 128, "block3")    #  56 -> 28
    x = conv_block(x, 256, "block4")    #  28 -> 14

    x = layers.GlobalAveragePooling2D(name="gap")(x)              # 14x14x256 -> 256
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)   # P(image is AI)

    model = models.Model(inputs, outputs, name="baseline_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def layer_table(model):
    rows = []
    for layer in model.layers:
        shape = tuple(layer.output.shape)
        rows.append({
            "Layer": layer.name,
            "Type": layer.__class__.__name__,
            "Output shape": str(shape).replace("None", "batch"),
            "Parameters": layer.count_params(),
        })
    df = pd.DataFrame(rows)
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    non_trainable = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    return df, trainable, non_trainable


if __name__ == "__main__":
    model = build_cnn()
    model.summary(line_length=100)

    df, trainable, non_trainable = layer_table(model)
    print("\n" + df.to_string(index=False))
    print(f"\nTotal params        : {trainable + non_trainable:,}")
    print(f"Trainable params    : {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")

    df.to_csv(OUT_DIR / "cnn_layer_table.csv", index=False)
    with open(OUT_DIR / "cnn_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s, **k: f.write(s + "\n"), line_length=100)
    print(f"\nSaved {OUT_DIR / 'cnn_layer_table.csv'} and {OUT_DIR / 'cnn_summary.txt'}")
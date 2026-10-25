from pathlib import Path

import pandas as pd
import tensorflow as tf

DATA_DIR = Path("data")
CLEAN_INDEX = Path("output/clean_index.csv")
SPLIT_DIR = Path("output/splits")
FIG_DIR = Path("output/figures")

IMG_SIZE = 224          # standard input size for ImageNet-pretrained models
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
LABELS = {"RealArt": 0, "AiArtData": 1}     # 1 = AI-generated (positive class)
CLASS_NAMES = {0: "Real", 1: "AI"}

def tf_can_decode(path):
    """Pillow read every file in the audit, but TensorFlow supports fewer formats."""
    try:
        tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return True
    except Exception:
        return False


def create_splits(force=False):
    if (SPLIT_DIR / "train.csv").exists() and not force:
        print("Split lists already exist - skipping (use force=True to rebuild).")
        return

    from sklearn.model_selection import train_test_split
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN_INDEX)
    df["label"] = df["class"].map(LABELS)
    df["filepath"] = df["path"].apply(lambda p: (DATA_DIR / p).as_posix())

    df["tf_ok"] = df["filepath"].apply(tf_can_decode)
    bad = df[~df["tf_ok"]]
    print(f"Files TensorFlow cannot decode: {len(bad)}")
    if len(bad):
        print(bad[["path", "format"]].to_string(index=False))
    df = df[df["tf_ok"]]

    # stratify = same AI/Real ratio in every split
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[["filepath", "label", "class"]].to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(part):4d} images | AI = {int(part['label'].sum()):3d} | "
            f"Real = {int((part['label'] == 0).sum()):3d}")


augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomRotation(0.05, fill_mode="constant", seed=SEED),           # +/- 18 deg
    tf.keras.layers.RandomZoom(0.10, fill_mode="constant", seed=SEED),               # +/- 10 %
    tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=SEED),  # +/- 5 %
    tf.keras.layers.RandomContrast(0.10, seed=SEED),
], name="augmentation")


def load_image(path, label):
    raw = tf.io.read_file(path)                                           # raw bytes
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)    # -> RGB pixels
    img.set_shape([None, None, 3])
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)               # keep aspect ratio
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)      # uint8 = cheap cache
    return img, label


def to_float(images, labels):
    return tf.cast(images, tf.float32), labels      # pixels stay 0-255; models rescale themselves


def make_dataset(split, training=False, batch_size=BATCH_SIZE):
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    ds = tf.data.Dataset.from_tensor_slices(
        (df["filepath"].values, df["label"].values.astype("float32")))

    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(to_float, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

def save_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # one augmented training batch
    images, labels = next(iter(make_dataset("train", training=True)))
    print("Batch shape:", images.shape, "| labels:", labels.shape,
        f"| pixel range: {float(tf.reduce_min(images)):.0f} - {float(tf.reduce_max(images)):.0f}")

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, img, lab in zip(axes.flat, images[:16], labels[:16]):
        ax.imshow(tf.clip_by_value(img / 255.0, 0, 1))
        ax.set_title(CLASS_NAMES[int(lab)], fontsize=10)
        ax.axis("off")
    fig.suptitle(f"One augmented training batch ({IMG_SIZE}x{IMG_SIZE}, padded)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "augmented_batch.png", dpi=150)
    plt.close(fig)


    clean_img, _ = next(iter(make_dataset("train", training=False).unbatch()))
    fig, axes = plt.subplots(1, 8, figsize=(18, 2.8))
    axes[0].imshow(tf.cast(clean_img, tf.uint8))
    axes[0].set_title("Original")
    axes[0].axis("off")
    for ax in axes[1:]:
        aug = augment(tf.expand_dims(clean_img, 0), training=True)[0]
        ax.imshow(tf.clip_by_value(aug / 255.0, 0, 1))
        ax.set_title("Augmented")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "augmentation_examples.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'augmented_batch.png'} and {FIG_DIR / 'augmentation_examples.png'}")

if __name__ == "__main__":
    create_splits()
    for s in ["train", "val", "test"]:
        n = len(pd.read_csv(SPLIT_DIR / f"{s}.csv"))
        print(f"{s:5s} dataset ready: {n} images")
    save_figures()
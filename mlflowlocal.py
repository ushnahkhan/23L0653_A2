import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import tensorflow as tf

SEED = 42
EXPERIMENT = "23L0653_A2_Ops_mlflow_lab"
LEARNING_RATES = [1e-3, 1e-2]          # the two runs we will compare
EPOCHS, BATCH_SIZE, UNITS = 5, 64, 128

mlflow.set_tracking_uri("sqlite:///mlflow.db")   # local database file in the project folder
mlflow.set_experiment(EXPERIMENT)

(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
x_tr, x_te = x_tr / 255.0, x_te / 255.0


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Send every metric to MLflow at the end of each epoch (step = epoch number)."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch + 1)


def build_model(lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(UNITS, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def log_keras_model(model):
    try:                                    # MLflow 3.x
        mlflow.tensorflow.log_model(model, name="model")
    except TypeError:                       # MLflow 2.x
        mlflow.tensorflow.log_model(model, artifact_path="model")


for lr in LEARNING_RATES:
    tf.keras.utils.set_random_seed(SEED)    # same start for both runs -> fair comparison
    with mlflow.start_run(run_name=f"lr_{lr}"):
        # ---- tags: labels describing the run ----
        mlflow.set_tags({"student": "23L0653", "task": "3.1",
                        "dataset": "fashion_mnist", "model_type": "tiny_dense"})
        # ---- parameters: settings chosen before training ----
        mlflow.log_params({"learning_rate": lr, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                        "units": UNITS, "optimizer": "adam", "seed": SEED})

        model = build_model(lr)
        hist = model.fit(x_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.2, callbacks=[MLflowEpochLogger()], verbose=2)

        # ---- final metric on the test set ----
        test_loss, test_acc = model.evaluate(x_te, y_te, verbose=0)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

        # ---- artifact 1: a plot ----
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, m in zip(axes, ["loss", "accuracy"]):
            ax.plot(range(1, EPOCHS + 1), hist.history[m], "o-", label="train")
            ax.plot(range(1, EPOCHS + 1), hist.history[f"val_{m}"], "o-", label="validation")
            ax.set_title(f"{m} (lr = {lr})")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        mlflow.log_figure(fig, "training_curves.png")
        plt.close(fig)

        # ---- artifact 2: the trained model ----
        log_keras_model(model)
        print(f"lr = {lr}: test accuracy = {test_acc:.4f}")

print("\nDone. Start the UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
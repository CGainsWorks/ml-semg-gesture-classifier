"""TensorFlow implementation of the shared architecture.

Two things are kept deliberately identical to the PyTorch side, because the
comparison is worthless otherwise:

* **Input layout.** Everything upstream produces (batch, electrodes, time).
  Keras' Conv2D wants channels last, so the reshape to a single-channel image
  happens inside the graph rather than in the caller. The exported TFLite
  model therefore takes the same input as the exported ONNX one.
* **Augmentation.** The perturbations come from
  :mod:`src.models.augment`, the same NumPy code the PyTorch loop calls,
  delivered through a ``PyDataset``. Using Keras' own preprocessing layers
  would have been tidier and would have meant the two models were trained on
  differently perturbed data.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

# Keras is noisy on import and none of it is actionable here.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

from src.config import TrainConfig  # noqa: E402
from src.models.architecture import ArchitectureSpec  # noqa: E402
from src.models.augment import Augmenter, from_config  # noqa: E402

# Keras and PyTorch define batch-norm momentum as complements of each other.
# PyTorch's default of 0.1 is the weight on the incoming batch; Keras' is the
# weight on the running estimate. 0.9 here is PyTorch's 0.1.
BATCH_NORM_MOMENTUM = 0.9


def build(spec: ArchitectureSpec) -> tf.keras.Model:
    """The Keras model, taking (batch, n_channels, window_samples)."""
    inputs = tf.keras.Input(
        shape=(spec.n_channels, spec.window_samples), name="emg_window"
    )
    # (batch, electrodes, time) -> (batch, electrodes, time, 1).
    x = tf.keras.layers.Reshape((spec.n_channels, spec.window_samples, 1))(inputs)
    kernel = (spec.kernel_electrodes, spec.kernel_time)
    for out_ch in spec.conv_channels:
        x = tf.keras.layers.Conv2D(out_ch, kernel, padding="same")(x)
        # momentum=0.9, not the Keras default of 0.99. PyTorch's BatchNorm2d
        # defaults to momentum=0.1, which means the opposite thing: it is the
        # weight given to the new batch, so it is equivalent to Keras' 0.9. Left
        # at the Keras default the running statistics update ten times more
        # slowly, which on the smaller within-subject folds means the model is
        # evaluated with statistics that never settled. The two frameworks are
        # only the same model if this matches.
        x = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(spec.dropout)(x)
    x = tf.keras.layers.Dense(spec.dense_units)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(spec.dropout)(x)
    outputs = tf.keras.layers.Dense(spec.n_classes, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="semg_gesture_cnn")


class _AugmentedBatches(tf.keras.utils.PyDataset):
    """Shuffled, augmented training batches from the shared NumPy augmenter."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        augmenter: Augmenter,
        seed: int,
    ):
        super().__init__()
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        self.batch_size = batch_size
        self.augmenter = augmenter
        self._rng = np.random.default_rng(seed)
        self._order = self._rng.permutation(self.x.shape[0])

    def __len__(self) -> int:
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, index: int):
        idx = self._order[index * self.batch_size : (index + 1) * self.batch_size]
        return self.augmenter(self.x[idx]), self.y[idx]

    def on_epoch_end(self) -> None:
        self._order = self._rng.permutation(self.x.shape[0])


def train_model(
    spec: ArchitectureSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    model: tf.keras.Model | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    augmenter: Augmenter | None = None,
    verbose: bool = True,
) -> tuple[tf.keras.Model, dict[str, object]]:
    """Train with the same recipe the PyTorch side uses.

    Same optimiser, learning rate, weight decay, batch size, epoch budget,
    early-stopping rule and augmentation. Any accuracy difference between the
    two is then down to the frameworks and their initialisers, which is the
    comparison worth making.
    """
    tf.keras.utils.set_random_seed(config.seed)
    epochs = config.epochs if epochs is None else epochs
    learning_rate = config.learning_rate if learning_rate is None else learning_rate
    augmenter = from_config(config) if augmenter is None else augmenter
    augmenter.reset()

    if model is None:
        model = build(spec)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=config.weight_decay
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=config.early_stopping_patience,
        restore_best_weights=True,
        mode="max",
    )
    validation = (
        np.ascontiguousarray(x_val, dtype=np.float32),
        np.ascontiguousarray(y_val, dtype=np.int64),
    )

    started = time.perf_counter()
    if augmenter.enabled:
        history = model.fit(
            _AugmentedBatches(x_train, y_train, config.batch_size, augmenter, config.seed),
            validation_data=validation,
            epochs=epochs,
            callbacks=[stopper],
            verbose=2 if verbose else 0,
        )
    else:
        history = model.fit(
            np.ascontiguousarray(x_train, dtype=np.float32),
            np.ascontiguousarray(y_train, dtype=np.int64),
            validation_data=validation,
            epochs=epochs,
            batch_size=config.batch_size,
            callbacks=[stopper],
            verbose=2 if verbose else 0,
            shuffle=True,
        )
    elapsed = time.perf_counter() - started

    val_accuracy = [float(v) for v in history.history.get("val_accuracy", [])]
    best_index = int(np.argmax(val_accuracy)) if val_accuracy else -1
    gpus = tf.config.list_physical_devices("GPU")
    return model, {
        "framework": "tensorflow",
        "device": "gpu" if gpus else "cpu",
        "epochs_run": len(val_accuracy),
        "best_epoch": best_index + 1,
        "best_val_accuracy": val_accuracy[best_index] if val_accuracy else float("nan"),
        "train_seconds": elapsed,
        "augmented": augmenter.enabled,
        "history": [
            {
                "epoch": i + 1,
                "train_loss": float(history.history["loss"][i]),
                "val_loss": float(history.history["val_loss"][i]),
                "val_accuracy": val_accuracy[i],
            }
            for i in range(len(val_accuracy))
        ],
    }


def predict_proba(
    model: tf.keras.Model, x: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    if x.shape[0] == 0:
        return np.empty((0, int(model.output_shape[-1])), dtype=np.float32)
    logits = model.predict(
        np.ascontiguousarray(x, dtype=np.float32), batch_size=batch_size, verbose=0
    )
    return tf.nn.softmax(logits).numpy().astype(np.float32)


def get_weights(model: tf.keras.Model) -> list:
    """A copy of the weights, for caching a trained model."""
    return [np.array(w, copy=True) for w in model.get_weights()]


def clone_with_weights(spec: ArchitectureSpec, weights: list) -> tf.keras.Model:
    """A fresh model carrying ``weights``.

    Used to hand the same trained base model to two protocols without letting
    the second one's fine-tuning mutate what the first one measured.
    """
    model = build(spec)
    model.set_weights(weights)
    return model


def save(model: tf.keras.Model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load(path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, compile=False)


def export_saved_model(
    model: tf.keras.Model, path: str | Path, batch_size: int = 1
) -> Path:
    """Write a SavedModel with a fixed batch dimension.

    Fixed, not dynamic, and this matters more than it looks. ``model.export()``
    leaves the batch dimension as ``None``. The reshape at the head of the
    network then cannot be resolved at conversion time, so the TFLite converter
    emits SHAPE, STRIDED_SLICE and PACK operators to compute the target shape
    at run time. Those are eight extra operators a TFLite Micro build has to
    register, they defeat static memory planning, and the failure they cause on
    the device is obscure.

    Pinning the signature to one window, which is all a control loop ever has,
    reduces the graph to convolutions, pooling and fully-connected layers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    spec = tf.TensorSpec(
        shape=(batch_size, *model.input_shape[1:]), dtype=tf.float32, name="emg_window"
    )
    archive = tf.keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint(
        name="serve",
        fn=lambda x: model(x, training=False),
        input_signature=[spec],
    )
    archive.write_out(str(path))
    return path

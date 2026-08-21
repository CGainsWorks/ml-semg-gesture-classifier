"""One interface over the four deployment artefacts.

TFLite float32, TFLite int8, ONNX float32 and ONNX int8 all classify a single
200 ms window and all get benchmarked the same way. Wrapping them in one
interface is what makes the latency table an apples-to-apples comparison
rather than four differently-shaped measurements.

Everything here is single-window, batch size one. Batching is the obvious way
to make throughput numbers look better and it is meaningless for this problem:
a control loop has exactly one window available at a time, and waiting to
accumulate a batch would add the very latency the model is being optimised to
avoid.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class InferenceRuntime(ABC):
    """A loaded model that classifies one window."""

    name: str
    path: Path

    @abstractmethod
    def infer(self, window: np.ndarray) -> np.ndarray:
        """Logits or scores for a single (n_channels, n_samples) window."""

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Class indices for (n_windows, n_channels, n_samples), one at a time."""
        if windows.ndim != 3:
            raise ValueError(f"expected (n, n_channels, n_samples), got {windows.shape}")
        return np.array(
            [int(np.argmax(self.infer(w))) for w in windows], dtype=np.int64
        )

    def scores(self, windows: np.ndarray) -> np.ndarray:
        if windows.ndim != 3:
            raise ValueError(f"expected (n, n_channels, n_samples), got {windows.shape}")
        return np.stack([self.infer(w) for w in windows]).astype(np.float32)

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def close(self) -> None:  # noqa: B027 - optional hook; not every backend holds resources
        """Release whatever the backend holds. Safe to call twice."""


class TFLiteRuntime(InferenceRuntime):
    """TFLite interpreter, the desktop stand-in for TFLite Micro.

    The desktop interpreter runs the same graph and the same quantisation as
    TFLite Micro would, but not the same kernels: Micro on a Cortex-M uses
    CMSIS-NN. Timings from here therefore say what the arithmetic costs on
    this machine, not what it costs on the target. See
    docs/LATENCY_BUDGET.md for how the two relate.
    """

    def __init__(self, path: str | Path, num_threads: int = 1):
        import tensorflow as tf

        self.path = Path(path)
        self.name = f"tflite:{self.path.stem}"
        self._interpreter = tf.lite.Interpreter(
            model_path=str(self.path), num_threads=num_threads
        )
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        self._in_scale, self._in_zero = self._input["quantization"]
        self._out_scale, self._out_zero = self._output["quantization"]
        self._quantised = np.dtype(self._input["dtype"]) == np.int8

    def infer(self, window: np.ndarray) -> np.ndarray:
        x = np.asarray(window, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        if self._quantised:
            # The int8 model expects its input already quantised. Doing this in
            # Python costs a few microseconds that a real deployment would
            # spend in the ADC path instead, so it is timed as part of
            # inference rather than excluded.
            x = np.clip(
                np.round(x / self._in_scale) + self._in_zero, -128, 127
            ).astype(np.int8)
        self._interpreter.set_tensor(self._input["index"], x)
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output["index"])[0]
        if np.dtype(self._output["dtype"]) == np.int8:
            return (raw.astype(np.float32) - self._out_zero) * self._out_scale
        return raw.astype(np.float32)


class ONNXRuntime(InferenceRuntime):
    """ONNX Runtime, the deployment path for a Linux-class board."""

    def __init__(self, path: str | Path, num_threads: int = 1):
        import onnxruntime as ort

        self.path = Path(path)
        self.name = f"onnx:{self.path.stem}"
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, window: np.ndarray) -> np.ndarray:
        x = np.asarray(window, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        return self._session.run(None, {self._input_name: x})[0][0].astype(np.float32)

    def close(self) -> None:
        self._session = None


def load_runtime(path: str | Path, num_threads: int = 1) -> InferenceRuntime:
    """Pick the backend from the file extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no deployment artefact at {path}")
    if path.suffix == ".tflite":
        return TFLiteRuntime(path, num_threads=num_threads)
    if path.suffix == ".onnx":
        return ONNXRuntime(path, num_threads=num_threads)
    raise ValueError(
        f"cannot deploy {path.name}: expected a .tflite or .onnx artefact"
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)

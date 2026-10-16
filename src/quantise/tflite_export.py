"""Export the TensorFlow model to TFLite, float and int8.

Two artefacts, because they answer different questions:

``*_float32.tflite``
    The same arithmetic as training. Establishes that the conversion itself
    did not change the answers.

``*_int8.tflite``
    Full-integer quantisation: weights, activations and the input and output
    tensors are all int8. This is the one that runs on a microcontroller,
    because TFLite Micro on a Cortex-M has no floating-point kernels worth
    using and CMSIS-NN only accelerates the integer path.

Full-integer rather than dynamic-range or float16. Dynamic-range quantisation
leaves activations in float and still needs an FPU; float16 halves the size
and buys nothing on a device with no float16 hardware. The cost of going all
the way to int8 is an accuracy drop, and measuring that drop rather than
assuming it is negligible is the point of ``--evaluate``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

from src.config import portable_path  # noqa: E402


def _representative_dataset_fn(windows: np.ndarray):
    """Yield one window at a time, as the converter expects.

    The converter runs these through the graph to observe the real range of
    every activation tensor. Feed it a class-balanced sample: calibrating on
    rest alone would set the activation ranges from the quietest signal in the
    dataset and clip every actual gesture.
    """
    if windows.ndim != 3:
        raise ValueError(
            f"representative windows must be (n, n_channels, n_samples), got {windows.shape}"
        )
    if windows.shape[0] == 0:
        raise ValueError("representative dataset is empty")

    def generator():
        for window in windows.astype(np.float32):
            yield [window[None, ...]]

    return generator


def convert_float32(saved_model_dir: str | Path) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = []
    return converter.convert()


def convert_int8(saved_model_dir: str | Path, representative: np.ndarray) -> bytes:
    """Full-integer quantisation with int8 input and output tensors."""
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_fn(representative)
    # Refusing to fall back to float ops is deliberate: a silent float fallback
    # produces a model that runs here and fails to build for TFLite Micro.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def export(
    saved_model_dir: str | Path,
    out_dir: str | Path,
    representative: np.ndarray,
    stem: str = "gesture_cnn",
) -> dict[str, Path]:
    """Write both variants and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    float_path = out_dir / f"{stem}_float32.tflite"
    float_path.write_bytes(convert_float32(saved_model_dir))
    paths["float32"] = float_path

    int8_path = out_dir / f"{stem}_int8.tflite"
    int8_path.write_bytes(convert_int8(saved_model_dir, representative))
    paths["int8"] = int8_path
    return paths


def operators(path: str | Path) -> dict[str, int]:
    """Which builtin operators the model uses, and how many of each.

    A TFLite Micro build has to register every operator the model contains,
    one by one, and the error you get for a missing one on the device is
    unhelpful. Recording the list here means the port can be set up from the
    export report instead of by trial and error.
    """
    from tensorflow.lite.python import schema_py_generated as schema

    model = schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel(Path(path).read_bytes(), 0)
    )
    names = {
        value: name
        for name, value in vars(schema.BuiltinOperator).items()
        if not name.startswith("_") and isinstance(value, int)
    }
    counts: dict[str, int] = {}
    for subgraph in model.subgraphs:
        for op in subgraph.operators:
            code = model.operatorCodes[op.opcodeIndex]
            builtin = max(code.builtinCode, getattr(code, "deprecatedBuiltinCode", 0))
            name = names.get(builtin, f"UNKNOWN_{builtin}")
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def describe(path: str | Path) -> dict[str, object]:
    """Tensor types, quantisation parameters and operator set of a model."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return {
        "path": portable_path(path),
        "bytes": Path(path).stat().st_size,
        "operators": operators(path),
        "input": {
            "shape": [int(v) for v in inp["shape"]],
            "dtype": np.dtype(inp["dtype"]).name,
            "quantization": [float(inp["quantization"][0]), int(inp["quantization"][1])],
        },
        "output": {
            "shape": [int(v) for v in out["shape"]],
            "dtype": np.dtype(out["dtype"]).name,
            "quantization": [float(out["quantization"][0]), int(out["quantization"][1])],
        },
        "all_tensors_bytes": _all_tensor_bytes(interpreter),
    }


def _all_tensor_bytes(interpreter: tf.lite.Interpreter) -> int:
    """Sum of the sizes of every tensor in the graph, weights included.

    Not a RAM figure and it was once mistaken for one. It counts the weights,
    which TFLite Micro reads from flash and it counts every activation as if
    all of them were live at once, when the Micro allocator reuses buffers
    whose lifetimes do not overlap. The real RAM requirement comes from
    building for the target and reading what the linker and the allocator
    report: see ``firmware/`` and ``benchmarks/renode_cortex_m4.json``. This
    is kept as a quick order-of-magnitude check on the exported graph.
    """
    total = 0
    for detail in interpreter.get_tensor_details():
        shape = detail.get("shape")
        if shape is None or len(shape) == 0:
            continue
        total += int(np.prod(shape)) * np.dtype(detail["dtype"]).itemsize
    return int(total)

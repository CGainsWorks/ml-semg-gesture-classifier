"""Export the PyTorch model to ONNX, float and int8.

ONNX Runtime is the deployment target for a Linux-class board such as a Jetson
Nano, where TFLite Micro would be the wrong tool. The two export paths mirror
each other so the TensorFlow and PyTorch halves of the project stay comparable
all the way to the deployed artefact.

Static quantisation, not dynamic. Dynamic quantisation picks activation
ranges at run time, which costs time on every inference and defeats the point
on a latency-bound control loop. Static calibration observes the ranges once,
offline, from the same representative sample the TFLite converter is given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from src.config import portable_path
from src.models.torch_cnn import TorchGestureCNN

OPSET = 17
INPUT_NAME = "emg_window"
OUTPUT_NAME = "logits"


class _WindowCalibrationReader(CalibrationDataReader):
    """Feeds the calibration windows to the quantiser once, then stops."""

    def __init__(self, windows: np.ndarray, input_name: str = INPUT_NAME):
        if windows.ndim != 3:
            raise ValueError(
                f"calibration windows must be (n, n_channels, n_samples), got {windows.shape}"
            )
        if windows.shape[0] == 0:
            raise ValueError("calibration dataset is empty")
        self._input_name = input_name
        self._iter = iter(windows.astype(np.float32))

    def get_next(self) -> dict[str, np.ndarray] | None:
        window = next(self._iter, None)
        if window is None:
            return None
        return {self._input_name: window[None, ...]}


def export_float32(model: TorchGestureCNN, path: str | Path) -> Path:
    """Write a batch-size-1 ONNX graph.

    The batch dimension is fixed rather than dynamic. A control loop
    classifies one window at a time, and a static shape lets the runtime plan
    its allocations once instead of on every call.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.zeros(
        1, model.spec.n_channels, model.spec.window_samples, dtype=torch.float32
    )
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def export_int8(
    float_path: str | Path, int8_path: str | Path, representative: np.ndarray
) -> Path:
    """Statically quantise an exported float graph to int8."""
    float_path, int8_path = Path(float_path), Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    # Shape inference and constant folding first. Without it the quantiser
    # leaves the convolutions in float and reports nothing.
    prepared = int8_path.with_suffix(".prepared.onnx")
    quant_pre_process(str(float_path), str(prepared), skip_symbolic_shape=False)

    quantize_static(
        model_input=str(prepared),
        model_output=str(int8_path),
        calibration_data_reader=_WindowCalibrationReader(representative),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    prepared.unlink(missing_ok=True)
    return int8_path


def export(
    model: TorchGestureCNN,
    out_dir: str | Path,
    representative: np.ndarray,
    stem: str = "gesture_cnn",
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    float_path = export_float32(model, out_dir / f"{stem}_float32.onnx")
    int8_path = export_int8(float_path, out_dir / f"{stem}_int8.onnx", representative)
    return {"float32": float_path, "int8": int8_path}


def describe(path: str | Path) -> dict[str, object]:
    model = onnx.load(str(path))
    ops: dict[str, int] = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    return {
        "path": portable_path(path),
        "bytes": Path(path).stat().st_size,
        "opset": max(o.version for o in model.opset_import),
        "operators": dict(sorted(ops.items())),
    }

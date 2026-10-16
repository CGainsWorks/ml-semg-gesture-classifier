"""Fixed test windows and reference outputs for the Cortex-M firmware.

The firmware under ``firmware/`` runs the int8 model inside TensorFlow Lite
Micro on an emulated Cortex-M4F and prints its raw int8 output for a fixed set
of int8 input windows. This module produces those windows and the answer the
desktop TFLite interpreter gives for each one, so the two can be compared
byte for byte:

    python -m src.embedded.firmware_vectors --config config/firmware.yaml

writes

* ``firmware/vectors/windows_int8.bin``: the windows, already quantised with
  the model's own input scale and zero point, back to back in the model's
  (channels, time) layout. The firmware embeds this file unchanged.
* ``firmware/vectors/expected.json``: for each window, the int8 output of the
  desktop interpreter with TFLite's reference kernels, plus digests of the
  model and the windows so a stale pair is detected rather than compared.

The windows are quantised here, once and committed. Recomputing them from the
float signal on every run would make the comparison depend on float rounding
in the preprocessing, which differs across platforms in the last place.

``reference_outputs`` is also what the test suite uses to prove the committed
expectations are still what the interpreter computes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.config import Config, load_config, portable_path, resolve

MODEL_PATH = "firmware/model/gesture_cnn_int8.tflite"
WINDOWS_PATH = "firmware/vectors/windows_int8.bin"
EXPECTED_PATH = "firmware/vectors/expected.json"
WINDOWS_PER_CLASS = 2


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _interpreter(model_path: str | Path, reference: bool):
    import tensorflow as tf

    resolver = (
        tf.lite.experimental.OpResolverType.BUILTIN_REF
        if reference
        else tf.lite.experimental.OpResolverType.AUTO
    )
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path), experimental_op_resolver_type=resolver, num_threads=1
    )
    interpreter.allocate_tensors()
    return interpreter


def quantisation(model_path: str | Path) -> dict[str, object]:
    """Input and output scale, zero point and shape of an int8 model."""
    interpreter = _interpreter(model_path, reference=True)
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if np.dtype(inp["dtype"]) != np.int8 or np.dtype(out["dtype"]) != np.int8:
        raise ValueError(f"{model_path} is not int8 in and out")
    return {
        "input_shape": [int(v) for v in inp["shape"]],
        "input_scale": float(inp["quantization"][0]),
        "input_zero_point": int(inp["quantization"][1]),
        "output_shape": [int(v) for v in out["shape"]],
        "output_scale": float(out["quantization"][0]),
        "output_zero_point": int(out["quantization"][1]),
    }


def reference_outputs(
    model_path: str | Path, windows: np.ndarray, reference: bool = True
) -> np.ndarray:
    """Raw int8 outputs of the desktop interpreter, one row per window.

    ``reference=True`` selects TFLite's reference kernels, the ones TFLite
    Micro's own kernels are written to match. ``False`` uses the default
    optimised kernels, which are allowed to round differently.
    """
    interpreter = _interpreter(model_path, reference=reference)
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    rows = []
    for window in windows:
        interpreter.set_tensor(inp["index"], window.reshape(inp["shape"]).astype(np.int8))
        interpreter.invoke()
        rows.append(interpreter.get_tensor(out["index"]).reshape(-1).copy())
    return np.stack(rows).astype(np.int8)


def load_windows(path: str | Path, input_shape: list[int]) -> np.ndarray:
    per_window = int(np.prod(input_shape[1:]))
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.int8)
    if raw.size % per_window:
        raise ValueError(f"{path} is not a whole number of {per_window}-byte windows")
    return raw.reshape(-1, *input_shape[1:])


def select_windows(config: Config, quant: dict[str, object]) -> tuple[np.ndarray, list[int]]:
    """A class-balanced, seeded selection of windows, quantised to int8."""
    from src.data.dataset import build_dataset, class_codes
    from src.preprocessing.normalise import ChannelScaler
    from src.train import deploy_dir

    scaler = ChannelScaler.load(deploy_dir(config) / "channel_scaler.npz")
    index = build_dataset(config)
    codes = class_codes(index)
    rng = np.random.default_rng([config.window.seed, 0xF1])
    picks = []
    for code in codes:
        members = np.flatnonzero(index.labels == code)
        picks.extend(rng.choice(members, size=WINDOWS_PER_CLASS, replace=False).tolist())
    picks = sorted(picks)
    subset = index.select(np.asarray(picks))
    x = scaler.transform(subset.materialise("ct")).astype(np.float32)
    q = np.round(x / quant["input_scale"]) + quant["input_zero_point"]
    return np.clip(q, -128, 127).astype(np.int8), [int(v) for v in subset.labels]


def write(config: Config, model_path: str | Path = MODEL_PATH) -> dict[str, object]:
    model_path = resolve(model_path)
    quant = quantisation(model_path)
    windows, labels = select_windows(config, quant)

    windows_path = resolve(WINDOWS_PATH)
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    windows_path.write_bytes(windows.tobytes())

    expected = reference_outputs(model_path, windows, reference=True)
    optimised = reference_outputs(model_path, windows, reference=False)
    diff = np.abs(expected.astype(np.int16) - optimised.astype(np.int16))

    import tensorflow as tf

    report = {
        "synthetic": True,
        "note": (
            "Synthetic windows from config/firmware.yaml. These outputs are what "
            "the desktop TFLite interpreter computes, not correct labels."
        ),
        "model": {
            "path": portable_path(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": sha256(model_path),
        },
        "windows": {
            "path": WINDOWS_PATH,
            "count": int(windows.shape[0]),
            "sha256": sha256(windows_path),
        },
        "quantisation": quant,
        "reference": {
            "runtime": f"TFLite interpreter, tensorflow {tf.__version__}",
            "kernels": "BUILTIN_REF (reference kernels)",
            "optimised_kernels_max_abs_diff_lsb": int(diff.max()),
        },
        "labels": labels,
        "outputs": expected.astype(int).tolist(),
    }
    out = resolve(EXPECTED_PATH)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/firmware.yaml")
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args(argv)
    report = write(load_config(resolve(args.config)), args.model)
    print(
        f"{report['windows']['count']} windows written to {WINDOWS_PATH}, "
        f"reference outputs to {EXPECTED_PATH}; optimised kernels differ from the "
        f"reference by at most {report['reference']['optimised_kernels_max_abs_diff_lsb']} LSB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Produce every deployment artefact and measure what quantisation cost.

Run after ``--protocol deploy`` training. Writes, under
``artifacts/deploy/<tag>/``:

* ``gesture_cnn_float32.tflite`` and ``gesture_cnn_int8.tflite``
* ``gesture_cnn_float32.onnx`` and ``gesture_cnn_int8.onnx``
* ``representative_windows.npy``, the calibration sample, kept so the
  quantisation is reproducible and so the latency benchmark feeds the same
  data
* ``export_report.json``

The accuracy in that report is measured through the deployed runtimes, one
window at a time, exactly as a device would run them. It is measured on the
held-out repetitions of the deployment split, so it says what quantisation
cost, not what the model scores on data it trained on. The number to compare
against the README's headline is still the leave-one-subject-out one; this
report answers a narrower question, which is whether going to int8 broke
anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import Config, load_config, resolve
from src.data.dataset import build_dataset, class_codes, encode_labels
from src.data.splits import train_validation_split
from src.embedded.runtime import load_runtime
from src.preprocessing.normalise import ChannelScaler
from src.quantise import onnx_export, tflite_export
from src.train import deploy_dir


def deployment_output_dir(config: Config) -> Path:
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    return resolve(config.artifacts_dir) / "deploy" / tag


def representative_sample(
    config: Config, scaler: ChannelScaler, limit: int = 512
) -> np.ndarray:
    """A class-balanced, scaled sample for calibrating both quantisers.

    Balanced on purpose. Calibrating on a random sample would over-weight
    rest, set every activation range from the quietest part of the signal, and
    clip real contractions once quantised.
    """
    index = build_dataset(config)
    codes = class_codes(index)
    rng = np.random.default_rng(config.window.seed)
    per_class = max(1, limit // max(1, codes.size))

    picks = []
    for code in codes:
        members = np.flatnonzero(index.labels == code)
        if members.size:
            picks.append(rng.choice(members, size=min(per_class, members.size), replace=False))
    selected = np.sort(np.concatenate(picks))
    return scaler.transform(index.select(selected).materialise("ct"))


def _holdout(config: Config, scaler: ChannelScaler) -> tuple[np.ndarray, np.ndarray]:
    """The deployment model's validation repetitions, scaled and encoded.

    The deployment model trained on everything except these, so they are the
    only windows available for an honest quantisation comparison without
    retraining.
    """
    index = build_dataset(config)
    codes = class_codes(index)
    _, val_idx = train_validation_split(index, config.train.val_fraction, config.train.seed)
    subset = index.select(val_idx)
    return scaler.transform(subset.materialise("ct")), encode_labels(subset.labels, codes)


def _accuracy_through_runtime(path: Path, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    runtime = load_runtime(path)
    try:
        predictions = runtime.predict(x)
    finally:
        runtime.close()
    return {
        "artefact": path.name,
        "bytes": path.stat().st_size,
        "accuracy": float((predictions == y).mean()),
        "n_windows": int(y.size),
    }


def run(config: Config, limit: int = 512, evaluate: bool = True) -> dict[str, object]:
    models_dir = deploy_dir(config)
    out_dir = deployment_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = models_dir / "channel_scaler.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"{scaler_path} not found. Train the deployment models first:\n"
            "  python -m src.train --protocol deploy --model tensorflow\n"
            "  python -m src.train --protocol deploy --model pytorch"
        )
    scaler = ChannelScaler.load(scaler_path)

    representative = representative_sample(config, scaler, limit=limit)
    np.save(out_dir / "representative_windows.npy", representative)
    print(f"calibration sample: {representative.shape[0]} windows")

    report: dict[str, object] = {
        "synthetic": config.dataset.source == "synthetic",
        "representative_windows": int(representative.shape[0]),
        "artefacts": {},
        "accuracy": [],
    }

    saved_model = models_dir / "tensorflow_saved_model"
    if saved_model.exists():
        print("converting TensorFlow -> TFLite")
        paths = tflite_export.export(saved_model, out_dir, representative)
        report["artefacts"]["tflite_float32"] = tflite_export.describe(paths["float32"])
        report["artefacts"]["tflite_int8"] = tflite_export.describe(paths["int8"])
    else:
        print(f"skipping TFLite: no SavedModel at {saved_model}")

    torch_path = models_dir / "pytorch_cnn.pt"
    if torch_path.exists():
        print("converting PyTorch -> ONNX")
        from src.models import torch_cnn

        model = torch_cnn.load(torch_path)
        paths = onnx_export.export(model, out_dir, representative)
        report["artefacts"]["onnx_float32"] = onnx_export.describe(paths["float32"])
        report["artefacts"]["onnx_int8"] = onnx_export.describe(paths["int8"])
        report["architecture"] = model.spec.describe()
    else:
        print(f"skipping ONNX: no checkpoint at {torch_path}")

    if not report["artefacts"]:
        raise FileNotFoundError(
            "nothing to export. Train a deployment model first: "
            "python -m src.train --protocol deploy --model all"
        )

    if evaluate:
        x, y = _holdout(config, scaler)
        print(f"scoring {len(report['artefacts'])} artefact(s) on {y.size} held-out windows")
        rows = []
        for path in sorted([*out_dir.glob("*.tflite"), *out_dir.glob("*.onnx")]):
            row = _accuracy_through_runtime(path, x, y)
            rows.append(row)
            print(f"  {row['artefact']:34s} accuracy {row['accuracy']:.4f}")
        report["accuracy"] = rows
        report["quantisation_cost"] = _quantisation_cost(rows)

    # The binaries stay under artifacts/ because they are regenerated. The
    # report goes to results/ and is committed, because the README quotes it
    # and src.check_readme has to be able to verify that on a fresh checkout.
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    report_path = resolve("results") / tag / "export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(f"report written to {report_path}")
    return report


def _quantisation_cost(rows: list[dict[str, object]]) -> dict[str, object]:
    """Accuracy and size deltas between each float artefact and its int8 twin."""
    by_name = {r["artefact"]: r for r in rows}
    out: dict[str, object] = {}
    for suffix in ("tflite", "onnx"):
        float_name = f"gesture_cnn_float32.{suffix}"
        int8_name = f"gesture_cnn_int8.{suffix}"
        if float_name in by_name and int8_name in by_name:
            f, q = by_name[float_name], by_name[int8_name]
            out[suffix] = {
                "float32_accuracy": f["accuracy"],
                "int8_accuracy": q["accuracy"],
                "accuracy_delta": q["accuracy"] - f["accuracy"],
                "float32_bytes": f["bytes"],
                "int8_bytes": q["bytes"],
                "size_ratio": q["bytes"] / f["bytes"] if f["bytes"] else float("nan"),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--limit", type=int, default=512, help="calibration windows to sample"
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="skip the accuracy comparison (conversion only)",
    )
    args = parser.parse_args(argv)
    run(load_config(args.config), limit=args.limit, evaluate=not args.no_evaluate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

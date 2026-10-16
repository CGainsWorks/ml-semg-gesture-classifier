"""Training entry point.

Two jobs, and they are different jobs:

``--protocol {within_subject, subject_independent, subject_independent_calibrated}``
    Cross-validate. Produces the numbers quoted in the README. Trains many
    models and keeps none of them.

``--protocol deploy``
    Train one model on every subject and keep it. This is the artefact that
    gets quantised, exported and benchmarked. It is *not* where the reported
    accuracy comes from: a model trained on all ten subjects has no held-out
    subject to be honest about, which is what the leave-one-subject-out run is
    for. The deployment model exists because shipping a model trained on nine
    tenths of your data would be silly, and the leave-one-subject-out result
    is the estimate of how it will behave on wearer eleven.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import Config, load_config, resolve
from src.data.dataset import build_dataset, class_codes, encode_labels, summarise
from src.data.gestures import gesture_names
from src.data.splits import train_validation_split
from src.experiment import MODELS, results_path, run_protocol
from src.models.architecture import ArchitectureSpec
from src.models.classical import ClassicalPipeline
from src.preprocessing.normalise import ChannelScaler

DEPLOY = "deploy"


def deploy_dir(config: Config) -> Path:
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    return resolve(config.artifacts_dir) / "models" / tag


def train_deployment_model(
    config: Config, model: str, verbose: bool = True
) -> dict[str, object]:
    """Fit on every subject and write the artefacts the deployment path needs."""
    index = build_dataset(config)
    codes = class_codes(index)
    names = gesture_names(config.dataset.exercise, [int(c) for c in codes])

    x = index.materialise("ct")
    y = encode_labels(index.labels, codes)
    scaler = ChannelScaler.fit(x)
    x_scaled = scaler.transform(x)

    out = deploy_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    scaler.save(out / "channel_scaler.npz")

    spec = ArchitectureSpec.from_config(
        config.train,
        n_channels=index.n_channels,
        window_samples=index.length,
        n_classes=codes.size,
    )

    if model.startswith("classical"):
        from src.experiment import _classical_name

        pipeline = ClassicalPipeline(
            classifier=_classical_name(model), features=config.features, seed=config.train.seed
        )
        pipeline.fit(x, y)
        pipeline.save(out / f"{model}.pkl")
        info: dict[str, object] = {
            "framework": "scikit-learn",
            "train_seconds": pipeline.fit_seconds,
        }
    else:
        train_idx, val_idx = train_validation_split(
            index, config.train.val_fraction, config.train.seed
        )
        if model == "pytorch":
            from src.models import torch_cnn

            net, info = torch_cnn.train_model(
                spec,
                x_scaled[train_idx],
                y[train_idx],
                x_scaled[val_idx],
                y[val_idx],
                config.train,
                verbose=verbose,
            )
            torch_cnn.save(net, out / "pytorch_cnn.pt")
        elif model == "tensorflow":
            from src.models import tf_cnn

            net, info = tf_cnn.train_model(
                spec,
                x_scaled[train_idx],
                y[train_idx],
                x_scaled[val_idx],
                y[val_idx],
                config.train,
                verbose=verbose,
            )
            tf_cnn.save(net, out / "tensorflow_cnn.keras")
            tf_cnn.export_saved_model(net, out / "tensorflow_saved_model")
        else:
            raise ValueError(f"unknown model {model!r}")

    manifest = {
        "model": model,
        "synthetic": config.dataset.source == "synthetic",
        "class_codes": [int(c) for c in codes],
        "class_names": names,
        "architecture": spec.describe(),
        "dataset": summarise(index),
        "training": info,
    }
    (out / f"{model}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    if verbose:
        print(f"deployment artefacts written to {out}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--model",
        default="pytorch",
        choices=[*MODELS, "all"],
        help="which model family to train",
    )
    parser.add_argument(
        "--protocol",
        default="subject_independent",
        choices=[
            "within_subject",
            "subject_independent",
            "subject_independent_calibrated",
            DEPLOY,
            "all",
        ],
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "suppress the per-epoch training output. One line per fold is still "
            "printed: a leave-one-subject-out rotation takes over an hour, and a "
            "stage that prints nothing for that long is indistinguishable from a "
            "hung one."
        ),
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    models = list(MODELS) if args.model == "all" else [args.model]
    protocols = (
        ["within_subject", "subject_independent", "subject_independent_calibrated"]
        if args.protocol == "all"
        else [args.protocol]
    )
    verbose = not args.quiet

    if config.dataset.source == "synthetic":
        print(
            "WARNING: dataset source is 'synthetic'. Accuracy from this run is a "
            "wiring check, not a result.",
            file=sys.stderr,
        )

    for model in models:
        # Shared between the two subject-independent protocols, which train the
        # same base model on the same windows. Scoped to one model so a stale
        # architecture cannot leak across families, and dropped between models
        # so the weights do not sit in memory longer than they are wanted.
        base_models: dict[tuple[str, str], tuple[object, dict]] = {}
        for protocol in protocols:
            if protocol == DEPLOY:
                train_deployment_model(config, model, verbose=verbose)
                continue
            run = run_protocol(
                config,
                model,
                protocol,
                # Fold progress always; per-epoch output only when asked for.
                verbose=True,
                train_verbose=verbose,
                base_models=base_models,
            )
            path = run.save(results_path(config, model, protocol))
            print(
                f"{model} / {protocol}: accuracy {run.pooled.accuracy:.4f}  "
                f"balanced {run.pooled.balanced_accuracy:.4f}  "
                f"macro-F1 {run.pooled.macro_f1:.4f}  -> {path}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

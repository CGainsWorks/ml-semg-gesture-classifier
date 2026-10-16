"""Fit the classifier to a new wearer from their own demonstrations.

This is the fitting appointment, as a command. It is also this project's
answer to "learning by demonstration" in the original plan and it is worth being
precise about what that phrase does and does not mean here.

**What this is.** The new wearer performs each gesture once. Those recordings
are the demonstrations. The deployed model, trained on everybody else, is
fine-tuned on them, and the result is measured on the wearer's *other*
repetitions, which the model never sees. That is supervised learning from
demonstrated examples, and it is exactly how a myoelectric prosthesis is
actually personalised.

**What this is not.** It is not imitation learning in the reinforcement
learning sense: no policy is cloned, there is no expert trajectory over
states and actions, and there is no environment. That would need an arm to
imitate and this repository has no arm. The original plan lists imitation
learning in its target stack; this is the part of it that applies to a classifier and the
part that does not is out of scope rather than quietly relabelled.

Usage:

    python -m src.calibrate --subject 3
    python -m src.calibrate --subject 3 --repetitions 1 2 --export

The reported before-and-after is a single subject and is noisier than the
ten-fold `subject_independent_calibrated` protocol in the README. For the
headline figure use that; for a demonstration of the workflow on one person,
use this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import Config, load_config, resolve
from src.data.dataset import build_dataset, class_codes, encode_labels
from src.data.gestures import gesture_names
from src.data.windows import WindowIndex
from src.eval.metrics import evaluate
from src.models.architecture import ArchitectureSpec
from src.preprocessing.normalise import ChannelScaler
from src.train import deploy_dir


def _paths(config: Config) -> tuple[Path, Path]:
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    return deploy_dir(config), resolve(config.artifacts_dir) / "calibrated" / tag


def split_demonstrations(
    config: Config,
    subject: int,
    repetitions: list[int],
    scaler: ChannelScaler,
    index: WindowIndex | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The wearer's demonstrations and their held-out repetitions.

    The two sets are disjoint by construction. Scoring on the repetitions the
    model was just fine-tuned on would say nothing about whether the
    calibration generalises to the next time the wearer makes that gesture.

    The scaler is a parameter rather than being loaded here, because which one
    is correct depends on which model is being calibrated. Applying the
    all-subject deployment scaler to a model trained without this wearer would
    put the wearer's own channel statistics back into the pipeline through the
    side door.
    """
    index = build_dataset(config) if index is None else index
    codes = class_codes(index)

    is_subject = index.subjects == subject
    if not is_subject.any():
        available = sorted(int(s) for s in np.unique(index.subjects))
        raise ValueError(f"subject {subject} is not in this dataset; have {available}")

    demonstrated = np.isin(index.repetitions, sorted(set(repetitions)))
    demo = index.select(is_subject & demonstrated)
    held_out = index.select(is_subject & ~demonstrated)

    if len(demo) == 0:
        have = sorted(int(r) for r in np.unique(index.repetitions[is_subject]))
        raise ValueError(
            f"subject {subject} has no windows in repetitions {repetitions}; have {have}"
        )
    if len(held_out) == 0:
        raise ValueError(
            f"calibrating on repetitions {repetitions} leaves subject {subject} "
            "with nothing to be scored on. Demonstrate fewer of them."
        )

    return (
        scaler.transform(demo.materialise("ct")),
        encode_labels(demo.labels, codes),
        scaler.transform(held_out.materialise("ct")),
        encode_labels(held_out.labels, codes),
        codes,
    )


def _train_without_subject(
    config: Config,
    subject: int,
    codes: np.ndarray,
    index: WindowIndex,
    verbose: bool,
):
    """Train a fresh model that has never seen this wearer.

    Slower than loading the shipped checkpoint, and the only way to get a
    "before" number that means anything: the shipped model was trained on
    every subject, this one included.
    """
    from src.data.splits import subject_independent_folds, train_validation_split
    from src.models import torch_cnn

    fold = next(
        f
        for f in subject_independent_folds(index, config)
        if f.held_out_subject == subject
    )
    if verbose:
        print(
            f"  training a model on the other {len(set(fold.train.subjects.tolist()))} "
            f"subjects ({len(fold.train)} windows); this takes a few minutes"
        )

    raw = fold.train.materialise("ct")
    scaler = ChannelScaler.fit(raw)
    x = scaler.transform(raw)
    y = encode_labels(fold.train.labels, codes)
    train_idx, val_idx = train_validation_split(
        fold.train, config.train.val_fraction, config.train.seed
    )
    spec = ArchitectureSpec.from_config(
        config.train, index.n_channels, index.length, codes.size
    )
    model, _ = torch_cnn.train_model(
        spec,
        x[train_idx],
        y[train_idx],
        x[val_idx],
        y[val_idx],
        config.train,
        verbose=False,
    )
    return model, scaler


def run(
    config: Config,
    subject: int,
    repetitions: list[int] | None = None,
    export: bool = False,
    holdout: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    from src.models import torch_cnn

    repetitions = repetitions or config.evaluation.calibration_reps
    models_dir, out_dir = _paths(config)
    checkpoint = models_dir / "pytorch_cnn.pt"
    if not holdout and not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found. Train the deployment model first:\n"
            "  python -m src.train --protocol deploy --model pytorch\n"
            "or pass --holdout to train one here that excludes this wearer."
        )

    # Built once and threaded through: it reads and decompresses a cache of
    # every subject, and the three call sites below would otherwise each pay
    # for that separately.
    index = build_dataset(config)
    codes = class_codes(index)
    if holdout:
        # A model that has genuinely never seen this wearer, with the scaler
        # that was fit alongside it. This is the before-and-after that means
        # something.
        model, scaler = _train_without_subject(config, subject, codes, index, verbose)
    else:
        # The shipped model, which was trained on every subject including this
        # one, with the scaler it was trained under. Faster, and its "before"
        # number is optimistic by construction. The report says so, and
        # --holdout is the way to avoid it.
        model = torch_cnn.load(checkpoint)
        scaler = ChannelScaler.load(models_dir / "channel_scaler.npz")

    x_demo, y_demo, x_test, y_test, codes = split_demonstrations(
        config, subject, repetitions, scaler, index=index
    )
    names = gesture_names(config.dataset.exercise, [int(c) for c in codes])
    spec: ArchitectureSpec = model.spec

    before = evaluate(
        y_test,
        torch_cnn.predict_proba(model, x_test, device=config.train.device).argmax(axis=1),
        codes,
        names,
    )
    if verbose:
        print(
            f"subject {subject}: {len(y_demo)} demonstration windows from repetitions "
            f"{repetitions}, {len(y_test)} held-out windows"
        )
        print(f"  before calibration: accuracy {before.accuracy:.4f}")

    model, info = torch_cnn.train_model(
        spec,
        x_demo,
        y_demo,
        x_demo,
        y_demo,
        config.train,
        model=model,
        epochs=config.evaluation.calibration_epochs,
        learning_rate=config.evaluation.calibration_learning_rate,
        verbose=False,
    )
    after = evaluate(
        y_test,
        torch_cnn.predict_proba(model, x_test, device=config.train.device).argmax(axis=1),
        codes,
        names,
    )
    if verbose:
        print(f"  after calibration:  accuracy {after.accuracy:.4f}")
        print(f"  change:             {after.accuracy - before.accuracy:+.4f}")

    report = {
        "synthetic": config.dataset.source == "synthetic",
        "subject": subject,
        "demonstration_repetitions": repetitions,
        "n_demonstration_windows": int(len(y_demo)),
        "n_held_out_windows": int(len(y_test)),
        "before": before.summary(),
        "after": after.summary(),
        "accuracy_change": after.accuracy - before.accuracy,
        "calibration_epochs": info["epochs_run"],
        "holdout": holdout,
        "caveat": (
            "The 'before' model has never seen this wearer, so the change is a "
            "genuine measurement of what the demonstrations bought."
            if holdout
            else "The 'before' model was trained on every subject including this "
            "one, so it is optimistic and the change understates the benefit. "
            "Pass --holdout for a model that has never seen this wearer, or read "
            "the subject_independent and subject_independent_calibrated rows in "
            "the README for the ten-fold version."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"calibration_s{subject:02d}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )

    if export:
        from src.quantise import onnx_export

        personalised = out_dir / f"s{subject:02d}"
        torch_cnn.save(model, personalised / "pytorch_cnn.pt")
        paths = onnx_export.export(model, personalised, x_demo[:512])
        report["artefacts"] = {k: str(v) for k, v in paths.items()}
        if verbose:
            print(f"  personalised artefacts written to {personalised}")

    if verbose:
        print(f"  report written to {out_dir / f'calibration_s{subject:02d}.json'}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--subject", type=int, required=True, help="the wearer being fitted"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        nargs="+",
        default=None,
        help="which repetitions count as demonstrations (default: from the config)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="also write a personalised ONNX artefact for this wearer",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help=(
            "train the 'before' model here, excluding this wearer, instead of "
            "loading the shipped one. Slower, and the only way the before-and-after "
            "means anything."
        ),
    )
    args = parser.parse_args(argv)

    run(
        load_config(args.config),
        subject=args.subject,
        repetitions=args.repetitions,
        export=args.export,
        holdout=args.holdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

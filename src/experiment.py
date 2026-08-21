"""Run one (model, protocol) combination and write the result.

The same fold loop drives all three model families and all three protocols,
which is the only way to be sure the classical baseline and the two networks
are being asked the same question on the same windows.

Ordering inside a fold, and the reason for it:

1. Materialise the training windows.
2. Fit the channel scaler on those windows and nothing else.
3. Split off a validation set by repetition, not at random.
4. Train.
5. Transform the test windows with the scaler from step 2 and score.

Step 2 before step 5 is the whole discipline of the subject-independent
protocol. Fitting the scaler on everything would hand the model the held-out
subject's channel gains, which is exactly the information it is not supposed
to have.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import Config, resolve, source_fingerprint
from src.data.dataset import build_dataset, class_codes, encode_labels, summarise
from src.data.gestures import gesture_names
from src.data.splits import Fold, make_folds, train_validation_split
from src.data.windows import WindowIndex
from src.eval.metrics import Result, evaluate, pool
from src.models.architecture import ArchitectureSpec
from src.models.classical import ClassicalPipeline
from src.preprocessing.normalise import ChannelScaler

MODELS = ("classical_lda", "classical_svm", "classical_rf", "tensorflow", "pytorch")


@dataclass
class ProtocolRun:
    model: str
    protocol: str
    pooled: Result
    per_fold: list[dict[str, object]]
    class_codes: list[int]
    class_names: list[str]
    dataset: dict[str, object]
    config: dict[str, object]
    wall_seconds: float
    synthetic: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "protocol": self.protocol,
            "synthetic": self.synthetic,
            # Which revision of src/ produced this. Two results are only
            # comparable if these match; see src.config.source_fingerprint.
            "source_fingerprint": source_fingerprint(),
            "wall_seconds": self.wall_seconds,
            "class_codes": self.class_codes,
            "class_names": self.class_names,
            "dataset": self.dataset,
            "pooled": self.pooled.to_dict(),
            "per_fold": self.per_fold,
            "config": self.config,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def _classical_name(model: str) -> str:
    return {"classical_lda": "lda", "classical_svm": "svm", "classical_rf": "random_forest"}[
        model
    ]


def _predict_indices(proba: np.ndarray) -> np.ndarray:
    return np.asarray(proba).argmax(axis=1)


def _run_classical(
    model: str,
    config: Config,
    fold: Fold,
    codes: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    info: dict[str, object] = {"framework": "scikit-learn"}

    if fold.calibration is not None:
        # There is no fine-tuning step for a scikit-learn estimator, so
        # calibration means refitting on the training subjects plus the new
        # wearer's demonstrations. That is what a commercial LDA-based
        # controller actually does at a fitting appointment. Skipping this and
        # reporting the uncalibrated model's score under a "calibrated" heading
        # would be reporting a number for a thing that never happened.
        x_calibration = fold.calibration.materialise("ct")
        y_calibration = encode_labels(fold.calibration.labels, codes)
        x_train = np.concatenate([x_train, x_calibration])
        y_train = np.concatenate([y_train, y_calibration])
        info["calibration"] = {"n_windows": int(y_calibration.size), "refit": True}

    pipeline = ClassicalPipeline(
        classifier=_classical_name(model),
        features=config.features,
        seed=config.train.seed,
    )
    pipeline.fit(x_train, y_train)
    proba = pipeline.predict_proba(x_test)
    # scikit-learn drops classes absent from the training fold. Re-expand to
    # the full label space so confusion matrices from different folds line up.
    full = np.zeros((proba.shape[0], codes.size), dtype=np.float32)
    full[:, pipeline.classes_.astype(int)] = proba
    info["train_seconds"] = pipeline.fit_seconds
    return full, info


def _calibration_arrays(
    fold: Fold, scaler: ChannelScaler, codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The held-out subject's calibration windows, scaled and encoded."""
    if fold.calibration is None:
        raise ValueError(f"fold {fold.name!r} has no calibration set")
    x = scaler.transform(fold.calibration.materialise("ct"))
    y = encode_labels(fold.calibration.labels, codes)
    return x, y


def _run_deep(
    model: str,
    config: Config,
    fold: Fold,
    spec: ArchitectureSpec,
    scaler: ChannelScaler,
    codes: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    verbose: bool,
    base_models: dict[tuple[str, str], tuple[object, dict]] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if model == "pytorch":
        from src.models import torch_cnn as backend
    elif model == "tensorflow":
        from src.models import tf_cnn as backend
    else:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")

    # The subject-independent and subject-independent-plus-calibration
    # protocols train exactly the same base model on exactly the same windows.
    # Given a cache, the second one reuses the first one's weights rather than
    # spending another hour arriving at them. The weights are cloned on the way
    # out so the calibration fine-tune cannot mutate what the uncalibrated
    # protocol already measured.
    key = (model, fold.name)
    cached = base_models.get(key) if base_models is not None else None
    if cached is not None:
        weights, info = cached
        net = backend.clone_with_weights(spec, weights)
        info = {**info, "reused_base_model": True}
        if verbose:
            print(f"  reusing the base model already trained for {fold.name}")
    else:
        train_idx, val_idx = train_validation_split(
            fold.train, config.train.val_fraction, config.train.seed
        )
        net, info = backend.train_model(
            spec,
            x_train[train_idx],
            y_train[train_idx],
            x_train[val_idx],
            y_train[val_idx],
            config.train,
            verbose=verbose,
        )
        if base_models is not None:
            base_models[key] = (backend.get_weights(net), info)

    if fold.calibration is not None:
        # Fine-tune on the new wearer's calibration repetition. The calibration
        # set is small, so it doubles as its own validation set. That would be
        # circular if the reported number came from it, but it does not: the
        # score below comes from the wearer's remaining repetitions, which the
        # model never sees. The only exposure is choosing a stopping epoch, and
        # the calibration budget is short and fixed for that reason.
        x_cal, y_cal = _calibration_arrays(fold, scaler, codes)
        net, calib_info = backend.train_model(
            spec,
            x_cal,
            y_cal,
            x_cal,
            y_cal,
            config.train,
            model=net,
            epochs=config.evaluation.calibration_epochs,
            learning_rate=config.evaluation.calibration_learning_rate,
            verbose=verbose,
        )
        info["calibration"] = {
            "n_windows": int(x_cal.shape[0]),
            "epochs": calib_info["epochs_run"],
        }

    return backend.predict_proba(net, x_test), info


def run_protocol(
    config: Config,
    model: str,
    protocol: str,
    index: WindowIndex | None = None,
    verbose: bool = True,
    base_models: dict[tuple[str, str], tuple[object, dict]] | None = None,
    train_verbose: bool | None = None,
) -> ProtocolRun:
    """Train and score ``model`` under ``protocol`` across every fold.

    ``base_models`` is an optional cache shared between the two
    subject-independent protocols, which train an identical base model on
    identical windows. Passing the same dictionary to both halves the deep
    models' running time and changes nothing about the result; a test asserts
    that the cached and uncached runs agree.

    ``train_verbose`` defaults to ``verbose`` and can be set independently, so
    a caller can keep the one-line-per-fold progress without the per-epoch
    output. A stage that prints nothing for half an hour is indistinguishable
    from a hung one, which is a real thing that happened here.
    """
    train_verbose = verbose if train_verbose is None else train_verbose
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")

    started = time.perf_counter()
    if index is None:
        index = build_dataset(config)

    codes = class_codes(index)
    names = gesture_names(config.dataset.exercise, [int(c) for c in codes])
    folds = make_folds(index, config, protocol)

    spec = ArchitectureSpec.from_config(
        config.train,
        n_channels=index.n_channels,
        window_samples=index.length,
        n_classes=codes.size,
    )

    results: list[Result] = []
    per_fold: list[dict[str, object]] = []

    for number, fold in enumerate(folds, start=1):
        if verbose:
            # Flushed: this is the only sign of life during a long fold loop,
            # and a block-buffered pipe would hold it back for half an hour.
            print(
                f"[{model} / {protocol}] fold {number}/{len(folds)} {fold.name}: "
                f"{len(fold.train)} train, {len(fold.test)} test",
                flush=True,
            )
        raw_train = fold.train.materialise("ct")
        raw_test = fold.test.materialise("ct")
        y_train = encode_labels(fold.train.labels, codes)
        y_test = encode_labels(fold.test.labels, codes)

        if model.startswith("classical"):
            # The classical pipeline derives its own per-channel scale from the
            # training fold, so it takes unscaled windows and no scaler is fit.
            proba, info = _run_classical(
                model, config, fold, codes, raw_train, y_train, raw_test
            )
        else:
            scaler = ChannelScaler.fit(raw_train)
            proba, info = _run_deep(
                model,
                config,
                fold,
                spec,
                scaler,
                codes,
                scaler.transform(raw_train),
                y_train,
                scaler.transform(raw_test),
                train_verbose,
                base_models=base_models,
            )

        result = evaluate(y_test, _predict_indices(proba), codes, names)
        results.append(result)
        entry = {**fold.describe(), **result.summary(), "training": info}
        entry["confusion"] = result.confusion.tolist()
        per_fold.append(entry)
        if verbose:
            print(
                f"  accuracy {result.accuracy:.4f}  balanced {result.balanced_accuracy:.4f}"
                f"  macro-F1 {result.macro_f1:.4f}",
                flush=True,
            )

    pooled = pool(results, codes, names)
    return ProtocolRun(
        model=model,
        protocol=protocol,
        pooled=pooled,
        per_fold=per_fold,
        class_codes=[int(c) for c in codes],
        class_names=names,
        dataset=summarise(index),
        config=config.to_dict(),
        wall_seconds=time.perf_counter() - started,
        synthetic=config.dataset.source == "synthetic",
    )


def results_path(config: Config, model: str, protocol: str) -> Path:
    """Where a scored run is written.

    Under ``results/`` at the repository root rather than under
    ``artifacts/``, and committed, because these files are the evidence for
    every number in the README. ``src.check_readme`` reads them, CI runs that
    checker, and a fresh checkout has to contain enough to run it. Trained
    weights, caches and exported binaries stay in ``artifacts/`` where they
    are regenerated rather than tracked.
    """
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    return resolve("results") / tag / f"{model}__{protocol}.json"

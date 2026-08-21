"""Evaluation protocols.

Three conditions, in increasing order of how much they resemble deploying a
prosthesis on somebody new:

``within_subject``
    Train and test on the same person, split by repetition. The Ninapro
    convention: repetitions 1, 3, 4 and 6 train, 2 and 5 test. This is the
    number most sEMG papers lead with and it is the least useful one, because
    it presumes the wearer already sat through a full recording session.

``subject_independent``
    Leave one subject out. Train on nine people, test on the tenth, who the
    model has never seen. This is what a prosthesis fitted to a new user
    actually faces on day one, and it is the number this repository leads with.

``subject_independent_calibrated``
    Leave one subject out, then fine-tune on a single repetition from the held
    out subject before testing on the rest. This is the realistic middle
    ground: a few minutes of fitting appointment, not a full session.

The one rule that matters throughout: nothing derived from the test set may
touch the training pipeline. Normalisation statistics in particular are fit on
the training windows of each fold and applied unchanged to the test windows.
Fitting them globally would leak the test subject's channel gains into the
model and inflate the subject-independent number, which is precisely the
number that is supposed to be hard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import Config
from src.data.windows import WindowIndex


@dataclass
class Fold:
    """One train/test division, plus optional calibration windows."""

    name: str
    protocol: str
    train: WindowIndex
    test: WindowIndex
    calibration: WindowIndex | None = None
    held_out_subject: int | None = None

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "held_out_subject": self.held_out_subject,
            "n_train": len(self.train),
            "n_test": len(self.test),
            "n_calibration": len(self.calibration) if self.calibration else 0,
            "train_subjects": sorted(int(s) for s in np.unique(self.train.subjects)),
            "test_subjects": sorted(int(s) for s in np.unique(self.test.subjects)),
        }


def _require_nonempty(fold: Fold) -> Fold:
    if len(fold.train) == 0:
        raise ValueError(f"fold {fold.name!r} has no training windows")
    if len(fold.test) == 0:
        raise ValueError(f"fold {fold.name!r} has no test windows")
    return fold


def within_subject_folds(index: WindowIndex, config: Config) -> list[Fold]:
    """One fold per subject, split by repetition."""
    train_reps = set(config.evaluation.within_subject_train_reps)
    test_reps = set(config.evaluation.within_subject_test_reps)
    overlap = train_reps & test_reps
    if overlap:
        raise ValueError(f"repetitions appear in both train and test: {sorted(overlap)}")

    folds = []
    for subject in sorted(int(s) for s in np.unique(index.subjects)):
        is_subject = index.subjects == subject
        train_mask = is_subject & np.isin(index.repetitions, sorted(train_reps))
        test_mask = is_subject & np.isin(index.repetitions, sorted(test_reps))
        folds.append(
            _require_nonempty(
                Fold(
                    name=f"s{subject:02d}",
                    protocol="within_subject",
                    train=index.select(train_mask),
                    test=index.select(test_mask),
                    held_out_subject=subject,
                )
            )
        )
    return folds


def subject_independent_folds(
    index: WindowIndex, config: Config, calibrated: bool = False
) -> list[Fold]:
    """Leave-one-subject-out, optionally reserving calibration repetitions.

    When ``calibrated`` is set, the held-out subject's calibration repetitions
    are pulled out of the test set and offered to the model separately. They
    are removed from the test set, not shared with it, so the calibrated
    number is not measured on data the model was just fine-tuned on.
    """
    subjects = sorted(int(s) for s in np.unique(index.subjects))
    if len(subjects) < 2:
        raise ValueError(
            f"leave-one-subject-out needs at least two subjects, got {subjects}"
        )

    calib_reps = sorted(set(config.evaluation.calibration_reps))
    protocol = (
        "subject_independent_calibrated" if calibrated else "subject_independent"
    )

    folds = []
    for held_out in subjects:
        is_held_out = index.subjects == held_out
        train_mask = ~is_held_out

        if calibrated:
            in_calib = np.isin(index.repetitions, calib_reps)
            calibration = index.select(is_held_out & in_calib)
            test = index.select(is_held_out & ~in_calib)
            if len(calibration) == 0:
                raise ValueError(
                    f"subject {held_out} has no windows in calibration repetitions {calib_reps}"
                )
        else:
            calibration = None
            test = index.select(is_held_out)

        folds.append(
            _require_nonempty(
                Fold(
                    name=f"loso_s{held_out:02d}",
                    protocol=protocol,
                    train=index.select(train_mask),
                    test=test,
                    calibration=calibration,
                    held_out_subject=held_out,
                )
            )
        )
    return folds


def make_folds(index: WindowIndex, config: Config, protocol: str) -> list[Fold]:
    if protocol == "within_subject":
        return within_subject_folds(index, config)
    if protocol == "subject_independent":
        return subject_independent_folds(index, config, calibrated=False)
    if protocol == "subject_independent_calibrated":
        return subject_independent_folds(index, config, calibrated=True)
    raise ValueError(
        f"unknown protocol {protocol!r}; expected one of within_subject, "
        f"subject_independent, subject_independent_calibrated"
    )


PROTOCOLS = (
    "within_subject",
    "subject_independent",
    "subject_independent_calibrated",
)


def train_validation_split(
    train: WindowIndex, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole repetitions for early stopping, never random windows.

    Consecutive windows overlap by three quarters at the default geometry. A
    random split would put a window in training and its near-duplicate in
    validation, and the validation curve would then say nothing about
    generalisation. Splitting by (subject, repetition) group avoids that.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    groups = (
        train.subjects.astype(np.int64) * 1000 + train.repetitions.astype(np.int64)
    )
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError(
            "cannot hold out a validation group: the training set contains a single "
            "(subject, repetition) group"
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_val = max(1, int(round(val_fraction * unique.size)))
    n_val = min(n_val, unique.size - 1)
    val_groups = set(shuffled[:n_val].tolist())

    is_val = np.isin(groups, sorted(val_groups))
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)

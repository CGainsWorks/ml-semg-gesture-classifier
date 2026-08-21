"""Scoring.

Accuracy alone is not enough to judge a prosthetic controller, for two
reasons this module tries to keep in view:

* Rest is over-represented even after balancing, and it is the easiest class.
  Balanced accuracy and macro F1 are reported alongside plain accuracy, and
  they are the ones to read.
* A confusion between two grasps that look similar costs the wearer a retry.
  A confusion between rest and a movement makes the hand move when they did
  not ask it to, which is worse. The rest-versus-movement error rates are
  therefore broken out separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class ClassReport:
    code: int
    name: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class Result:
    """Everything measured on one fold, or pooled over several."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    cohen_kappa: float
    confusion: np.ndarray
    per_class: list[ClassReport]
    n_samples: int
    # Movement windows classified as rest: the arm fails to respond.
    missed_activation_rate: float = float("nan")
    # Rest windows classified as a movement: the arm moves unbidden. In a
    # prosthesis this is the error users complain about.
    false_activation_rate: float = float("nan")
    extra: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "cohen_kappa": self.cohen_kappa,
            "n_samples": self.n_samples,
            "false_activation_rate": self.false_activation_rate,
            "missed_activation_rate": self.missed_activation_rate,
            **self.extra,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.summary(),
            "confusion": self.confusion.tolist(),
            "per_class": [c.__dict__ for c in self.per_class],
        }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_codes: np.ndarray,
    class_names: list[str],
    rest_index: int | None = 0,
) -> Result:
    """Score predictions given as contiguous class indices.

    ``class_codes`` and ``class_names`` describe the label space in the order
    the model emits, so a class absent from a particular test fold still gets
    a row in the confusion matrix rather than shifting every later class up
    by one.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    n_classes = len(class_codes)
    if len(class_names) != n_classes:
        raise ValueError(
            f"{len(class_names)} names for {n_classes} classes"
        )
    if y_true.size == 0:
        raise ValueError("nothing to evaluate")

    labels = np.arange(n_classes)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = [
        ClassReport(
            code=int(class_codes[i]),
            name=class_names[i],
            support=int(support[i]),
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
        )
        for i in range(n_classes)
    ]

    # Restrict the aggregate scores to classes that actually appear, so a fold
    # missing a class is not penalised for a recall of zero on something it was
    # never asked about.
    present = support > 0
    macro_f1 = float(f1[present].mean()) if present.any() else float("nan")

    false_activation = missed_activation = float("nan")
    if rest_index is not None and 0 <= rest_index < n_classes:
        rest_total = int(matrix[rest_index].sum())
        if rest_total:
            false_activation = float(
                (rest_total - matrix[rest_index, rest_index]) / rest_total
            )
        movement_rows = [i for i in range(n_classes) if i != rest_index]
        movement_total = int(matrix[movement_rows].sum())
        if movement_total:
            missed_activation = float(
                matrix[movement_rows, rest_index].sum() / movement_total
            )

    return Result(
        accuracy=float((y_true == y_pred).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=macro_f1,
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        confusion=matrix,
        per_class=per_class,
        n_samples=int(y_true.size),
        false_activation_rate=false_activation,
        missed_activation_rate=missed_activation,
    )


def pool(results: list[Result], class_codes: np.ndarray, class_names: list[str]) -> Result:
    """Combine per-fold results by summing their confusion matrices.

    Summing confusions rather than averaging accuracies weights each fold by
    its size, which is what you want when folds differ in length. The mean and
    standard deviation across folds are kept in ``extra`` because for the
    subject-independent condition the spread between subjects is as
    informative as the mean.
    """
    if not results:
        raise ValueError("nothing to pool")

    total = np.sum([r.confusion for r in results], axis=0)
    y_true = np.repeat(np.arange(total.shape[0]), total.sum(axis=1))
    y_pred = np.concatenate(
        [np.repeat(np.arange(total.shape[1]), row) for row in total]
    )

    pooled = evaluate(y_true, y_pred, class_codes, class_names)
    accuracies = np.array([r.accuracy for r in results], dtype=float)
    balanced = np.array([r.balanced_accuracy for r in results], dtype=float)
    pooled.extra.update(
        {
            "n_folds": len(results),
            "fold_accuracy_mean": float(accuracies.mean()),
            "fold_accuracy_sd": float(accuracies.std(ddof=1)) if len(results) > 1 else 0.0,
            "fold_accuracy_min": float(accuracies.min()),
            "fold_accuracy_max": float(accuracies.max()),
            "fold_balanced_accuracy_mean": float(balanced.mean()),
            "fold_balanced_accuracy_sd": (
                float(balanced.std(ddof=1)) if len(results) > 1 else 0.0
            ),
        }
    )
    return pooled


def majority_vote(predictions: np.ndarray, k: int) -> np.ndarray:
    """Smooth a decision stream with a k-decision majority vote.

    A real controller does not act on a single window. Voting over the last k
    decisions removes most single-window flicker at the cost of (k-1) stride
    periods of extra latency, which is accounted for in the latency budget.
    The first k-1 outputs vote over however many decisions exist so far.
    """
    predictions = np.asarray(predictions).reshape(-1)
    if k <= 1:
        return predictions.copy()
    n = predictions.size
    out = np.empty_like(predictions)
    n_classes = int(predictions.max()) + 1 if n else 1
    for i in range(n):
        window = predictions[max(0, i - k + 1) : i + 1]
        out[i] = np.bincount(window, minlength=n_classes).argmax()
    return out

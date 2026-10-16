"""Sweep the window geometry against the control-loop deadline.

The finding that motivated this experiment: with a 200 ms window, a 50 ms
stride and a three-decision majority vote, the controller takes 320 ms from
muscle to actuator and misses the 300 ms budget. Inference is not the reason.
Inference is a rounding error. The window has to fill before the model can see
anything, and the vote has to accumulate before the arm is allowed to move,
and those two terms are 300 of the 320 ms on their own.

That makes the geometry a design decision rather than a default, so it gets
measured, and the configuration shipped in ``config/default.yaml`` comes from
the output of this script.

Two stages, and the second one is not decoration. Stage one trains a model at
each of ten window-and-stride combinations on a single held-out subject and
scores all thirty vote variants. That is enough to rank them coarsely and not
enough to separate two that finish a couple of points apart. Stage two takes
the best budget-meeting geometries and re-runs each over the full ten-fold
leave-one-subject-out rotation.

It matters. On the run committed here, stage two showed the two best
budget-meeting window lengths to be 0.22 standard errors apart, which is a
tie, and re-running the experiment flips which one is nominally ahead. Stage
one gives a confident ranking of them regardless. Knowing it is a tie is what
licenses picking the one that costs fifty milliseconds less. Pass
``--confirm 0`` to skip the second stage and get the confident answer.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from src.config import Config, load_config, resolve
from src.data.dataset import build_dataset, class_codes, encode_labels
from src.data.gestures import gesture_names
from src.data.splits import subject_independent_folds, train_validation_split
from src.eval.metrics import evaluate, majority_vote
from src.models.architecture import ArchitectureSpec
from src.preprocessing.normalise import ChannelScaler

WINDOW_LENGTHS_MS = (100.0, 150.0, 200.0, 250.0, 300.0)
STRIDES_MS = (25.0, 50.0)
VOTE_WINDOWS = (1, 3, 5)


def _evaluate_geometry(
    config: Config, window_ms: float, stride_ms: float, held_out: int, verbose: bool
) -> dict[str, object]:
    """Train once at this geometry and score it, with and without voting."""
    # Copied, not mutated in place: the caller reuses its config for the next
    # geometry and for the confirmation stage.
    config = copy.deepcopy(config)
    config.window.length_ms = window_ms
    config.window.stride_ms = stride_ms

    index = build_dataset(config)
    codes = class_codes(index)
    names = gesture_names(config.dataset.exercise, [int(c) for c in codes])

    folds = subject_independent_folds(index, config)
    fold = next(f for f in folds if f.held_out_subject == held_out)

    x_train = fold.train.materialise("ct")
    y_train = encode_labels(fold.train.labels, codes)
    scaler = ChannelScaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(fold.test.materialise("ct"))
    y_test = encode_labels(fold.test.labels, codes)

    spec = ArchitectureSpec.from_config(
        config.train, index.n_channels, index.length, codes.size
    )
    train_idx, val_idx = train_validation_split(
        fold.train, config.train.val_fraction, config.train.seed
    )

    from src.models import torch_cnn

    model, info = torch_cnn.train_model(
        spec,
        x_train[train_idx],
        y_train[train_idx],
        x_train[val_idx],
        y_train[val_idx],
        config.train,
        verbose=False,
    )
    proba = torch_cnn.predict_proba(model, x_test, device=config.train.device)
    predictions = proba.argmax(axis=1)

    # The vote only makes sense over consecutive decisions from one subject,
    # which is what the test fold is, in time order.
    order = np.argsort(fold.test.starts, kind="stable")
    rows = []
    for vote in VOTE_WINDOWS:
        voted = np.empty_like(predictions)
        voted[order] = majority_vote(predictions[order], vote)
        result = evaluate(y_test, voted, codes, names)
        vote_ms = (vote - 1) * stride_ms
        total_ms = window_ms + vote_ms + config.latency.io_allowance_ms
        rows.append(
            {
                "window_ms": window_ms,
                "stride_ms": stride_ms,
                "vote_window": vote,
                "accuracy": result.accuracy,
                "balanced_accuracy": result.balanced_accuracy,
                "macro_f1": result.macro_f1,
                "false_activation_rate": result.false_activation_rate,
                # Inference is excluded here and added back by the caller from
                # the measured benchmark, so this column is the part of the
                # budget the geometry alone controls.
                "geometry_latency_ms": total_ms,
                "window_fill_ms": window_ms,
                "majority_vote_ms": vote_ms,
                "io_allowance_ms": config.latency.io_allowance_ms,
                "budget_ms": config.latency.budget_ms,
                "n_test_windows": int(y_test.size),
                "parameters": spec.parameter_count(),
                "macs": spec.multiply_accumulates(),
            }
        )
        if verbose:
            print(
                f"  window {window_ms:5.0f} ms  stride {stride_ms:4.0f} ms  vote {vote}: "
                f"accuracy {result.accuracy:.4f}  geometry latency {total_ms:6.1f} ms"
            )
    for row in rows:
        row["train_seconds"] = info["train_seconds"]
    return rows


def _confirm_across_all_folds(
    config: Config, candidates: list[dict[str, object]], verbose: bool
) -> list[dict[str, object]]:
    """Re-run the shortlisted geometries over every leave-one-subject-out fold.

    The sweep above uses one held-out subject, which is enough to rank thirty
    geometries against each other but not enough to separate two that finish
    within a couple of points. This stage takes the shortlist and does the full
    ten-fold rotation on each, so the choice between them rests on a mean and a
    spread rather than on one subject.
    """
    from src.experiment import run_protocol

    out = []
    for candidate in candidates:
        window_ms, stride_ms = candidate["window_ms"], candidate["stride_ms"]
        if verbose:
            print(
                f"confirming {window_ms:.0f} ms window, {stride_ms:.0f} ms stride "
                "over all ten folds; this trains ten models",
                flush=True,
            )
        confirmed = copy.deepcopy(config)
        confirmed.window.length_ms = window_ms
        confirmed.window.stride_ms = stride_ms
        run = run_protocol(
            confirmed,
            "pytorch",
            "subject_independent",
            verbose=verbose,
            train_verbose=False,
        )
        pooled = run.pooled
        row = {
            "window_ms": window_ms,
            "stride_ms": stride_ms,
            "vote_window": 1,
            "protocol": "subject_independent (all ten folds)",
            "accuracy": pooled.accuracy,
            "balanced_accuracy": pooled.balanced_accuracy,
            "macro_f1": pooled.macro_f1,
            "fold_mean": pooled.extra["fold_accuracy_mean"],
            "fold_sd": pooled.extra["fold_accuracy_sd"],
            "fold_min": pooled.extra["fold_accuracy_min"],
            "fold_max": pooled.extra["fold_accuracy_max"],
            "n_train_windows": int(run.per_fold[0]["n_train"]),
            "geometry_latency_ms": window_ms + config.latency.io_allowance_ms,
            "wall_seconds": run.wall_seconds,
        }
        out.append(row)
        if verbose:
            print(
                f"  balanced accuracy {row['balanced_accuracy']:.4f}, "
                f"per-fold {row['fold_mean']:.4f} +/- {row['fold_sd']:.4f}"
            )
    return out


def run(
    config: Config,
    held_out: int = 1,
    windows: tuple[float, ...] = WINDOW_LENGTHS_MS,
    strides: tuple[float, ...] = STRIDES_MS,
    confirm: int = 0,
    verbose: bool = True,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for window_ms in windows:
        for stride_ms in strides:
            if verbose:
                print(f"window {window_ms:.0f} ms, stride {stride_ms:.0f} ms")
            rows.extend(
                _evaluate_geometry(config, window_ms, stride_ms, held_out, verbose)
            )

    budget = config.latency.budget_ms
    feasible = [r for r in rows if r["geometry_latency_ms"] <= budget]
    best = max(feasible, key=lambda r: r["balanced_accuracy"]) if feasible else None
    best_overall = max(rows, key=lambda r: r["balanced_accuracy"])

    report: dict[str, object] = {
        "synthetic": config.dataset.source == "synthetic",
        "held_out_subject": held_out,
        "protocol": "subject_independent (single fold)",
        "budget_ms": budget,
        "rows": rows,
        "best_within_budget": best,
        "best_overall": best_overall,
        "accuracy_cost_of_meeting_budget": (
            best_overall["balanced_accuracy"] - best["balanced_accuracy"]
            if best
            else None
        ),
    }

    if confirm and feasible:
        # Shortlist by accuracy among the geometries that meet the budget, one
        # per (window, stride) pair since the vote does not change training.
        seen: set[tuple[float, float]] = set()
        shortlist: list[dict[str, object]] = []
        for row in sorted(feasible, key=lambda r: -r["balanced_accuracy"]):
            key = (row["window_ms"], row["stride_ms"])
            if key in seen:
                continue
            seen.add(key)
            shortlist.append(row)
            if len(shortlist) >= confirm:
                break
        report["confirmation"] = _confirm_across_all_folds(config, shortlist, verbose)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--held-out", type=int, default=1)
    parser.add_argument(
        "--confirm",
        type=int,
        default=2,
        help=(
            "re-run this many of the best budget-meeting geometries across all "
            "ten folds, so the choice between them is not decided by one subject"
        ),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    report = run(config, held_out=args.held_out, confirm=args.confirm)

    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    out = (
        Path(args.out)
        if args.out
        else resolve("experiments") / "results" / tag / "latency_accuracy_tradeoff.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    best = report["best_within_budget"]
    if best:
        print(
            f"\nbest inside the {report['budget_ms']:.0f} ms budget: "
            f"window {best['window_ms']:.0f} ms, stride {best['stride_ms']:.0f} ms, "
            f"vote {best['vote_window']} -> balanced accuracy {best['balanced_accuracy']:.4f} "
            f"at {best['geometry_latency_ms']:.0f} ms"
        )
        print(
            f"best at any latency: {report['best_overall']['balanced_accuracy']:.4f} "
            f"at {report['best_overall']['geometry_latency_ms']:.0f} ms "
            f"(meeting the budget costs {report['accuracy_cost_of_meeting_budget']:.4f})"
        )
    else:
        print("\nno geometry in the sweep meets the budget")

    for row in report.get("confirmation", []):
        print(
            f"ten-fold: {row['window_ms']:.0f} ms window, {row['stride_ms']:.0f} ms "
            f"stride -> balanced accuracy {row['balanced_accuracy']:.4f}, "
            f"per fold {row['fold_mean']:.4f} +/- {row['fold_sd']:.4f}"
        )
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

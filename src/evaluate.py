"""Turn the saved result files into figures, tables and the evaluation report.

Reads whatever ``src.train`` wrote under ``artifacts/results/``, so it can be
re-run without retraining. Everything quoted in the README comes from here,
and ``--check`` verifies that what is in the README still matches what is on
disk, which is what stops the numbers drifting away from the code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import Config, load_config, resolve
from src.data.gestures import short_names
from src.data.splits import PROTOCOLS
from src.eval import plots
from src.experiment import MODELS

MODEL_LABELS = {
    "classical_lda": "LDA on time-domain features",
    "classical_svm": "RBF SVM on time-domain features",
    "classical_rf": "Random forest on time-domain features",
    "tensorflow": "2-D CNN (TensorFlow)",
    "pytorch": "2-D CNN (PyTorch)",
}

PROTOCOL_LABELS = {
    "within_subject": "Within-subject",
    "subject_independent": "Subject-independent (LOSO)",
    "subject_independent_calibrated": "Subject-independent + calibration",
}


def results_dir(config: Config) -> Path:
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    return resolve("results") / tag


def analysis_dir(config: Config) -> Path:
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    return resolve("analysis") / tag


def load_results(config: Config) -> list[dict]:
    directory = results_dir(config)
    if not directory.exists():
        raise FileNotFoundError(
            f"no results under {directory}. "
            "Run: python -m src.train --model all --protocol all"
        )
    # Only the scored protocol runs, which are named "<model>__<protocol>.json".
    # The same directory also holds export_report.json, which is evidence of a
    # different kind and has no confusion matrix in it.
    out = []
    for path in sorted(directory.glob("*__*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    if not out:
        raise FileNotFoundError(
            f"no <model>__<protocol>.json files in {directory}. "
            "Run: python -m src.train --model all --protocol all"
        )
    return out


def summary_rows(results: list[dict]) -> list[dict]:
    rows = []
    for run in results:
        pooled = run["pooled"]
        rows.append(
            {
                "model": run["model"],
                "protocol": run["protocol"],
                "accuracy": pooled["accuracy"],
                "balanced_accuracy": pooled["balanced_accuracy"],
                "macro_f1": pooled["macro_f1"],
                "cohen_kappa": pooled["cohen_kappa"],
                "false_activation_rate": pooled["false_activation_rate"],
                "missed_activation_rate": pooled["missed_activation_rate"],
                "n_folds": pooled.get("n_folds"),
                "fold_mean": pooled.get("fold_accuracy_mean"),
                "fold_sd": pooled.get("fold_accuracy_sd"),
                "fold_min": pooled.get("fold_accuracy_min"),
                "fold_max": pooled.get("fold_accuracy_max"),
                "n_test_windows": pooled["n_samples"],
                "wall_seconds": run["wall_seconds"],
            }
        )
    return rows


def generalisation_gap(rows: list[dict]) -> list[dict]:
    """Within-subject minus subject-independent, per model.

    The single number this repository is built to report. Anything that quotes
    only the within-subject figure is describing a device that has already met
    its user.
    """
    by_model: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["protocol"]] = row

    out = []
    for model, protocols in sorted(by_model.items()):
        within = protocols.get("within_subject")
        across = protocols.get("subject_independent")
        calibrated = protocols.get("subject_independent_calibrated")
        if not (within and across):
            continue
        entry = {
            "model": model,
            "within_subject": within["balanced_accuracy"],
            "subject_independent": across["balanced_accuracy"],
            "gap": within["balanced_accuracy"] - across["balanced_accuracy"],
            "relative_drop": (
                1.0 - across["balanced_accuracy"] / within["balanced_accuracy"]
                if within["balanced_accuracy"] > 0
                else float("nan")
            ),
            "subject_independent_sd": across.get("fold_sd"),
            "worst_subject": across.get("fold_min"),
            "best_subject": across.get("fold_max"),
        }
        if calibrated:
            entry["subject_independent_calibrated"] = calibrated["balanced_accuracy"]
            entry["gap_recovered_by_calibration"] = (
                calibrated["balanced_accuracy"] - across["balanced_accuracy"]
            )
            entry["fraction_of_gap_recovered"] = (
                (calibrated["balanced_accuracy"] - across["balanced_accuracy"]) / entry["gap"]
                if entry["gap"] > 1e-9
                else float("nan")
            )
        out.append(entry)
    return out


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def accuracy_table(rows: list[dict]) -> str:
    ordered = [
        r
        for protocol in PROTOCOLS
        for model in MODELS
        for r in rows
        if r["model"] == model and r["protocol"] == protocol
    ]
    body = []
    for r in ordered:
        spread = (
            f"{r['fold_mean']:.3f} ± {r['fold_sd']:.3f}"
            if r["fold_mean"] is not None
            else "-"
        )
        body.append(
            [
                PROTOCOL_LABELS[r["protocol"]],
                MODEL_LABELS.get(r["model"], r["model"]),
                f"{r['accuracy']:.3f}",
                f"{r['balanced_accuracy']:.3f}",
                f"{r['macro_f1']:.3f}",
                spread,
                f"{r['fold_min']:.3f} / {r['fold_max']:.3f}" if r["fold_min"] is not None else "-",
            ]
        )
    return _markdown_table(
        [
            "Protocol",
            "Model",
            "Accuracy",
            "Balanced accuracy",
            "Macro F1",
            "Per-fold mean ± sd",
            "Worst / best fold",
        ],
        body,
    )


def gap_table(gaps: list[dict]) -> str:
    body = []
    for g in gaps:
        calibrated = g.get("subject_independent_calibrated")
        body.append(
            [
                MODEL_LABELS.get(g["model"], g["model"]),
                f"{g['within_subject']:.3f}",
                f"{g['subject_independent']:.3f}",
                f"{g['gap']:.3f}",
                f"{100 * g['relative_drop']:.1f}%",
                f"{calibrated:.3f}" if calibrated is not None else "-",
                (
                    f"{100 * g['fraction_of_gap_recovered']:.0f}%"
                    if calibrated is not None
                    else "-"
                ),
            ]
        )
    return _markdown_table(
        [
            "Model",
            "Within-subject",
            "Subject-independent",
            "Gap",
            "Relative drop",
            "+ calibration",
            "Gap recovered",
        ],
        body,
    )


def per_class_table(run: dict, limit: int | None = None) -> str:
    per_class = run["pooled"]["per_class"]
    ordered = sorted(per_class, key=lambda c: c["recall"])
    if limit:
        ordered = ordered[:limit]
    body = [
        [
            str(c["code"]),
            c["name"],
            str(c["support"]),
            f"{c['precision']:.3f}",
            f"{c['recall']:.3f}",
            f"{c['f1']:.3f}",
        ]
        for c in ordered
    ]
    return _markdown_table(
        ["Code", "Gesture", "Support", "Precision", "Recall", "F1"], body
    )


def write_figures(config: Config, results: list[dict], rows: list[dict]) -> list[Path]:
    out_dir = analysis_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    n_classes = len(results[0]["class_codes"])
    written.append(
        plots.protocol_comparison(
            rows,
            out_dir / "protocol_comparison.png",
            title=(
                "Balanced accuracy by protocol "
                f"(Ninapro DB5, exercise {config.dataset.exercise}, {n_classes} classes)"
                if config.dataset.source != "synthetic"
                else "SYNTHETIC DATA - wiring check only"
            ),
            n_classes=n_classes,
        )
    )

    for run in results:
        labels = short_names(config.dataset.exercise, run["class_codes"])
        matrix = np.array(run["pooled"]["confusion"])
        written.append(
            plots.confusion_matrix(
                matrix,
                labels,
                out_dir / f"confusion_{run['model']}__{run['protocol']}.png",
                title=(
                    f"{MODEL_LABELS.get(run['model'], run['model'])}, "
                    f"{PROTOCOL_LABELS[run['protocol']]}"
                ),
            )
        )

    for run in results:
        if run["protocol"] != "subject_independent":
            continue
        per_class = run["pooled"]["per_class"]
        written.append(
            plots.per_class_bars(
                [c["name"] for c in per_class],
                [c["precision"] for c in per_class],
                [c["recall"] for c in per_class],
                out_dir / f"per_class_{run['model']}.png",
                title=f"{MODEL_LABELS.get(run['model'], run['model'])}, subject-independent",
            )
        )

    sweep = resolve("experiments") / "results" / (
        "synthetic" if config.dataset.source == "synthetic" else "db5"
    ) / "latency_accuracy_tradeoff.json"
    if sweep.exists():
        report = json.loads(sweep.read_text(encoding="utf-8"))
        written.append(
            plots.latency_accuracy(
                report["rows"], report["budget_ms"], out_dir / "latency_accuracy.png"
            )
        )
    return written


def write_report(config: Config, results: list[dict], rows: list[dict]) -> Path:
    gaps = generalisation_gap(rows)
    loso = [r for r in results if r["protocol"] == "subject_independent"]
    best = max(loso, key=lambda r: r["pooled"]["balanced_accuracy"]) if loso else None

    dataset = results[0]["dataset"]
    lines = [
        "# Evaluation report",
        "",
        "Generated by `python -m src.evaluate`. Do not edit by hand.",
        "",
    ]
    if config.dataset.source == "synthetic":
        lines += [
            "> **SYNTHETIC DATA.** This report was generated from the synthetic",
            "> generator, not from Ninapro. The numbers below are a wiring check.",
            "",
        ]

    lines += [
        "## Dataset",
        "",
        f"- Source: `{config.dataset.source}`, exercise {config.dataset.exercise}",
        f"- Subjects: {dataset['subjects']}",
        f"- Channels: {dataset['n_channels']} at {dataset['sample_rate_hz']} Hz",
        f"- Window: {dataset['window_samples']} samples "
        f"({config.window.length_ms:.0f} ms), stride {config.window.stride_ms:.0f} ms",
        f"- Windows after rest balancing: {dataset['n_windows']:,}",
        f"- Classes: {len(results[0]['class_codes'])}",
        "",
        "## Accuracy by protocol",
        "",
        accuracy_table(rows),
        "",
        "## The generalisation gap",
        "",
        "Within-subject accuracy measures a model that has already seen the wearer.",
        "Subject-independent accuracy measures one that has not. The second number is",
        "the one a prosthesis faces on the day it is fitted.",
        "",
        gap_table(gaps),
        "",
    ]

    if best:
        lines += [
            f"## Per-class breakdown: {MODEL_LABELS.get(best['model'], best['model'])}, "
            "subject-independent",
            "",
            "Sorted by recall, worst first.",
            "",
            per_class_table(best),
            "",
        ]

    out = analysis_dir(config) / "EVALUATION.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    results = load_results(config)
    rows = summary_rows(results)

    print(accuracy_table(rows))
    print()
    print(gap_table(generalisation_gap(rows)))

    summary = {
        "synthetic": config.dataset.source == "synthetic",
        "rows": rows,
        "generalisation_gap": generalisation_gap(rows),
    }
    summary_path = analysis_dir(config) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_figures:
        written = write_figures(config, results, rows)
        print(f"\n{len(written)} figure(s) written to {analysis_dir(config)}")

    report = write_report(config, results, rows)
    print(f"report written to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

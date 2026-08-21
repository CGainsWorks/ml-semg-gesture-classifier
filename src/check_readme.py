"""Keep the README honest.

The brief this repository was built to says every quantitative claim in the
README must be reproducible from a script in the repository. Saying so is
easy. Keeping it true after the fourth retraining run is not, which is why it
is enforced here rather than promised.

Every generated table and figure in the README sits between a pair of
markers:

    <!-- BEGIN:accuracy -->
    ...generated content...
    <!-- END:accuracy -->

This module regenerates each block from the committed JSON under
``results/``, ``benchmarks/``, ``experiments/results/`` and ``data/``, and
either checks it against what is in the README or writes it back:

    python -m src.check_readme            # verify; non-zero exit on drift
    python -m src.check_readme --write    # regenerate the blocks in place

CI runs the first form. A number that changes without the README being
regenerated fails the build.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.config import Config, load_config, resolve
from src.evaluate import (
    MODEL_LABELS,
    PROTOCOL_LABELS,
    _markdown_table,
    accuracy_table,
    gap_table,
    generalisation_gap,
    load_results,
    summary_rows,
)

README = "README.md"
BACKSLASH = chr(92)
MARKER = re.compile(
    r"(<!-- BEGIN:(?P<name>[a-z0-9_-]+) -->\n)(?P<body>.*?)(<!-- END:(?P=name) -->)",
    re.DOTALL,
)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _tag(config: Config) -> str:
    return "synthetic" if config.dataset.source == "synthetic" else "db5"


def _artefact_name(path: str) -> str:
    """Basename of a path that may have been recorded on another platform.

    ``pathlib`` only understands the separators of the host it runs on, so a
    Windows path read back on Linux comes out as one long filename and the
    lookup silently misses, leaving the latency columns blank. Benchmarks are
    produced on one machine and checked on another, which is precisely when
    that happens.
    """
    return str(path).replace(BACKSLASH, "/").rsplit("/", 1)[-1]


# --------------------------------------------------------------------------
# Block builders. One per marker name in the README.
# --------------------------------------------------------------------------


def block_dataset(config: Config) -> str:
    stats = _read_json(resolve("data/dataset_stats.json"))
    if stats is None:
        raise FileNotFoundError(
            "data/dataset_stats.json is missing. Run: python -m src.data.build_report"
        )
    ages = stats["age_range"]
    circumference = stats["forearm_circumference_range_cm"]
    genders = ", ".join(f"{n} {k}" for k, n in sorted(stats["gender_counts"].items()))
    return _markdown_table(
        ["Property", "Value"],
        [
            ["Dataset", f"Ninapro DB5, exercise {stats['exercise']}"],
            ["Sensor", "two Thalmic Myo armbands, 16 dry electrodes"],
            [
                "Sampling",
                f"{stats['n_channels']} channels at {stats['sample_rate_hz']} Hz",
            ],
            ["Subjects", f"{len(stats['subjects'])} intact-limbed adults ({genders})"],
            ["Ages", f"{ages[0]} to {ages[1]}"],
            [
                "Forearm circumference",
                f"{circumference[0]} to {circumference[1]} cm",
            ],
            ["Recording", f"{stats['total_recording_hours']} hours"],
            ["Classes", f"{stats['n_classes']} (rest plus 17 movements)"],
            [
                "Window",
                f"{stats['window_samples']} samples "
                f"({stats['window_ms']:.0f} ms), {stats['stride_ms']:.0f} ms stride",
            ],
            ["Windows after rest balancing", f"{stats['windows']:,}"],
            [
                "Windows per class",
                f"{stats['class_count_min']:,} to {stats['class_count_max']:,}",
            ],
        ],
    )


def block_accuracy(config: Config) -> str:
    return accuracy_table(summary_rows(load_results(config)))


def block_gap(config: Config) -> str:
    return gap_table(generalisation_gap(summary_rows(load_results(config))))


def block_per_class(config: Config) -> str:
    """The five worst and five best classes under the leave-one-subject-out protocol."""
    results = load_results(config)
    loso = [r for r in results if r["protocol"] == "subject_independent"]
    if not loso:
        raise FileNotFoundError("no subject_independent results to summarise")
    best = max(loso, key=lambda r: r["pooled"]["balanced_accuracy"])
    per_class = sorted(best["pooled"]["per_class"], key=lambda c: c["recall"])
    # Half the classes at most from each end, so a small label space does not
    # produce a table that lists the same gesture as both the worst and the
    # best.
    take = min(5, len(per_class) // 2)
    selected = per_class[:take] + per_class[len(per_class) - take :]

    body = [
        [
            "worst" if i < take else "best",
            str(c["code"]),
            c["name"],
            f"{c['precision']:.3f}",
            f"{c['recall']:.3f}",
            f"{c['f1']:.3f}",
        ]
        for i, c in enumerate(selected)
    ]
    label = MODEL_LABELS.get(best["model"], best["model"])
    # The figure is emitted from here rather than written into the README by
    # hand, so it always shows the same model the table describes. Which model
    # wins the subject-independent protocol is a result, not something to
    # hard-code in prose and then forget to update.
    figure = f"analysis/db5/per_class_{best['model']}.png"
    return (
        f"Model: {label}, {PROTOCOL_LABELS[best['protocol']]}. This is whichever\n"
        "model came out with the best subject-independent balanced accuracy.\n\n"
        + _markdown_table(["", "Code", "Gesture", "Precision", "Recall", "F1"], body)
        + f"\n\n![Per-class precision and recall, subject-independent]({figure})"
    )


def block_architecture(config: Config) -> str:
    report = _read_json(
        resolve("experiments") / "results" / _tag(config) / "architecture_comparison.json"
    )
    if report is None:
        raise FileNotFoundError(
            "experiments/results/*/architecture_comparison.json is missing. "
            "Run: python -m experiments.architecture_comparison"
        )
    labels = {
        "conv1d": "1-D convolution over time",
        "conv2d": "2-D over electrodes and time (shipped)",
        "conv2d_circular": "2-D, electrode axis wrapped",
        "conv2d_noaug": "2-D, no augmentation",
        "conv2d_electrode_shift": "2-D, plus random electrode rotation",
    }
    within = {r["variant"]: r for r in report["results"]["within_subject"]}
    across = {r["variant"]: r for r in report["results"]["subject_independent"]}
    body = [
        [
            labels.get(variant, variant),
            f"{within[variant]['test_accuracy']:.3f}",
            f"{across[variant]['test_accuracy']:.3f}",
            f"{within[variant]['parameters']:,}",
        ]
        for variant in labels
        if variant in within and variant in across
    ]
    return _markdown_table(
        ["Variant", "Within-subject", "Subject-independent", "Parameters"], body
    )


def block_quantisation(config: Config) -> str:
    report = _read_json(resolve("results") / _tag(config) / "export_report.json")
    if report is None:
        raise FileNotFoundError(
            "results/*/export_report.json is missing. "
            "Run: python -m src.quantise.export"
        )
    latency = _read_json(resolve("benchmarks") / "latency_host.json")
    latencies = {}
    if latency:
        for row in latency["results"]:
            latencies[_artefact_name(row["artefact"])] = row

    body = []
    for row in report["accuracy"]:
        name = row["artefact"]
        measured = latencies.get(name)
        body.append(
            [
                f"`{name}`",
                f"{row['bytes'] / 1024:.0f} KiB",
                f"{row['accuracy']:.4f}",
                f"{measured['median_ms']:.3f}" if measured else "-",
                f"{measured['p99_ms']:.3f}" if measured else "-",
            ]
        )
    return _markdown_table(
        ["Artefact", "Size", "Accuracy", "p50 latency (ms)", "p99 latency (ms)"], body
    )


def block_latency_budget(config: Config) -> str:
    tag = _tag(config)
    loop = _read_json(resolve("benchmarks") / f"control_loop_{tag}.json")
    if loop is None:
        raise FileNotFoundError(
            f"benchmarks/control_loop_{tag}.json is missing. "
            "Run: python -m src.embedded.control_loop"
        )
    latency = loop["latency"]
    verdict = "**meets**" if latency["meets_budget"] else "**MISSES**"
    body = [
        ["Window fill", f"{latency['window_fill_ms']:.1f}", "measured geometry"],
        [
            "Majority vote",
            f"{latency['majority_vote_ms']:.1f}",
            "not used; see the geometry sweep"
            if loop["vote_window"] <= 1
            else f"{loop['vote_window']} decisions",
        ],
        [
            "Inference (p99)",
            f"{latency['p99_inference_ms']:.3f}",
            "measured, this machine, int8",
        ],
        [
            "Acquisition and actuation",
            f"{latency['io_allowance_ms']:.1f}",
            "**assumed**, not implemented here",
        ],
        ["**Total**", f"**{latency['total_ms']:.1f}**", f"{verdict} the budget"],
        [
            "Budget",
            f"{latency['budget_ms']:.0f}",
            f"{latency['headroom_ms']:.1f} ms headroom",
        ],
    ]
    return _markdown_table(["Term", "Milliseconds", "Provenance"], body)


def block_on_target(config: Config) -> str:
    report = _read_json(resolve("benchmarks") / "latency_targets.json")
    if report is None:
        raise FileNotFoundError(
            "benchmarks/latency_targets.json is missing. "
            "Run: python -m src.embedded.estimate_report"
        )
    body = []
    for row in report["estimates"]:
        memory = row["memory"]
        body.append(
            [
                row["target"],
                row["runtime"],
                f"{row['estimated_inference_ms']:.2f}",
                f"{row['estimated_total_ms']:.1f}",
                "yes" if row["estimated_to_meet_budget"] else "no",
                "yes" if memory["fits_flash"] else "no",
            ]
        )
    return (
        "**Estimates, not measurements.** No STM32 or Jetson was available for\n"
        "this work. Every row below is arithmetic from the model's\n"
        f"{report['artefact']['macs']:,} multiply-accumulates and a documented\n"
        "throughput assumption per board; see `src/embedded/estimate.py` for the\n"
        "assumptions and `benchmarks/README.md` for how to replace a row with a\n"
        "real measurement.\n\n"
        + _markdown_table(
            [
                "Target",
                "Runtime",
                "Est. inference (ms)",
                "Est. total (ms)",
                "Inside budget?",
                "Fits flash?",
            ],
            body,
        )
    )


def block_control_loop(config: Config) -> str:
    tag = _tag(config)
    loop = _read_json(resolve("benchmarks") / f"control_loop_{tag}.json")
    if loop is None:
        raise FileNotFoundError(f"benchmarks/control_loop_{tag}.json is missing")
    single = loop["single_window"]
    voted = loop["majority_vote"]
    body = [
        [
            "Single window",
            f"{single['accuracy']:.4f}",
            f"{single['balanced_accuracy']:.4f}",
            f"{single['false_activation_rate']:.4f}",
        ],
        [
            f"{loop['vote_window']}-decision majority vote",
            f"{voted['accuracy']:.4f}",
            f"{voted['balanced_accuracy']:.4f}",
            f"{voted['false_activation_rate']:.4f}",
        ],
    ]
    return _markdown_table(
        ["Decision rule", "Accuracy", "Balanced accuracy", "False activation rate"],
        body,
    )


def block_geometry(config: Config) -> str:
    report = _read_json(
        resolve("experiments")
        / "results"
        / _tag(config)
        / "latency_accuracy_tradeoff.json"
    )
    if report is None:
        raise FileNotFoundError(
            "experiments/results/*/latency_accuracy_tradeoff.json is missing. "
            "Run: python -m experiments.latency_accuracy_tradeoff"
        )
    rows = sorted(report["rows"], key=lambda r: -r["balanced_accuracy"])[:8]
    body = [
        [
            f"{r['window_ms']:.0f}",
            f"{r['stride_ms']:.0f}",
            str(r["vote_window"]),
            f"{r['balanced_accuracy']:.4f}",
            f"{r['geometry_latency_ms']:.0f}",
            "yes" if r["geometry_latency_ms"] <= report["budget_ms"] else "no",
        ]
        for r in rows
    ]
    best = report["best_within_budget"]
    cost = report["accuracy_cost_of_meeting_budget"]
    suffix = ""
    if best is not None and cost is not None:
        suffix = (
            f"\n\nBest geometry inside the {report['budget_ms']:.0f} ms budget: "
            f"a {best['window_ms']:.0f} ms window, a {best['stride_ms']:.0f} ms stride "
            f"and a {best['vote_window']}-decision vote. Meeting the budget costs "
            f"{cost:.4f} balanced accuracy against the best geometry at any latency."
        )

    confirmation = report.get("confirmation")
    if confirmation:
        ranked = sorted(confirmation, key=lambda r: -r["balanced_accuracy"])
        suffix += (
            "\n\nThose rows come from one held-out subject, which is enough to rank\n"
            "thirty geometries and not enough to separate two that finish a couple\n"
            "of points apart. The shortlist was therefore re-run over the full\n"
            "ten-fold rotation:\n\n"
            + _markdown_table(
                [
                    "Window (ms)",
                    "Stride (ms)",
                    "Balanced accuracy",
                    "Per-fold mean +/- sd",
                    "Worst / best fold",
                    "Training windows",
                ],
                [
                    [
                        f"{r['window_ms']:.0f}",
                        f"{r['stride_ms']:.0f}",
                        f"{r['balanced_accuracy']:.4f}",
                        f"{r['fold_mean']:.4f} +/- {r['fold_sd']:.4f}",
                        f"{r['fold_min']:.3f} / {r['fold_max']:.3f}",
                        f"{r['n_train_windows']:,}",
                    ]
                    for r in ranked
                ],
            )
        )
    return (
        "Top eight geometries by accuracy, out of the "
        f"{len(report['rows'])} in the sweep. Latency here is window fill plus vote "
        "plus the input/output allowance; inference is under a tenth of a "
        "millisecond and does not move the column.\n\n"
        + _markdown_table(
            [
                "Window (ms)",
                "Stride (ms)",
                "Vote",
                "Balanced accuracy",
                "Latency (ms)",
                "Inside budget?",
            ],
            body,
        )
        + suffix
    )


BUILDERS = {
    "dataset": block_dataset,
    "accuracy": block_accuracy,
    "gap": block_gap,
    "per_class": block_per_class,
    "architecture": block_architecture,
    "quantisation": block_quantisation,
    "latency_budget": block_latency_budget,
    "on_target": block_on_target,
    "control_loop": block_control_loop,
    "geometry": block_geometry,
}


def regenerate(text: str, config: Config) -> tuple[str, list[str]]:
    """Rewrite every marked block. Returns the new text and the names that changed.

    The marker set is validated first, before any builder runs. A structural
    problem with the README is cheap to detect, and reporting it as whatever
    missing-data error the first builder happens to raise is unhelpful.
    """
    present = {match.group("name") for match in MARKER.finditer(text)}

    unknown = present - set(BUILDERS)
    if unknown:
        raise ValueError(
            "README has block(s) with no builder in src/check_readme.py: "
            f"{sorted(unknown)}. Known blocks: {sorted(BUILDERS)}"
        )
    missing = set(BUILDERS) - present
    if missing:
        raise ValueError(
            f"src/check_readme.py knows how to build {sorted(missing)} but the README "
            "has no marker for them. Add the markers or remove the builders."
        )

    changed: list[str] = []

    def replace(match: re.Match) -> str:
        name = match.group("name")
        fresh = BUILDERS[name](config).strip() + "\n"
        if fresh != match.group("body"):
            changed.append(name)
        return match.group(1) + fresh + match.group(4)

    return MARKER.sub(replace, text), changed

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--readme", default=README)
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the blocks in place instead of checking them",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    path = resolve(args.readme)
    text = path.read_text(encoding="utf-8")

    try:
        new_text, changed = regenerate(text, config)
    except FileNotFoundError as exc:
        print(f"cannot check the README: {exc}", file=sys.stderr)
        return 2

    if args.write:
        if changed:
            path.write_text(new_text, encoding="utf-8")
            print(f"regenerated: {', '.join(changed)}")
        else:
            print("already up to date")
        return 0

    if changed:
        print(
            "README is out of date with the committed results.\n"
            f"  stale blocks: {', '.join(changed)}\n"
            "  fix with: python -m src.check_readme --write",
            file=sys.stderr,
        )
        return 1

    print(f"README matches the committed results ({len(BUILDERS)} blocks checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

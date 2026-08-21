"""Write the on-target latency estimate table.

Reads the exported artefacts and the measured host benchmark, then applies
:mod:`src.embedded.estimate` to produce a per-board figure.

Every row it writes carries ``"measured": false``. That is the point. No
STM32 or Jetson was available for this work, the brief asks for on-target
latency, and the honest answer is arithmetic with its assumptions written
down rather than a number that looks like a measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import Config, load_config, resolve
from src.embedded.estimate import estimate_all


def _artefact_facts(config: Config) -> dict[str, object]:
    """Model size, MAC count and arena floor, read from what was exported."""
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    report_path = resolve("results") / tag / "export_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} not found. Run `python -m src.quantise.export` first."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    architecture = report.get("architecture")
    if not architecture:
        raise ValueError(
            f"{report_path} has no architecture block; re-run the export with the "
            "PyTorch deployment model present so the MAC count is recorded"
        )

    tflite_int8 = report["artefacts"].get("tflite_int8")
    if tflite_int8 is None:
        raise ValueError(
            f"{report_path} has no int8 TFLite artefact; the on-target estimate is "
            "for the integer model and there is nothing to size"
        )

    return {
        "macs": int(architecture["macs_per_inference"]),
        "parameters": int(architecture["parameters"]),
        "model_bytes": int(tflite_int8["bytes"]),
        "arena_bytes": int(tflite_int8["arena_bytes"]),
    }


def _measured_host(config: Config) -> dict[str, object] | None:
    path = resolve("benchmarks") / "latency_host.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    for row in blob.get("results", []):
        if row["artefact"].endswith("gesture_cnn_int8.tflite"):
            return {
                "label": blob.get("label", "host"),
                "host": blob.get("host"),
                "median_ms": row["median_ms"],
                "p99_ms": row["p99_ms"],
                "measured": True,
            }
    return None


def run(config: Config) -> dict[str, object]:
    facts = _artefact_facts(config)
    vote_ms = (config.evaluation.vote_window - 1) * config.window.stride_ms

    rows = estimate_all(
        macs=facts["macs"],
        model_bytes=facts["model_bytes"],
        arena_bytes=facts["arena_bytes"],
        window_ms=config.window.length_ms,
        vote_ms=vote_ms,
        io_ms=config.latency.io_allowance_ms,
        budget_ms=config.latency.budget_ms,
    )

    return {
        "synthetic": config.dataset.source == "synthetic",
        "disclaimer": (
            "Estimates, not measurements. No STM32 or Jetson was available. "
            "Every row is arithmetic from the MAC count and a documented "
            "throughput assumption; see src/embedded/estimate.py and "
            "benchmarks/README.md."
        ),
        "artefact": facts,
        "measured_host": _measured_host(config),
        "estimates": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    report = run(config)

    out = Path(args.out) if args.out else resolve("benchmarks") / "latency_targets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    measured = report["measured_host"]
    if measured:
        print(
            f"measured on {measured['label']}: p50 {measured['median_ms']:.3f} ms, "
            f"p99 {measured['p99_ms']:.3f} ms"
        )
    print(
        f"\nESTIMATES ONLY, no board was measured. "
        f"{report['artefact']['macs']:,} MACs, "
        f"{report['artefact']['model_bytes'] / 1024:.0f} KiB int8 model.\n"
    )
    for row in report["estimates"]:
        verdict = "meets" if row["estimated_to_meet_budget"] else "MISSES"
        print(
            f"  {row['target']:44s} {row['estimated_inference_ms']:7.2f} ms inference, "
            f"{row['estimated_total_ms']:7.1f} ms total, {verdict} the budget"
        )
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

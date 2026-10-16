"""Write the on-target latency and memory table.

Reads the exported artefacts, the measured host benchmark and the emulated
firmware report, then applies :mod:`src.embedded.estimate` to produce a
per-board figure.

Every row it writes carries ``"measured": false``. No STM32 or Jetson board
was used for this work. The latency columns are estimates: arithmetic from the
MAC count for every board, plus for the Cortex-M4 a cycle count from the
emulated firmware, which is an emulator estimate. The RAM column comes from
that firmware build (what the linker placed in RAM, including the arena the
TFLite Micro allocator reported), which is exact for the build and not an
estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import Config, load_config, resolve
from src.embedded.estimate import estimate_all


def _artefact_facts(config: Config) -> dict[str, object]:
    """Model size and MAC count, read from what was exported."""
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
        "all_tensors_bytes": int(tflite_int8["all_tensors_bytes"]),
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


EMULATOR_REPORT = "renode_cortex_m4.json"
EMULATED_TARGET_PREFIX = "STM32F411"


def _emulator() -> dict[str, object]:
    path = resolve("benchmarks") / EMULATOR_REPORT
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build and run the firmware under Renode: "
            "bash firmware/scripts/build_and_run.sh inside the firmware image "
            "(docker/firmware.Dockerfile), then copy build/firmware/"
            f"{EMULATOR_REPORT} to benchmarks/."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run(config: Config) -> dict[str, object]:
    facts = _artefact_facts(config)
    emulator = _emulator()
    vote_ms = (config.evaluation.vote_window - 1) * config.window.stride_ms
    memory = emulator["memory"]
    ram_source = (
        "RAM placed by the linker for the Renode firmware build: statics, a "
        f"{memory['arena_reserved_bytes']:,}-byte tensor arena "
        f"({memory['arena_used_bytes']:,} bytes used), heap and stack"
    )

    rows = estimate_all(
        macs=facts["macs"],
        model_bytes=facts["model_bytes"],
        ram_bytes=int(memory["ram_used_bytes"]),
        ram_source=ram_source,
        window_ms=config.window.length_ms,
        vote_ms=vote_ms,
        io_ms=config.latency.io_allowance_ms,
        budget_ms=config.latency.budget_ms,
    )

    cycles = emulator["cycles"]
    for row in rows:
        if row["target"].startswith(EMULATED_TARGET_PREFIX):
            inference = cycles["mean"] / (row["clock_mhz"] * 1e3)
            total = row["window_fill_ms"] + row["majority_vote_ms"] + inference
            total += row["io_allowance_ms"]
            row["emulator"] = {
                "label": "emulator estimate",
                "source": f"benchmarks/{EMULATOR_REPORT}",
                "cycles_mean": cycles["mean"],
                "estimated_inference_ms": inference,
                "estimated_total_ms": total,
                "estimated_to_meet_budget": bool(total <= row["budget_ms"]),
            }

    return {
        "synthetic": config.dataset.source == "synthetic",
        "disclaimer": (
            "Estimates, not measurements. No STM32 or Jetson board was used. "
            "Latency is arithmetic from the MAC count and a documented throughput "
            "assumption (src/embedded/estimate.py), plus for the STM32F411 an "
            "emulator estimate from Renode. RAM is from the emulated firmware "
            "build. See benchmarks/README.md."
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
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

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
        emulated = row.get("emulator")
        if emulated:
            verdict = "meets" if emulated["estimated_to_meet_budget"] else "MISSES"
            print(
                f"  {'  emulator estimate (Renode)':44s} "
                f"{emulated['estimated_inference_ms']:7.2f} ms inference, "
                f"{emulated['estimated_total_ms']:7.1f} ms total, {verdict} the budget"
            )
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Latency measurement.

What is measured, and what is not:

**Measured.** The wall-clock cost of classifying one window, on the machine
running this script, single-threaded, batch size one. Reported as a
distribution rather than a mean, because a control loop is not judged on its
average: it is judged on whether it ever stalls. The 99th percentile is the
number that matters.

**Not measured.** Anything on an STM32 or a Jetson. No such board was
available for this work. ``python -m src.embedded.benchmark --artefact ...``
run on a board produces the same JSON in the same format, and
``benchmarks/README.md`` says how. Until somebody does that, the on-target
figures in this repository are arithmetic from the operation count, they are
labelled as estimates, and they are not results.

Timing uses ``time.perf_counter_ns`` per iteration rather than dividing a
total, so the tails survive. A garbage collection pause during one inference
is real latency the control loop would see, so the collector is left running.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import Config, load_config, portable_path, resolve
from src.embedded.runtime import InferenceRuntime, load_runtime


@dataclass
class LatencyStats:
    artefact: str
    backend: str
    size_bytes: int
    iterations: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    host: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def host_description() -> dict[str, object]:
    """Enough about the machine that a reader knows what the numbers mean."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def measure(
    runtime: InferenceRuntime,
    windows: np.ndarray,
    iterations: int = 2000,
    warmup: int = 200,
) -> LatencyStats:
    """Time ``iterations`` single-window inferences.

    Windows are cycled rather than reused, so the measurement includes the
    cost of touching cold input data. Reusing one window would leave it in L1
    and flatter the result.
    """
    if windows.ndim != 3 or windows.shape[0] == 0:
        raise ValueError(
            f"need a non-empty (n, n_channels, n_samples) array, got {windows.shape}"
        )
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")

    pool = np.ascontiguousarray(windows, dtype=np.float32)
    n = pool.shape[0]

    for i in range(warmup):
        runtime.infer(pool[i % n])

    samples_ns: list[int] = []
    for i in range(iterations):
        window = pool[i % n]
        start = time.perf_counter_ns()
        runtime.infer(window)
        samples_ns.append(time.perf_counter_ns() - start)

    ms = sorted(s / 1e6 for s in samples_ns)
    return LatencyStats(
        artefact=portable_path(runtime.path),
        backend=runtime.name.split(":")[0],
        size_bytes=runtime.size_bytes,
        iterations=iterations,
        mean_ms=float(statistics.fmean(ms)),
        median_ms=float(statistics.median(ms)),
        p90_ms=float(ms[min(len(ms) - 1, int(0.90 * len(ms)))]),
        p99_ms=float(ms[min(len(ms) - 1, int(0.99 * len(ms)))]),
        min_ms=float(ms[0]),
        max_ms=float(ms[-1]),
        stdev_ms=float(statistics.stdev(ms)) if len(ms) > 1 else 0.0,
        host=host_description(),
    )


def budget_check(stats: LatencyStats, config: Config) -> dict[str, object]:
    """Set the measured inference cost against the control-loop deadline.

    The deadline is not the inference time. It is everything between the
    wearer's muscle contracting and the actuator moving, and inference is the
    smallest term in it. The window has to fill first, and the majority vote
    has to see enough decisions to agree. Those are counted here so the
    headline is the number the wearer would feel, not the flattering one.
    """
    window_ms = config.window.length_ms
    vote_ms = (config.evaluation.vote_window - 1) * config.window.stride_ms
    io_ms = config.latency.io_allowance_ms
    total = window_ms + vote_ms + stats.p99_ms + io_ms
    return {
        "window_fill_ms": window_ms,
        "majority_vote_ms": vote_ms,
        "inference_p99_ms": stats.p99_ms,
        "io_allowance_ms": io_ms,
        "total_ms": total,
        "budget_ms": config.latency.budget_ms,
        "headroom_ms": config.latency.budget_ms - total,
        "meets_budget": bool(total <= config.latency.budget_ms),
        "max_decision_rate_hz": 1000.0 / stats.p99_ms if stats.p99_ms > 0 else float("inf"),
        "required_decision_rate_hz": 1000.0 / config.window.stride_ms,
        "keeps_up_with_stride": bool(stats.p99_ms <= config.window.stride_ms),
    }


def benchmark_artefacts(
    config: Config,
    artefacts: list[Path],
    windows: np.ndarray,
    num_threads: int = 1,
) -> list[dict[str, object]]:
    rows = []
    for path in artefacts:
        runtime = load_runtime(path, num_threads=num_threads)
        try:
            stats = measure(
                runtime,
                windows,
                iterations=config.latency.bench_iterations,
                warmup=config.latency.bench_warmup,
            )
        finally:
            runtime.close()
        rows.append({**stats.to_dict(), "budget": budget_check(stats, config)})
        print(
            f"  {path.name:34s} p50 {stats.median_ms:7.3f} ms  p99 {stats.p99_ms:7.3f} ms  "
            f"{stats.size_bytes / 1024:8.1f} KiB"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--artefact",
        nargs="*",
        default=None,
        help="paths to .tflite or .onnx files (default: everything under artifacts/deploy)",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--label",
        default="host",
        help="name for the machine these numbers came from, e.g. 'jetson-nano'",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    deploy = resolve(config.artifacts_dir) / "deploy" / tag

    if args.artefact:
        artefacts = [Path(a) for a in args.artefact]
    else:
        artefacts = sorted(
            [*deploy.glob("*.tflite"), *deploy.glob("*.onnx")],
            key=lambda p: (p.suffix, p.name),
        )
    if not artefacts:
        print(f"no artefacts found under {deploy}; run `make export` first")
        return 1

    windows_path = deploy / "representative_windows.npy"
    if not windows_path.exists():
        print(f"missing {windows_path}; run `make export` first")
        return 1
    windows = np.load(windows_path)

    print(f"benchmarking {len(artefacts)} artefact(s), {args.threads} thread(s)")
    rows = benchmark_artefacts(config, artefacts, windows, num_threads=args.threads)

    out = Path(args.out) if args.out else resolve("benchmarks") / f"latency_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "label": args.label,
                "threads": args.threads,
                "synthetic": config.dataset.source == "synthetic",
                "host": host_description(),
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

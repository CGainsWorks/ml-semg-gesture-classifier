"""The controller as it would actually run, and what that costs.

Every accuracy figure produced offline in this repository is measured on
windows cut from a signal filtered with ``scipy.signal.sosfiltfilt``. That
filter runs forwards and then backwards, so it uses samples from the future.
A prosthesis has no future samples. If the offline numbers are going to mean
anything about the deployed system, somebody has to check what happens when
the filter is made causal, and that is what this module does.

It streams the recording block by block, exactly as an acquisition board would
deliver it:

1. Filter the new block with a one-directional Butterworth, carrying the
   filter state across blocks.
2. Push it into a ring buffer.
3. Once the buffer holds a full window and a stride has elapsed, scale it,
   run the quantised model, and take a decision.
4. Feed that decision to a majority vote before the arm is allowed to move.

The reported accuracy is therefore end to end: causal filtering, int8
arithmetic, majority voting, all of it. It is the number a wearer would
experience, and it is the one worth arguing about.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

from src.config import Config, load_config, resolve
from src.data.dataset import build_dataset, class_codes, load_raw_recording
from src.data.gestures import gesture_names
from src.embedded.benchmark import portable_path
from src.embedded.runtime import InferenceRuntime, load_runtime, softmax
from src.eval.metrics import evaluate
from src.preprocessing.filters import bandpass_sos
from src.preprocessing.normalise import ChannelScaler


@dataclass
class Decision:
    """One classifier output, before and after the vote."""

    sample_index: int
    raw_class: int
    voted_class: int
    confidence: float
    true_class: int
    inference_ms: float


@dataclass
class LoopResult:
    decisions: list[Decision] = field(default_factory=list)
    block_samples: int = 0
    dropped_incomplete: int = 0

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = np.array([d.raw_class for d in self.decisions], dtype=np.int64)
        voted = np.array([d.voted_class for d in self.decisions], dtype=np.int64)
        true = np.array([d.true_class for d in self.decisions], dtype=np.int64)
        latency = np.array([d.inference_ms for d in self.decisions], dtype=np.float64)
        return raw, voted, true, latency


class StreamingController:
    """Causal filter, ring buffer, model and vote, driven one block at a time."""

    def __init__(
        self,
        runtime: InferenceRuntime,
        scaler: ChannelScaler,
        config: Config,
        n_channels: int,
    ):
        self.runtime = runtime
        self.scaler = scaler
        self.config = config
        self.window = config.window_samples
        self.stride = config.stride_samples
        self.vote = max(1, config.evaluation.vote_window)

        self._sos = bandpass_sos(
            config.preprocessing.bandpass_low_hz,
            config.preprocessing.bandpass_high_hz,
            config.dataset.sample_rate_hz,
            config.preprocessing.bandpass_order,
        )
        self._filter_state = np.zeros((self._sos.shape[0], 2, n_channels))
        self._buffer = np.zeros((self.window, n_channels), dtype=np.float32)
        self._filled = 0
        self._since_decision = 0
        self._recent: list[int] = []

    def push(self, block: np.ndarray) -> tuple[bool, np.ndarray | None, float]:
        """Feed one block. Returns (decided, scores, inference_ms).

        A block shorter than the stride is legal and simply will not trigger a
        decision, which is what a real acquisition board's jitter looks like.
        """
        filtered, self._filter_state = sp_signal.sosfilt(
            self._sos, block, axis=0, zi=self._filter_state
        )
        filtered = filtered.astype(np.float32)

        n = filtered.shape[0]
        if n >= self.window:
            self._buffer[:] = filtered[-self.window :]
            self._filled = self.window
        else:
            self._buffer[:-n] = self._buffer[n:]
            self._buffer[-n:] = filtered
            self._filled = min(self.window, self._filled + n)
        self._since_decision += n

        if self._filled < self.window or self._since_decision < self.stride:
            return False, None, 0.0
        self._since_decision = 0

        # (window, channels) -> (channels, window), the layout both exported
        # graphs take.
        window = np.ascontiguousarray(self._buffer.T)[None, ...]
        scaled = self.scaler.transform(window)[0]

        start = time.perf_counter_ns()
        scores = self.runtime.infer(scaled)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        return True, scores, elapsed_ms

    def vote_on(self, class_index: int) -> int:
        self._recent.append(int(class_index))
        if len(self._recent) > self.vote:
            self._recent.pop(0)
        return int(np.bincount(self._recent).argmax())


def run_stream(
    runtime: InferenceRuntime,
    scaler: ChannelScaler,
    config: Config,
    emg: np.ndarray,
    labels: np.ndarray,
    codes: np.ndarray,
    block_samples: int | None = None,
) -> LoopResult:
    """Stream a whole recording through the controller.

    Decisions taken while the buffer straddles a label change are discarded,
    matching the offline convention. Keeping them would measure the
    controller's reaction time, which is a real and interesting quantity, but
    mixing it into an accuracy figure would make that figure mean two things
    at once.
    """
    block_samples = block_samples or config.stride_samples
    if block_samples < 1:
        raise ValueError(f"block size must be positive, got {block_samples}")

    controller = StreamingController(runtime, scaler, config, emg.shape[1])
    result = LoopResult(block_samples=block_samples)
    window = config.window_samples
    code_set = {int(c) for c in codes}

    for start in range(0, emg.shape[0] - block_samples + 1, block_samples):
        stop = start + block_samples
        decided, scores, elapsed_ms = controller.push(emg[start:stop])
        if not decided:
            continue

        span = labels[max(0, stop - window) : stop]
        if span.size < window or np.unique(span).size != 1:
            result.dropped_incomplete += 1
            continue
        true_code = int(span[0])
        if true_code not in code_set:
            result.dropped_incomplete += 1
            continue

        raw_class = int(np.argmax(scores))
        result.decisions.append(
            Decision(
                sample_index=stop,
                raw_class=raw_class,
                voted_class=controller.vote_on(raw_class),
                confidence=float(softmax(scores)[raw_class]),
                true_class=int(np.searchsorted(codes, true_code)),
                inference_ms=elapsed_ms,
            )
        )
    return result


def _deploy_paths(config: Config) -> tuple[Path, Path]:
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    root = resolve(config.artifacts_dir)
    return root / "deploy" / tag, root / "models" / tag


def run(
    config: Config,
    artefact: str | None = None,
    subjects: list[int] | None = None,
) -> dict[str, object]:
    deploy, models = _deploy_paths(config)
    path = Path(artefact) if artefact else deploy / "gesture_cnn_int8.tflite"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make export` first")

    scaler = ChannelScaler.load(models / "channel_scaler.npz")
    index = build_dataset(config)
    codes = class_codes(index)
    names = gesture_names(config.dataset.exercise, [int(c) for c in codes])
    del index

    subjects = subjects or config.dataset.subjects
    runtime = load_runtime(path)
    try:
        per_subject = []
        all_raw, all_voted, all_true, all_latency = [], [], [], []
        for subject in subjects:
            recording = load_raw_recording(config, subject)
            # The raw, unfiltered signal: the controller does its own filtering,
            # causally, which is the entire point of this measurement.
            result = run_stream(
                runtime,
                scaler,
                config,
                recording.emg,
                recording.restimulus,
                codes,
            )
            raw, voted, true, latency = result.arrays()
            if true.size == 0:
                continue
            per_subject.append(
                {
                    "subject": subject,
                    "n_decisions": int(true.size),
                    "raw_accuracy": float((raw == true).mean()),
                    "voted_accuracy": float((voted == true).mean()),
                    "median_inference_ms": float(np.median(latency)),
                }
            )
            all_raw.append(raw)
            all_voted.append(voted)
            all_true.append(true)
            all_latency.append(latency)
            row = per_subject[-1]
            print(
                f"  s{subject:02d}: {true.size:6d} decisions  "
                f"raw {row['raw_accuracy']:.4f}  voted {row['voted_accuracy']:.4f}"
            )
    finally:
        runtime.close()

    if not all_true:
        raise RuntimeError("the stream produced no usable decisions")

    raw = np.concatenate(all_raw)
    voted = np.concatenate(all_voted)
    true = np.concatenate(all_true)
    latency = np.concatenate(all_latency)

    raw_result = evaluate(true, raw, codes, names)
    voted_result = evaluate(true, voted, codes, names)
    vote_ms = (config.evaluation.vote_window - 1) * config.window.stride_ms
    p99 = float(np.percentile(latency, 99))
    total_ms = config.window.length_ms + vote_ms + p99 + config.latency.io_allowance_ms

    return {
        "artefact": portable_path(path),
        "synthetic": config.dataset.source == "synthetic",
        "subjects": subjects,
        "causal_filter": True,
        "n_decisions": int(true.size),
        "single_window": raw_result.to_dict(),
        "majority_vote": voted_result.to_dict(),
        "vote_window": config.evaluation.vote_window,
        "latency": {
            "median_inference_ms": float(np.median(latency)),
            "p99_inference_ms": p99,
            "window_fill_ms": config.window.length_ms,
            "majority_vote_ms": vote_ms,
            "io_allowance_ms": config.latency.io_allowance_ms,
            "total_ms": total_ms,
            "budget_ms": config.latency.budget_ms,
            "headroom_ms": config.latency.budget_ms - total_ms,
            "meets_budget": bool(total_ms <= config.latency.budget_ms),
        },
        "per_subject": per_subject,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--artefact", default=None)
    parser.add_argument("--subjects", type=int, nargs="*", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print("streaming the recordings through the deployed model (causal filter)")
    report = run(config, artefact=args.artefact, subjects=args.subjects)

    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    out = (
        Path(args.out)
        if args.out
        else resolve("benchmarks") / f"control_loop_{tag}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    latency = report["latency"]
    print(
        f"\nsingle window {report['single_window']['accuracy']:.4f}  "
        f"majority vote {report['majority_vote']['accuracy']:.4f}"
    )
    print(
        f"end-to-end {latency['total_ms']:.1f} ms against a {latency['budget_ms']:.0f} ms "
        f"budget ({'meets' if latency['meets_budget'] else 'MISSES'} it, "
        f"{latency['headroom_ms']:.1f} ms headroom)"
    )
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

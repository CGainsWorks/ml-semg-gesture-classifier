"""Figures that come from the signal rather than from a results file.

The README opens with the signal figure this writes. A wall of text about
bandpass filters is less use than one picture of what a gesture looks like
before and after preprocessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import Config, load_config, resolve
from src.data.dataset import load_raw_recording
from src.eval import plots
from src.preprocessing.filters import envelope, preprocess


def figures_dir(config: Config) -> Path:
    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    return resolve("analysis") / tag


def signal_figure(
    config: Config, subject: int = 1, channel: int = 0, seconds: float = 30.0
) -> Path:
    """Raw, bandpassed and enveloped traces of one channel, with the labels.

    Thirty seconds by default: long enough to show several gestures with the
    rest periods between them, short enough that individual bursts are still
    visible.
    """
    recording = load_raw_recording(config, subject)
    n = min(recording.emg.shape[0], int(seconds * recording.sample_rate_hz))

    # Start at the first movement rather than at sample zero, which is rest.
    movement = np.flatnonzero(recording.restimulus > 0)
    start = max(0, int(movement[0]) - recording.sample_rate_hz) if movement.size else 0
    stop = min(recording.emg.shape[0], start + n)

    raw = recording.emg[start:stop]
    filtered = preprocess(
        raw,
        sample_rate_hz=recording.sample_rate_hz,
        bandpass_low_hz=config.preprocessing.bandpass_low_hz,
        bandpass_high_hz=config.preprocessing.bandpass_high_hz,
        bandpass_order=config.preprocessing.bandpass_order,
        notch_hz=config.preprocessing.notch_hz,
    )
    linear_envelope = envelope(filtered, recording.sample_rate_hz, cutoff_hz=5.0)

    return plots.signal_example(
        raw,
        filtered,
        linear_envelope,
        recording.restimulus[start:stop],
        recording.sample_rate_hz,
        figures_dir(config) / "signal_example.png",
        channel=channel,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    path = signal_figure(
        config, subject=args.subject, channel=args.channel, seconds=args.seconds
    )
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

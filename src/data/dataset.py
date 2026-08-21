"""Assemble the windowed dataset, with a cache.

Filtering ten subjects takes long enough to be annoying when iterating on
models, so the filtered signal and the window index are cached per subject
under ``artifacts/cache``. The cache key covers everything that changes the
contents, so a config edit invalidates it rather than silently reusing stale
arrays.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from src.config import Config, resolve
from src.data import synthetic
from src.data.ninapro import Recording, load_recording
from src.data.windows import (
    WindowIndex,
    balance_rest,
    build_window_index,
    concatenate,
    filter_gestures,
)
from src.preprocessing.filters import preprocess


def cache_key(config: Config, subject: int) -> str:
    """Digest of every setting that changes what lands in the cache."""
    payload = {
        "source": config.dataset.source,
        "exercise": config.dataset.exercise,
        "subject": subject,
        "sample_rate_hz": config.dataset.sample_rate_hz,
        "preprocessing": config.preprocessing.__dict__,
        "window_samples": config.window_samples,
        "stride_samples": config.stride_samples,
        "drop_mixed_label": config.window.drop_mixed_label,
        # Gesture filtering and rest balancing happen after the cache is read,
        # so they are deliberately excluded from the key.
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_raw_recording(config: Config, subject: int) -> Recording:
    """Read one subject from whichever source the config names."""
    source = config.dataset.source
    if source == "ninapro_db5":
        return load_recording(
            resolve(config.dataset.raw_dir), subject, config.dataset.exercise
        )
    if source == "synthetic":
        return synthetic.generate_recording(
            subject=subject,
            exercise=config.dataset.exercise,
            gestures=config.dataset.gestures,
            sample_rate_hz=config.dataset.sample_rate_hz,
            seed=config.window.seed,
        )
    raise ValueError(
        f"unknown dataset source {source!r}; expected 'ninapro_db5' or 'synthetic'"
    )


def build_subject(config: Config, subject: int, use_cache: bool = True) -> WindowIndex:
    """Filtered signal plus window index for one subject."""
    cache_dir = resolve(config.dataset.cache_dir)
    path = cache_dir / f"s{subject}_{cache_key(config, subject)}.npz"

    if use_cache and path.exists():
        with np.load(path) as blob:
            return WindowIndex(
                signal=blob["signal"],
                starts=blob["starts"],
                labels=blob["labels"],
                repetitions=blob["repetitions"],
                subjects=blob["subjects"],
                length=int(blob["length"]),
                sample_rate_hz=int(blob["sample_rate_hz"]),
            )

    recording = load_raw_recording(config, subject)
    if recording.sample_rate_hz != config.dataset.sample_rate_hz:
        raise ValueError(
            f"subject {subject} was recorded at {recording.sample_rate_hz} Hz but the "
            f"config says {config.dataset.sample_rate_hz} Hz"
        )

    filtered = preprocess(
        recording.emg,
        sample_rate_hz=recording.sample_rate_hz,
        bandpass_low_hz=config.preprocessing.bandpass_low_hz,
        bandpass_high_hz=config.preprocessing.bandpass_high_hz,
        bandpass_order=config.preprocessing.bandpass_order,
        notch_hz=config.preprocessing.notch_hz,
        notch_quality=config.preprocessing.notch_quality,
    )
    index = build_window_index(
        recording,
        filtered,
        length=config.window_samples,
        stride=config.stride_samples,
        drop_mixed_label=config.window.drop_mixed_label,
    )

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            signal=index.signal,
            starts=index.starts,
            labels=index.labels,
            repetitions=index.repetitions,
            subjects=index.subjects,
            length=index.length,
            sample_rate_hz=index.sample_rate_hz,
        )
    return index


def build_dataset(
    config: Config, subjects: list[int] | None = None, use_cache: bool = True
) -> WindowIndex:
    """Every requested subject, gesture-filtered and rest-balanced, in one index."""
    subjects = subjects if subjects is not None else config.dataset.subjects
    if not subjects:
        raise ValueError("no subjects requested")

    per_subject = []
    for subject in subjects:
        index = build_subject(config, subject, use_cache=use_cache)
        index = filter_gestures(index, config.dataset.gestures)
        per_subject.append(index)

    combined = concatenate(per_subject)
    return balance_rest(combined, config.window.rest_ratio, config.window.seed)


def class_codes(index: WindowIndex) -> np.ndarray:
    """Sorted restimulus codes present, which fixes the model's output order."""
    return np.unique(index.labels)


def encode_labels(labels: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Map restimulus codes onto contiguous class indices.

    ``np.searchsorted`` on a sorted code list rather than a dictionary lookup,
    and the result is checked, so an unseen code fails loudly instead of being
    mapped to a neighbouring class.
    """
    idx = np.searchsorted(codes, labels)
    idx = np.clip(idx, 0, codes.size - 1)
    if not np.array_equal(codes[idx], labels):
        unseen = sorted(set(np.asarray(labels).tolist()) - set(codes.tolist()))
        raise ValueError(f"labels contain codes not in the class list: {unseen}")
    return idx.astype(np.int64)


def summarise(index: WindowIndex) -> dict[str, object]:
    """Counts used by the dataset card and by the report headers."""
    codes, counts = np.unique(index.labels, return_counts=True)
    return {
        "n_windows": len(index),
        "n_channels": index.n_channels,
        "window_samples": index.length,
        "sample_rate_hz": index.sample_rate_hz,
        "subjects": sorted(int(s) for s in np.unique(index.subjects)),
        "repetitions": sorted(int(r) for r in np.unique(index.repetitions)),
        "class_counts": {int(c): int(n) for c, n in zip(codes, counts)},
    }

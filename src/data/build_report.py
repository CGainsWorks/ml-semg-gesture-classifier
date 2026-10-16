"""Regenerate the dataset statistics quoted in the dataset card.

Exists so that every number in ``data/DATASET_CARD.md`` comes from a script
rather than from something somebody counted once and never checked again.
Writes ``data/dataset_stats.json`` and prints a summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import Config, load_config, resolve
from src.data.dataset import build_dataset, load_raw_recording, summarise
from src.data.gestures import gesture_names
from src.data.ninapro import sha256, subject_zip_path


def subject_metadata(config: Config) -> list[dict[str, object]]:
    """Per-subject demographics and recording length, read from the files."""
    rows = []
    for subject in config.dataset.subjects:
        recording = load_raw_recording(config, subject)
        metadata = recording.metadata
        rows.append(
            {
                "subject": subject,
                "age": metadata.get("age"),
                "gender": metadata.get("gender"),
                "height_cm": metadata.get("height_cm"),
                "weight_kg": metadata.get("weight_kg"),
                "forearm_circumference_cm": metadata.get("forearm_circumference_cm"),
                "laterality": metadata.get("laterality"),
                "sensor": metadata.get("sensor"),
                "duration_s": round(recording.duration_s, 1),
                "n_samples": int(recording.emg.shape[0]),
                "exercise_field_in_file": metadata.get("exercise_field_in_file"),
            }
        )
    return rows


def archive_digests(config: Config) -> dict[str, str]:
    if config.dataset.source != "ninapro_db5":
        return {}
    raw_dir = resolve(config.dataset.raw_dir)
    out = {}
    for subject in config.dataset.subjects:
        path = subject_zip_path(raw_dir, subject)
        if path.exists():
            out[path.name] = sha256(path)
    return out


def build(config: Config) -> dict[str, object]:
    index = build_dataset(config)
    summary = summarise(index)
    codes = sorted(summary["class_counts"])
    names = gesture_names(config.dataset.exercise, codes)

    subjects = subject_metadata(config)
    ages = [s["age"] for s in subjects if isinstance(s["age"], int | float)]
    circumferences = [
        s["forearm_circumference_cm"]
        for s in subjects
        if isinstance(s["forearm_circumference_cm"], int | float)
    ]
    genders: dict[str, int] = {}
    for s in subjects:
        key = str(s["gender"])
        genders[key] = genders.get(key, 0) + 1

    counts = np.array([summary["class_counts"][c] for c in codes])
    return {
        "source": config.dataset.source,
        "exercise": config.dataset.exercise,
        "window_ms": config.window.length_ms,
        "stride_ms": config.window.stride_ms,
        "rest_ratio": config.window.rest_ratio,
        "windows": summary["n_windows"],
        "n_channels": summary["n_channels"],
        "window_samples": summary["window_samples"],
        "sample_rate_hz": summary["sample_rate_hz"],
        "n_classes": len(codes),
        "class_counts": {
            str(code): {"name": name, "windows": summary["class_counts"][code]}
            for code, name in zip(codes, names)
        },
        "class_count_min": int(counts.min()),
        "class_count_max": int(counts.max()),
        "subjects": subjects,
        "age_range": [min(ages), max(ages)] if ages else None,
        "forearm_circumference_range_cm": (
            [min(circumferences), max(circumferences)] if circumferences else None
        ),
        "gender_counts": genders,
        "total_recording_hours": round(
            sum(s["duration_s"] for s in subjects) / 3600.0, 2
        ),
        "archive_sha256": archive_digests(config),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="data/dataset_stats.json")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    stats = build(config)

    out = Path(args.out)
    if not out.is_absolute():
        out = resolve(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8", newline="\n")

    print(f"source            {stats['source']} exercise {stats['exercise']}")
    print(f"subjects          {len(stats['subjects'])}")
    print(f"channels          {stats['n_channels']} at {stats['sample_rate_hz']} Hz")
    print(
        f"window            {stats['window_samples']} samples "
        f"({stats['window_ms']:.0f} ms), stride {stats['stride_ms']:.0f} ms"
    )
    print(f"windows           {stats['windows']:,}")
    print(f"classes           {stats['n_classes']}")
    print(
        f"per class         {stats['class_count_min']:,} to {stats['class_count_max']:,}"
    )
    if stats["age_range"]:
        print(f"ages              {stats['age_range'][0]} to {stats['age_range'][1]}")
    print(f"gender            {stats['gender_counts']}")
    print(f"recording         {stats['total_recording_hours']} hours")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

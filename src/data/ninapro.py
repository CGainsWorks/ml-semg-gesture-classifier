"""Ninapro DB5 loader.

DB5 is the double-Myo-armband database: 16 channels at 200 Hz, 10 intact
subjects, three exercises of six repetitions each. It is distributed as one
zip per subject containing three MATLAB v5 files.

Two things about the distribution are worth knowing before trusting anything
read out of it, and both are handled here:

1. The ``exercise`` field inside a DB5 ``.mat`` file does not match the
   exercise the file actually contains. ``S1_E1_A1.mat`` reports
   ``exercise == 3``. The filename is authoritative and the field is ignored.
2. ``rerepetition`` is 0 throughout every rest period, so rest cannot be split
   by repetition as it stands. :func:`assign_rest_repetitions` gives each rest
   segment the repetition index of the movement it leads into, which keeps a
   repetition-wise train/test split honest for the rest class too.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

DB5_BASE_URL = "https://ninapro.hevs.ch/files/DB5_Preproc"
DB5_SUBJECTS = list(range(1, 11))
DB5_CHANNELS = 16
DB5_SAMPLE_RATE_HZ = 200
DB5_EXERCISES = ("E1", "E2", "E3")

# Ninapro DB5 records the raw Myo output as signed 8-bit counts stored in a
# float array. Dividing by this brings channels into roughly [-1, 1] before
# any per-channel standardisation, which keeps the filter numerics sane.
MYO_FULL_SCALE = 128.0


@dataclass
class Recording:
    """One subject's continuous recording for one exercise."""

    subject: int
    exercise: str
    emg: np.ndarray  # (n_samples, n_channels) float32
    restimulus: np.ndarray  # (n_samples,) int16, 0 = rest
    repetition: np.ndarray  # (n_samples,) int16, rest already assigned
    sample_rate_hz: int
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        n = self.emg.shape[0]
        if self.restimulus.shape != (n,) or self.repetition.shape != (n,):
            raise ValueError(
                "emg, restimulus and repetition must agree on length: "
                f"{self.emg.shape=} {self.restimulus.shape=} {self.repetition.shape=}"
            )

    @property
    def n_channels(self) -> int:
        return int(self.emg.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.emg.shape[0]) / self.sample_rate_hz


def subject_zip_path(raw_dir: Path, subject: int) -> Path:
    return Path(raw_dir) / f"s{subject}.zip"


def subject_zip_url(subject: int) -> str:
    return f"{DB5_BASE_URL}/s{subject}.zip"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def assign_rest_repetitions(
    restimulus: np.ndarray, rerepetition: np.ndarray
) -> np.ndarray:
    """Give every rest sample the repetition index of a real movement block.

    Ninapro leaves ``rerepetition`` at 0 during rest, which would put every
    rest window into whichever side of a repetition split happens to catch
    repetition 0. Each contiguous rest segment is instead attributed to the
    movement repetition that follows it, or to the one before it when the
    recording ends on rest.

    Returns a copy; the input is not modified.
    """
    restimulus = np.asarray(restimulus).reshape(-1)
    rerepetition = np.asarray(rerepetition).reshape(-1)
    out = rerepetition.astype(np.int16, copy=True)

    is_rest = restimulus == 0
    if not is_rest.any():
        return out

    # Boundaries of contiguous runs of rest.
    padded = np.concatenate(([False], is_rest, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = edges[0::2], edges[1::2]

    # Repetition of the movement block on each side of every rest run.
    for start, stop in zip(starts, stops):
        after = out[stop] if stop < out.size else 0
        before = out[start - 1] if start > 0 else 0
        rep = after if after > 0 else before
        out[start:stop] = rep

    # A recording that is entirely rest, or rest before any movement, leaves
    # zeros behind. Those samples belong to no repetition and are dropped
    # downstream rather than being silently folded into repetition 1.
    return out


def _scalar(value: object, default: object = None) -> object:
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    flat = arr.reshape(-1)[0]
    if isinstance(flat, np.generic):
        return flat.item()
    return flat


def parse_mat(
    data: bytes, subject: int, exercise: str, source: str
) -> Recording:
    """Turn the bytes of a DB5 ``.mat`` file into a :class:`Recording`."""
    mat = sio.loadmat(io.BytesIO(data))
    missing = {"emg", "restimulus", "rerepetition"} - set(mat)
    if missing:
        raise ValueError(f"{source}: missing field(s) {sorted(missing)}")

    emg = np.asarray(mat["emg"], dtype=np.float32)
    if emg.ndim != 2:
        raise ValueError(f"{source}: expected a 2-D emg array, got {emg.shape}")
    if emg.shape[1] != DB5_CHANNELS:
        raise ValueError(
            f"{source}: expected {DB5_CHANNELS} channels, got {emg.shape[1]}"
        )
    emg = emg / MYO_FULL_SCALE

    restimulus = np.asarray(mat["restimulus"], dtype=np.int16).reshape(-1)
    rerepetition = np.asarray(mat["rerepetition"], dtype=np.int16).reshape(-1)
    repetition = assign_rest_repetitions(restimulus, rerepetition)

    frequency = _scalar(mat.get("frequency"), DB5_SAMPLE_RATE_HZ)
    metadata = {
        "source_file": source,
        # The in-file `exercise` field is deliberately not used; see the module
        # docstring. It is recorded so the discrepancy stays visible.
        "exercise_field_in_file": _scalar(mat.get("exercise")),
        "age": _scalar(mat.get("age")),
        "gender": _scalar(mat.get("gender")),
        "height_cm": _scalar(mat.get("height")),
        "weight_kg": _scalar(mat.get("weight")),
        "forearm_circumference_cm": _scalar(mat.get("circumference")),
        "laterality": _scalar(mat.get("laterality")),
        "sensor": _scalar(mat.get("sensor")),
        "n_samples": int(emg.shape[0]),
    }

    return Recording(
        subject=subject,
        exercise=exercise,
        emg=emg,
        restimulus=restimulus,
        repetition=repetition,
        sample_rate_hz=int(frequency),
        metadata=metadata,
    )


def load_recording(raw_dir: str | Path, subject: int, exercise: str) -> Recording:
    """Read one subject-exercise pair straight out of the distributed zip.

    The zips are not unpacked. Keeping them sealed means the checksums in the
    dataset card stay checkable against exactly what was downloaded.
    """
    if exercise not in DB5_EXERCISES:
        raise ValueError(f"unknown exercise {exercise!r}; expected one of {DB5_EXERCISES}")
    path = subject_zip_path(Path(raw_dir), subject)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` (or python -m src.data.download) "
            f"to fetch Ninapro DB5 from {subject_zip_url(subject)}"
        )

    member = f"s{subject}/S{subject}_{exercise}_A1.mat"
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        if member not in names:
            raise FileNotFoundError(
                f"{path} does not contain {member}; it holds {sorted(names)}"
            )
        payload = zf.read(member)

    return parse_mat(payload, subject=subject, exercise=exercise, source=member)


def available_subjects(raw_dir: str | Path) -> list[int]:
    """Subjects whose zip is present on disk."""
    return [s for s in DB5_SUBJECTS if subject_zip_path(Path(raw_dir), s).exists()]

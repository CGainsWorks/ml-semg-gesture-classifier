"""Cut a continuous recording into labelled windows.

Windows are represented by their start index rather than materialised, and the
array itself is produced with :func:`numpy.lib.stride_tricks.sliding_window_view`,
which is a view. Ten subjects of DB5 exercise B come to roughly 180 000
overlapping 200 ms windows; materialising them all at once would cost about
half a gigabyte for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.ninapro import Recording


@dataclass
class WindowIndex:
    """Where the windows are and what they contain.

    ``signal`` is the filtered continuous recording. ``starts`` indexes into
    it. The four parallel arrays are all the same length as ``starts``.
    """

    signal: np.ndarray  # (n_samples, n_channels) float32
    starts: np.ndarray  # (n_windows,) int64
    labels: np.ndarray  # (n_windows,) int16, restimulus code
    repetitions: np.ndarray  # (n_windows,) int16
    subjects: np.ndarray  # (n_windows,) int16
    length: int
    sample_rate_hz: int

    def __post_init__(self) -> None:
        n = self.starts.size
        for name in ("labels", "repetitions", "subjects"):
            arr = getattr(self, name)
            if arr.size != n:
                raise ValueError(f"{name} has {arr.size} entries, expected {n}")

    def __len__(self) -> int:
        return int(self.starts.size)

    @property
    def n_channels(self) -> int:
        return int(self.signal.shape[1])

    def materialise(self, order: str = "ct") -> np.ndarray:
        """Build the (n_windows, n_channels, length) tensor of window contents.

        Only one layout, ``"ct"``. Both framework implementations take
        (batch, electrodes, time) and reshape to an image inside their own
        graph, so there is nothing left to transpose here and no opportunity
        for the two of them to be fed differently arranged numbers.
        """
        if order != "ct":
            raise ValueError(
                f"order must be 'ct' (n_windows, n_channels, length), got {order!r}"
            )
        if len(self) == 0:
            return np.empty((0, self.n_channels, self.length), dtype=np.float32)

        view = np.lib.stride_tricks.sliding_window_view(
            self.signal, self.length, axis=0
        )  # (n_valid, n_channels, length)
        return np.ascontiguousarray(view[self.starts], dtype=np.float32)

    def select(self, mask: np.ndarray) -> WindowIndex:
        """A new index over the same signal, keeping only the masked windows."""
        mask = np.asarray(mask)
        if mask.dtype == bool and mask.size != len(self):
            raise ValueError(f"mask has {mask.size} entries, expected {len(self)}")
        return WindowIndex(
            signal=self.signal,
            starts=self.starts[mask],
            labels=self.labels[mask],
            repetitions=self.repetitions[mask],
            subjects=self.subjects[mask],
            length=self.length,
            sample_rate_hz=self.sample_rate_hz,
        )


def _homogeneous_runs(values: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Start indices of every window over which ``values`` never changes.

    Computed from run boundaries rather than by testing each window, which
    keeps it linear in the length of the recording.
    """
    n = values.size
    if n < length:
        return np.empty(0, dtype=np.int64)

    # last_change[i] is the index of the most recent sample where the value
    # differed from the one before it.
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    changed[1:] = values[1:] != values[:-1]
    last_change = np.maximum.accumulate(np.where(changed, np.arange(n), 0))

    starts = np.arange(0, n - length + 1, stride, dtype=np.int64)
    # A window [s, s+length) is homogeneous when nothing changed after s.
    return starts[last_change[starts + length - 1] <= starts]


def build_window_index(
    recording: Recording,
    signal: np.ndarray,
    length: int,
    stride: int,
    drop_mixed_label: bool = True,
) -> WindowIndex:
    """Index every window of ``signal`` that carries a single, valid label.

    Windows are dropped when they straddle a label change, when they straddle
    a repetition change, or when they fall in a rest period that could not be
    attributed to a repetition (see
    :func:`src.data.ninapro.assign_rest_repetitions`).
    """
    if signal.shape[0] != recording.emg.shape[0]:
        raise ValueError(
            f"filtered signal has {signal.shape[0]} samples but the recording has "
            f"{recording.emg.shape[0]}"
        )
    if length <= 0 or stride <= 0:
        raise ValueError(f"length and stride must be positive, got {length=} {stride=}")

    n = signal.shape[0]
    if n < length:
        return WindowIndex(
            signal=signal,
            starts=np.empty(0, dtype=np.int64),
            labels=np.empty(0, dtype=np.int16),
            repetitions=np.empty(0, dtype=np.int16),
            subjects=np.empty(0, dtype=np.int16),
            length=length,
            sample_rate_hz=recording.sample_rate_hz,
        )

    if drop_mixed_label:
        # A window must be homogeneous in the label and in the repetition. The
        # pair is combined into one key so a single run scan covers both.
        key = recording.restimulus.astype(np.int64) * 1000 + recording.repetition.astype(
            np.int64
        )
        starts = _homogeneous_runs(key, length, stride)
    else:
        starts = np.arange(0, n - length + 1, stride, dtype=np.int64)

    labels = recording.restimulus[starts].astype(np.int16)
    repetitions = recording.repetition[starts].astype(np.int16)

    # Repetition 0 survives only where rest could not be attributed to a
    # movement block, which is the head of the recording. Those windows belong
    # to no fold and are dropped rather than being assigned arbitrarily.
    keep = repetitions > 0
    starts, labels, repetitions = starts[keep], labels[keep], repetitions[keep]

    return WindowIndex(
        signal=signal,
        starts=starts,
        labels=labels,
        repetitions=repetitions,
        subjects=np.full(starts.size, recording.subject, dtype=np.int16),
        length=length,
        sample_rate_hz=recording.sample_rate_hz,
    )


def filter_gestures(index: WindowIndex, gestures: list[int] | None) -> WindowIndex:
    """Keep only the requested gesture codes. Rest is always kept."""
    if not gestures:
        return index
    keep_codes = {int(g) for g in gestures} | {0}
    mask = np.isin(index.labels, sorted(keep_codes))
    return index.select(mask)


def balance_rest(index: WindowIndex, ratio: float, seed: int) -> WindowIndex:
    """Subsample rest to ``ratio`` times the mean movement-class count.

    Ninapro alternates a movement with a rest period of comparable length, so
    rest arrives at roughly seventeen times the rate of any single movement in
    exercise B. Left alone, a classifier that always answers rest scores well
    on accuracy while being useless. Subsampling is applied per subject and per
    repetition so no fold is starved of rest.
    """
    if ratio <= 0:
        return index.select(index.labels != 0)

    movement_labels = index.labels[index.labels != 0]
    if movement_labels.size == 0:
        return index
    _, counts = np.unique(movement_labels, return_counts=True)
    target = max(1, int(round(ratio * counts.mean())))

    rng = np.random.default_rng(seed)
    keep = np.zeros(len(index), dtype=bool)
    keep[index.labels != 0] = True

    rest_idx = np.flatnonzero(index.labels == 0)
    if rest_idx.size == 0:
        return index

    # Spread the quota evenly over (subject, repetition) groups so a
    # repetition-wise split does not end up with all the rest on one side.
    groups = (
        index.subjects[rest_idx].astype(np.int64) * 1000
        + index.repetitions[rest_idx].astype(np.int64)
    )
    unique_groups = np.unique(groups)
    per_group = max(1, target // max(1, unique_groups.size))
    for g in unique_groups:
        members = rest_idx[groups == g]
        if members.size <= per_group:
            keep[members] = True
        else:
            keep[rng.choice(members, size=per_group, replace=False)] = True

    return index.select(keep)


def concatenate(indices: list[WindowIndex]) -> WindowIndex:
    """Join per-subject indices into one, offsetting starts into a shared signal.

    The signals are stacked with a gap of ``length`` zero samples between them
    so no window can ever span two subjects.
    """
    indices = [ix for ix in indices if ix is not None]
    if not indices:
        raise ValueError("nothing to concatenate")
    length = indices[0].length
    n_channels = indices[0].n_channels
    rate = indices[0].sample_rate_hz
    for ix in indices:
        if ix.length != length:
            raise ValueError(f"window lengths differ: {ix.length} vs {length}")
        if ix.n_channels != n_channels:
            raise ValueError(f"channel counts differ: {ix.n_channels} vs {n_channels}")
        if ix.sample_rate_hz != rate:
            raise ValueError(f"sample rates differ: {ix.sample_rate_hz} vs {rate}")

    gap = np.zeros((length, n_channels), dtype=np.float32)
    signals: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    offset = 0
    for ix in indices:
        signals.append(ix.signal)
        starts.append(ix.starts + offset)
        offset += ix.signal.shape[0]
        signals.append(gap)
        offset += length

    return WindowIndex(
        signal=np.concatenate(signals[:-1], axis=0),
        starts=np.concatenate(starts),
        labels=np.concatenate([ix.labels for ix in indices]),
        repetitions=np.concatenate([ix.repetitions for ix in indices]),
        subjects=np.concatenate([ix.subjects for ix in indices]),
        length=length,
        sample_rate_hz=rate,
    )

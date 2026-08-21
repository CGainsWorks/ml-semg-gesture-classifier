"""Filtering applied to the continuous signal, before it is cut into windows.

Everything here is a zero-phase (``filtfilt``) filter, which is legitimate
offline but *not* what a real-time controller can do: ``filtfilt`` runs the
filter backwards as well as forwards and so needs samples from the future.
:func:`causal_bandpass` is the one-directional equivalent used by the
embedded runtime, and :mod:`src.embedded.control_loop` checks that swapping to
it does not move the accuracy, so the offline numbers stay meaningful for the
deployed system.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _validate_band(low_hz: float, high_hz: float, sample_rate_hz: float) -> tuple[float, float]:
    nyquist = sample_rate_hz / 2.0
    if not 0 < low_hz < high_hz:
        raise ValueError(f"need 0 < low < high, got low={low_hz} high={high_hz}")
    if high_hz >= nyquist:
        raise ValueError(
            f"bandpass high cut {high_hz} Hz is at or above the Nyquist frequency "
            f"{nyquist} Hz for a {sample_rate_hz} Hz signal"
        )
    return low_hz / nyquist, high_hz / nyquist


def bandpass_sos(
    low_hz: float, high_hz: float, sample_rate_hz: float, order: int = 4
) -> np.ndarray:
    """Second-order sections for a Butterworth bandpass.

    Second-order sections rather than transfer-function coefficients because a
    high-order ``ba`` design is numerically unstable, and this filter is
    exported to the embedded runtime where that would bite silently.
    """
    low, high = _validate_band(low_hz, high_hz, sample_rate_hz)
    return signal.butter(order, [low, high], btype="bandpass", output="sos")


def notch_sos(freq_hz: float, sample_rate_hz: float, quality: float = 30.0) -> np.ndarray:
    """Second-order sections for a mains-frequency notch."""
    nyquist = sample_rate_hz / 2.0
    if not 0 < freq_hz < nyquist:
        raise ValueError(
            f"notch frequency {freq_hz} Hz must sit below the Nyquist frequency {nyquist} Hz"
        )
    b, a = signal.iirnotch(freq_hz / nyquist, quality)
    return signal.tf2sos(b, a)


def apply_sos(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Zero-phase filter along the time axis of an (n_samples, n_channels) array."""
    if x.ndim != 2:
        raise ValueError(f"expected (n_samples, n_channels), got {x.shape}")
    padlen = 3 * (sos.shape[0] * 2)
    if x.shape[0] <= padlen:
        raise ValueError(
            f"signal of {x.shape[0]} samples is too short for this filter "
            f"(needs more than {padlen})"
        )
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)


def preprocess(
    x: np.ndarray,
    sample_rate_hz: float,
    bandpass_low_hz: float = 10.0,
    bandpass_high_hz: float = 99.0,
    bandpass_order: int = 4,
    notch_hz: float | None = None,
    notch_quality: float = 30.0,
) -> np.ndarray:
    """Bandpass, and optionally notch, a continuous multi-channel recording.

    Rectification is deliberately absent. The convolutional models read the
    oscillating signal directly and rectifying throws away the sign structure
    they use; the classical feature set rectifies internally where a feature
    calls for it (mean absolute value, integrated absolute value).
    """
    sos = bandpass_sos(bandpass_low_hz, bandpass_high_hz, sample_rate_hz, bandpass_order)
    out = apply_sos(x, sos)
    if notch_hz is not None:
        out = apply_sos(out, notch_sos(notch_hz, sample_rate_hz, notch_quality))
    return out


def causal_bandpass(
    x: np.ndarray,
    sample_rate_hz: float,
    low_hz: float = 10.0,
    high_hz: float = 99.0,
    order: int = 4,
    state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-directional bandpass, for streaming use.

    Returns the filtered block and the filter state, so a caller can hand the
    state back on the next block and get the same answer it would have got
    from filtering the whole stream at once.
    """
    sos = bandpass_sos(low_hz, high_hz, sample_rate_hz, order)
    if state is None:
        state = np.zeros((sos.shape[0], 2, x.shape[1]), dtype=np.float64)
    out, state = signal.sosfilt(sos, x, axis=0, zi=state)
    return out.astype(np.float32), state


def rectify(x: np.ndarray) -> np.ndarray:
    """Full-wave rectification."""
    return np.abs(x)


def envelope(x: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 5.0) -> np.ndarray:
    """Linear envelope: rectify, then low-pass.

    Used for the README figure rather than by any model. It is what the Otto
    Bock amplifiers behind Ninapro DB1 emit in hardware, and showing it next to
    the raw Myo trace makes the difference between the databases obvious.
    """
    nyquist = sample_rate_hz / 2.0
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"envelope cutoff {cutoff_hz} Hz must sit below {nyquist} Hz")
    sos = signal.butter(2, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return apply_sos(rectify(x), sos)

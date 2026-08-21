"""Classical time-domain sEMG features.

The core four are Hudgins' set (mean absolute value, waveform length, zero
crossings, slope sign changes), which has been the reference feature set for
myoelectric control since 1993 and is still what most commercial prostheses
run on. The rest are the usual additions.

Everything is vectorised across (n_windows, n_channels, n_samples) at once.
A per-window Python loop over 180 000 windows would dominate the runtime of
the whole project.

Three of the features need an amplitude threshold to reject baseline noise.
Fixing that threshold in signal units cannot work across subjects, because
electrode gain varies by a factor of two between people, so the thresholds are
expressed as a multiple of each channel's training-set standard deviation and
that vector is carried alongside the model.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "mav",
    "rms",
    "wl",
    "zc",
    "ssc",
    "wamp",
    "var",
    "ssi",
    "iav",
    "log_detector",
    "mfl",
    "hjorth_mobility",
    "hjorth_complexity",
)


def _check(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"expected (n_windows, n_channels, n_samples), got {x.shape}")
    if x.shape[2] < 3:
        raise ValueError(
            f"windows of {x.shape[2]} samples are too short; slope sign changes and "
            "the Hjorth parameters need at least three"
        )
    return x


def channel_sigma(x: np.ndarray) -> np.ndarray:
    """Per-channel standard deviation over a set of windows.

    Fit this on the training fold only, then pass the result to
    :func:`extract` for every fold.
    """
    x = _check(x)
    sigma = x.std(axis=(0, 2))
    # A dead channel would otherwise make every threshold zero and turn zero
    # crossings into a noise counter.
    return np.maximum(sigma, 1e-6).astype(np.float32)


def mav(x: np.ndarray) -> np.ndarray:
    """Mean absolute value: the amplitude estimator prostheses have used for decades."""
    return np.abs(x).mean(axis=2)


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(x).mean(axis=2))


def waveform_length(x: np.ndarray) -> np.ndarray:
    """Cumulative path length: amplitude and frequency content in one number."""
    return np.abs(np.diff(x, axis=2)).sum(axis=2)


def zero_crossings(x: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Sign changes whose amplitude clears the noise threshold.

    A frequency proxy. The threshold is what stops it counting baseline
    wander, and is why it is expressed relative to each channel's own scale.
    """
    thr = np.asarray(threshold, dtype=np.float32).reshape(1, -1, 1)
    a, b = x[:, :, :-1], x[:, :, 1:]
    crossed = (a * b < 0) & (np.abs(a - b) >= thr)
    return crossed.sum(axis=2).astype(np.float32)


def slope_sign_changes(x: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Turning points, again gated by amplitude. A second frequency proxy."""
    thr = np.asarray(threshold, dtype=np.float32).reshape(1, -1, 1)
    prev, mid, nxt = x[:, :, :-2], x[:, :, 1:-1], x[:, :, 2:]
    d1, d2 = mid - prev, mid - nxt
    turning = (d1 * d2 > 0) & ((np.abs(d1) >= thr) | (np.abs(d2) >= thr))
    return turning.sum(axis=2).astype(np.float32)


def willison_amplitude(x: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Count of sample-to-sample jumps above threshold: a firing-rate proxy."""
    thr = np.asarray(threshold, dtype=np.float32).reshape(1, -1, 1)
    return (np.abs(np.diff(x, axis=2)) >= thr).sum(axis=2).astype(np.float32)


def variance(x: np.ndarray) -> np.ndarray:
    return x.var(axis=2)


def simple_square_integral(x: np.ndarray) -> np.ndarray:
    """Energy of the window."""
    return np.square(x).sum(axis=2)


def integrated_absolute_value(x: np.ndarray) -> np.ndarray:
    return np.abs(x).sum(axis=2)


def log_detector(x: np.ndarray) -> np.ndarray:
    """exp(mean(log|x|)): a geometric-mean amplitude estimate, robust to spikes."""
    return np.exp(np.log(np.abs(x) + 1e-8).mean(axis=2))


def myopulse_free_length(x: np.ndarray) -> np.ndarray:
    """log of waveform length. Compresses the dynamic range for linear models."""
    return np.log(waveform_length(x) + 1e-8)


def hjorth(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hjorth mobility and complexity: spectral shape from time-domain moments."""
    d1 = np.diff(x, axis=2)
    d2 = np.diff(d1, axis=2)
    v0 = x.var(axis=2) + 1e-12
    v1 = d1.var(axis=2) + 1e-12
    v2 = d2.var(axis=2) + 1e-12
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    return mobility, complexity


def ar_coefficients(x: np.ndarray, order: int = 4) -> np.ndarray:
    """Autoregressive coefficients per channel, by the Yule-Walker equations.

    Returns (n_windows, n_channels, order). Solved batched with
    ``numpy.linalg.solve`` over the Toeplitz autocorrelation matrices; a
    per-window Levinson-Durbin in Python would be an order of magnitude
    slower for no gain in accuracy.
    """
    x = _check(x)
    if order < 1:
        raise ValueError(f"AR order must be at least 1, got {order}")
    n_win, n_ch, n = x.shape
    if n <= order:
        raise ValueError(f"window of {n} samples cannot support an order-{order} model")

    centred = x - x.mean(axis=2, keepdims=True)
    # Autocorrelation up to the model order, normalised by lag-0.
    acf = np.empty((n_win, n_ch, order + 1), dtype=np.float64)
    for lag in range(order + 1):
        acf[:, :, lag] = (centred[:, :, lag:] * centred[:, :, : n - lag]).sum(axis=2)
    r0 = acf[:, :, :1]
    acf = acf / np.where(np.abs(r0) < 1e-12, 1.0, r0)

    idx = np.abs(np.arange(order)[:, None] - np.arange(order)[None, :])
    toeplitz = acf[:, :, idx]  # (n_win, n_ch, order, order)
    # Ridge on the diagonal: a near-silent channel gives a singular system.
    toeplitz = toeplitz + 1e-6 * np.eye(order)[None, None, :, :]
    rhs = acf[:, :, 1 : order + 1]

    coeffs = np.linalg.solve(toeplitz, rhs[..., None])[..., 0]
    return coeffs.astype(np.float32)


def extract(
    x: np.ndarray,
    sigma: np.ndarray,
    zc_threshold_sd: float = 0.05,
    ssc_threshold_sd: float = 0.05,
    wamp_threshold_sd: float = 0.10,
    ar_order: int = 4,
) -> np.ndarray:
    """Full feature matrix, (n_windows, n_channels * (13 + ar_order)).

    Channel-major layout: all thirteen scalar features for channel 0, then the
    AR coefficients for channel 0, then channel 1, and so on. Keeping a
    channel's features contiguous makes per-channel ablation a slice.
    """
    x = _check(x)
    sigma = np.asarray(sigma, dtype=np.float32)
    if sigma.shape != (x.shape[1],):
        raise ValueError(
            f"sigma must have one entry per channel: got {sigma.shape}, "
            f"expected ({x.shape[1]},)"
        )

    mobility, complexity = hjorth(x)
    scalar = np.stack(
        [
            mav(x),
            rms(x),
            waveform_length(x),
            zero_crossings(x, sigma * zc_threshold_sd),
            slope_sign_changes(x, sigma * ssc_threshold_sd),
            willison_amplitude(x, sigma * wamp_threshold_sd),
            variance(x),
            simple_square_integral(x),
            integrated_absolute_value(x),
            log_detector(x),
            myopulse_free_length(x),
            mobility,
            complexity,
        ],
        axis=2,
    )  # (n_windows, n_channels, 13)

    parts = [scalar]
    if ar_order > 0:
        parts.append(ar_coefficients(x, ar_order))
    stacked = np.concatenate(parts, axis=2)
    return stacked.reshape(x.shape[0], -1).astype(np.float32)


def feature_names(n_channels: int, ar_order: int = 4) -> list[str]:
    """Column names matching :func:`extract`, for feature-importance output."""
    names = []
    for ch in range(n_channels):
        names += [f"ch{ch:02d}_{f}" for f in FEATURE_NAMES]
        names += [f"ch{ch:02d}_ar{k + 1}" for k in range(ar_order)]
    return names

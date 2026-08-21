"""Per-channel normalisation, fit on training data only.

The scaler is a separate object with an explicit fit step rather than a
function, because the whole point is that the statistics come from the
training fold and nowhere else. Making that a parameter you have to pass makes
it hard to leak by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ChannelScaler:
    """Standardise each channel over all its samples and windows.

    Per channel rather than per sample position: a channel's gain is a
    property of where the electrode sits, and is what differs between people.
    Normalising it away is the cheapest thing that helps a model transfer.
    """

    mean: np.ndarray  # (n_channels,)
    scale: np.ndarray  # (n_channels,)

    @classmethod
    def fit(cls, x: np.ndarray) -> ChannelScaler:
        """Fit on (n_windows, n_channels, n_samples)."""
        if x.ndim != 3:
            raise ValueError(f"expected (n_windows, n_channels, n_samples), got {x.shape}")
        if x.shape[0] == 0:
            raise ValueError("cannot fit a scaler on zero windows")
        mean = x.mean(axis=(0, 2)).astype(np.float32)
        scale = x.std(axis=(0, 2)).astype(np.float32)
        return cls(mean=mean, scale=np.maximum(scale, 1e-6))

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError(f"expected (n_windows, n_channels, n_samples), got {x.shape}")
        if x.shape[1] != self.mean.size:
            raise ValueError(
                f"scaler was fit on {self.mean.size} channels, got {x.shape[1]}"
            )
        return (
            (x - self.mean[None, :, None]) / self.scale[None, :, None]
        ).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path: str | Path) -> ChannelScaler:
        with np.load(path) as blob:
            return cls(mean=blob["mean"], scale=blob["scale"])


@dataclass
class FeatureScaler:
    """Standardise a feature matrix column-wise, for the classical baseline."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> FeatureScaler:
        if x.ndim != 2:
            raise ValueError(f"expected (n_windows, n_features), got {x.shape}")
        if x.shape[0] == 0:
            raise ValueError("cannot fit a scaler on zero windows")
        mean = x.mean(axis=0).astype(np.float32)
        scale = x.std(axis=0).astype(np.float32)
        # A constant feature contributes nothing; keeping its scale at 1 leaves
        # it at zero after centring rather than producing infinities.
        return cls(mean=mean, scale=np.where(scale < 1e-6, 1.0, scale).astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"expected (n_windows, n_features), got {x.shape}")
        if x.shape[1] != self.mean.size:
            raise ValueError(
                f"scaler was fit on {self.mean.size} features, got {x.shape[1]}"
            )
        return ((x - self.mean) / self.scale).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path: str | Path) -> FeatureScaler:
        with np.load(path) as blob:
            return cls(mean=blob["mean"], scale=blob["scale"])

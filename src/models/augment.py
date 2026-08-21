"""Training-time augmentation, shared by both frameworks.

Written in NumPy and called from both training loops so the TensorFlow and
PyTorch models see the same perturbations from the same seed. Putting it in
each framework's own preprocessing layers would have been tidier and would
have quietly made the comparison meaningless.

Two transforms, both chosen because they correspond to something that
actually varies between recording sessions:

``noise``
    Additive Gaussian noise. Stands in for electrode-contact quality and
    amplifier noise floor, which differ between sessions and between people.

``scale``
    A per-channel gain multiplier, one draw per window. Stands in for
    contraction effort and for electrode-skin impedance, which is the largest
    single source of between-subject variation in sEMG amplitude.

**Why these two are worth their cost.** They are not free: on
`experiments/architecture_comparison.py` they cost 0.044 of within-subject
accuracy (0.650 without, 0.606 with). They buy 0.054 of subject-independent
accuracy (0.216 without, 0.270 with), and the subject-independent number is
the one this repository leads with. That is the trade augmentation is supposed
to make, and it is measured rather than assumed.

**What is deliberately not here, and the result is more interesting than
expected.** Circularly shifting each armband's eight electrodes to simulate a
rotated band is the obvious third transform. It was tried. It takes
within-subject accuracy from 0.606 to 0.396, because for one wearer with the
band in one position, which electrode sees which muscle *is* the signal, and a
model forced to ignore it has thrown away most of what separates the gestures.

The complication: on the single cross-subject fold measured, it scored 0.297
against the shipped model's 0.270. The invariance is real, because subjects
genuinely do wear the band differently. It was still not adopted, for two
reasons. One fold at +0.027 is inside the fold-to-fold spread of the ten-fold
run, so that is not evidence yet. And the deployed system is calibrated to its
wearer, at which point the band's position is fixed and known and a model that
cannot exploit it is giving up the thing calibration is for. Whether that
second argument survives being measured is an open question and a good next
experiment; it has not been run here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Augmenter:
    """Perturbs a batch of (n_windows, n_channels, n_samples) windows.

    Applies to training batches only. Never to validation or test data, where
    it would make the reported number a measurement of the augmentation.
    """

    noise_sd: float = 0.0
    scale_sd: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.noise_sd < 0 or self.scale_sd < 0:
            raise ValueError(
                f"augmentation strengths must be non-negative, got "
                f"noise_sd={self.noise_sd} scale_sd={self.scale_sd}"
            )
        self._rng = np.random.default_rng(self.seed)

    @property
    def enabled(self) -> bool:
        return self.noise_sd > 0 or self.scale_sd > 0

    def reset(self) -> None:
        """Restart the stream, so a rerun with the same seed augments identically."""
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return batch
        if batch.ndim != 3:
            raise ValueError(
                f"expected (n_windows, n_channels, n_samples), got {batch.shape}"
            )

        out = np.asarray(batch, dtype=np.float32)
        if self.scale_sd > 0:
            # One gain per window per channel, held constant across the window:
            # electrode gain does not change within 200 ms.
            gains = 1.0 + self._rng.normal(
                scale=self.scale_sd, size=(out.shape[0], out.shape[1], 1)
            )
            out = out * gains.astype(np.float32)
        if self.noise_sd > 0:
            out = out + self._rng.normal(scale=self.noise_sd, size=out.shape).astype(
                np.float32
            )
        return out.astype(np.float32)


def from_config(train_config, seed_offset: int = 0) -> Augmenter:
    """Build an augmenter from a :class:`src.config.TrainConfig`."""
    return Augmenter(
        noise_sd=train_config.augment_noise_sd,
        scale_sd=train_config.augment_scale_sd,
        seed=train_config.seed + seed_offset,
    )

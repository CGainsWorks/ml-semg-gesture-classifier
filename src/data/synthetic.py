"""Synthetic sEMG generator.

This exists so the pipeline, the tests and CI can run without the 200 MB
Ninapro download. **No accuracy figure produced from this source is a
result.** It is a wiring check, and every script that touches it stamps
``SYNTHETIC`` on its output so a number from it cannot be mistaken for one
measured on real recordings.

The generator is not arbitrary noise. It reproduces the two properties that
make the real problem hard in the way it is hard:

* A gesture is a pattern of activation across forearm muscles, and the
  armband observes those muscles through a mixing matrix set by where the
  electrodes happen to sit.
* That mixing matrix is different for every subject, because arms differ and
  armbands are not put on in the same rotation twice. This is why a model
  trained on other people transfers worse than one trained on you, and it is
  the effect the subject-independent split is there to expose.

Interference-pattern sEMG is close to band-limited Gaussian noise amplitude-
modulated by the muscle's activation level, which is what is synthesised here.
"""

from __future__ import annotations

import numpy as np

from src.data.ninapro import Recording, assign_rest_repetitions

N_CHANNELS = 16
N_MUSCLES = 6
MOVEMENT_SECONDS = 3.0
REST_SECONDS = 2.0


def _rng(seed: int, subject: int) -> np.random.Generator:
    return np.random.default_rng([seed, subject])


def _muscle_patterns(n_gestures: int, seed: int) -> np.ndarray:
    """Activation of each muscle for each gesture, shared across subjects.

    Row 0 is rest. Patterns are sparse so gestures stay distinguishable at the
    muscle level, which is the level at which they genuinely differ.
    """
    rng = np.random.default_rng([seed, 0xA17])
    patterns = rng.random((n_gestures + 1, N_MUSCLES)) ** 3
    patterns[0] = 0.0  # rest
    # Guarantee every non-rest gesture has at least one strongly driven muscle,
    # otherwise a gesture can end up indistinguishable from rest by accident.
    for g in range(1, n_gestures + 1):
        patterns[g, rng.integers(N_MUSCLES)] = 1.0
    return patterns.astype(np.float32)


def _subject_mixing(subject: int, seed: int) -> np.ndarray:
    """(N_CHANNELS, N_MUSCLES) map from muscle activation to electrode signal.

    Built as a ring of electrodes with a subject-specific rotation offset and
    per-channel gain, which is the physical story behind cross-subject
    degradation.
    """
    rng = _rng(seed, subject)
    electrode_angle = np.linspace(0, 2 * np.pi, N_CHANNELS, endpoint=False)
    muscle_angle = np.linspace(0, 2 * np.pi, N_MUSCLES, endpoint=False)
    offset = rng.uniform(0, 2 * np.pi)
    spread = rng.uniform(0.5, 0.9)

    delta = electrode_angle[:, None] - (muscle_angle[None, :] + offset)
    mixing = np.exp((np.cos(delta) - 1.0) / spread**2)
    gains = rng.uniform(0.6, 1.4, size=(N_CHANNELS, 1))
    return (mixing * gains).astype(np.float32)


def _bandlimited_noise(
    n: int, n_channels: int, sample_rate_hz: int, rng: np.random.Generator
) -> np.ndarray:
    """Gaussian noise shaped to the 20-95 Hz band the Myo actually passes."""
    spectrum = rng.normal(size=(n_channels, n // 2 + 1)) + 1j * rng.normal(
        size=(n_channels, n // 2 + 1)
    )
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate_hz)
    band = ((freqs >= 20.0) & (freqs <= min(95.0, sample_rate_hz / 2 - 5))).astype(float)
    signal = np.fft.irfft(spectrum * band, n=n).T
    sd = signal.std()
    return (signal / (sd if sd > 0 else 1.0)).astype(np.float32)


def generate_recording(
    subject: int,
    exercise: str = "E2",
    gestures: list[int] | None = None,
    repetitions: int = 6,
    sample_rate_hz: int = 200,
    seed: int = 7,
) -> Recording:
    """Synthesise one subject's recording with the DB5 field layout."""
    gestures = list(gestures) if gestures else [1, 2, 3, 4, 5, 6]
    if any(g <= 0 for g in gestures):
        raise ValueError("gesture codes must be positive; rest is added automatically")

    rng = _rng(seed, subject)
    patterns = _muscle_patterns(max(gestures), seed)
    mixing = _subject_mixing(subject, seed)

    move_n = int(MOVEMENT_SECONDS * sample_rate_hz)
    rest_n = int(REST_SECONDS * sample_rate_hz)

    activation_blocks: list[np.ndarray] = []
    stim_blocks: list[np.ndarray] = []
    rep_blocks: list[np.ndarray] = []

    for rep in range(1, repetitions + 1):
        for gesture in gestures:
            # Rest, then the movement. A trapezoidal envelope stands in for the
            # ramp on and off of a real contraction.
            activation_blocks.append(np.zeros((rest_n, N_MUSCLES), dtype=np.float32))
            stim_blocks.append(np.zeros(rest_n, dtype=np.int16))
            rep_blocks.append(np.zeros(rest_n, dtype=np.int16))

            ramp = int(0.2 * move_n)
            envelope = np.ones(move_n, dtype=np.float32)
            envelope[:ramp] = np.linspace(0, 1, ramp, dtype=np.float32)
            envelope[-ramp:] = np.linspace(1, 0, ramp, dtype=np.float32)
            effort = rng.uniform(0.75, 1.25)
            activation_blocks.append(
                envelope[:, None] * patterns[gesture][None, :] * effort
            )
            stim_blocks.append(np.full(move_n, gesture, dtype=np.int16))
            rep_blocks.append(np.full(move_n, rep, dtype=np.int16))

    activation_blocks.append(np.zeros((rest_n, N_MUSCLES), dtype=np.float32))
    stim_blocks.append(np.zeros(rest_n, dtype=np.int16))
    rep_blocks.append(np.zeros(rest_n, dtype=np.int16))

    activation = np.concatenate(activation_blocks, axis=0)
    restimulus = np.concatenate(stim_blocks)
    rerepetition = np.concatenate(rep_blocks)
    n = activation.shape[0]

    carrier = _bandlimited_noise(n, N_CHANNELS, sample_rate_hz, rng)
    channel_activation = activation @ mixing.T  # (n, N_CHANNELS)
    baseline = rng.uniform(0.01, 0.03)
    emg = carrier * (channel_activation + baseline) * 0.25
    emg += rng.normal(scale=0.004, size=emg.shape).astype(np.float32)

    return Recording(
        subject=subject,
        exercise=exercise,
        emg=emg.astype(np.float32),
        restimulus=restimulus,
        repetition=assign_rest_repetitions(restimulus, rerepetition),
        sample_rate_hz=sample_rate_hz,
        metadata={
            "source_file": f"synthetic:s{subject}:{exercise}",
            "synthetic": True,
            "seed": seed,
            "n_samples": n,
            "sensor": "synthetic 16-channel ring",
        },
    )

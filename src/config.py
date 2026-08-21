"""Central configuration.

Every tunable lives here or in ``config/default.yaml``. Scripts take a
``--config`` path so a run is fully described by one file plus the git commit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


@dataclass
class DatasetConfig:
    """Which recordings to read and where they live."""

    # "ninapro_db5" is the real dataset. "synthetic" is a generator used by the
    # test suite and CI so the pipeline can be exercised without a 200 MB
    # download. Numbers from the synthetic source are never quoted as results.
    source: str = "ninapro_db5"
    raw_dir: str = "data/raw/DB5"
    cache_dir: str = "artifacts/cache"
    # DB5 ships three files per subject. E2 is exercise B: 17 basic hand and
    # wrist movements, which is the set a wrist-and-grip prosthesis needs.
    exercise: str = "E2"
    subjects: list[int] = field(default_factory=lambda: list(range(1, 11)))
    # None keeps every movement in the exercise. A list keeps only those
    # restimulus codes, with rest (0) always retained.
    gestures: list[int] | None = None
    sample_rate_hz: int = 200


@dataclass
class PreprocessingConfig:
    """Filtering applied to the continuous signal before windowing."""

    bandpass_low_hz: float = 10.0
    # Nyquist is 100 Hz at the DB5 rate, so the top of the band sits just under
    # it. Raised automatically for higher-rate sources such as DB2.
    bandpass_high_hz: float = 99.0
    bandpass_order: int = 4
    # The Myo armband filters mains interference in hardware, so the notch is
    # off for DB5. At the DB5 rate a 50 Hz notch would also gouge a hole in the
    # middle of the useful band.
    notch_hz: float | None = None
    notch_quality: float = 30.0


@dataclass
class WindowConfig:
    """Sliding window geometry.

    ``length_ms`` is the dominant term in the control-loop latency budget: the
    classifier cannot emit a decision until the window it needs has been
    filled. See docs/LATENCY_BUDGET.md.
    """

    length_ms: float = 200.0
    # 50 ms beats 25 ms at four of the five window lengths swept in
    # experiments/latency_accuracy_tradeoff.py. Halving the stride doubles the
    # training windows, but they then overlap by seven eighths and are largely
    # the same data again, at twice the cost.
    stride_ms: float = 50.0
    # Windows straddling a label change are ambiguous. Dropping them is the
    # usual Ninapro convention and avoids training on transitions.
    drop_mixed_label: bool = True
    # Rest dominates the raw recordings. Cap it at this multiple of the mean
    # movement-class count so the classifier cannot win by always saying rest.
    rest_ratio: float = 1.0
    seed: int = 20250821


@dataclass
class FeatureConfig:
    """Classical time-domain feature set."""

    # Threshold parameters for zero crossings, slope sign changes and Willison
    # amplitude, expressed as a fraction of each channel's training-set
    # standard deviation so they transfer across subjects and gains.
    zc_threshold_sd: float = 0.05
    ssc_threshold_sd: float = 0.05
    wamp_threshold_sd: float = 0.10
    ar_order: int = 4


@dataclass
class TrainConfig:
    """Shared hyperparameters for both deep models.

    The TensorFlow and PyTorch models take identical values so the comparison
    measures the frameworks, not two different training recipes.
    """

    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    # 0.4 rather than a smaller value because the network overfits this dataset
    # badly. Even at 0.4 the within-subject train-test gap is 0.17 and the
    # cross-subject gap is 0.47. Measured in
    # experiments/architecture_comparison.py.
    dropout: float = 0.4
    # Feature maps for the three convolution blocks.
    conv_channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    # Kernel across electrodes and across time. Three electrodes because
    # adjacent electrodes on the armband see overlapping muscles; five samples
    # is 25 ms at the DB5 rate.
    kernel_electrodes: int = 3
    kernel_time: int = 5
    dense_units: int = 128
    # Training-time augmentation, applied identically in both frameworks.
    # Additive noise stands in for contact quality; the per-channel gain
    # stands in for electrode impedance, which is the biggest single source of
    # between-subject amplitude variation. Circular electrode shifts were
    # tried and are catastrophic; see src/models/augment.py.
    augment_noise_sd: float = 0.15
    augment_scale_sd: float = 0.25
    early_stopping_patience: int = 8
    # Fraction of the training repetitions held out for early stopping. Split
    # by repetition, never at random, so a window and its neighbours cannot
    # land on both sides.
    val_fraction: float = 0.2
    seed: int = 20250821
    device: str = "auto"


@dataclass
class EvalConfig:
    """Protocol definitions."""

    # Ninapro convention: train on repetitions 1, 3, 4 and 6, test on 2 and 5.
    within_subject_train_reps: list[int] = field(default_factory=lambda: [1, 3, 4, 6])
    within_subject_test_reps: list[int] = field(default_factory=lambda: [2, 5])
    # Repetitions of the held-out subject used to calibrate in the
    # subject-independent-plus-calibration condition.
    calibration_reps: list[int] = field(default_factory=lambda: [1])
    calibration_epochs: int = 10
    calibration_learning_rate: float = 2e-4
    # Majority vote over this many consecutive decisions before the arm moves.
    # 1 means no vote, and that is the measured default: consecutive windows
    # overlap by three quarters, so their decisions are highly correlated and
    # voting corrects little, while each extra decision costs a stride of
    # latency. Dropping it is what brought the controller inside its 300 ms
    # deadline. The mechanism is kept because it is cheap and because the
    # offline protocol drops the transition windows where it would plausibly
    # earn its keep.
    vote_window: int = 1


@dataclass
class LatencyConfig:
    """The control-loop deadline this project is held to."""

    # Total controller delay from muscle activity to actuator command. 300 ms
    # is the ceiling that the prosthetics literature treats as the point where
    # users start to perceive the device as unresponsive.
    budget_ms: float = 300.0
    # Allowance for reading the electrodes and driving the actuators, which
    # this repository does not implement and therefore cannot measure.
    io_allowance_ms: float = 20.0
    bench_iterations: int = 2000
    bench_warmup: int = 200


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    artifacts_dir: str = "artifacts"

    @property
    def window_samples(self) -> int:
        return int(round(self.window.length_ms * self.dataset.sample_rate_hz / 1000.0))

    @property
    def stride_samples(self) -> int:
        return max(
            1, int(round(self.window.stride_ms * self.dataset.sample_rate_hz / 1000.0))
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


_SECTIONS = {
    "dataset": DatasetConfig,
    "preprocessing": PreprocessingConfig,
    "window": WindowConfig,
    "features": FeatureConfig,
    "train": TrainConfig,
    "evaluation": EvalConfig,
    "latency": LatencyConfig,
}


def _build_section(name: str, raw: dict[str, Any]) -> Any:
    cls = _SECTIONS[name]
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in config section '{name}': {sorted(unknown)}"
        )
    return cls(**raw)


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config, falling back to the packaged default."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    unknown_top = set(raw) - set(_SECTIONS) - {"artifacts_dir"}
    if unknown_top:
        raise ValueError(f"unknown top-level config key(s): {sorted(unknown_top)}")

    kwargs: dict[str, Any] = {}
    for name in _SECTIONS:
        if name in raw:
            kwargs[name] = _build_section(name, raw[name] or {})
    if "artifacts_dir" in raw:
        kwargs["artifacts_dir"] = raw["artifacts_dir"]
    return Config(**kwargs)


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


_SOURCE_FINGERPRINT: str | None = None


def source_fingerprint() -> str:
    """A digest of every source file under ``src/``.

    Stamped into each result so that two results can be shown to have come
    from the same code, not merely the same config. The config alone does not
    settle it: a training process that survived being cancelled once wrote
    half a results directory from an older revision, with an identical config
    and a materially different batch-norm setting, and nothing in the output
    said so.

    Computed once per process. It covers ``src/`` only; the experiment
    scripts do not write these files.
    """
    global _SOURCE_FINGERPRINT
    if _SOURCE_FINGERPRINT is None:
        digest = hashlib.sha256()
        for path in sorted((REPO_ROOT / "src").rglob("*.py")):
            digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        _SOURCE_FINGERPRINT = digest.hexdigest()[:16]
    return _SOURCE_FINGERPRINT

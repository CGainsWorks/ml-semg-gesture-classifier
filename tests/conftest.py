"""Shared fixtures.

Every test runs against the synthetic generator. Nothing here needs the
Ninapro download, so the suite runs in CI and on a clean checkout. The tests
that would need real data are the ones that assert on accuracy, and this
repository does not have any: accuracy is measured by
``python -m src.train``, reported with its provenance, and not asserted in a
unit test where it would either be trivially true or a flake.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config, DatasetConfig, TrainConfig, load_config
from src.data import synthetic
from src.data.dataset import build_dataset


@pytest.fixture(scope="session")
def ci_config() -> Config:
    return load_config("config/ci.yaml")


@pytest.fixture
def small_config(tmp_path) -> Config:
    """A four-subject synthetic config with the cache pointed at a temp dir."""
    config = load_config("config/ci.yaml")
    config.dataset = DatasetConfig(
        source="synthetic",
        raw_dir=str(tmp_path / "raw"),
        cache_dir=str(tmp_path / "cache"),
        exercise="E2",
        subjects=[1, 2, 3],
        gestures=[1, 2, 3, 4],
        sample_rate_hz=200,
    )
    config.train = TrainConfig(
        epochs=2,
        batch_size=64,
        early_stopping_patience=2,
        val_fraction=0.34,
        seed=11,
        device="cpu",
    )
    return config


@pytest.fixture
def recording():
    return synthetic.generate_recording(subject=1, gestures=[1, 2, 3], seed=5)


@pytest.fixture
def small_index(small_config):
    return build_dataset(small_config)


@pytest.fixture
def windows(small_index) -> np.ndarray:
    return small_index.materialise("ct")[:256]

"""The augmenter, and the property that makes the framework comparison valid.

Both training loops call the same NumPy augmenter. If they did not, the
TensorFlow and PyTorch models would be trained on differently perturbed data
and the accuracy difference between them would mean nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import TrainConfig
from src.models.augment import Augmenter, from_config


@pytest.fixture
def batch() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(64, 16, 40)).astype(np.float32)


class TestAugmenter:
    def test_disabled_by_default_returns_the_input_untouched(self, batch):
        augmenter = Augmenter()
        assert not augmenter.enabled
        assert augmenter(batch) is batch

    def test_enabled_changes_the_data(self, batch):
        out = Augmenter(noise_sd=0.1, scale_sd=0.2, seed=1)(batch)
        assert out.shape == batch.shape
        assert out.dtype == np.float32
        assert not np.allclose(out, batch)

    def test_the_same_seed_gives_the_same_perturbation(self, batch):
        first = Augmenter(noise_sd=0.1, scale_sd=0.2, seed=7)(batch)
        second = Augmenter(noise_sd=0.1, scale_sd=0.2, seed=7)(batch)
        np.testing.assert_allclose(first, second)

    def test_different_seeds_give_different_perturbations(self, batch):
        first = Augmenter(noise_sd=0.1, scale_sd=0.2, seed=1)(batch)
        second = Augmenter(noise_sd=0.1, scale_sd=0.2, seed=2)(batch)
        assert not np.allclose(first, second)

    def test_reset_restarts_the_stream(self, batch):
        augmenter = Augmenter(noise_sd=0.1, scale_sd=0.2, seed=3)
        first = augmenter(batch)
        augmenter(batch)  # advance the generator
        augmenter.reset()
        np.testing.assert_allclose(augmenter(batch), first)

    def test_consecutive_calls_differ(self, batch):
        """Otherwise every epoch would see identical noise and buy nothing."""
        augmenter = Augmenter(noise_sd=0.1, scale_sd=0.2, seed=3)
        assert not np.allclose(augmenter(batch), augmenter(batch))

    def test_the_gain_is_constant_within_a_window(self):
        """Electrode gain does not change inside 200 ms, so neither does the scale."""
        constant = np.ones((8, 4, 20), dtype=np.float32)
        out = Augmenter(noise_sd=0.0, scale_sd=0.3, seed=5)(constant)
        # Each (window, channel) row should be one value repeated across time.
        spread = out.std(axis=2)
        np.testing.assert_allclose(spread, 0.0, atol=1e-6)
        # And different channels should have taken different gains.
        assert out[0, :, 0].std() > 0

    def test_noise_only_leaves_the_mean_where_it_was(self, batch):
        out = Augmenter(noise_sd=0.1, scale_sd=0.0, seed=4)(batch)
        assert abs(float(out.mean() - batch.mean())) < 0.01

    def test_a_negative_strength_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Augmenter(noise_sd=-0.1)

    def test_a_wrongly_shaped_batch_is_rejected(self):
        with pytest.raises(ValueError, match="n_windows, n_channels, n_samples"):
            Augmenter(noise_sd=0.1)(np.zeros((4, 40), dtype=np.float32))

    def test_from_config_reads_the_training_settings(self):
        config = TrainConfig(augment_noise_sd=0.11, augment_scale_sd=0.22, seed=99)
        augmenter = from_config(config)
        assert augmenter.noise_sd == 0.11
        assert augmenter.scale_sd == 0.22
        assert augmenter.seed == 99
        assert augmenter.enabled


class TestSharedAcrossFrameworks:
    def test_both_training_loops_accept_an_augmenter(self):
        """Passing one in is how a caller guarantees the two see the same recipe."""
        import inspect

        from src.models import tf_cnn, torch_cnn

        for module in (torch_cnn, tf_cnn):
            parameters = inspect.signature(module.train_model).parameters
            assert "augmenter" in parameters, module.__name__

    def test_a_shared_augmenter_produces_identical_batches(self, batch):
        """The property the comparison rests on, asserted rather than assumed."""
        config = TrainConfig(augment_noise_sd=0.15, augment_scale_sd=0.25, seed=42)
        first, second = from_config(config), from_config(config)
        first.reset()
        second.reset()
        for _ in range(3):
            np.testing.assert_allclose(first(batch), second(batch))


class TestAugmentedTrainingPaths:
    """Both loops take a different code path when augmentation is on.

    PyTorch keeps the data on the host and transfers each perturbed batch;
    Keras switches from a NumPy array to a ``PyDataset``. Neither is exercised
    by the CI config, which turns augmentation off to keep the smoke run
    quick, so they are exercised here.
    """

    @pytest.fixture
    def tiny(self):
        from src.models.architecture import ArchitectureSpec

        rng = np.random.default_rng(0)
        spec = ArchitectureSpec.from_config(
            TrainConfig(conv_channels=[4, 8], dense_units=16),
            n_channels=8,
            window_samples=16,
            n_classes=3,
        )
        x = rng.normal(scale=0.1, size=(200, 8, 16)).astype(np.float32)
        y = rng.integers(0, 3, size=200)
        for i, label in enumerate(y):
            x[i, label * 2] += 3.0
        return spec, x, y

    @pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
    def test_training_with_augmentation_runs_and_learns(self, tiny, framework):
        spec, x, y = tiny
        if framework == "pytorch":
            from src.models import torch_cnn as backend
        else:
            from src.models import tf_cnn as backend

        config = TrainConfig(
            epochs=6,
            batch_size=32,
            device="cpu",
            seed=0,
            augment_noise_sd=0.05,
            augment_scale_sd=0.1,
            conv_channels=[4, 8],
            dense_units=16,
            early_stopping_patience=6,
        )
        model, info = backend.train_model(
            spec, x[:160], y[:160], x[160:], y[160:], config, verbose=False
        )
        assert info["augmented"] is True
        accuracy = (backend.predict_proba(model, x[160:]).argmax(axis=1) == y[160:]).mean()
        assert accuracy > 1.0 / 3

    @pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
    def test_augmentation_can_be_switched_off(self, tiny, framework):
        spec, x, y = tiny
        if framework == "pytorch":
            from src.models import torch_cnn as backend
        else:
            from src.models import tf_cnn as backend

        config = TrainConfig(
            epochs=2,
            batch_size=32,
            device="cpu",
            seed=0,
            augment_noise_sd=0.0,
            augment_scale_sd=0.0,
            conv_channels=[4, 8],
            dense_units=16,
        )
        _, info = backend.train_model(
            spec, x[:160], y[:160], x[160:], y[160:], config, verbose=False
        )
        assert info["augmented"] is False

    def test_keras_batches_cover_every_window_once_per_epoch(self, tiny):
        """A PyDataset that silently drops the tail would train on less data."""
        from src.models.tf_cnn import _AugmentedBatches

        _, x, y = tiny
        batches = _AugmentedBatches(x, y, batch_size=32, augmenter=Augmenter(), seed=0)
        seen = sum(len(batches[i][1]) for i in range(len(batches)))
        assert seen == x.shape[0]

    def test_keras_batches_reshuffle_between_epochs(self, tiny):
        from src.models.tf_cnn import _AugmentedBatches

        _, x, y = tiny
        batches = _AugmentedBatches(x, y, batch_size=32, augmenter=Augmenter(), seed=0)
        first = batches[0][1].copy()
        batches.on_epoch_end()
        assert not np.array_equal(first, batches[0][1])

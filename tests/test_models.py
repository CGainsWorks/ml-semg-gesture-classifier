"""Models, metrics and the two frameworks agreeing with each other."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.config import TrainConfig
from src.data.dataset import class_codes, encode_labels
from src.eval.metrics import evaluate, majority_vote, pool
from src.models.architecture import ArchitectureSpec
from src.models.classical import CLASSIFIERS, ClassicalPipeline
from src.preprocessing.normalise import ChannelScaler


@pytest.fixture
def spec() -> ArchitectureSpec:
    return ArchitectureSpec.from_config(
        TrainConfig(), n_channels=16, window_samples=40, n_classes=18
    )


class TestArchitectureSpec:
    def test_both_axes_halve_once_per_block(self, spec):
        # 16 electrodes and 40 samples, three pooling stages.
        assert spec.pooled_shape == (2, 5)
        assert spec.flattened_units == 2 * 5 * 64

    def test_a_window_too_short_for_the_pooling_stack_is_rejected(self):
        with pytest.raises(ValueError, match="pooling stages"):
            ArchitectureSpec.from_config(
                TrainConfig(), n_channels=16, window_samples=4, n_classes=3
            )

    def test_too_few_electrodes_for_the_pooling_stack_is_rejected(self):
        with pytest.raises(ValueError, match="electrodes cannot survive"):
            ArchitectureSpec.from_config(
                TrainConfig(), n_channels=4, window_samples=40, n_classes=3
            )

    def test_one_class_is_rejected(self):
        with pytest.raises(ValueError, match="at least two classes"):
            ArchitectureSpec.from_config(
                TrainConfig(), n_channels=16, window_samples=40, n_classes=1
            )

    def test_mac_count_is_positive_and_dominated_by_the_convolutions(self, spec):
        assert spec.multiply_accumulates() > spec.parameter_count()


class TestFrameworkParity:
    """The comparison is only meaningful if the two models are the same model."""

    def test_pytorch_matches_the_derived_parameter_count(self, spec):
        from src.models.torch_cnn import TorchGestureCNN

        model = TorchGestureCNN(spec)
        counted = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # The derived count excludes batch-norm running statistics, which are
        # buffers rather than parameters, so the two should agree exactly.
        assert counted == spec.parameter_count()

    def test_tensorflow_matches_the_derived_parameter_count(self, spec):
        from src.models.tf_cnn import build

        model = build(spec)
        # Trainable weights only. Keras' count_params() also counts the
        # batch-norm running mean and variance, which PyTorch keeps as buffers
        # and the derived count deliberately excludes.
        trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
        assert trainable == spec.parameter_count()

    def test_both_frameworks_agree_on_the_parameter_count(self, spec):
        import numpy as np_

        from src.models.tf_cnn import build
        from src.models.torch_cnn import TorchGestureCNN

        keras = sum(int(np_.prod(w.shape)) for w in build(spec).trainable_weights)
        torch = sum(p.numel() for p in TorchGestureCNN(spec).parameters() if p.requires_grad)
        assert keras == torch

    def test_batch_norm_momentum_means_the_same_thing_in_both(self, spec):
        """The two frameworks define this parameter as complements.

        PyTorch's ``BatchNorm2d(momentum=0.1)`` weights the incoming batch by
        0.1. Keras' ``BatchNormalization(momentum=0.99)`` weights the running
        estimate by 0.99, so it weights the batch by 0.01. Left at both
        defaults the running statistics converge ten times more slowly in
        Keras, and on the smaller folds the model is then evaluated with
        statistics that never settled: within-subject accuracy came out at
        0.40 against PyTorch's 0.59 before this was matched, and at 0.60
        against 0.59 after.
        """
        import torch

        from src.models.tf_cnn import build
        from src.models.torch_cnn import TorchGestureCNN

        keras_momenta = {
            layer.momentum
            for layer in build(spec).layers
            if isinstance(layer, tf.keras.layers.BatchNormalization)
        }
        torch_momenta = {
            module.momentum
            for module in TorchGestureCNN(spec).modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        }
        assert len(keras_momenta) == 1 and len(torch_momenta) == 1
        keras, pytorch = keras_momenta.pop(), torch_momenta.pop()
        assert keras == pytest.approx(1.0 - pytorch), (
            f"Keras batch-norm momentum {keras} is not the complement of "
            f"PyTorch's {pytorch}; the two models are not the same model"
        )

    def test_both_frameworks_take_the_same_input_shape(self, spec):
        import torch

        from src.models.tf_cnn import build
        from src.models.torch_cnn import TorchGestureCNN

        x = np.zeros((3, spec.n_channels, spec.window_samples), dtype=np.float32)
        torch_out = TorchGestureCNN(spec).eval()(torch.from_numpy(x))
        tf_out = build(spec)(x, training=False)
        assert tuple(torch_out.shape) == (3, spec.n_classes)
        assert tuple(tf_out.shape) == (3, spec.n_classes)

    def test_both_emit_logits_not_probabilities(self, spec):
        """Softmax is the caller's job, so the exported graphs stay unambiguous."""
        import torch

        from src.models.tf_cnn import build
        from src.models.torch_cnn import TorchGestureCNN

        rng = np.random.default_rng(0)
        x = rng.normal(size=(8, spec.n_channels, spec.window_samples)).astype(np.float32)
        torch_out = TorchGestureCNN(spec).eval()(torch.from_numpy(x)).detach().numpy()
        tf_out = np.asarray(build(spec)(x, training=False))
        # Probabilities would sum to one along the class axis. Logits do not.
        assert not np.allclose(torch_out.sum(axis=1), 1.0)
        assert not np.allclose(tf_out.sum(axis=1), 1.0)


class TestTorchTraining:
    def test_training_learns_a_separable_problem(self, spec):
        from src.models.torch_cnn import predict_proba, train_model

        rng = np.random.default_rng(0)
        small = ArchitectureSpec.from_config(
            TrainConfig(conv_channels=[8, 8], dense_units=16),
            n_channels=4,
            window_samples=16,
            n_classes=3,
        )
        # Three classes distinguished by which channel is loud.
        x = rng.normal(scale=0.1, size=(300, 4, 16)).astype(np.float32)
        y = rng.integers(0, 3, size=300)
        for i, label in enumerate(y):
            x[i, label] += 3.0

        config = TrainConfig(epochs=12, batch_size=32, device="cpu", seed=0)
        model, info = train_model(small, x[:240], y[:240], x[240:], y[240:], config, verbose=False)
        accuracy = (predict_proba(model, x[240:], device="cpu").argmax(axis=1) == y[240:]).mean()
        assert accuracy > 0.9
        assert info["framework"] == "pytorch"

    def test_checkpoint_round_trips(self, spec, tmp_path):
        from src.models import torch_cnn

        model = torch_cnn.TorchGestureCNN(spec).eval()
        x = np.zeros((2, spec.n_channels, spec.window_samples), dtype=np.float32)
        before = torch_cnn.predict_proba(model, x, device="cpu")

        torch_cnn.save(model, tmp_path / "m.pt")
        after = torch_cnn.predict_proba(torch_cnn.load(tmp_path / "m.pt"), x, device="cpu")
        np.testing.assert_allclose(before, after, atol=1e-6)

    def test_predict_proba_rows_sum_to_one(self, spec):
        from src.models import torch_cnn

        model = torch_cnn.TorchGestureCNN(spec).eval()
        x = np.random.default_rng(0).normal(size=(5, 16, 40)).astype(np.float32)
        proba = torch_cnn.predict_proba(model, x, device="cpu")
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_requesting_a_missing_cuda_device_fails_loudly(self):
        import torch

        from src.models.torch_cnn import resolve_device

        if torch.cuda.is_available():
            pytest.skip("CUDA is present, so this cannot be exercised here")
        with pytest.raises(RuntimeError, match="no CUDA device"):
            resolve_device("cuda")


class TestClassicalPipeline:
    @pytest.mark.parametrize("name", CLASSIFIERS)
    def test_every_classifier_fits_and_predicts(self, name, small_index):
        codes = class_codes(small_index)
        x = small_index.materialise("ct")[:600]
        y = encode_labels(small_index.labels[:600], codes)

        pipeline = ClassicalPipeline(classifier=name).fit(x, y)
        proba = pipeline.predict_proba(x[:50])
        assert proba.shape[0] == 50
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_extracting_before_fitting_is_an_error(self, windows):
        with pytest.raises(RuntimeError, match="fit the pipeline"):
            ClassicalPipeline().predict(windows)

    def test_unknown_classifier_is_rejected(self):
        with pytest.raises(ValueError, match="unknown classifier"):
            ClassicalPipeline(classifier="knn")

    def test_pipeline_round_trips(self, small_index, tmp_path):
        codes = class_codes(small_index)
        x = small_index.materialise("ct")[:400]
        y = encode_labels(small_index.labels[:400], codes)
        pipeline = ClassicalPipeline("lda").fit(x, y)
        pipeline.save(tmp_path / "p.pkl")
        loaded = ClassicalPipeline.load(tmp_path / "p.pkl")
        np.testing.assert_array_equal(pipeline.predict(x[:20]), loaded.predict(x[:20]))


class TestMetrics:
    def test_perfect_predictions_score_one(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        result = evaluate(y, y, np.array([0, 1, 2]), ["rest", "a", "b"])
        assert result.accuracy == 1.0
        assert result.balanced_accuracy == 1.0
        assert result.macro_f1 == 1.0
        assert result.false_activation_rate == 0.0
        assert result.missed_activation_rate == 0.0

    def test_false_and_missed_activation_are_distinguished(self):
        # Class 0 is rest. Two rest windows are called movements, and two
        # movement windows are called rest.
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 2, 0, 0, 1, 1])
        result = evaluate(y_true, y_pred, np.array([0, 1, 2]), ["rest", "a", "b"])
        assert result.false_activation_rate == pytest.approx(0.5)
        assert result.missed_activation_rate == pytest.approx(0.5)

    def test_confusion_matrix_covers_classes_absent_from_the_fold(self):
        y = np.array([0, 0, 1, 1])
        result = evaluate(y, y, np.array([0, 1, 2]), ["rest", "a", "b"])
        assert result.confusion.shape == (3, 3)
        assert len(result.per_class) == 3
        assert result.per_class[2].support == 0

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate(np.zeros(4), np.zeros(3), np.array([0, 1]), ["a", "b"])

    def test_pooling_sums_confusions_and_records_the_spread(self):
        codes, names = np.array([0, 1]), ["rest", "a"]
        good = evaluate(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), codes, names)
        bad = evaluate(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]), codes, names)
        pooled = pool([good, bad], codes, names)
        assert pooled.accuracy == pytest.approx(0.5)
        assert pooled.extra["n_folds"] == 2
        assert pooled.extra["fold_accuracy_min"] == 0.0
        assert pooled.extra["fold_accuracy_max"] == 1.0
        assert pooled.confusion.sum() == 8

    def test_majority_vote_removes_a_single_window_glitch(self):
        stream = np.array([1, 1, 1, 2, 1, 1, 1])
        voted = majority_vote(stream, k=3)
        assert voted[3] == 1
        assert (voted == 1).all()

    def test_majority_vote_of_one_changes_nothing(self):
        stream = np.array([0, 1, 2, 1])
        np.testing.assert_array_equal(majority_vote(stream, k=1), stream)


class TestNoLeakage:
    def test_the_scaler_is_never_fit_on_test_windows(self, small_index, small_config):
        """The property the subject-independent number depends on.

        A scaler fit on the training subjects will not standardise the held-out
        subject to zero mean and unit variance, because that subject's channel
        gains differ. If it did, the statistics would have come from the test
        set.
        """
        from src.data.splits import subject_independent_folds

        fold = subject_independent_folds(small_index, small_config)[0]
        scaler = ChannelScaler.fit(fold.train.materialise("ct"))
        transformed = scaler.transform(fold.test.materialise("ct"))
        assert not np.allclose(transformed.std(axis=(0, 2)), 1.0, atol=0.02)

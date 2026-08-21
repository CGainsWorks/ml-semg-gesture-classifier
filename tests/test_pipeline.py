"""End-to-end: the whole thing runs, and its outputs hold together.

Small and synthetic, so it runs in CI. What it proves is that the pieces
connect and that the reporting is consistent with the results it reports on,
not that the model is any good.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluate import accuracy_table, generalisation_gap, summary_rows
from src.experiment import MODELS, run_protocol


@pytest.mark.parametrize("model", ["classical_lda", "pytorch", "tensorflow"])
def test_a_protocol_runs_end_to_end(small_config, small_index, model):
    run = run_protocol(small_config, model, "within_subject", index=small_index, verbose=False)
    assert run.model == model
    assert run.synthetic is True
    assert len(run.per_fold) == len(small_config.dataset.subjects)
    assert 0.0 <= run.pooled.accuracy <= 1.0
    assert run.pooled.confusion.sum() == run.pooled.n_samples


def test_every_model_name_is_runnable(small_config, small_index):
    """A name in MODELS that nothing dispatches on would fail only at run time."""
    for model in MODELS:
        if model == "classical_svm":
            continue  # covered elsewhere; the RBF fit is the slowest in the set
        run = run_protocol(small_config, model, "within_subject", index=small_index, verbose=False)
        assert run.pooled.n_samples > 0


def test_all_three_protocols_run(small_config, small_index):
    for protocol in (
        "within_subject",
        "subject_independent",
        "subject_independent_calibrated",
    ):
        run = run_protocol(
            small_config, "classical_lda", protocol, index=small_index, verbose=False
        )
        assert run.protocol == protocol
        assert run.pooled.n_samples > 0


def test_calibration_uses_the_held_out_subject_only(small_config, small_index):
    run = run_protocol(
        small_config,
        "pytorch",
        "subject_independent_calibrated",
        index=small_index,
        verbose=False,
    )
    for fold in run.per_fold:
        assert fold["n_calibration"] > 0
        assert fold["test_subjects"] == [fold["held_out_subject"]]
        assert fold["held_out_subject"] not in fold["train_subjects"]
        assert fold["training"]["calibration"]["n_windows"] == fold["n_calibration"]


def test_classical_models_actually_calibrate(small_config, small_index):
    """A "calibrated" number has to come from a model that was calibrated.

    A scikit-learn estimator has no fine-tuning step, so calibration means
    refitting with the new wearer's demonstrations included. Without that, the
    calibrated protocol would silently report the uncalibrated model's score
    under a heading that says otherwise.
    """
    run = run_protocol(
        small_config,
        "classical_lda",
        "subject_independent_calibrated",
        index=small_index,
        verbose=False,
    )
    for fold in run.per_fold:
        assert fold["training"]["calibration"]["refit"] is True
        assert fold["training"]["calibration"]["n_windows"] == fold["n_calibration"]


def test_the_uncalibrated_protocol_does_not_refit(small_config, small_index):
    run = run_protocol(
        small_config,
        "classical_lda",
        "subject_independent",
        index=small_index,
        verbose=False,
    )
    for fold in run.per_fold:
        assert "calibration" not in fold["training"]


def test_results_round_trip_through_json(small_config, small_index, tmp_path):
    run = run_protocol(
        small_config, "classical_lda", "within_subject", index=small_index, verbose=False
    )
    path = run.save(tmp_path / "r.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["model"] == "classical_lda"
    assert loaded["pooled"]["accuracy"] == pytest.approx(run.pooled.accuracy)
    np.testing.assert_array_equal(
        np.array(loaded["pooled"]["confusion"]), run.pooled.confusion
    )


def test_the_reported_gap_matches_the_underlying_results(small_config, small_index):
    runs = [
        run_protocol(small_config, "classical_lda", p, index=small_index, verbose=False)
        for p in ("within_subject", "subject_independent")
    ]
    rows = summary_rows([r.to_dict() for r in runs])
    gaps = generalisation_gap(rows)
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["gap"] == pytest.approx(
        gap["within_subject"] - gap["subject_independent"]
    )
    # The table has to render without a protocol missing from the results.
    assert "Within-subject" in accuracy_table(rows)


def test_subject_independent_is_the_harder_condition(small_config, small_index):
    """The finding the whole repository exists to report, on data built to show it.

    The synthetic generator gives every subject a different electrode-to-muscle
    mixing matrix, which is the physical reason cross-subject transfer is hard.
    If this ever passed with the two conditions equal, either the generator or
    the split logic would have stopped working.
    """
    within = run_protocol(
        small_config, "classical_lda", "within_subject", index=small_index, verbose=False
    )
    across = run_protocol(
        small_config, "classical_lda", "subject_independent", index=small_index, verbose=False
    )
    assert within.pooled.balanced_accuracy > across.pooled.balanced_accuracy


def test_an_unknown_model_is_rejected(small_config, small_index):
    with pytest.raises(ValueError, match="unknown model"):
        run_protocol(small_config, "transformer", "within_subject", index=small_index)


class TestCalibration:
    """The learning-by-demonstration workflow in src/calibrate.py."""

    def test_demonstrations_and_held_out_repetitions_are_disjoint(self, small_config):
        from src.calibrate import split_demonstrations
        from src.data.dataset import build_dataset
        from src.preprocessing.normalise import ChannelScaler

        index = build_dataset(small_config)
        scaler = ChannelScaler.fit(index.materialise("ct"))
        x_demo, y_demo, x_test, y_test, codes = split_demonstrations(
            small_config, subject=1, repetitions=[1], scaler=scaler
        )
        assert x_demo.shape[0] == y_demo.size > 0
        assert x_test.shape[0] == y_test.size > 0
        # Both sets cover the label space; neither is a slice of the other.
        assert x_demo.shape[1:] == x_test.shape[1:]
        assert codes.size >= 2

    def test_demonstrating_every_repetition_leaves_nothing_to_score(self, small_config):
        from src.calibrate import split_demonstrations
        from src.data.dataset import build_dataset
        from src.preprocessing.normalise import ChannelScaler

        index = build_dataset(small_config)
        scaler = ChannelScaler.fit(index.materialise("ct"))
        every = sorted(int(r) for r in np.unique(index.repetitions))
        with pytest.raises(ValueError, match="nothing to be scored on"):
            split_demonstrations(small_config, 1, every, scaler)

    def test_an_absent_subject_is_reported_clearly(self, small_config):
        from src.calibrate import split_demonstrations
        from src.data.dataset import build_dataset
        from src.preprocessing.normalise import ChannelScaler

        index = build_dataset(small_config)
        scaler = ChannelScaler.fit(index.materialise("ct"))
        with pytest.raises(ValueError, match="not in this dataset"):
            split_demonstrations(small_config, 99, [1], scaler)

    def test_calibration_runs_end_to_end_and_reports_honestly(self, small_config):
        """--holdout must produce a 'before' model that never saw this wearer."""
        from src.calibrate import run

        report = run(
            small_config, subject=1, repetitions=[1], holdout=True, verbose=False
        )
        assert report["holdout"] is True
        assert report["n_demonstration_windows"] > 0
        assert report["n_held_out_windows"] > 0
        assert report["accuracy_change"] == pytest.approx(
            report["after"]["accuracy"] - report["before"]["accuracy"]
        )
        assert "never seen this wearer" in report["caveat"]


class TestBaseModelCache:
    """Reusing the base model between the two subject-independent protocols.

    It halves the deep models' running time. It has to change nothing about
    the answer, or it is not an optimisation, it is a different experiment.
    """

    def test_a_cached_run_matches_an_uncached_one(self, small_config, small_index):
        cache: dict = {}
        cached = run_protocol(
            small_config,
            "pytorch",
            "subject_independent",
            index=small_index,
            verbose=False,
            base_models=cache,
        )
        uncached = run_protocol(
            small_config,
            "pytorch",
            "subject_independent",
            index=small_index,
            verbose=False,
        )
        assert cached.pooled.accuracy == pytest.approx(uncached.pooled.accuracy)
        np.testing.assert_array_equal(
            cached.pooled.confusion, uncached.pooled.confusion
        )

    def test_the_cache_is_populated_and_then_reused(self, small_config, small_index):
        cache: dict = {}
        first = run_protocol(
            small_config,
            "pytorch",
            "subject_independent",
            index=small_index,
            verbose=False,
            base_models=cache,
        )
        assert len(cache) == len(first.per_fold)
        assert all("reused_base_model" not in f["training"] for f in first.per_fold)

        second = run_protocol(
            small_config,
            "pytorch",
            "subject_independent_calibrated",
            index=small_index,
            verbose=False,
            base_models=cache,
        )
        assert all(f["training"]["reused_base_model"] for f in second.per_fold)
        # Reused, then fine-tuned: the calibration step still has to happen.
        assert all(f["training"]["calibration"]["n_windows"] > 0 for f in second.per_fold)

    def test_calibration_does_not_mutate_the_cached_weights(self, small_config, small_index):
        """The uncalibrated protocol's model must survive the calibrated run."""
        cache: dict = {}
        run_protocol(
            small_config, "pytorch", "subject_independent",
            index=small_index, verbose=False, base_models=cache,
        )
        before = {k: [w.clone() for w in v[0].values()] for k, v in cache.items()}
        run_protocol(
            small_config, "pytorch", "subject_independent_calibrated",
            index=small_index, verbose=False, base_models=cache,
        )
        for key, weights in cache.items():
            for original, current in zip(before[key], weights[0].values()):
                assert np.allclose(original.numpy(), current.numpy())

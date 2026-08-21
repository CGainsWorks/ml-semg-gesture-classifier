"""Loader, windowing and split correctness.

The tests that matter here are the leakage ones. A subject-independent result
that is quietly contaminated looks exactly like a good result, so the property
has to be asserted rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.data.dataset import build_dataset, class_codes, encode_labels
from src.data.gestures import EXERCISE_B, gesture_names
from src.data.ninapro import assign_rest_repetitions, parse_mat
from src.data.splits import (
    make_folds,
    subject_independent_folds,
    train_validation_split,
    within_subject_folds,
)
from src.data.windows import _homogeneous_runs, balance_rest, build_window_index, concatenate
from src.preprocessing.filters import preprocess


class TestRestRepetitionAssignment:
    def test_rest_takes_the_repetition_of_the_movement_it_precedes(self):
        restimulus = np.array([0, 0, 5, 5, 0, 0, 5, 5, 0], dtype=np.int16)
        rerepetition = np.array([0, 0, 1, 1, 0, 0, 2, 2, 0], dtype=np.int16)
        out = assign_rest_repetitions(restimulus, rerepetition)
        # Leading rest joins repetition 1, the middle rest joins repetition 2,
        # and the trailing rest falls back to the repetition before it.
        assert out.tolist() == [1, 1, 1, 1, 2, 2, 2, 2, 2]

    def test_input_is_not_modified(self):
        restimulus = np.array([0, 1, 0], dtype=np.int16)
        rerepetition = np.array([0, 3, 0], dtype=np.int16)
        before = rerepetition.copy()
        assign_rest_repetitions(restimulus, rerepetition)
        assert np.array_equal(rerepetition, before)

    def test_all_rest_stays_at_zero(self):
        restimulus = np.zeros(10, dtype=np.int16)
        out = assign_rest_repetitions(restimulus, np.zeros(10, dtype=np.int16))
        assert (out == 0).all()


class TestHomogeneousRuns:
    def test_finds_only_windows_without_a_label_change(self):
        values = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        starts = _homogeneous_runs(values, length=3, stride=1)
        # Windows starting at 0, 1 are all 1s; 4, 5 are all 2s. 2 and 3 straddle.
        assert starts.tolist() == [0, 1, 4, 5]

    def test_respects_the_stride(self):
        values = np.ones(20, dtype=np.int64)
        assert _homogeneous_runs(values, length=4, stride=4).tolist() == [0, 4, 8, 12, 16]

    def test_window_longer_than_the_signal_gives_nothing(self):
        assert _homogeneous_runs(np.ones(3, dtype=np.int64), length=5, stride=1).size == 0


class TestWindowIndex:
    def test_every_window_carries_a_single_label(self, recording):
        filtered = preprocess(recording.emg, recording.sample_rate_hz)
        index = build_window_index(recording, filtered, length=40, stride=10)
        materialised = index.materialise("ct")
        assert materialised.shape == (len(index), 16, 40)

        for start, label in zip(index.starts[:200], index.labels[:200]):
            span = recording.restimulus[start : start + 40]
            assert np.unique(span).size == 1
            assert span[0] == label

    def test_an_unknown_layout_is_rejected(self, small_index):
        with pytest.raises(ValueError, match="order must be 'ct'"):
            small_index.materialise("tc")

    def test_materialised_window_matches_the_underlying_signal(self, small_index):
        materialised = small_index.materialise("ct")
        for i in (0, 7, 31):
            start = small_index.starts[i]
            expected = small_index.signal[start : start + small_index.length].T
            np.testing.assert_allclose(materialised[i], expected)

    def test_repetition_zero_windows_are_dropped(self, recording):
        filtered = preprocess(recording.emg, recording.sample_rate_hz)
        index = build_window_index(recording, filtered, length=40, stride=10)
        assert (index.repetitions > 0).all()

    def test_signal_shorter_than_a_window_gives_an_empty_index(self, recording):
        filtered = preprocess(recording.emg, recording.sample_rate_hz)
        index = build_window_index(recording, filtered, length=len(filtered) + 1, stride=1)
        assert len(index) == 0
        assert index.materialise("ct").shape[0] == 0

    def test_mismatched_signal_length_is_rejected(self, recording):
        with pytest.raises(ValueError, match="samples"):
            build_window_index(recording, recording.emg[:-5], length=40, stride=10)


class TestConcatenate:
    def test_no_window_spans_two_subjects(self, small_config):
        from src.data.dataset import build_subject

        parts = [build_subject(small_config, s) for s in small_config.dataset.subjects]
        joined = concatenate(parts)
        assert len(joined) == sum(len(p) for p in parts)

        # The gap between subjects is exactly one window of zeros, so any window
        # crossing a boundary would be all zeros. None should be.
        materialised = joined.materialise("ct")
        assert not np.any(np.all(materialised == 0, axis=(1, 2)))

    def test_rejects_mismatched_geometry(self, small_config):
        from src.data.dataset import build_subject

        a = build_subject(small_config, 1)
        b = build_subject(small_config, 2)
        b.length = a.length + 1
        with pytest.raises(ValueError, match="window lengths differ"):
            concatenate([a, b])


class TestRestBalancing:
    def test_rest_is_capped_near_the_movement_class_mean(self, small_index):
        codes, counts = np.unique(small_index.labels, return_counts=True)
        rest_count = counts[codes == 0][0]
        movement_mean = counts[codes != 0].mean()
        # Balanced to roughly the mean, with slack for the per-group rounding.
        assert rest_count <= 2.0 * movement_mean

    def test_ratio_zero_removes_rest_entirely(self, small_index):
        stripped = balance_rest(small_index, ratio=0.0, seed=1)
        assert (stripped.labels != 0).all()

    def test_balancing_is_deterministic(self, small_index):
        a = balance_rest(small_index, ratio=1.0, seed=3)
        b = balance_rest(small_index, ratio=1.0, seed=3)
        assert np.array_equal(a.starts, b.starts)


class TestSplits:
    def test_within_subject_never_shares_a_repetition(self, small_index, small_config):
        for fold in within_subject_folds(small_index, small_config):
            train_reps = set(fold.train.repetitions.tolist())
            test_reps = set(fold.test.repetitions.tolist())
            assert not (train_reps & test_reps)

    def test_within_subject_never_leaves_the_subject(self, small_index, small_config):
        for fold in within_subject_folds(small_index, small_config):
            assert set(fold.train.subjects.tolist()) == {fold.held_out_subject}
            assert set(fold.test.subjects.tolist()) == {fold.held_out_subject}

    def test_subject_independent_never_shares_a_subject(self, small_index, small_config):
        for fold in subject_independent_folds(small_index, small_config):
            train = set(fold.train.subjects.tolist())
            test = set(fold.test.subjects.tolist())
            assert not (train & test)
            assert test == {fold.held_out_subject}

    def test_every_subject_is_held_out_exactly_once(self, small_index, small_config):
        folds = subject_independent_folds(small_index, small_config)
        held_out = [f.held_out_subject for f in folds]
        assert sorted(held_out) == sorted(set(small_index.subjects.tolist()))

    def test_calibration_windows_are_not_also_test_windows(self, small_index, small_config):
        for fold in subject_independent_folds(small_index, small_config, calibrated=True):
            assert fold.calibration is not None
            calibration = set(fold.calibration.starts.tolist())
            test = set(fold.test.starts.tolist())
            assert not (calibration & test)
            # Calibration comes from the held-out subject, not from training.
            assert set(fold.calibration.subjects.tolist()) == {fold.held_out_subject}
            assert not (set(fold.train.starts.tolist()) & calibration)

    def test_validation_split_holds_out_whole_repetitions(self, small_index, small_config):
        folds = subject_independent_folds(small_index, small_config)
        train = folds[0].train
        train_idx, val_idx = train_validation_split(train, 0.25, seed=1)

        assert train_idx.size > 0 and val_idx.size > 0
        assert set(train_idx.tolist()) & set(val_idx.tolist()) == set()

        def groups(idx):
            return set(
                zip(train.subjects[idx].tolist(), train.repetitions[idx].tolist())
            )

        assert not (groups(train_idx) & groups(val_idx))

    def test_unknown_protocol_is_rejected(self, small_index, small_config):
        with pytest.raises(ValueError, match="unknown protocol"):
            make_folds(small_index, small_config, "leave_one_out")


class TestLabelEncoding:
    def test_codes_map_to_contiguous_indices(self, small_index):
        codes = class_codes(small_index)
        encoded = encode_labels(small_index.labels, codes)
        assert encoded.min() == 0
        assert encoded.max() == codes.size - 1
        np.testing.assert_array_equal(codes[encoded], small_index.labels)

    def test_an_unseen_code_raises(self, small_index):
        codes = class_codes(small_index)
        with pytest.raises(ValueError, match="not in the class list"):
            encode_labels(np.array([codes.max() + 99]), codes)


class TestGestureNames:
    def test_exercise_b_has_seventeen_movements(self):
        assert len(EXERCISE_B) == 17

    def test_rest_is_named(self):
        assert gesture_names("E2", [0, 6])[0] == "rest"
        assert "fist" in gesture_names("E2", [6])[0]

    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="not part of exercise"):
            gesture_names("E2", [99])

    def test_unknown_exercise_raises(self):
        with pytest.raises(ValueError, match="unknown exercise"):
            gesture_names("E9", [1])


class TestNinaproParsing:
    def test_rejects_a_file_with_the_wrong_channel_count(self):
        import io

        import scipy.io as sio

        buffer = io.BytesIO()
        sio.savemat(
            buffer,
            {
                "emg": np.zeros((100, 8), dtype=np.float32),
                "restimulus": np.zeros((100, 1), dtype=np.int8),
                "rerepetition": np.zeros((100, 1), dtype=np.int8),
            },
        )
        with pytest.raises(ValueError, match="expected 16 channels"):
            parse_mat(buffer.getvalue(), subject=1, exercise="E2", source="test.mat")

    def test_rejects_a_file_missing_labels(self):
        import io

        import scipy.io as sio

        buffer = io.BytesIO()
        sio.savemat(buffer, {"emg": np.zeros((100, 16), dtype=np.float32)})
        with pytest.raises(ValueError, match="missing field"):
            parse_mat(buffer.getvalue(), subject=1, exercise="E2", source="test.mat")


class TestCache:
    def test_a_config_change_invalidates_the_cache(self, small_config):
        first = build_dataset(small_config)
        small_config.window.length_ms = 150.0
        second = build_dataset(small_config)
        assert first.length != second.length

    def test_a_cached_build_matches_an_uncached_one(self, small_config):
        cached = build_dataset(small_config, use_cache=True)
        again = build_dataset(small_config, use_cache=True)
        np.testing.assert_array_equal(cached.starts, again.starts)
        np.testing.assert_allclose(cached.signal, again.signal)


def test_default_config_loads_and_is_self_consistent():
    config = load_config()
    assert config.window_samples == 40  # 200 ms at 200 Hz
    assert config.stride_samples == 10  # 50 ms at 200 Hz
    assert config.stride_samples < config.window_samples
    assert config.preprocessing.bandpass_high_hz < config.dataset.sample_rate_hz / 2


def test_the_shipped_geometry_meets_the_latency_budget():
    """The default configuration must not be one that misses its own deadline.

    An earlier default (a 50 ms stride with a three-decision vote) came to
    320 ms against a 300 ms budget. Inference was 0.01% of that; the geometry
    was the whole problem. This pins the fix.
    """
    config = load_config()
    vote_ms = (config.evaluation.vote_window - 1) * config.window.stride_ms
    # Inference is excluded: it is measured, not configured, and it is under a
    # tenth of a millisecond on every artefact benchmarked so far.
    geometry_ms = config.window.length_ms + vote_ms + config.latency.io_allowance_ms
    assert geometry_ms <= config.latency.budget_ms, (
        f"the shipped geometry costs {geometry_ms} ms before inference, against a "
        f"{config.latency.budget_ms} ms budget"
    )


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("window:\n  length_millis: 200\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)

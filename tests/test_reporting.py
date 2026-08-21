"""The reporting layer, and the README consistency check itself.

The README checker is the mechanism that keeps "every quantitative claim is
reproducible from a script" true after the fourth retraining run, so it is
worth testing rather than trusting.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.check_readme import BUILDERS, MARKER, regenerate
from src.eval.metrics import evaluate, pool
from src.eval.plots import confusion_matrix, latency_accuracy, per_class_bars, protocol_comparison
from src.evaluate import accuracy_table, gap_table, generalisation_gap, summary_rows


def _fake_run(model: str, protocol: str, accuracy: float) -> dict:
    """A results file shaped like the real thing, without training anything."""
    n = 100
    correct = int(round(accuracy * n))
    y_true = np.array([0] * (n // 2) + [1] * (n - n // 2))
    y_pred = y_true.copy()
    y_pred[correct:] = 1 - y_pred[correct:]
    result = evaluate(y_true, y_pred, np.array([0, 1]), ["rest", "fist"])
    pooled = pool([result], np.array([0, 1]), ["rest", "fist"])
    return {
        "model": model,
        "protocol": protocol,
        "synthetic": True,
        "wall_seconds": 1.0,
        "class_codes": [0, 1],
        "class_names": ["rest", "fist"],
        "dataset": {
            "n_windows": n,
            "n_channels": 16,
            "window_samples": 40,
            "sample_rate_hz": 200,
            "subjects": [1],
            "repetitions": [1],
            "class_counts": {0: 50, 1: 50},
        },
        "pooled": pooled.to_dict(),
        "per_fold": [],
        "config": {},
    }


class TestSummaryTables:
    def test_the_gap_is_the_difference_between_the_two_protocols(self):
        rows = summary_rows(
            [
                _fake_run("pytorch", "within_subject", 0.9),
                _fake_run("pytorch", "subject_independent", 0.6),
            ]
        )
        gaps = generalisation_gap(rows)
        assert len(gaps) == 1
        assert gaps[0]["gap"] == pytest.approx(
            gaps[0]["within_subject"] - gaps[0]["subject_independent"]
        )
        assert gaps[0]["relative_drop"] > 0

    def test_a_model_with_only_one_protocol_is_skipped(self):
        rows = summary_rows([_fake_run("pytorch", "within_subject", 0.9)])
        assert generalisation_gap(rows) == []

    def test_calibration_recovery_is_reported_as_a_fraction_of_the_gap(self):
        rows = summary_rows(
            [
                _fake_run("pytorch", "within_subject", 0.9),
                _fake_run("pytorch", "subject_independent", 0.6),
                _fake_run("pytorch", "subject_independent_calibrated", 0.75),
            ]
        )
        gap = generalisation_gap(rows)[0]
        assert 0.0 < gap["fraction_of_gap_recovered"] < 1.0

    def test_tables_render_without_a_missing_protocol_breaking_them(self):
        rows = summary_rows(
            [
                _fake_run("classical_lda", "within_subject", 0.8),
                _fake_run("pytorch", "within_subject", 0.7),
                _fake_run("pytorch", "subject_independent", 0.4),
            ]
        )
        table = accuracy_table(rows)
        # Header, rule, then one row per result.
        assert len(table.splitlines()) == 2 + 3
        assert "Within-subject" in table
        # Only one model has both protocols, so the gap table has a single row.
        assert len(gap_table(generalisation_gap(rows)).splitlines()) == 2 + 1


class TestPlots:
    def test_confusion_matrix_writes_a_file(self, tmp_path):
        matrix = np.eye(4) * 10
        path = confusion_matrix(matrix, ["a", "b", "c", "d"], tmp_path / "c.png")
        assert path.exists() and path.stat().st_size > 1000

    def test_a_non_square_matrix_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="square"):
            confusion_matrix(np.zeros((2, 3)), ["a", "b"], tmp_path / "c.png")

    def test_label_count_must_match(self, tmp_path):
        with pytest.raises(ValueError, match="labels for a"):
            confusion_matrix(np.eye(3), ["a", "b"], tmp_path / "c.png")

    def test_per_class_bars_writes_a_file(self, tmp_path):
        path = per_class_bars(["a", "b"], [0.5, 0.7], [0.6, 0.8], tmp_path / "p.png")
        assert path.exists()

    def test_protocol_comparison_writes_a_file(self, tmp_path):
        rows = summary_rows(
            [
                _fake_run("pytorch", "within_subject", 0.8),
                _fake_run("pytorch", "subject_independent", 0.4),
            ]
        )
        assert protocol_comparison(rows, tmp_path / "b.png").exists()

    def test_latency_accuracy_writes_a_file(self, tmp_path):
        rows = [
            {"vote_window": 1, "geometry_latency_ms": 120.0, "balanced_accuracy": 0.27},
            {"vote_window": 3, "geometry_latency_ms": 220.0, "balanced_accuracy": 0.28},
        ]
        assert latency_accuracy(rows, 300.0, tmp_path / "l.png").exists()


class TestReadmeMarkers:
    def test_the_readme_has_a_marker_for_every_builder(self):
        from src.config import resolve

        text = resolve("README.md").read_text(encoding="utf-8")
        found = {m.group("name") for m in MARKER.finditer(text)}
        assert found == set(BUILDERS), (
            f"markers without builders: {sorted(found - set(BUILDERS))}; "
            f"builders without markers: {sorted(set(BUILDERS) - found)}"
        )

    def test_every_marker_is_closed(self):
        from src.config import resolve

        text = resolve("README.md").read_text(encoding="utf-8")
        assert text.count("<!-- BEGIN:") == text.count("<!-- END:")
        assert text.count("<!-- BEGIN:") == len(BUILDERS)

    def test_an_unknown_marker_is_reported(self, small_config):
        text = "<!-- BEGIN:not_a_block -->\nx\n<!-- END:not_a_block -->\n"
        with pytest.raises(ValueError, match="no builder"):
            regenerate(text, small_config)

    def test_a_missing_marker_is_reported(self, small_config):
        # A README with only one of the ten blocks should be rejected, because a
        # builder with nowhere to write is a silently dropped claim.
        text = "<!-- BEGIN:accuracy -->\nx\n<!-- END:accuracy -->\n"
        with pytest.raises(ValueError, match="no marker"):
            regenerate(text, small_config)


class TestResultSerialisation:
    def test_a_result_survives_a_json_round_trip(self):
        run = _fake_run("pytorch", "subject_independent", 0.6)
        restored = json.loads(json.dumps(run))
        assert restored["pooled"]["accuracy"] == pytest.approx(run["pooled"]["accuracy"])
        assert np.array(restored["pooled"]["confusion"]).sum() == 100

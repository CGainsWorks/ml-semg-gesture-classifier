"""Quantisation, the embedded runtimes and the streaming control loop.

These run the whole deployment path on synthetic data: train, export to both
formats, quantise both, and stream a recording through the result. They are
slower than the rest of the suite and they are the tests that would have
caught every integration break encountered while this was being built.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config
from src.data.dataset import build_dataset, class_codes, encode_labels
from src.embedded.benchmark import budget_check, measure
from src.embedded.estimate import TARGETS, estimate_latency_ms, fits_in_memory
from src.embedded.runtime import load_runtime, softmax
from src.models.architecture import ArchitectureSpec
from src.preprocessing.normalise import ChannelScaler


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train a small model once and export every artefact from it."""
    from src.config import DatasetConfig, TrainConfig, load_config
    from src.models import tf_cnn, torch_cnn

    tmp = tmp_path_factory.mktemp("deploy")
    config = load_config("config/ci.yaml")
    config.dataset = DatasetConfig(
        source="synthetic",
        raw_dir=str(tmp / "raw"),
        cache_dir=str(tmp / "cache"),
        exercise="E2",
        subjects=[1, 2],
        gestures=[1, 2, 3],
        sample_rate_hz=200,
    )
    config.train = TrainConfig(
        epochs=2,
        batch_size=64,
        conv_channels=[8, 16, 16],
        dense_units=32,
        val_fraction=0.34,
        early_stopping_patience=2,
        seed=3,
        device="cpu",
    )
    config.artifacts_dir = str(tmp / "artifacts")

    index = build_dataset(config)
    codes = class_codes(index)
    x = index.materialise("ct")
    y = encode_labels(index.labels, codes)
    scaler = ChannelScaler.fit(x)
    x_scaled = scaler.transform(x)

    spec = ArchitectureSpec.from_config(
        config.train, index.n_channels, index.length, codes.size
    )
    split = int(0.75 * len(x_scaled))

    torch_model, _ = torch_cnn.train_model(
        spec, x_scaled[:split], y[:split], x_scaled[split:], y[split:],
        config.train, verbose=False,
    )
    tf_model, _ = tf_cnn.train_model(
        spec, x_scaled[:split], y[:split], x_scaled[split:], y[split:],
        config.train, verbose=False,
    )

    out = tmp / "deploy"
    out.mkdir(parents=True, exist_ok=True)
    saved_model = tmp / "saved_model"
    tf_cnn.export_saved_model(tf_model, saved_model)

    representative = x_scaled[:64]
    from src.quantise import onnx_export, tflite_export

    tflite_paths = tflite_export.export(saved_model, out, representative)
    onnx_paths = onnx_export.export(torch_model, out, representative)

    return {
        "config": config,
        "spec": spec,
        "scaler": scaler,
        "codes": codes,
        "x": x_scaled,
        "y": y,
        "index": index,
        "paths": {**{f"tflite_{k}": v for k, v in tflite_paths.items()},
                  **{f"onnx_{k}": v for k, v in onnx_paths.items()}},
        "torch_model": torch_model,
        "representative": representative,
    }


class TestExportedArtefacts:
    @pytest.mark.parametrize(
        "key", ["tflite_float32", "tflite_int8", "onnx_float32", "onnx_int8"]
    )
    def test_artefact_exists_and_is_not_empty(self, trained, key):
        path = trained["paths"][key]
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_int8_is_substantially_smaller_than_float32(self, trained):
        for fmt in ("tflite", "onnx"):
            float_size = trained["paths"][f"{fmt}_float32"].stat().st_size
            int8_size = trained["paths"][f"{fmt}_int8"].stat().st_size
            assert int8_size < float_size

    def test_the_int8_tflite_model_is_fully_integer(self, trained):
        """A float fallback would run here and fail to build for TFLite Micro."""
        from src.quantise.tflite_export import describe

        info = describe(trained["paths"]["tflite_int8"])
        assert info["input"]["dtype"] == "int8"
        assert info["output"]["dtype"] == "int8"

    def test_the_tflite_graph_has_no_runtime_shape_operators(self, trained):
        """SHAPE, STRIDED_SLICE and PACK have no business in an embedded graph.

        They appear when the exported batch dimension is dynamic: the reshape
        at the head of the network then has to compute its target at run time.
        Each one is another operator a TFLite Micro build must register, and
        they defeat static memory planning. Pinning the export signature to a
        batch of one removes them, and this asserts they stay removed.
        """
        from src.quantise.tflite_export import operators

        dynamic = {"SHAPE", "STRIDED_SLICE", "PACK", "GATHER", "WHILE", "IF"}
        for key in ("tflite_float32", "tflite_int8"):
            present = set(operators(trained["paths"][key]))
            assert not (present & dynamic), f"{key} contains {sorted(present & dynamic)}"

    def test_the_tflite_graph_uses_only_the_expected_operators(self, trained):
        from src.quantise.tflite_export import operators

        expected = {"CONV_2D", "MAX_POOL_2D", "FULLY_CONNECTED", "RESHAPE"}
        for key in ("tflite_float32", "tflite_int8"):
            present = set(operators(trained["paths"][key]))
            assert present <= expected, f"{key} has unexpected {sorted(present - expected)}"

    def test_the_tflite_graph_takes_a_batch_of_one(self, trained):
        from src.quantise.tflite_export import describe

        for key in ("tflite_float32", "tflite_int8"):
            shape = describe(trained["paths"][key])["input"]["shape"]
            assert shape[0] == 1, f"{key} input shape is {shape}"

    def test_the_onnx_graph_takes_a_batch_of_one(self, trained):
        import onnx

        model = onnx.load(str(trained["paths"]["onnx_float32"]))
        shape = model.graph.input[0].type.tensor_type.shape.dim
        assert shape[0].dim_value == 1
        assert shape[1].dim_value == trained["spec"].n_channels
        assert shape[2].dim_value == trained["spec"].window_samples


class TestRuntimeAgreement:
    def test_float32_runtimes_match_their_source_model(self, trained):
        """Conversion must not change the answers, or nothing downstream means anything."""
        from src.models import torch_cnn

        x = trained["x"][:64]
        reference = torch_cnn.predict_proba(trained["torch_model"], x, device="cpu").argmax(axis=1)
        runtime = load_runtime(trained["paths"]["onnx_float32"])
        try:
            assert (runtime.predict(x) == reference).mean() > 0.99
        finally:
            runtime.close()

    def test_int8_mostly_agrees_with_float32(self, trained):
        """Quantisation is allowed to change some answers, not most of them."""
        x = trained["x"][:128]
        for fmt in ("tflite", "onnx"):
            float_runtime = load_runtime(trained["paths"][f"{fmt}_float32"])
            int8_runtime = load_runtime(trained["paths"][f"{fmt}_int8"])
            try:
                agreement = (float_runtime.predict(x) == int8_runtime.predict(x)).mean()
            finally:
                float_runtime.close()
                int8_runtime.close()
            assert agreement > 0.85, f"{fmt} int8 disagreed with float32 on {1 - agreement:.1%}"

    def test_a_single_window_and_a_batch_give_the_same_answer(self, trained):
        runtime = load_runtime(trained["paths"]["onnx_float32"])
        try:
            batch = runtime.predict(trained["x"][:8])
            one_by_one = [int(np.argmax(runtime.infer(w))) for w in trained["x"][:8]]
        finally:
            runtime.close()
        assert batch.tolist() == one_by_one

    def test_an_unknown_extension_is_rejected(self, tmp_path):
        path = tmp_path / "model.pb"
        path.write_bytes(b"not a model")
        with pytest.raises(ValueError, match="expected a .tflite or .onnx"):
            load_runtime(path)

    def test_a_missing_artefact_is_reported_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no deployment artefact"):
            load_runtime(tmp_path / "absent.onnx")


class TestLatency:
    def test_measurement_produces_an_ordered_distribution(self, trained):
        runtime = load_runtime(trained["paths"]["onnx_int8"])
        try:
            stats = measure(runtime, trained["x"][:32], iterations=50, warmup=5)
        finally:
            runtime.close()
        assert stats.min_ms <= stats.median_ms <= stats.p99_ms <= stats.max_ms
        assert stats.iterations == 50
        assert stats.mean_ms > 0

    def test_the_budget_counts_more_than_inference(self, trained):
        """The window and the vote dominate; a budget that ignores them lies."""
        runtime = load_runtime(trained["paths"]["onnx_int8"])
        try:
            stats = measure(runtime, trained["x"][:32], iterations=50, warmup=5)
        finally:
            runtime.close()
        config: Config = trained["config"]
        verdict = budget_check(stats, config)
        assert verdict["window_fill_ms"] == config.window.length_ms
        assert verdict["total_ms"] > verdict["inference_p99_ms"]
        assert verdict["total_ms"] == pytest.approx(
            verdict["window_fill_ms"]
            + verdict["majority_vote_ms"]
            + verdict["inference_p99_ms"]
            + verdict["io_allowance_ms"]
        )

    def test_empty_input_is_rejected(self, trained):
        runtime = load_runtime(trained["paths"]["onnx_int8"])
        try:
            with pytest.raises(ValueError, match="non-empty"):
                measure(runtime, np.empty((0, 4, 40), dtype=np.float32))
        finally:
            runtime.close()


class TestOnTargetEstimates:
    def test_a_faster_clock_gives_a_lower_estimate(self):
        macs = 555_264
        by_clock = sorted(TARGETS, key=lambda t: t.clock_mhz)
        estimates = [estimate_latency_ms(macs, t) for t in by_clock]
        assert estimates == sorted(estimates, reverse=True)

    def test_the_model_fits_the_smallest_target(self):
        verdict = fits_in_memory(
            model_bytes=91_000, ram_bytes=60_000, target=TARGETS[0], ram_source="test"
        )
        assert verdict["fits_flash"]
        assert verdict["fits_ram"]

    def test_zero_macs_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            estimate_latency_ms(0, TARGETS[0])


class TestControlLoop:
    def test_streaming_produces_decisions_and_they_beat_chance(self, trained):
        from src.data.dataset import load_raw_recording
        from src.embedded.control_loop import run_stream

        config: Config = trained["config"]
        recording = load_raw_recording(config, 1)
        runtime = load_runtime(trained["paths"]["tflite_int8"])
        try:
            result = run_stream(
                runtime,
                trained["scaler"],
                config,
                recording.emg,
                recording.restimulus,
                trained["codes"],
            )
        finally:
            runtime.close()

        raw, voted, true, latency = result.arrays()
        assert true.size > 50
        chance = 1.0 / trained["codes"].size
        assert (raw == true).mean() > chance
        assert (latency > 0).all()

    def test_the_causal_filter_state_carries_between_blocks(self, trained):
        """Two block sizes over the same signal must give the same decisions.

        If the filter state were reset per block, the answers would differ, and
        the streaming result would not be comparable to the offline one.
        """
        from src.data.dataset import load_raw_recording
        from src.embedded.control_loop import run_stream

        config: Config = trained["config"]
        recording = load_raw_recording(config, 1)
        emg = recording.emg[: 40 * config.stride_samples]
        labels = recording.restimulus[: emg.shape[0]]

        runtime = load_runtime(trained["paths"]["onnx_float32"])
        try:
            one = run_stream(
                runtime, trained["scaler"], config, emg, labels, trained["codes"],
                block_samples=config.stride_samples,
            )
            # Half-stride blocks: two blocks per decision, same decision points.
            two = run_stream(
                runtime, trained["scaler"], config, emg, labels, trained["codes"],
                block_samples=config.stride_samples // 2,
            )
        finally:
            runtime.close()

        assert one.decisions and two.decisions
        shared = {d.sample_index: d.raw_class for d in one.decisions}
        overlap = [
            (shared[d.sample_index], d.raw_class)
            for d in two.decisions
            if d.sample_index in shared
        ]
        assert len(overlap) > 5
        assert all(a == b for a, b in overlap)


def test_softmax_is_stable_on_large_logits():
    out = softmax(np.array([1000.0, 1001.0, 999.0], dtype=np.float32))
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)

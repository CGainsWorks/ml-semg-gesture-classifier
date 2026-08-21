"""Filters, features and normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.time_domain import (
    ar_coefficients,
    channel_sigma,
    extract,
    feature_names,
    mav,
    rms,
    slope_sign_changes,
    waveform_length,
    zero_crossings,
)
from src.preprocessing.filters import (
    bandpass_sos,
    causal_bandpass,
    envelope,
    notch_sos,
    preprocess,
    rectify,
)
from src.preprocessing.normalise import ChannelScaler, FeatureScaler

RATE = 200.0


def _tone(freq_hz: float, n: int = 2000, channels: int = 4, rate: float = RATE) -> np.ndarray:
    t = np.arange(n) / rate
    return np.tile(np.sin(2 * np.pi * freq_hz * t)[:, None], (1, channels)).astype(np.float32)


class TestBandpass:
    def test_passes_in_band_and_rejects_out_of_band(self):
        in_band = _tone(50.0)
        below = _tone(2.0)
        passed = preprocess(in_band, RATE, 10.0, 99.0)
        blocked = preprocess(below, RATE, 10.0, 99.0)
        # Ignore the filter's edge transient at both ends.
        assert passed[200:-200].std() > 0.5 * in_band.std()
        assert blocked[200:-200].std() < 0.1 * below.std()

    def test_high_cut_at_or_above_nyquist_is_rejected(self):
        with pytest.raises(ValueError, match="Nyquist"):
            bandpass_sos(10.0, 100.0, RATE)

    def test_inverted_band_is_rejected(self):
        with pytest.raises(ValueError, match="low < high"):
            bandpass_sos(90.0, 20.0, RATE)

    def test_a_signal_shorter_than_the_filter_padding_is_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            preprocess(np.zeros((10, 4), dtype=np.float32), RATE)

    def test_output_shape_and_dtype_are_preserved(self):
        x = _tone(40.0)
        out = preprocess(x, RATE)
        assert out.shape == x.shape
        assert out.dtype == np.float32


class TestCausalBandpass:
    def test_block_by_block_matches_a_single_pass(self):
        x = _tone(45.0, n=1200)
        whole, _ = causal_bandpass(x, RATE)

        state = None
        blocks = []
        for start in range(0, x.shape[0], 100):
            block, state = causal_bandpass(x[start : start + 100], RATE, state=state)
            blocks.append(block)
        streamed = np.concatenate(blocks)
        np.testing.assert_allclose(whole, streamed, atol=1e-5)

    def test_causal_output_lags_the_zero_phase_one(self):
        """The reason the streaming controller has to be measured separately.

        ``filtfilt`` is zero-phase because it runs the filter both ways. The
        causal filter cannot be, and the resulting group delay is real latency
        the wearer feels.
        """
        # A burst, not a continuous tone: a steady sinusoid repeats every few
        # samples at this rate, so any delay in it is ambiguous. The onset of a
        # burst is not.
        n = 2000
        carrier = _tone(40.0, n=n, channels=1)
        gate = np.zeros((n, 1), dtype=np.float32)
        gate[800:1200] = 1.0
        x = carrier * gate

        zero_phase = preprocess(x, RATE)
        causal = causal_bandpass(x, RATE)[0]

        def energy_centroid(signal: np.ndarray) -> float:
            """Where the burst's energy sits in time."""
            energy = np.square(signal[:, 0])
            return float((np.arange(energy.size) * energy).sum() / energy.sum())

        burst_centre = 1000.0
        # filtfilt is zero-phase by construction, so its output sits exactly
        # where the burst did.
        assert energy_centroid(zero_phase) == pytest.approx(burst_centre, abs=0.2)

        # The causal filter cannot respond before the burst starts, so its
        # energy arrives later. The delay is small at this rate, around three
        # milliseconds, which is why it is a footnote in the latency budget
        # rather than a problem. It is measured here rather than assumed.
        delay_samples = energy_centroid(causal) - energy_centroid(zero_phase)
        assert 0.0 < delay_samples < 5.0


class TestNotch:
    def test_removes_the_notched_frequency(self):
        x = _tone(50.0, n=3000)
        from src.preprocessing.filters import apply_sos

        out = apply_sos(x, notch_sos(50.0, RATE, quality=30.0))
        assert out[400:-400].std() < 0.2 * x.std()

    def test_frequency_above_nyquist_is_rejected(self):
        with pytest.raises(ValueError, match="Nyquist"):
            notch_sos(150.0, RATE)


class TestEnvelope:
    def test_envelope_is_non_negative_and_smooth(self):
        x = _tone(60.0, n=3000)
        out = envelope(preprocess(x, RATE), RATE, cutoff_hz=5.0)
        interior = out[400:-400]
        assert (interior >= -1e-4).all()
        # The envelope of a constant-amplitude tone should barely move.
        assert interior.std() < 0.2 * np.abs(interior).mean()

    def test_rectify_matches_abs(self):
        x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        np.testing.assert_array_equal(rectify(x), np.abs(x))


class TestTimeDomainFeatures:
    def test_mav_and_rms_on_a_known_signal(self):
        x = np.ones((2, 3, 10), dtype=np.float32) * 2.0
        np.testing.assert_allclose(mav(x), 2.0)
        np.testing.assert_allclose(rms(x), 2.0)

    def test_waveform_length_of_a_ramp(self):
        ramp = np.arange(10, dtype=np.float32)[None, None, :]
        assert waveform_length(ramp)[0, 0] == pytest.approx(9.0)

    def test_zero_crossings_counts_sign_changes(self):
        alternating = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)[None, None, :]
        counted = zero_crossings(alternating, np.array([0.1], dtype=np.float32))
        assert counted[0, 0] == 4.0

    def test_zero_crossings_ignores_sub_threshold_wobble(self):
        tiny = np.array([0.01, -0.01, 0.01, -0.01], dtype=np.float32)[None, None, :]
        assert zero_crossings(tiny, np.array([1.0], dtype=np.float32))[0, 0] == 0.0

    def test_slope_sign_changes_counts_turning_points(self):
        zigzag = np.array([0.0, 2.0, 0.0, 2.0, 0.0], dtype=np.float32)[None, None, :]
        counted = slope_sign_changes(zigzag, np.array([0.1], dtype=np.float32))
        assert counted[0, 0] == 3.0

    def test_ar_coefficients_recover_a_known_process(self):
        """An AR(1) process with a=0.8 should give a first coefficient near 0.8."""
        rng = np.random.default_rng(0)
        n = 4000
        signal = np.zeros(n, dtype=np.float32)
        for i in range(1, n):
            signal[i] = 0.8 * signal[i - 1] + rng.normal(scale=0.1)
        coeffs = ar_coefficients(signal[None, None, :], order=2)
        assert coeffs[0, 0, 0] == pytest.approx(0.8, abs=0.08)

    def test_feature_matrix_shape_and_names_agree(self, windows):
        sigma = channel_sigma(windows)
        features = extract(windows, sigma, ar_order=4)
        names = feature_names(windows.shape[1], ar_order=4)
        assert features.shape == (windows.shape[0], len(names))
        assert features.shape[1] == windows.shape[1] * (13 + 4)

    def test_features_are_finite(self, windows):
        features = extract(windows, channel_sigma(windows))
        assert np.isfinite(features).all()

    def test_sigma_shape_is_checked(self, windows):
        with pytest.raises(ValueError, match="one entry per channel"):
            extract(windows, np.ones(3, dtype=np.float32))

    def test_a_two_sample_window_is_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            channel_sigma(np.zeros((1, 2, 2), dtype=np.float32))

    def test_thresholds_scale_with_the_channel(self, windows):
        """Doubling a channel's gain should not change its zero-crossing count.

        This is the property that lets the feature set transfer between people
        whose electrodes sit differently.
        """
        scaled = windows.copy()
        scaled[:, 0, :] *= 2.0
        a = zero_crossings(windows, channel_sigma(windows) * 0.05)
        b = zero_crossings(scaled, channel_sigma(scaled) * 0.05)
        assert abs(a[:, 0].mean() - b[:, 0].mean()) < 0.6


class TestScalers:
    def test_channel_scaler_standardises_per_channel(self, windows):
        scaler = ChannelScaler.fit(windows)
        out = scaler.transform(windows)
        np.testing.assert_allclose(out.mean(axis=(0, 2)), 0.0, atol=1e-4)
        np.testing.assert_allclose(out.std(axis=(0, 2)), 1.0, atol=1e-3)

    def test_channel_scaler_round_trips_through_disk(self, windows, tmp_path):
        scaler = ChannelScaler.fit(windows)
        scaler.save(tmp_path / "s.npz")
        loaded = ChannelScaler.load(tmp_path / "s.npz")
        np.testing.assert_allclose(scaler.transform(windows), loaded.transform(windows))

    def test_channel_count_mismatch_is_rejected(self, windows):
        scaler = ChannelScaler.fit(windows)
        with pytest.raises(ValueError, match="channels"):
            scaler.transform(windows[:, :4, :])

    def test_a_dead_channel_does_not_produce_infinities(self):
        x = np.random.default_rng(0).normal(size=(20, 3, 40)).astype(np.float32)
        x[:, 1, :] = 0.0
        out = ChannelScaler.fit(x).transform(x)
        assert np.isfinite(out).all()

    def test_feature_scaler_leaves_a_constant_column_finite(self):
        x = np.ones((10, 3), dtype=np.float32)
        x[:, 0] = np.arange(10)
        out = FeatureScaler.fit(x).transform(x)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-6)

    def test_fitting_on_nothing_is_rejected(self):
        with pytest.raises(ValueError, match="zero windows"):
            ChannelScaler.fit(np.empty((0, 4, 40), dtype=np.float32))

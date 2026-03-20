import numpy as np
import pytest

from capture.mock import (
    generate_drone_signal,
    generate_ambient_noise,
    generate_mock_recording,
    DRONE_PROFILES,
    ENVIRONMENT_PROFILES,
)


class TestDroneSignal:
    def test_output_shape_4ch(self):
        signal = generate_drone_signal("fpv_5inch", 50, 1.0, 48000, 4)
        assert signal.shape == (48000, 4)

    def test_output_shape_2ch(self):
        signal = generate_drone_signal("fpv_5inch", 50, 1.0, 48000, 2)
        assert signal.shape == (48000, 2)

    def test_distance_attenuation(self):
        close = generate_drone_signal("fpv_5inch", 10, 1.0, 48000, 1)
        far = generate_drone_signal("fpv_5inch", 100, 1.0, 48000, 1)
        assert np.std(close) > np.std(far)

    def test_all_drone_types(self):
        for drone_type in DRONE_PROFILES:
            signal = generate_drone_signal(drone_type, 50, 0.5, 48000, 2)
            assert signal.shape[0] == 24000
            assert not np.all(signal == 0)

    def test_no_nan_or_inf(self):
        signal = generate_drone_signal("fpv_5inch", 50, 2.0)
        assert not np.any(np.isnan(signal))
        assert not np.any(np.isinf(signal))


class TestAmbientNoise:
    def test_output_shape(self):
        noise = generate_ambient_noise("open_field", 1.0, 48000, 4)
        assert noise.shape == (48000, 4)

    def test_suburban_louder_than_field(self):
        field = generate_ambient_noise("open_field", 1.0)
        suburban = generate_ambient_noise("suburban", 1.0)
        assert np.std(suburban) > np.std(field)

    def test_all_environments(self):
        for env in ENVIRONMENT_PROFILES:
            noise = generate_ambient_noise(env, 0.5, 48000, 2)
            assert noise.shape == (24000, 2)


class TestMockRecording:
    def test_default_output(self):
        rec, _ = generate_mock_recording()
        assert rec.shape == (480000, 4)  # 10s * 48000

    def test_2ch_output(self):
        rec, _ = generate_mock_recording(channels=2)
        assert rec.shape == (480000, 2)

    def test_no_clipping(self):
        rec, _ = generate_mock_recording(distance_m=5)
        assert np.max(np.abs(rec)) <= 1.0

    def test_custom_duration(self):
        rec, _ = generate_mock_recording(duration_s=5.0)
        assert rec.shape[0] == 240000


class TestProcessing:
    def test_spectrogram_output(self):
        from processing.spectrogram import compute_spectrogram
        rec, _ = generate_mock_recording(duration_s=2.0, channels=1)
        data = rec.T  # (channels, samples)
        f, t, Sxx = compute_spectrogram(data, 48000, channel=0)
        assert len(f) > 0
        assert len(t) > 0
        assert Sxx.shape[0] == len(f)

    def test_snr_positive_for_close_drone(self):
        from processing.spectrogram import compute_snr
        rec, _ = generate_mock_recording(distance_m=10, duration_s=2.0, channels=1)
        data = rec.T
        snr = compute_snr(data, 48000)
        assert snr > 0

    def test_snr_lower_for_far_drone(self):
        from processing.spectrogram import compute_snr
        close, _ = generate_mock_recording(distance_m=10, duration_s=2.0, channels=1)
        far, _ = generate_mock_recording(distance_m=200, duration_s=2.0, channels=1)
        snr_close = compute_snr(close.T, 48000)
        snr_far = compute_snr(far.T, 48000)
        assert snr_close > snr_far

    def test_peak_detection(self):
        from processing.spectrogram import detect_peaks
        rec, _ = generate_mock_recording(
            drone_type="fpv_5inch", distance_m=20, duration_s=2.0, channels=1
        )
        data = rec.T
        peaks = detect_peaks(data, 48000)
        assert len(peaks) > 0
        fundamental = DRONE_PROFILES["fpv_5inch"]["fundamental_hz"]
        closest = min(peaks, key=lambda p: abs(p["frequency_hz"] - fundamental))
        assert abs(closest["frequency_hz"] - fundamental) < 50

    def test_mfcc_output(self):
        from processing.spectrogram import compute_mfcc
        rec, _ = generate_mock_recording(duration_s=2.0, channels=1)
        data = rec.T
        mfcc = compute_mfcc(data, 48000)
        assert mfcc.shape[0] == 13


class TestRadar:
    def test_mock_radar_frame(self):
        from radar.mmwave import AWR1843
        radar = AWR1843("", "", mock=True)
        radar.configure()
        frame = radar.read_frame()
        assert frame is not None
        assert frame.frame_number == 1
        radar.close()

    def test_mock_radar_capture(self):
        from radar.mmwave import AWR1843
        radar = AWR1843("", "", mock=True)
        radar.configure()
        frames = radar.capture_frames(duration_s=1.0)
        assert isinstance(frames, list)
        radar.close()


class TestPeakFundamentals:
    def test_each_drone_fundamental_detected(self):
        from processing.spectrogram import detect_peaks
        for name, profile in DRONE_PROFILES.items():
            rec, _ = generate_mock_recording(
                drone_type=name, distance_m=20, duration_s=2.0, channels=1, seed=42
            )
            peaks = detect_peaks(rec.T, 48000)
            assert len(peaks) > 0, f"No peaks for {name}"
            fundamental = profile["fundamental_hz"]
            closest = min(peaks, key=lambda p: abs(p["frequency_hz"] - fundamental))
            assert abs(closest["frequency_hz"] - fundamental) < 50, (
                f"{name}: expected ~{fundamental} Hz, got {closest['frequency_hz']:.0f} Hz"
            )


class TestReproducibility:
    def test_seed_produces_identical_output(self):
        a, _ = generate_mock_recording(duration_s=1.0, channels=1, seed=123)
        b, _ = generate_mock_recording(duration_s=1.0, channels=1, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a, _ = generate_mock_recording(duration_s=1.0, channels=1, seed=1)
        b, _ = generate_mock_recording(duration_s=1.0, channels=1, seed=2)
        assert not np.array_equal(a, b)


class TestEdgeCases:
    def test_zero_duration(self):
        signal = generate_drone_signal("fpv_5inch", 50, 0.0, 48000, 1)
        assert signal.shape == (0, 1)

    def test_very_large_distance(self):
        rec, _ = generate_mock_recording(distance_m=10000, duration_s=1.0, channels=1)
        assert not np.any(np.isnan(rec))
        assert not np.any(np.isinf(rec))

    def test_minimum_distance_clamp(self):
        rec, _ = generate_mock_recording(distance_m=0.1, duration_s=1.0, channels=1)
        assert np.max(np.abs(rec)) <= 1.0


class TestDoppler:
    def test_doppler_preserves_shape(self):
        no_doppler = generate_drone_signal("fpv_5inch", 50, 1.0, 48000, 1, approaching=False)
        with_doppler = generate_drone_signal("fpv_5inch", 50, 1.0, 48000, 1, approaching=True)
        assert no_doppler.shape == with_doppler.shape

    def test_doppler_no_nan(self):
        signal = generate_drone_signal("fpv_5inch", 50, 2.0, 48000, 4, approaching=True)
        assert not np.any(np.isnan(signal))


class TestLoadAudio:
    def test_wav_round_trip(self, tmp_path):
        from capture.recorder import save_wav
        from processing.spectrogram import load_audio
        data, _ = generate_mock_recording(duration_s=1.0, channels=2, seed=42)
        wav_path = tmp_path / "test.wav"
        save_wav(data, wav_path, 48000)
        y, sr = load_audio(wav_path, sr=48000)
        assert sr == 48000
        assert y.shape == (2, 48000)
        # int16 quantization + save(32767)/load(32768) asymmetry
        np.testing.assert_allclose(y, data.T, atol=5e-5, rtol=0)

    def test_mono_downmix(self, tmp_path):
        from capture.recorder import save_wav
        from processing.spectrogram import load_audio
        data, _ = generate_mock_recording(duration_s=0.5, channels=4, seed=42)
        wav_path = tmp_path / "test.wav"
        save_wav(data, wav_path, 48000)
        y, sr = load_audio(wav_path, sr=48000, mono=True)
        assert y.shape[0] == 1

    def test_single_channel(self, tmp_path):
        from capture.recorder import save_wav
        from processing.spectrogram import load_audio
        data, _ = generate_mock_recording(duration_s=0.5, channels=1, seed=42)
        wav_path = tmp_path / "test.wav"
        save_wav(data, wav_path, 48000)
        y, sr = load_audio(wav_path, sr=48000)
        assert y.shape == (1, 24000)


class TestAdversarialConditions:
    def test_quiet_props_preserves_shape(self):
        from experiments.runner import _apply_adversarial_condition
        data, _ = generate_mock_recording(duration_s=1.0, channels=2, seed=10)
        result = _apply_adversarial_condition(data, "quiet_props", 48000)
        assert result.shape == data.shape

    def test_low_throttle_preserves_shape(self):
        from experiments.runner import _apply_adversarial_condition
        data, _ = generate_mock_recording(duration_s=1.0, channels=2, seed=10)
        result = _apply_adversarial_condition(data, "low_throttle", 48000)
        assert result.shape == data.shape

    def test_low_throttle_reduces_amplitude(self):
        from experiments.runner import _apply_adversarial_condition
        data, _ = generate_mock_recording(duration_s=1.0, channels=1, seed=10)
        result = _apply_adversarial_condition(data, "low_throttle", 48000)
        assert np.std(result) < np.std(data)

    def test_standard_props_is_identity(self):
        from experiments.runner import _apply_adversarial_condition
        data, _ = generate_mock_recording(duration_s=0.5, channels=1, seed=10)
        result = _apply_adversarial_condition(data, "standard_props", 48000)
        np.testing.assert_array_equal(result, data)

    def test_low_throttle_no_nan(self):
        from experiments.runner import _apply_adversarial_condition
        data, _ = generate_mock_recording(duration_s=2.0, channels=4, seed=10)
        result = _apply_adversarial_condition(data, "low_throttle", 48000)
        assert not np.any(np.isnan(result))


class TestFirstDetectionDistance:
    def test_detects_close_drone(self, tmp_path):
        from capture.recorder import save_wav, save_metadata
        from processing.spectrogram import first_detection_distance
        data, _ = generate_mock_recording(
            distance_m=10, duration_s=3.0, channels=1, seed=42
        )
        save_wav(data, tmp_path / "recording.wav", 48000)
        result = first_detection_distance(tmp_path, snr_threshold=3.0)
        assert result["first_detection_time_s"] is not None
        assert result["first_detection_snr_db"] >= 3.0
        assert len(result["snr_timeline"]) > 0

    def test_no_detection_for_noise_only(self, tmp_path):
        from capture.recorder import save_wav
        from processing.spectrogram import first_detection_distance
        # Very far drone = effectively noise
        data, _ = generate_mock_recording(
            distance_m=50000, duration_s=2.0, channels=1, seed=42
        )
        save_wav(data, tmp_path / "recording.wav", 48000)
        result = first_detection_distance(tmp_path, snr_threshold=10.0)
        assert result["first_detection_time_s"] is None


class TestSNREstimator:
    def test_noise_only_gives_negative_snr(self):
        from processing.spectrogram import compute_snr
        from capture.mock import generate_ambient_noise
        rng = np.random.default_rng(42)
        noise = generate_ambient_noise("open_field", 2.0, 48000, 1, rng)
        snr = compute_snr(noise.T, 48000)
        assert snr < 0, f"Noise-only SNR should be negative, got {snr:.1f}"

    def test_close_drone_higher_snr_than_noise(self):
        from processing.spectrogram import compute_snr
        from capture.mock import generate_ambient_noise
        rng = np.random.default_rng(42)
        noise = generate_ambient_noise("open_field", 2.0, 48000, 1, rng)
        drone, _ = generate_mock_recording(distance_m=10, duration_s=2.0, channels=1, seed=42)
        snr_noise = compute_snr(noise.T, 48000)
        snr_drone = compute_snr(drone.T, 48000)
        assert snr_drone > snr_noise


class TestStatistics:
    def test_confidence_interval(self):
        from processing.statistics import confidence_interval_95
        data = [10.0, 11.0, 12.0, 13.0, 14.0]
        lo, hi = confidence_interval_95(data)
        assert lo < 12.0 < hi
        assert lo > 8.0
        assert hi < 16.0

    def test_cohens_d_large_effect(self):
        from processing.statistics import cohens_d
        g1 = np.array([10, 11, 12, 13, 14], dtype=float)
        g2 = np.array([20, 21, 22, 23, 24], dtype=float)
        d = cohens_d(g1, g2)
        assert abs(d) > 2.0  # Very large effect

    def test_welch_ttest_significant(self):
        from processing.statistics import welch_ttest
        g1 = np.random.default_rng(1).normal(10, 1, 30)
        g2 = np.random.default_rng(2).normal(15, 1, 30)
        result = welch_ttest(g1, g2)
        assert result["significant"] is True
        assert result["p_value"] < 0.01

    def test_welch_ttest_not_significant(self):
        from processing.statistics import welch_ttest
        g1 = np.random.default_rng(1).normal(10, 1, 30)
        g2 = np.random.default_rng(2).normal(10, 1, 30)
        result = welch_ttest(g1, g2)
        assert result["p_value"] > 0.01

    def test_roc_perfect_separation(self):
        from processing.statistics import compute_roc
        pos = np.array([10, 11, 12, 13, 14], dtype=float)
        neg = np.array([0, 1, 2, 3, 4], dtype=float)
        roc = compute_roc(pos, neg)
        assert roc["auc"] > 0.99

    def test_roc_no_separation(self):
        from processing.statistics import compute_roc
        rng = np.random.default_rng(42)
        pos = rng.normal(5, 2, 100)
        neg = rng.normal(5, 2, 100)
        roc = compute_roc(pos, neg)
        assert 0.3 < roc["auc"] < 0.7

    def test_summarize_condition(self):
        from processing.statistics import summarize_condition
        data = list(np.random.default_rng(42).normal(10, 2, 30))
        s = summarize_condition(data)
        assert s["n"] == 30
        assert 8 < s["mean"] < 12
        assert s["ci_95_low"] < s["mean"] < s["ci_95_high"]

    def test_detection_rate(self):
        from processing.statistics import detection_rate
        snrs = [1.0, 2.0, 3.0, 4.0, 5.0]
        rate = detection_rate(snrs, threshold=3.0)
        assert rate == 0.6  # 3.0, 4.0, 5.0 are >= 3.0

    def test_bonferroni_correction(self):
        from processing.statistics import bonferroni_correct
        corrected = bonferroni_correct([0.01, 0.03, 0.05])
        assert abs(corrected[0] - 0.03) < 1e-10
        assert abs(corrected[1] - 0.09) < 1e-10
        assert abs(corrected[2] - 0.15) < 1e-10

    def test_csv_export(self, tmp_path):
        from processing.statistics import export_experiment_csv
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        path = tmp_path / "test.csv"
        export_experiment_csv(rows, path)
        assert path.exists()
        import csv
        with open(path) as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["a"] == "1"


class TestExperimentSeeds:
    def test_exp1_reproducible(self):
        from experiments.runner import _condition_seed, BASE_SEED
        s1 = _condition_seed(BASE_SEED, "exp1", "fpv_5inch", "open_field", 25, 0)
        s2 = _condition_seed(BASE_SEED, "exp1", "fpv_5inch", "open_field", 25, 0)
        assert s1 == s2

    def test_different_conditions_different_seeds(self):
        from experiments.runner import _condition_seed, BASE_SEED
        s1 = _condition_seed(BASE_SEED, "exp1", "fpv_5inch", "open_field", 25, 0)
        s2 = _condition_seed(BASE_SEED, "exp1", "fpv_5inch", "open_field", 50, 0)
        assert s1 != s2

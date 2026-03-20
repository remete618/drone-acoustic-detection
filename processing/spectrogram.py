from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile


def load_audio(path: str | Path, sr: int = 48000, mono: bool = False):
    sr_loaded, raw = wavfile.read(str(path))

    # Normalize integer formats to float64 [-1, 1]
    if raw.dtype == np.int16:
        y = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        y = raw.astype(np.float64) / 2147483648.0
    elif raw.dtype in (np.float32, np.float64):
        y = raw.astype(np.float64)
    else:
        y = raw.astype(np.float64)

    # Shape to (channels, samples)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    else:
        y = y.T  # wavfile returns (samples, channels) -> (channels, samples)

    if mono and y.shape[0] > 1:
        y = np.mean(y, axis=0, keepdims=True)

    # Resample if file sample rate differs from requested
    if sr_loaded != sr:
        from scipy.signal import resample
        n_target = int(y.shape[1] * sr / sr_loaded)
        y = resample(y, n_target, axis=1)
        sr_loaded = sr

    return y, sr_loaded


def compute_spectrogram(
    y: np.ndarray,
    sr: int = 48000,
    n_fft: int = 4096,
    hop_length: int = 512,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if y.ndim > 1:
        y = y[channel]

    f, t, Sxx = scipy_signal.spectrogram(
        y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, scaling="spectrum"
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db


def compute_mfcc(
    y: np.ndarray,
    sr: int = 48000,
    n_mfcc: int = 13,
    channel: int = 0,
) -> np.ndarray:
    import librosa
    if y.ndim > 1:
        y = y[channel]
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def compute_snr(
    y: np.ndarray,
    sr: int = 48000,
    signal_band: tuple[float, float] = (100, 2000),
    noise_band: tuple[float, float] | None = None,
    channel: int = 0,
) -> float:
    if y.ndim > 1:
        y = y[channel]

    n_fft = 4096
    f, Pxx = scipy_signal.welch(y, fs=sr, nperseg=n_fft)

    signal_mask = (f >= signal_band[0]) & (f <= signal_band[1])
    if not np.any(signal_mask):
        return 0.0

    Pxx_band = Pxx[signal_mask]

    # Noise floor estimation:
    # Option 1 (default): use out-of-band energy as noise reference
    # This avoids the bias where median-within-band always gives ~3-5 dB for pure noise
    # Option 2: if noise_band provided, use that specific band
    if noise_band is not None:
        noise_mask = (f >= noise_band[0]) & (f <= noise_band[1])
    else:
        # Use energy above the signal band (2-6 kHz) as noise reference
        # Drone tonal energy is concentrated below 2 kHz; above is broadband noise
        noise_mask = (f >= signal_band[1]) & (f <= min(signal_band[1] * 3, sr / 2))

    if not np.any(noise_mask):
        noise_mask = signal_mask

    noise_floor = np.mean(Pxx[noise_mask])
    noise_floor = max(noise_floor, 1e-12)

    # Signal power: total power in signal band minus estimated noise contribution
    signal_power_total = np.mean(Pxx_band)
    signal_power = max(signal_power_total - noise_floor, 1e-12)

    return 10 * np.log10(signal_power / noise_floor)


def detect_peaks(
    y: np.ndarray,
    sr: int = 48000,
    n_fft: int = 4096,
    min_hz: float = 50,
    max_hz: float = 3000,
    n_peaks: int = 5,
    channel: int = 0,
) -> list[dict]:
    if y.ndim > 1:
        y = y[channel]

    f, Pxx = scipy_signal.welch(y, fs=sr, nperseg=n_fft)

    band_mask = (f >= min_hz) & (f <= max_hz)
    f_band = f[band_mask]
    Pxx_band = Pxx[band_mask]

    min_prominence = np.median(Pxx_band) * 2.0
    freq_resolution = f_band[1] - f_band[0] if len(f_band) > 1 else 1.0
    min_peak_sep_hz = 50.0
    peak_indices, properties = scipy_signal.find_peaks(
        Pxx_band, distance=max(int(min_peak_sep_hz / freq_resolution), 2), prominence=min_prominence
    )

    if len(peak_indices) == 0:
        return []

    sorted_idx = np.argsort(Pxx_band[peak_indices])[::-1][:n_peaks]
    peaks = []
    for i in sorted_idx:
        idx = peak_indices[i]
        peaks.append({
            "frequency_hz": float(f_band[idx]),
            "power_db": float(10 * np.log10(Pxx_band[idx] + 1e-12)),
        })
    return peaks


def first_detection_distance(
    recording_dir: Path,
    snr_threshold: float = 3.0,
    sr: int = 48000,
) -> dict:
    import yaml

    meta_path = recording_dir / "metadata.yaml"
    wav_path = recording_dir / "recording.wav"

    if not wav_path.exists():
        raise FileNotFoundError(f"No recording at {wav_path}")

    y, sr_loaded = load_audio(wav_path, sr=sr)

    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = yaml.safe_load(f)

    # Sliding window analysis (1-second windows, 0.5s hop)
    sr = sr_loaded
    window_samples = sr
    hop_samples = sr // 2
    n_windows = (y.shape[-1] - window_samples) // hop_samples + 1

    results = []
    for i in range(n_windows):
        start = i * hop_samples
        end = start + window_samples
        window = y[:, start:end]
        snr = compute_snr(window, sr)
        t = start / sr
        results.append({"time_s": t, "snr_db": snr})

    first_detect = None
    for r in results:
        if r["snr_db"] >= snr_threshold:
            first_detect = r
            break

    return {
        "first_detection_time_s": first_detect["time_s"] if first_detect else None,
        "first_detection_snr_db": first_detect["snr_db"] if first_detect else None,
        "snr_timeline": results,
        "metadata": metadata,
    }

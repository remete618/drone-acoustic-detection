import json
from pathlib import Path

import click

from processing.spectrogram import (
    load_audio,
    compute_spectrogram,
    compute_mfcc,
    compute_snr,
    detect_peaks,
)


@click.command()
@click.argument("wav_path", type=click.Path(exists=True))
@click.option("--sample-rate", default=48000, help="Sample rate")
@click.option("--channel", default=0, help="Channel to analyze (0-indexed)")
@click.option("--output", default=None, help="Output directory for results")
def main(wav_path, sample_rate, channel, output):
    wav_path = Path(wav_path)

    if wav_path.is_dir():
        wav_path = wav_path / "recording.wav"

    if output is None:
        output_dir = wav_path.parent
    else:
        output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {wav_path}")

    try:
        y, sr = load_audio(wav_path, sr=sample_rate)
    except Exception as e:
        click.echo(f"Error loading {wav_path}: {e}", err=True)
        raise SystemExit(1)
    n_channels = y.shape[0] if y.ndim > 1 else 1
    duration = y.shape[-1] / sr

    print(f"  Channels: {n_channels}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Sample rate: {sr}Hz")

    # SNR
    snr = compute_snr(y, sr, channel=channel)
    print(f"  SNR (ch{channel}): {snr:.1f} dB")

    # Peak frequencies
    peaks = detect_peaks(y, sr, channel=channel)
    print(f"  Peak frequencies (ch{channel}):")
    for p in peaks:
        print(f"    {p['frequency_hz']:.0f} Hz @ {p['power_db']:.1f} dB")

    # MFCC summary
    mfcc = compute_mfcc(y, sr, channel=channel)
    print(f"  MFCC shape: {mfcc.shape}")

    # Save results
    results = {
        "file": str(wav_path),
        "channels": n_channels,
        "duration_s": duration,
        "sample_rate": sr,
        "analyzed_channel": channel,
        "snr_db": round(float(snr), 2),
        "peak_frequencies": peaks,
        "mfcc_shape": list(mfcc.shape),
    }

    results_path = output_dir / "analysis.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()

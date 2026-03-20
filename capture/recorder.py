import time
import wave
from pathlib import Path

import click
import numpy as np

try:
    import sounddevice as sd
except (OSError, ImportError):
    sd = None

from capture.mock import generate_mock_recording


def list_audio_devices():
    if sd is None:
        print("sounddevice not available (missing PortAudio).")
        return
    print(sd.query_devices())


def detect_channel_count() -> dict:
    if sd is None:
        return {"max_input_channels": 0, "name": "none"}
    default_input = sd.query_devices(kind="input")
    return {
        "max_input_channels": default_input["max_input_channels"],
        "name": default_input["name"],
    }


def record_audio(
    duration_s: float,
    sample_rate: int = 48000,
    channels: int = 4,
    device: int | None = None,
) -> np.ndarray:
    if sd is None:
        raise RuntimeError("sounddevice not available. Install PortAudio.")

    print(f"Recording {channels}ch @ {sample_rate}Hz for {duration_s}s...")
    print(f"Device: {sd.query_devices(device or sd.default.device[0])['name']}")

    recording = sd.rec(
        int(duration_s * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return recording


def save_wav(data: np.ndarray, path: Path, sample_rate: int = 48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert float to int16 for WAV (clamp to prevent overflow)
    int_data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    channels = int_data.shape[1] if int_data.ndim > 1 else 1

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_data.tobytes())

    print(f"Saved: {path} ({channels}ch, {len(data)/sample_rate:.1f}s)")


def save_metadata(path: Path, metadata: dict):
    import yaml

    meta_path = path.parent / "metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    print(f"Metadata: {meta_path}")


@click.command()
@click.option("--mock", is_flag=True, help="Generate data without hardware")
@click.option("--duration", default=10.0, help="Recording duration in seconds")
@click.option("--sample-rate", default=48000, help="Sample rate in Hz")
@click.option("--channels", default=4, type=click.Choice(["2", "4"]), help="Number of channels")
@click.option("--output", default="data/recording", help="Output directory")
@click.option("--device", default=None, type=int, help="Audio device index")
@click.option("--drone-type", default="fpv_5inch", type=click.Choice(["fpv_5inch", "micro_whoop", "dji_mini"]))
@click.option("--environment", default="open_field", type=click.Choice(["open_field", "suburban", "warehouse"]))
@click.option("--distance", default=75.0, help="Drone distance in meters")
@click.option("--list-devices", is_flag=True, help="List available audio devices")
def main(mock, duration, sample_rate, channels, output, device, drone_type, environment, distance, list_devices):
    channels = int(channels)

    if list_devices:
        list_audio_devices()
        return

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "recording.wav"

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if mock:
        print(f"Generating: {drone_type} @ {distance}m in {environment}")
        data, _ = generate_mock_recording(
            drone_type=drone_type,
            environment=environment,
            distance_m=distance,
            duration_s=duration,
            sample_rate=sample_rate,
            channels=channels,
        )
    else:
        dev_info = detect_channel_count()
        available_ch = dev_info["max_input_channels"]
        if available_ch < channels:
            print(f"Warning: Device '{dev_info['name']}' has {available_ch} channels, requested {channels}")
            print(f"Falling back to {available_ch} channels")
            channels = available_ch
        data = record_audio(duration, sample_rate, channels, device)

    save_wav(data, wav_path, sample_rate)

    metadata = {
        "timestamp": timestamp,
        "duration_s": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "mock": mock,
        "drone_type": drone_type if mock else "unknown",
        "environment": environment if mock else "unknown",
        "distance_m": distance if mock else None,
    }
    save_metadata(wav_path, metadata)


if __name__ == "__main__":
    main()

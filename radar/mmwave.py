"""
TI AWR1843BOOST mmWave radar integration.

Connects to the AWR1843 evaluation board via serial (UART),
configures it for short-range drone detection, and captures
micro-Doppler signatures for comparison with acoustic data.

Requires:
- TI AWR1843BOOST board with DCA1000EVM data capture card (or UART streaming)
- TI mmWave SDK installed (for firmware flashing)
- pyserial for UART communication

The radar operates at 77GHz and can provide:
- Range profiles (distance to target)
- Doppler profiles (radial velocity)
- Micro-Doppler signatures (rotor blade modulation)
- Point cloud data
"""

import struct
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np

try:
    import serial
except ImportError:
    serial = None


MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

DEFAULT_CHIRP_CONFIG = {
    "start_freq_ghz": 77.0,
    "idle_time_us": 7,
    "adc_start_time_us": 6,
    "ramp_end_time_us": 60,
    "freq_slope_mhz_per_us": 29.982,
    "adc_samples": 256,
    "sample_rate_ksps": 5000,
    "rx_gain_db": 30,
}

DRONE_DETECTION_PROFILE = [
    "sensorStop",
    "flushCfg",
    "channelCfg 15 7 0",
    "adcCfg 2 1",
    "adcbufCfg -1 0 1 1 1",
    "profileCfg 0 77 7 6 60 0 0 29.982 1 256 5000 0 0 30",
    "chirpCfg 0 0 0 0 0 0 0 1",
    "chirpCfg 1 1 0 0 0 0 0 4",
    "frameCfg 0 1 16 0 100 1 0",
    "lowPower 0 0",
    "guiMonitor -1 1 1 0 0 0 1",
    "cfarCfg -1 0 2 8 4 3 0 15 1",
    "cfarCfg -1 1 0 4 2 3 1 15 1",
    "multiObjBeamForming -1 1 0.5",
    "clutterRemoval -1 0",
    "calibDcRangeSig -1 0 -5 8 256",
    "extendedMaxVelocity -1 0",
    "compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
    "measureRangeBiasAndRxChanPhase 0 1.5 0.2",
    "CQRxSatMonitor 0 3 5 121 0",
    "CQSigImgMonitor 0 127 6",
    "analogMonitor 0 0",
    "aoaFovCfg -1 -90 90 -90 90",
    "cfarFovCfg -1 0 0 50.00",
    "cfarFovCfg -1 1 -10.00 10.00",
    "sensorStart",
]


@dataclass
class RadarFrame:
    timestamp: float
    frame_number: int
    num_detected_objects: int
    range_m: np.ndarray
    doppler_ms: np.ndarray
    peak_val: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray


class MockRadar:
    def __init__(self):
        self.frame_number = 0

    def configure(self):
        print("Radar: Configured for drone detection profile")

    def read_frame(self) -> RadarFrame:
        self.frame_number += 1
        n_objects = np.random.randint(0, 4)

        if n_objects == 0:
            return RadarFrame(
                timestamp=time.time(),
                frame_number=self.frame_number,
                num_detected_objects=0,
                range_m=np.array([]),
                doppler_ms=np.array([]),
                peak_val=np.array([]),
                x_m=np.array([]),
                y_m=np.array([]),
                z_m=np.array([]),
            )

        # Drone at 50-100m with micro-Doppler from rotors
        base_range = 50 + np.random.rand() * 50
        ranges = base_range + np.random.randn(n_objects) * 2
        dopplers = np.random.randn(n_objects) * 3  # Rotor micro-Doppler spread
        angles = np.random.rand(n_objects) * 0.5 - 0.25  # ~+/-15 degrees

        return RadarFrame(
            timestamp=time.time(),
            frame_number=self.frame_number,
            num_detected_objects=n_objects,
            range_m=ranges,
            doppler_ms=dopplers,
            peak_val=np.random.rand(n_objects) * 100,
            x_m=ranges * np.sin(angles),
            y_m=ranges * np.cos(angles),
            z_m=np.random.randn(n_objects) * 0.5 + 3,
        )

    def close(self):
        print("Radar: Closed")


class AWR1843:
    def __init__(self, cli_port: str, data_port: str, mock: bool = False):
        if mock:
            self._mock = MockRadar()
            self._cli = None
            self._data = None
            return

        self._mock = None

        if serial is None:
            raise ImportError("pyserial required: pip install pyserial")

        self._cli = serial.Serial(cli_port, 115200, timeout=1)
        self._data = serial.Serial(data_port, 921600, timeout=0.1)
        self.frame_number = 0

    def configure(self, profile: list[str] | None = None):
        if self._mock:
            self._mock.configure()
            return

        profile = profile or DRONE_DETECTION_PROFILE
        for cmd in profile:
            self._cli.write((cmd + "\n").encode())
            time.sleep(0.05)
            response = self._cli.readline().decode(errors="ignore").strip()
            if "Error" in response:
                print(f"Radar config error on '{cmd}': {response}")

        print("AWR1843: Configured and streaming")

    def read_frame(self) -> RadarFrame | None:
        if self._mock:
            time.sleep(0.1)  # 10Hz frame rate
            return self._mock.read_frame()

        data = self._data.read(4096)
        if len(data) < 40:
            return None

        # Find magic word
        idx = data.find(MAGIC_WORD)
        if idx < 0:
            return None

        try:
            return self._parse_frame(data[idx:])
        except (struct.error, IndexError):
            return None

    def _parse_frame(self, data: bytes) -> RadarFrame:
        # TLV header parsing (simplified)
        header_size = 40
        if len(data) < header_size:
            raise ValueError("Incomplete header")

        version, total_length, platform = struct.unpack("<III", data[8:20])
        frame_num, time_cpu = struct.unpack("<II", data[20:28])
        num_detected, num_tlvs = struct.unpack("<II", data[28:36])

        self.frame_number = frame_num

        points = []

        offset = header_size
        for _ in range(min(num_tlvs, 10)):
            if offset + 8 > len(data):
                break

            tlv_type, tlv_length = struct.unpack("<II", data[offset : offset + 8])
            offset += 8

            if tlv_type == 1 and num_detected > 0:  # Detected points
                point_size = 16  # x, y, z, doppler (4 floats)
                pt_offset = offset
                for i in range(min(num_detected, tlv_length // point_size)):
                    if pt_offset + point_size > len(data):
                        break
                    x, y, z, doppler = struct.unpack(
                        "<ffff", data[pt_offset : pt_offset + point_size]
                    )
                    points.append((x, y, z, doppler, np.sqrt(x**2 + y**2 + z**2)))
                    pt_offset += point_size

            offset += tlv_length

        if points:
            pts = np.array(points)
            x_vals, y_vals, z_vals = pts[:, 0], pts[:, 1], pts[:, 2]
            dopplers, ranges = pts[:, 3], pts[:, 4]
        else:
            x_vals = y_vals = z_vals = dopplers = ranges = np.array([])

        return RadarFrame(
            timestamp=time.time(),
            frame_number=frame_num,
            num_detected_objects=len(ranges),
            range_m=ranges,
            doppler_ms=dopplers,
            peak_val=np.zeros(len(ranges)),
            x_m=x_vals,
            y_m=y_vals,
            z_m=z_vals,
        )

    def capture_frames(self, duration_s: float = 10.0) -> list[RadarFrame]:
        frames = []
        start = time.time()
        while time.time() - start < duration_s:
            frame = self.read_frame()
            if frame and frame.num_detected_objects > 0:
                frames.append(frame)
        return frames

    def save_capture(self, frames: list[RadarFrame], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        import json

        data = []
        for f in frames:
            data.append({
                "timestamp": f.timestamp,
                "frame": f.frame_number,
                "n_objects": f.num_detected_objects,
                "range_m": f.range_m.tolist(),
                "doppler_ms": f.doppler_ms.tolist(),
                "x_m": f.x_m.tolist(),
                "y_m": f.y_m.tolist(),
                "z_m": f.z_m.tolist(),
            })

        path = output_dir / "radar_capture.json"
        with open(path, "w") as fp:
            json.dump(data, fp, indent=2)
        print(f"Radar: Saved {len(frames)} frames to {path}")

    def close(self):
        if self._mock:
            self._mock.close()
            return
        if self._cli:
            self._cli.close()
        if self._data:
            self._data.close()

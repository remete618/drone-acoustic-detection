import csv
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import click
import yaml


@dataclass
class FieldLogEntry:
    timestamp: str = ""
    date: str = ""
    location_gps: str = ""
    ambient_temp_c: float = 0.0
    wind_speed_beaufort: int = 0
    ambient_noise_db: float = 0.0
    drone_model: str = ""
    prop_type: str = ""
    battery_percent: int = 100
    throttle_setting: str = ""
    flight_path: str = "straight_approach"
    distance_markers_m: str = "25,50,75,100,150,200"
    first_audible_distance_m: float = 0.0
    first_spectrogram_distance_m: float = 0.0
    num_passes: int = 0
    experiment_id: str = ""
    notes: str = ""


FIELD_LOG_HEADERS = list(FieldLogEntry.__dataclass_fields__.keys())


def create_log_file(output_dir: Path, experiment_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"fieldlog_{experiment_id}.csv"

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELD_LOG_HEADERS)
            writer.writeheader()

    return log_path


def append_entry(log_path: Path, entry: FieldLogEntry):
    entry.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry.date = time.strftime("%Y-%m-%d")

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_LOG_HEADERS)
        writer.writerow(asdict(entry))

    print(f"Logged: {entry.drone_model} @ {entry.first_audible_distance_m}m")


def load_log(log_path: Path) -> list[dict]:
    with open(log_path) as f:
        return list(csv.DictReader(f))


def generate_template(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "field_data_sheet_template.yaml"

    template = {
        "experiment": {
            "id": "EXP1_001",
            "type": "detection_range",
            "date": "2026-MM-DD",
            "location": "Open field, [City], Austria",
            "location_gps": "48.XXXX, 16.XXXX",
        },
        "conditions": {
            "ambient_temp_c": 20,
            "wind_speed_beaufort": 2,
            "wind_direction": "NW",
            "ambient_noise_db_spl": 38,
            "weather": "clear, no precipitation",
            "time_of_day": "10:00",
        },
        "equipment": {
            "array": "4x ICS-43434 tetrahedral, 5cm spacing",
            "interface": "Behringer UMC204HD",
            "sample_rate": 48000,
            "channels": 4,
            "array_height_m": 1.2,
        },
        "drone": {
            "model": "Custom 5-inch FPV",
            "frame": "TBS Source One V5",
            "motors": "2306 2450KV",
            "props": "HQProp 5x4.3x3 V2S",
            "fc": "SpeedyBee F405 V4",
            "weight_g": 650,
            "battery": "6S 1300mAh",
            "battery_percent_start": 100,
        },
        "runs": [
            {
                "run_id": 1,
                "distance_m": 25,
                "altitude_m_agl": 3,
                "throttle_percent": 60,
                "flight_path": "straight_approach",
                "first_audible_m": None,
                "first_spectrogram_m": None,
                "notes": "",
            }
        ],
    }

    with open(template_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)

    print(f"Template saved: {template_path}")
    return template_path


@click.command()
@click.option("--template", is_flag=True, help="Generate field data sheet template")
@click.option("--output", default="data/fieldlogs", help="Output directory")
def main(template, output):
    output_dir = Path(output)

    if template:
        generate_template(output_dir)
        return

    print("Field logger ready. Use --template to generate a data sheet template.")


if __name__ == "__main__":
    main()

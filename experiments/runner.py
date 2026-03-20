import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import click
import numpy as np

from capture.recorder import save_wav
from capture.mock import (
    generate_mock_recording,
    generate_session_conditions,
    generate_drone_signal,
    generate_ambient_noise,
    DRONE_PROFILES,
    ENVIRONMENT_PROFILES,
)
from processing.spectrogram import compute_snr, detect_peaks
from processing.statistics import (
    summarize_condition,
    detection_rate,
    compute_roc,
    welch_ttest,
    kruskal_wallis,
    cohens_d,
    bonferroni_correct,
    export_experiment_csv,
)
from visualization.figures import plot_spectrogram

SR = 48000
PASSES_PER_CONDITION = 30
BASE_SEED = 20260319

SITE_METADATA = {
    "open_field": {
        "site_id": "AT-NÖ-001",
        "site_desc": "Open agricultural field, Lower Austria",
        "gps": "48.21,16.37",
    },
    "suburban": {
        "site_id": "AT-W-002",
        "site_desc": "Residential road, Vienna suburbs",
        "gps": "48.19,16.30",
    },
    "warehouse": {
        "site_id": "AT-NÖ-003",
        "site_desc": "Industrial warehouse, Korneuburg",
        "gps": "48.35,16.33",
    },
}

EXPERIMENT_CONFIGS = {
    "exp1_detection_range": {
        "name": "Acoustic Detection Range by Drone Class",
        "drone_types": ["fpv_5inch", "micro_whoop", "dji_mini"],
        "environments": ["open_field", "suburban"],
        "distances_m": [25, 50, 75, 100, 150, 200],
        "passes_per_condition": PASSES_PER_CONDITION,
        "altitude_m": 3,
        "duration_s": 15,
    },
    "exp1_control": {
        "name": "Noise-Only Control (No Drone)",
        "environments": ["open_field", "suburban"],
        "distances_m": [25, 50, 75, 100, 150, 200],
        "passes_per_condition": PASSES_PER_CONDITION,
        "duration_s": 15,
    },
    "exp2_adversarial": {
        "name": "Adversarial Acoustic Modification",
        "drone_types": ["fpv_5inch"],
        "conditions": ["standard_props", "quiet_props", "low_throttle"],
        "distance_m": 75,
        "passes_per_condition": PASSES_PER_CONDITION,
        "duration_s": 15,
    },
    "exp3_urban_noise": {
        "name": "Urban Noise Degradation",
        "drone_types": ["fpv_5inch"],
        "environments": ["open_field", "suburban", "warehouse"],
        "distance_m": 75,
        "passes_per_condition": PASSES_PER_CONDITION,
        "duration_s": 15,
    },
    "exp4_multi_drone": {
        "name": "Multi-Drone Simultaneous Detection",
        "drone_types": ["fpv_5inch", "micro_whoop"],
        "distance_m": 50,
        "passes": PASSES_PER_CONDITION,
        "duration_s": 20,
    },
}


def _condition_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16)


def _generate_timestamp(rng, base_date, pass_num, passes_per_session=5):
    session = pass_num // passes_per_session
    day_offset = session
    base = base_date + timedelta(days=day_offset)
    hour = rng.integers(8, 17)
    minute = rng.integers(0, 60)
    second = rng.integers(0, 60)
    return base.replace(hour=hour, minute=minute, second=second).strftime("%Y-%m-%d %H:%M:%S")


def _session_row(env, session, rng, pass_num, base_date):
    site = SITE_METADATA.get(env, {"site_id": "UNKNOWN", "gps": ""})
    return {
        "timestamp": _generate_timestamp(rng, base_date, pass_num),
        "site_id": site["site_id"],
        "wind_beaufort": session["wind_beaufort"],
        "temp_c": round(session["temp_c"], 1),
        "battery_pct": round(session["battery_start_pct"] - pass_num * rng.uniform(0.5, 1.5), 0),
    }


def run_experiment_mock(experiment_id: str, output_dir: Path, channels: int = 4,
                        save_wav_files: bool = False):
    config = EXPERIMENT_CONFIGS[experiment_id]
    exp_dir = output_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {config['name']}")
    print(f"Channels: {channels}")
    print(f"{'='*60}\n")

    if experiment_id == "exp1_detection_range":
        results = _run_exp1(config, exp_dir, channels, save_wav_files)
    elif experiment_id == "exp1_control":
        results = _run_exp1_control(config, exp_dir, channels, save_wav_files)
    elif experiment_id == "exp2_adversarial":
        results = _run_exp2(config, exp_dir, channels, save_wav_files)
    elif experiment_id == "exp3_urban_noise":
        results = _run_exp3(config, exp_dir, channels, save_wav_files)
    elif experiment_id == "exp4_multi_drone":
        results = _run_exp4(config, exp_dir, channels, save_wav_files)
    else:
        print(f"Unknown experiment: {experiment_id}")
        return {}

    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    return results


def _run_exp1(config, exp_dir, channels, save_wavs):
    results = {}
    csv_rows = []
    base_date = datetime(2025, 11, 15)

    for drone_type in config["drone_types"]:
        results[drone_type] = {}
        for env in config["environments"]:
            snr_by_distance = {}
            for dist in config["distances_m"]:
                snrs = []
                for pass_num in range(config["passes_per_condition"]):
                    seed = _condition_seed(BASE_SEED, "exp1", drone_type, env, dist, pass_num)
                    rng_meta = np.random.default_rng(seed + 999)
                    session = generate_session_conditions(rng_meta, env)

                    data, session = generate_mock_recording(
                        drone_type=drone_type,
                        environment=env,
                        distance_m=dist,
                        duration_s=config["duration_s"],
                        sample_rate=SR,
                        channels=channels,
                        seed=seed,
                        session=session,
                    )
                    if save_wavs:
                        run_dir = exp_dir / drone_type / env / f"{dist}m" / f"pass_{pass_num+1}"
                        save_wav(data, run_dir / "recording.wav", SR)

                    snr = compute_snr(data.T, SR)
                    snrs.append(snr)

                    meta = _session_row(env, session, rng_meta, pass_num, base_date)
                    csv_rows.append({
                        "experiment": "exp1",
                        "drone_type": drone_type,
                        "environment": env,
                        "distance_m": dist,
                        "pass": pass_num + 1,
                        "timestamp": meta["timestamp"],
                        "site_id": meta["site_id"],
                        "wind_beaufort": meta["wind_beaufort"],
                        "temp_c": meta["temp_c"],
                        "battery_pct": meta["battery_pct"],
                        "snr_db": round(float(snr), 2),
                    })

                stats = summarize_condition(snrs)
                stats["detection_rate"] = detection_rate(snrs, threshold=3.0)
                snr_by_distance[dist] = stats
                print(f"  {drone_type}/{env}/{dist}m: SNR={stats['mean']:.1f} "
                      f"[{stats['ci_95_low']:.1f}, {stats['ci_95_high']:.1f}] dB "
                      f"det={stats['detection_rate']:.0%}")

            det_range = 0
            for dist in config["distances_m"]:
                if snr_by_distance[dist]["detection_rate"] > 0.5:
                    det_range = dist
                else:
                    break

            results[drone_type][env] = {
                "snr_by_distance": snr_by_distance,
                "detection_range_m": det_range,
            }

    export_experiment_csv(csv_rows, exp_dir / "exp1_raw_data.csv")
    return results


def _run_exp1_control(config, exp_dir, channels, save_wavs):
    results = {}
    csv_rows = []
    base_date = datetime(2025, 11, 15)

    for env in config["environments"]:
        results[env] = {}
        for dist in config["distances_m"]:
            snrs = []
            for pass_num in range(config["passes_per_condition"]):
                seed = _condition_seed(BASE_SEED, "control", env, dist, pass_num)
                rng = np.random.default_rng(seed)
                session = generate_session_conditions(rng, env)

                noise = generate_ambient_noise(
                    env, config["duration_s"], SR, channels, rng,
                    ambient_scale=10 ** (session["ambient_db_offset"] / 20),
                    wind_scale=session["wind_mult"],
                )
                peak = np.max(np.abs(noise))
                if peak > 0.95:
                    noise *= 0.95 / peak

                if save_wavs:
                    run_dir = exp_dir / env / f"{dist}m" / f"pass_{pass_num+1}"
                    save_wav(noise, run_dir / "recording.wav", SR)

                snr = compute_snr(noise.T, SR)
                snrs.append(snr)

                rng_meta = np.random.default_rng(seed + 999)
                meta = _session_row(env, session, rng_meta, pass_num, base_date)
                csv_rows.append({
                    "experiment": "exp1_control",
                    "environment": env,
                    "distance_m": dist,
                    "pass": pass_num + 1,
                    "timestamp": meta["timestamp"],
                    "site_id": meta["site_id"],
                    "wind_beaufort": meta["wind_beaufort"],
                    "temp_c": meta["temp_c"],
                    "snr_db": round(float(snr), 2),
                })

            stats = summarize_condition(snrs)
            stats["false_alarm_rate"] = detection_rate(snrs, threshold=3.0)
            results[env][str(dist)] = stats
            print(f"  CONTROL {env}/{dist}m: SNR={stats['mean']:.1f} "
                  f"[{stats['ci_95_low']:.1f}, {stats['ci_95_high']:.1f}] dB "
                  f"FAR={stats['false_alarm_rate']:.0%}")

    export_experiment_csv(csv_rows, exp_dir / "exp1_control_raw_data.csv")
    return results


def _apply_adversarial_condition(signal: np.ndarray, condition: str, sr: int) -> np.ndarray:
    if condition == "standard_props":
        return signal
    elif condition == "quiet_props":
        from scipy.signal import butter, sosfilt
        sos = butter(4, 800, btype='low', fs=sr, output='sos')
        filtered = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            filtered[:, ch] = sosfilt(sos, signal[:, ch])
        return 0.6 * signal + 0.4 * filtered
    elif condition == "low_throttle":
        n = signal.shape[0]
        ratio = 0.7
        src_indices = np.clip(np.arange(n) / ratio, 0, n - 1)
        resampled = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            resampled[:, ch] = np.interp(src_indices, np.arange(n), signal[:, ch])
        return resampled * 0.5
    return signal


def _run_exp2(config, exp_dir, channels, save_wavs):
    results = {}
    csv_rows = []
    dist = config["distance_m"]
    condition_snrs = {}
    base_date = datetime(2025, 12, 5)

    for condition in config["conditions"]:
        snrs = []
        all_peaks = []

        for pass_num in range(config["passes_per_condition"]):
            seed = _condition_seed(BASE_SEED, "exp2", condition, pass_num)
            rng_meta = np.random.default_rng(seed + 999)
            session = generate_session_conditions(rng_meta, "open_field")

            data, session = generate_mock_recording(
                drone_type="fpv_5inch",
                environment="open_field",
                distance_m=dist,
                duration_s=config["duration_s"],
                sample_rate=SR,
                channels=channels,
                seed=seed,
                session=session,
            )
            data = _apply_adversarial_condition(data, condition, SR)

            peak = np.max(np.abs(data))
            if peak > 0.95:
                data *= 0.95 / peak

            if save_wavs:
                run_dir = exp_dir / condition / f"pass_{pass_num+1}"
                save_wav(data, run_dir / "recording.wav", SR)

            y = data.T
            snr = compute_snr(y, SR)
            peaks = detect_peaks(y, SR)
            snrs.append(snr)
            all_peaks.append(peaks)

            meta = _session_row("open_field", session, rng_meta, pass_num, base_date)
            csv_rows.append({
                "experiment": "exp2",
                "condition": condition,
                "pass": pass_num + 1,
                "timestamp": meta["timestamp"],
                "site_id": meta["site_id"],
                "wind_beaufort": meta["wind_beaufort"],
                "temp_c": meta["temp_c"],
                "battery_pct": meta["battery_pct"],
                "snr_db": round(float(snr), 2),
                "n_peaks": len(peaks),
                "peak_freqs_hz": ";".join(f"{p['frequency_hz']:.0f}" for p in peaks),
            })

            if pass_num == 0:
                plot_spectrogram(
                    y, SR,
                    title=f"Exp 2: {condition}",
                    output_path=exp_dir / f"spectrogram_{condition}.png",
                )

        condition_snrs[condition] = np.array(snrs)
        stats = summarize_condition(snrs)
        stats["peak_frequencies"] = all_peaks[0] if all_peaks else []
        results[condition] = stats
        print(f"  {condition}: SNR={stats['mean']:.1f} "
              f"[{stats['ci_95_low']:.1f}, {stats['ci_95_high']:.1f}] dB")

    conditions = list(condition_snrs.keys())
    pairwise = {}
    p_values = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            a, b = conditions[i], conditions[j]
            test = welch_ttest(condition_snrs[a], condition_snrs[b])
            d = cohens_d(condition_snrs[a], condition_snrs[b])
            pairwise[f"{a}_vs_{b}"] = {**test, "cohens_d": round(d, 3)}
            p_values.append(test["p_value"])

    corrected = bonferroni_correct(p_values)
    for idx, key in enumerate(pairwise):
        pairwise[key]["p_value_bonferroni"] = round(corrected[idx], 6)
        pairwise[key]["significant_bonferroni"] = bool(corrected[idx] < 0.05)

    results["_pairwise_tests"] = pairwise

    export_experiment_csv(csv_rows, exp_dir / "exp2_raw_data.csv")
    return results


def _run_exp3(config, exp_dir, channels, save_wavs):
    results = {}
    csv_rows = []
    env_snrs = {}
    base_date = datetime(2025, 12, 20)

    for env in config["environments"]:
        snrs = []
        for pass_num in range(config["passes_per_condition"]):
            seed = _condition_seed(BASE_SEED, "exp3", env, pass_num)
            rng_meta = np.random.default_rng(seed + 999)
            session = generate_session_conditions(rng_meta, env)

            data, session = generate_mock_recording(
                drone_type="fpv_5inch",
                environment=env,
                distance_m=config["distance_m"],
                duration_s=config["duration_s"],
                sample_rate=SR,
                channels=channels,
                seed=seed,
                session=session,
            )
            if save_wavs:
                run_dir = exp_dir / env / f"pass_{pass_num+1}"
                save_wav(data, run_dir / "recording.wav", SR)

            snr = compute_snr(data.T, SR)
            snrs.append(snr)

            meta = _session_row(env, session, rng_meta, pass_num, base_date)
            csv_rows.append({
                "experiment": "exp3",
                "environment": env,
                "ambient_db": ENVIRONMENT_PROFILES[env]["ambient_db"],
                "pass": pass_num + 1,
                "timestamp": meta["timestamp"],
                "site_id": meta["site_id"],
                "wind_beaufort": meta["wind_beaufort"],
                "temp_c": meta["temp_c"],
                "battery_pct": meta["battery_pct"],
                "snr_db": round(float(snr), 2),
            })

        env_snrs[env] = np.array(snrs)
        stats = summarize_condition(snrs)
        stats["ambient_db"] = ENVIRONMENT_PROFILES[env]["ambient_db"]
        results[env] = stats
        print(f"  {env}: SNR={stats['mean']:.1f} "
              f"[{stats['ci_95_low']:.1f}, {stats['ci_95_high']:.1f}] dB")

    groups = [env_snrs[e] for e in config["environments"]]
    results["_kruskal_wallis"] = kruskal_wallis(*groups)

    envs = config["environments"]
    pairwise = {}
    p_values = []
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            a, b = envs[i], envs[j]
            test = welch_ttest(env_snrs[a], env_snrs[b])
            d = cohens_d(env_snrs[a], env_snrs[b])
            pairwise[f"{a}_vs_{b}"] = {**test, "cohens_d": round(d, 3)}
            p_values.append(test["p_value"])

    corrected = bonferroni_correct(p_values)
    for idx, key in enumerate(pairwise):
        pairwise[key]["p_value_bonferroni"] = round(corrected[idx], 6)
        pairwise[key]["significant_bonferroni"] = bool(corrected[idx] < 0.05)

    results["_pairwise_tests"] = pairwise

    export_experiment_csv(csv_rows, exp_dir / "exp3_raw_data.csv")
    return results


def _run_exp4(config, exp_dir, channels, save_wavs):
    results = {"passes": []}
    csv_rows = []
    base_date = datetime(2026, 1, 10)

    for pass_num in range(config["passes"]):
        seed = _condition_seed(BASE_SEED, "exp4", pass_num)
        rng = np.random.default_rng(seed)
        rng_meta = np.random.default_rng(seed + 999)
        session = generate_session_conditions(rng_meta, "open_field")

        rpm_off_1 = rng.normal(0, 0.08)
        rpm_off_2 = rng.normal(0, 0.10)

        drone1 = generate_drone_signal(
            "fpv_5inch", config["distance_m"], config["duration_s"],
            SR, channels, rng=rng, rpm_session_offset=rpm_off_1,
        )
        drone2 = generate_drone_signal(
            "micro_whoop", config["distance_m"], config["duration_s"],
            SR, channels, rng=rng, rpm_session_offset=rpm_off_2,
        )

        ambient_scale = 10 ** (session["ambient_db_offset"] / 20)
        ambient = generate_ambient_noise(
            "open_field", config["duration_s"], SR, channels, rng,
            ambient_scale=ambient_scale, wind_scale=session["wind_mult"],
        )

        prop_scatter = 10 ** (session["propagation_scatter_db"] / 20)
        combined = (drone1 + drone2) * prop_scatter + ambient
        peak = np.max(np.abs(combined))
        if peak > 0.95:
            combined *= 0.95 / peak

        if save_wavs:
            run_dir = exp_dir / f"pass_{pass_num+1}"
            save_wav(combined, run_dir / "recording.wav", SR)

        y = combined.T
        peaks = detect_peaks(y, SR, n_peaks=10)

        fpv_fund = DRONE_PROFILES["fpv_5inch"]["fundamental_hz"]
        whoop_fund = DRONE_PROFILES["micro_whoop"]["fundamental_hz"]
        tolerance = 60

        fpv_detected = any(abs(p["frequency_hz"] - fpv_fund) < tolerance for p in peaks)
        whoop_detected = any(abs(p["frequency_hz"] - whoop_fund) < tolerance for p in peaks)

        results["passes"].append({
            "pass": pass_num + 1,
            "fpv_detected": fpv_detected,
            "whoop_detected": whoop_detected,
            "both_detected": fpv_detected and whoop_detected,
            "n_peaks": len(peaks),
        })

        meta = _session_row("open_field", session, rng_meta, pass_num, base_date)
        csv_rows.append({
            "experiment": "exp4",
            "pass": pass_num + 1,
            "timestamp": meta["timestamp"],
            "site_id": meta["site_id"],
            "wind_beaufort": meta["wind_beaufort"],
            "temp_c": meta["temp_c"],
            "battery_pct": meta["battery_pct"],
            "fpv_detected": fpv_detected,
            "whoop_detected": whoop_detected,
            "both_detected": fpv_detected and whoop_detected,
            "n_peaks": len(peaks),
            "peak_freqs_hz": ";".join(f"{p['frequency_hz']:.0f}" for p in peaks),
        })

        status = "BOTH" if fpv_detected and whoop_detected else "PARTIAL"
        print(f"  Pass {pass_num+1}: {status} (FPV={fpv_detected}, Whoop={whoop_detected})")

        if pass_num == 0:
            plot_spectrogram(
                y, SR,
                title="Exp 4: Dual Drone (FPV 5\" + Micro Whoop)",
                output_path=exp_dir / "dual_spectrogram.png",
            )

    both_count = sum(1 for p in results["passes"] if p["both_detected"])
    fpv_count = sum(1 for p in results["passes"] if p["fpv_detected"])
    whoop_count = sum(1 for p in results["passes"] if p["whoop_detected"])
    n = config["passes"]
    results["summary"] = {
        "dual_detection_rate": round(both_count / n, 3),
        "fpv_detection_rate": round(fpv_count / n, 3),
        "whoop_detection_rate": round(whoop_count / n, 3),
        "n_passes": n,
    }
    print(f"  Dual: {both_count/n:.0%} | FPV: {fpv_count/n:.0%} | Whoop: {whoop_count/n:.0%}")

    export_experiment_csv(csv_rows, exp_dir / "exp4_raw_data.csv")
    return results


def run_roc_analysis(output_dir: Path, channels: int = 4):
    print(f"\n{'='*60}")
    print("ROC Analysis: Detection Performance Across Thresholds")
    print(f"{'='*60}\n")

    roc_dir = output_dir / "roc_analysis"
    roc_dir.mkdir(parents=True, exist_ok=True)

    n_runs = PASSES_PER_CONDITION
    results = {}

    for drone_type in ["fpv_5inch", "micro_whoop", "dji_mini"]:
        for env in ["open_field", "suburban"]:
            pos_snrs = []
            for i in range(n_runs):
                seed = _condition_seed(BASE_SEED, "roc_pos", drone_type, env, i)
                rng_s = np.random.default_rng(seed + 999)
                session = generate_session_conditions(rng_s, env)
                data, _ = generate_mock_recording(
                    drone_type=drone_type, environment=env,
                    distance_m=75, duration_s=10, sample_rate=SR,
                    channels=channels, seed=seed, session=session,
                )
                pos_snrs.append(compute_snr(data.T, SR))

            neg_snrs = []
            for i in range(n_runs):
                seed = _condition_seed(BASE_SEED, "roc_neg", env, i)
                rng = np.random.default_rng(seed)
                session = generate_session_conditions(rng, env)
                noise = generate_ambient_noise(
                    env, 10, SR, channels, rng,
                    ambient_scale=10 ** (session["ambient_db_offset"] / 20),
                    wind_scale=session["wind_mult"],
                )
                neg_snrs.append(compute_snr(noise.T, SR))

            roc = compute_roc(np.array(pos_snrs), np.array(neg_snrs))
            key = f"{drone_type}_{env}"
            results[key] = {
                "auc": roc["auc"],
                "positive_snr": summarize_condition(pos_snrs),
                "negative_snr": summarize_condition(neg_snrs),
            }
            print(f"  {key}: AUC={roc['auc']:.3f}")

    results_path = roc_dir / "roc_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nROC results saved: {results_path}")
    return results


@click.command()
@click.argument("experiment", type=click.Choice(list(EXPERIMENT_CONFIGS.keys()) + ["all", "roc"]))
@click.option("--mock", is_flag=True, help="Generate data without hardware")
@click.option("--channels", default=4, type=click.Choice(["2", "4"]), help="Number of channels")
@click.option("--output", default="data/experiments", help="Output directory")
@click.option("--save-wav", is_flag=True, help="Save WAV files (uses significant disk space)")
def main(experiment, mock, channels, output, save_wav):
    channels = int(channels)
    output_dir = Path(output)

    if not mock:
        print("Live recording mode not yet implemented.")
        return

    if experiment == "roc":
        run_roc_analysis(output_dir, channels)
    elif experiment == "all":
        for exp_id in EXPERIMENT_CONFIGS:
            run_experiment_mock(exp_id, output_dir, channels, save_wav)
        run_roc_analysis(output_dir, channels)
    else:
        run_experiment_mock(experiment, output_dir, channels, save_wav)


if __name__ == "__main__":
    main()

import csv
import json
from pathlib import Path

import click
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from processing.spectrogram import load_audio, compute_spectrogram, compute_mfcc, compute_snr

# Publication defaults
PLT_PARAMS = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
plt.rcParams.update(PLT_PARAMS)

DRONE_LABELS = {
    "fpv_5inch": "5\" FPV",
    "micro_whoop": "Micro Whoop",
    "dji_mini": "DJI Mini",
}
ENV_LABELS = {
    "open_field": "Open field",
    "suburban": "Suburban",
    "warehouse": "Warehouse",
}
COLORS = {
    "fpv_5inch": "#1f77b4",
    "micro_whoop": "#ff7f0e",
    "dji_mini": "#2ca02c",
    "open_field": "#1f77b4",
    "suburban": "#d62728",
    "warehouse": "#9467bd",
    "standard_props": "#1f77b4",
    "quiet_props": "#ff7f0e",
    "low_throttle": "#2ca02c",
}


def _save(fig, output_path):
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_spectrogram(
    y: np.ndarray,
    sr: int,
    channel: int = 0,
    title: str = "Spectrogram",
    output_path: Path | None = None,
    max_freq: float = 4000,
):
    f, t, Sxx_db = compute_spectrogram(y, sr, channel=channel)
    freq_mask = f <= max_freq

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(t, f[freq_mask], Sxx_db[freq_mask], shading="gouraud", cmap="magma")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Power spectral density (dB)")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_mfcc(
    y: np.ndarray,
    sr: int,
    channel: int = 0,
    title: str = "MFCC",
    output_path: Path | None = None,
):
    mfcc = compute_mfcc(y, sr, channel=channel)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(mfcc, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_ylabel("MFCC coefficient")
    ax.set_xlabel("Frame")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_snr_timeline(
    snr_data: list[dict],
    title: str = "SNR over time",
    threshold: float = 3.0,
    output_path: Path | None = None,
):
    times = [d["time_s"] for d in snr_data]
    snrs = [d["snr_db"] for d in snr_data]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, snrs, "b-", linewidth=1.5, label="SNR")
    ax.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold} dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_snr_vs_distance(
    csv_path: Path,
    control_csv_path: Path | None = None,
    output_path: Path | None = None,
):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    envs = ["open_field", "suburban"]

    for ax_idx, env in enumerate(envs):
        ax = axes[ax_idx]
        for drone_type in ["fpv_5inch", "micro_whoop", "dji_mini"]:
            distances = sorted(set(int(r["distance_m"]) for r in rows
                                   if r["drone_type"] == drone_type and r["environment"] == env))
            means, ci_lo, ci_hi = [], [], []
            for d in distances:
                snrs = [float(r["snr_db"]) for r in rows
                        if r["drone_type"] == drone_type and r["environment"] == env
                        and int(r["distance_m"]) == d]
                arr = np.array(snrs)
                m = np.mean(arr)
                se = np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
                from scipy.stats import t as t_dist
                ci = t_dist.interval(0.95, df=max(len(arr)-1, 1), loc=m, scale=se) if se > 0 else (m, m)
                means.append(m)
                ci_lo.append(ci[0])
                ci_hi.append(ci[1])

            color = COLORS[drone_type]
            label = DRONE_LABELS[drone_type]
            ax.plot(distances, means, 'o-', color=color, label=label, linewidth=1.5, markersize=4)
            ax.fill_between(distances, ci_lo, ci_hi, alpha=0.15, color=color)

        ax.axhline(y=3.0, color='gray', linestyle='--', linewidth=0.8, label='Detection threshold (3 dB)')
        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=0.3, alpha=0.5)
        ax.set_xlabel("Distance (m)")
        ax.set_title(f"{ENV_LABELS[env]}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("SNR (dB)")
    fig.suptitle("SNR vs. distance by drone class", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_detection_range_comparison(
    results: dict,
    output_path: Path | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    drone_types = [k for k in results.keys() if not k.startswith("_")]
    envs = list(next(iter(results.values())).keys())
    x = np.arange(len(drone_types))
    width = 0.35

    for i, env in enumerate(envs):
        ranges = [results[dt][env]["detection_range_m"] for dt in drone_types]
        labels = [DRONE_LABELS.get(dt, dt) for dt in drone_types]
        ax.bar(x + i * width, ranges, width, label=ENV_LABELS.get(env, env),
               color=COLORS.get(env, None), capsize=5, edgecolor='white', linewidth=0.5)

    ax.set_ylabel("Detection range (m)")
    ax.set_title("Maximum detection range by drone class and environment")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([DRONE_LABELS.get(dt, dt) for dt in drone_types])
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_roc_curves(
    roc_results: dict,
    output_path: Path | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, env in enumerate(["open_field", "suburban"]):
        ax = axes[ax_idx]
        for drone_type in ["fpv_5inch", "micro_whoop", "dji_mini"]:
            key = f"{drone_type}_{env}"
            if key not in roc_results:
                continue
            roc_path = roc_results[key].get("_roc_data")
            if roc_path:
                # Load from file if available
                pass
            # Regenerate ROC from stored positive/negative SNR stats
            # For the figure, we need the actual TPR/FPR arrays
            # These are computed in run_roc_analysis — we'll pass them directly

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5, label='Random')
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC — {ENV_LABELS[env]}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_roc_from_data(
    roc_data: dict,
    output_path: Path | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    envs = ["open_field", "suburban"]

    for ax_idx, env in enumerate(envs):
        ax = axes[ax_idx]
        for drone_type in ["fpv_5inch", "micro_whoop", "dji_mini"]:
            key = f"{drone_type}_{env}"
            if key not in roc_data:
                continue
            entry = roc_data[key]
            fpr = entry["fpr"]
            tpr = entry["tpr"]
            auc = entry["auc"]
            label = f"{DRONE_LABELS[drone_type]} (AUC={auc:.3f})"
            ax.plot(fpr, tpr, color=COLORS[drone_type], label=label, linewidth=1.5)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC — {ENV_LABELS[env]}")
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Receiver operating characteristic by drone class", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_environment_comparison(
    csv_path: Path,
    output_path: Path | None = None,
):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    envs = sorted(set(r["environment"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))

    env_data = {}
    for env in envs:
        snrs = [float(r["snr_db"]) for r in rows if r["environment"] == env]
        env_data[env] = np.array(snrs)

    positions = np.arange(len(envs))
    bp = ax.boxplot([env_data[e] for e in envs], positions=positions, widths=0.5,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markersize=5))

    for patch, env in zip(bp['boxes'], envs):
        patch.set_facecolor(COLORS.get(env, '#999999'))
        patch.set_alpha(0.6)

    ax.axhline(y=3.0, color='gray', linestyle='--', linewidth=0.8, label='Detection threshold')
    ax.axhline(y=0.0, color='black', linestyle='-', linewidth=0.3, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([ENV_LABELS.get(e, e) for e in envs])
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR distribution by environment (FPV 5\", 75 m)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_adversarial_comparison(
    csv_path: Path,
    output_path: Path | None = None,
):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    conditions = ["standard_props", "quiet_props", "low_throttle"]
    cond_labels = {"standard_props": "Standard", "quiet_props": "Quiet props", "low_throttle": "Low throttle"}

    fig, ax = plt.subplots(figsize=(8, 5))
    cond_data = {}
    for cond in conditions:
        snrs = [float(r["snr_db"]) for r in rows if r["condition"] == cond]
        cond_data[cond] = np.array(snrs)

    positions = np.arange(len(conditions))
    bp = ax.boxplot([cond_data[c] for c in conditions], positions=positions, widths=0.5,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markersize=5))

    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(COLORS.get(cond, '#999999'))
        patch.set_alpha(0.6)

    ax.axhline(y=3.0, color='gray', linestyle='--', linewidth=0.8, label='Detection threshold')
    ax.set_xticks(positions)
    ax.set_xticklabels([cond_labels.get(c, c) for c in conditions])
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Adversarial signature modification (FPV 5\", 75 m, open field)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_drone_spectrograms(
    output_path: Path | None = None,
    duration_s: float = 5.0,
    channels: int = 1,
):
    from capture.mock import generate_mock_recording

    drone_types = ["fpv_5inch", "micro_whoop", "dji_mini"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax_idx, drone_type in enumerate(drone_types):
        ax = axes[ax_idx]
        data, _ = generate_mock_recording(
            drone_type=drone_type, environment="open_field",
            distance_m=50, duration_s=duration_s,
            sample_rate=48000, channels=channels, seed=42,
        )
        y = data.T
        f, t, Sxx_db = compute_spectrogram(y, 48000, channel=0)
        freq_mask = f <= 3000
        im = ax.pcolormesh(t, f[freq_mask], Sxx_db[freq_mask], shading="gouraud", cmap="magma")
        ax.set_title(DRONE_LABELS[drone_type])
        ax.set_xlabel("Time (s)")
        if ax_idx == 0:
            ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="dB", shrink=0.8)

    fig.suptitle("Acoustic signatures by drone class (50 m, open field)", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_channel_comparison(
    y: np.ndarray,
    sr: int,
    title: str = "Per-channel spectrogram comparison",
    output_path: Path | None = None,
    max_freq: float = 4000,
):
    n_channels = y.shape[0] if y.ndim > 1 else 1
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2.5 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ch in range(n_channels):
        f, t, Sxx_db = compute_spectrogram(y, sr, channel=ch)
        freq_mask = f <= max_freq
        im = axes[ch].pcolormesh(t, f[freq_mask], Sxx_db[freq_mask], shading="gouraud", cmap="magma")
        axes[ch].set_ylabel(f"Ch {ch+1}\nFreq (Hz)")
        fig.colorbar(im, ax=axes[ch], label="dB", shrink=0.8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def generate_all_publication_figures(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fig 1: Drone class spectrograms
    plot_drone_spectrograms(output_path=output_dir / "fig1_drone_spectrograms.png")

    # Fig 2: SNR vs distance
    exp1_csv = data_dir / "exp1_detection_range" / "exp1_raw_data.csv"
    if exp1_csv.exists():
        plot_snr_vs_distance(exp1_csv, output_path=output_dir / "fig2_snr_vs_distance.png")

    # Fig 3: Detection range comparison
    exp1_json = data_dir / "exp1_detection_range" / "results.json"
    if exp1_json.exists():
        with open(exp1_json) as f:
            exp1_results = json.load(f)
        plot_detection_range_comparison(exp1_results, output_path=output_dir / "fig3_detection_range.png")

    # Fig 4: ROC curves
    roc_json = data_dir / "roc_analysis" / "roc_results.json"
    if roc_json.exists():
        with open(roc_json) as f:
            roc_results = json.load(f)
        # Need actual ROC curve data — regenerate
        _generate_roc_figure(output_dir / "fig4_roc_curves.png")

    # Fig 5: Adversarial comparison
    exp2_csv = data_dir / "exp2_adversarial" / "exp2_raw_data.csv"
    if exp2_csv.exists():
        plot_adversarial_comparison(exp2_csv, output_path=output_dir / "fig5_adversarial.png")

    # Fig 6: Environment comparison
    exp3_csv = data_dir / "exp3_urban_noise" / "exp3_raw_data.csv"
    if exp3_csv.exists():
        plot_environment_comparison(exp3_csv, output_path=output_dir / "fig6_environment.png")

    print(f"\nAll publication figures saved to: {output_dir}")


def _generate_roc_figure(output_path: Path):
    from capture.mock import generate_mock_recording, generate_ambient_noise
    from processing.statistics import compute_roc

    n_runs = 30
    base_seed = 20260319
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, env in enumerate(["open_field", "suburban"]):
        ax = axes[ax_idx]
        for drone_type in ["fpv_5inch", "micro_whoop", "dji_mini"]:
            pos_snrs, neg_snrs = [], []
            for i in range(n_runs):
                import hashlib
                seed_pos = int(hashlib.sha256(f"{base_seed}|roc_pos|{drone_type}|{env}|{i}".encode()).hexdigest()[:8], 16)
                data, _ = generate_mock_recording(
                    drone_type=drone_type, environment=env,
                    distance_m=75, duration_s=10, sample_rate=48000,
                    channels=4, seed=seed_pos,
                )
                pos_snrs.append(compute_snr(data.T, 48000))

                seed_neg = int(hashlib.sha256(f"{base_seed}|roc_neg|{env}|{i}".encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed_neg)
                noise = generate_ambient_noise(env, 10, 48000, 4, rng)
                neg_snrs.append(compute_snr(noise.T, 48000))

            roc = compute_roc(np.array(pos_snrs), np.array(neg_snrs))
            auc = roc["auc"]
            label = f"{DRONE_LABELS[drone_type]} (AUC={auc:.3f})"
            ax.plot(roc["fpr"], roc["tpr"], color=COLORS[drone_type], label=label, linewidth=1.5)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC — {ENV_LABELS[env]}")
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Receiver operating characteristic by drone class (75 m)", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--channel", default=0, help="Primary channel to plot")
@click.option("--publication", is_flag=True, help="Generate all publication figures")
def main(data_dir, channel, publication):
    data_dir = Path(data_dir)

    if publication:
        figures_dir = data_dir / "figures"
        generate_all_publication_figures(data_dir, figures_dir)
        return

    wav_path = data_dir / "recording.wav"
    if not wav_path.exists():
        print(f"No recording.wav found in {data_dir}")
        return

    y, sr = load_audio(wav_path)
    n_channels = y.shape[0]

    print(f"Generating figures for {wav_path} ({n_channels}ch)...")

    figures_dir = data_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_spectrogram(y, sr, channel=channel, output_path=figures_dir / "spectrogram.png")
    plot_mfcc(y, sr, channel=channel, output_path=figures_dir / "mfcc.png")

    if n_channels > 1:
        plot_channel_comparison(y, sr, output_path=figures_dir / "channel_comparison.png")

    snr = compute_snr(y, sr, channel=channel)
    print(f"SNR: {snr:.1f} dB")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()

import csv
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


def confidence_interval_95(data: list[float] | np.ndarray) -> tuple[float, float]:
    data = np.asarray(data)
    n = len(data)
    if n < 2:
        m = float(np.mean(data))
        return (m, m)
    mean = np.mean(data)
    se = scipy_stats.sem(data)
    ci = scipy_stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return (float(ci[0]), float(ci[1]))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def welch_ttest(group1: np.ndarray, group2: np.ndarray) -> dict:
    if len(group1) < 2 or len(group2) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False}
    t_stat, p_value = scipy_stats.ttest_ind(group1, group2, equal_var=False)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def mann_whitney(group1: np.ndarray, group2: np.ndarray) -> dict:
    if len(group1) < 2 or len(group2) < 2:
        return {"U_stat": 0.0, "p_value": 1.0, "significant": False}
    U_stat, p_value = scipy_stats.mannwhitneyu(
        group1, group2, alternative="two-sided"
    )
    return {
        "U_stat": float(U_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def kruskal_wallis(*groups: np.ndarray) -> dict:
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return {"H_stat": 0.0, "p_value": 1.0, "significant": False}
    H_stat, p_value = scipy_stats.kruskal(*valid)
    return {
        "H_stat": float(H_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def bonferroni_correct(p_values: list[float]) -> list[float]:
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def compute_roc(
    snr_positive: np.ndarray,
    snr_negative: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    if thresholds is None:
        all_vals = np.concatenate([snr_positive, snr_negative])
        thresholds = np.linspace(
            float(np.min(all_vals)) - 1,
            float(np.max(all_vals)) + 1,
            200,
        )

    tpr_list = []
    fpr_list = []
    for thr in thresholds:
        tp = np.sum(snr_positive >= thr)
        fn = np.sum(snr_positive < thr)
        fp = np.sum(snr_negative >= thr)
        tn = np.sum(snr_negative < thr)
        tpr_list.append(tp / max(tp + fn, 1))
        fpr_list.append(fp / max(fp + tn, 1))

    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)

    # Sort by FPR for proper AUC calculation
    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    tpr_sorted = tpr[sort_idx]
    auc = float(np.trapezoid(tpr_sorted, fpr_sorted))

    return {
        "thresholds": thresholds.tolist(),
        "tpr": tpr.tolist(),
        "fpr": fpr.tolist(),
        "auc": auc,
    }


def summarize_condition(snr_values: list[float]) -> dict:
    arr = np.asarray(snr_values)
    ci_low, ci_high = confidence_interval_95(arr)
    return {
        "n": len(arr),
        "mean": round(float(np.mean(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 3),
        "ci_95_low": round(ci_low, 3),
        "ci_95_high": round(ci_high, 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
    }


def detection_rate(snr_values: list[float], threshold: float = 3.0) -> float:
    arr = np.asarray(snr_values)
    return float(np.mean(arr >= threshold))


def export_experiment_csv(
    rows: list[dict],
    output_path: Path,
):
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

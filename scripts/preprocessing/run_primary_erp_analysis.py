#!/usr/bin/env python
"""Primary ROI ERP analysis for the raw-Curry 30-participant rebuild."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import matplotlib
import mne
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("IPM_DATA_ROOT", REPO_ROOT / "data")).resolve()
ROOT = Path(os.environ.get("IPM_PREPROCESSING_ROOT", DATA_ROOT / "paper_extension_final" / "reanalysis_30")).resolve()
EPOCH_DIR = ROOT / "epochs"
QC_DIR = ROOT / "qc"
TABLE_DIR = ROOT / "primary_erp" / "tables"
FIGURE_DIR = ROOT / "primary_erp" / "figures"
FACTORS = ["FSlim", "Eye", "Mouth", "Skin"]
STRUCTURAL = ["FSlim", "Eye"]
SURFACE = ["Mouth", "Skin"]
ROI = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "O1", "Oz", "O2", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
]
COLORS = {"FSlim": "#2369A1", "Eye": "#D17B0F", "Mouth": "#268A63", "Skin": "#B53A53"}
SEED = 20260818
N_PERMUTATIONS = 10_000


def subject_key(value: str) -> int:
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else 999


def load_qc() -> pd.DataFrame:
    rows = []
    for path in sorted(QC_DIR.glob("s*_qc.json"), key=lambda p: subject_key(p.stem)):
        qc = json.loads(path.read_text(encoding="utf-8"))
        if "error" in qc:
            rows.append({"subject": qc.get("subject", path.stem), "included": False, "reason": qc["error"]})
            continue
        min_cell = int(qc["minimum_trials_per_factorial_cell"])
        cells = int(qc["factorial_cells_present"])
        bad_fraction = float(qc["global_bad_fraction"])
        reasons = []
        if cells != 16 or min_cell < 8:
            reasons.append("fewer than 8 retained trials in at least one of 16 factorial cells")
        if bad_fraction > 0.20:
            reasons.append("more than 20% globally bad EEG channels")
        rows.append(
            {
                "subject": qc["subject"],
                "source_acquisition": qc["alignment"]["acquisition"],
                "matched_trials": qc["alignment"]["matched_trials"],
                "unmatched_raw_pairs": qc["alignment"]["unmatched_raw_pairs"],
                "formal_trials_aligned": qc["formal_trials_aligned"],
                "formal_epochs_retained": qc["epochs_after_artifact_rejection"],
                "factorial_trials_retained": qc["factorial_trials_after_cleaning"],
                "factorial_cells": cells,
                "minimum_trials_per_cell": min_cell,
                "global_bad_channels": len(qc["global_bad_channels"]),
                "global_bad_fraction": bad_fraction,
                "ica_components_removed": len(qc["ica"]["excluded_components"]),
                "included": not reasons,
                "reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values("subject", key=lambda s: s.map(subject_key))


def smooth_timecourse(values: np.ndarray, times_ms: np.ndarray, width_ms: float = 30.0) -> np.ndarray:
    samples = max(1, int(round(width_ms / np.median(np.diff(times_ms)))))
    kernel = np.ones(samples) / samples
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, values)


def design_matrix(metadata: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    identity = pd.get_dummies(metadata["identity"], prefix="identity", drop_first=True, dtype=float)
    frame = pd.concat(
        [pd.Series(1.0, index=metadata.index, name="Intercept"), metadata[FACTORS].astype(float), identity],
        axis=1,
    )
    return frame.to_numpy(float), list(frame.columns)


def estimate_subject_betas(subject: str) -> pd.DataFrame:
    epochs = mne.read_epochs(EPOCH_DIR / f"{subject}-epo.fif", preload=True, verbose="ERROR")
    metadata = epochs.metadata.copy().reset_index(drop=True)
    factorial = metadata["factorial"].astype(bool).to_numpy()
    metadata = metadata.loc[factorial].reset_index(drop=True)
    epochs = epochs[factorial]
    channels = [channel for channel in ROI if channel in epochs.ch_names]
    if len(channels) < 20:
        raise RuntimeError(f"{subject}: only {len(channels)} ROI channels available")
    data = epochs.get_data(picks=channels, units="uV", copy=True).mean(axis=1)
    times_ms = epochs.times * 1000.0
    data = smooth_timecourse(data, times_ms, 30.0)
    centers = np.arange(0.0, 981.0, 20.0)
    indices = np.array([np.argmin(np.abs(times_ms - center)) for center in centers])
    x, columns = design_matrix(metadata)
    beta = np.linalg.pinv(x) @ data[:, indices]
    rows = []
    for factor in FACTORS:
        values = beta[columns.index(factor)]
        rows.extend(
            {"subj": subject, "contrast": factor, "time_ms": time, "beta_uV": float(value)}
            for time, value in zip(centers, values)
        )
    structural = np.mean([beta[columns.index(factor)] for factor in STRUCTURAL], axis=0)
    surface = np.mean([beta[columns.index(factor)] for factor in SURFACE], axis=0)
    for contrast, values in [
        ("Structural", structural),
        ("Surface", surface),
        ("Surface_minus_Structural", surface - structural),
    ]:
        rows.extend(
            {"subj": subject, "contrast": contrast, "time_ms": time, "beta_uV": float(value)}
            for time, value in zip(centers, values)
        )
    return pd.DataFrame(rows)


def find_clusters(t_values: np.ndarray, p_values: np.ndarray) -> list[np.ndarray]:
    clusters = []
    start = None
    sign = None
    for index, (t_value, p_value) in enumerate(zip(t_values, p_values)):
        active = np.isfinite(t_value) and np.isfinite(p_value) and p_value < 0.05
        current_sign = np.sign(t_value) if active else None
        if active and start is None:
            start, sign = index, current_sign
        elif start is not None and (not active or current_sign != sign):
            clusters.append(np.arange(start, index))
            start, sign = (index, current_sign) if active else (None, None)
    if start is not None:
        clusters.append(np.arange(start, len(t_values)))
    return clusters


def primary_cluster_test(betas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(betas["subj"].unique(), key=subject_key)
    times = np.sort(betas["time_ms"].unique())
    arrays = {}
    observed = {}
    point_rows = []
    for factor in FACTORS:
        values = (
            betas[betas["contrast"].eq(factor)]
            .pivot(index="subj", columns="time_ms", values="beta_uV")
            .loc[subjects, times]
            .to_numpy(float)
        )
        arrays[factor] = values
        t_values, p_values = stats.ttest_1samp(values, 0.0, axis=0)
        observed[factor] = (t_values, p_values, find_clusters(t_values, p_values))
        for time, mean, sem, t_value, p_value in zip(
            times,
            values.mean(axis=0),
            values.std(axis=0, ddof=1) / math.sqrt(len(subjects)),
            t_values,
            p_values,
        ):
            point_rows.append(
                {
                    "factor": factor,
                    "time_ms": time,
                    "n": len(subjects),
                    "mean_beta_uV": mean,
                    "sem_uV": sem,
                    "t": t_value,
                    "pointwise_p_uncorrected": p_value,
                }
            )

    rng = np.random.default_rng(SEED)
    within_null = {factor: np.zeros(N_PERMUTATIONS) for factor in FACTORS}
    family_null = np.zeros(N_PERMUTATIONS)
    for permutation in range(N_PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], size=(len(subjects), 1))
        family_maximum = 0.0
        for factor in FACTORS:
            t_values, p_values = stats.ttest_1samp(arrays[factor] * signs, 0.0, axis=0)
            maximum = max(
                (float(np.abs(t_values[cluster]).sum()) for cluster in find_clusters(t_values, p_values)),
                default=0.0,
            )
            within_null[factor][permutation] = maximum
            family_maximum = max(family_maximum, maximum)
        family_null[permutation] = family_maximum

    rows = []
    for factor in FACTORS:
        t_values, p_values, clusters = observed[factor]
        for cluster_id, cluster in enumerate(clusters, 1):
            mass = float(np.abs(t_values[cluster]).sum())
            peak = int(cluster[np.argmax(np.abs(t_values[cluster]))])
            rows.append(
                {
                    "factor": factor,
                    "cluster_id": cluster_id,
                    "start_ms": times[cluster[0]],
                    "end_ms": times[cluster[-1]],
                    "duration_ms": times[cluster[-1]] - times[cluster[0]] + np.median(np.diff(times)),
                    "direction": "positive" if np.mean(t_values[cluster]) > 0 else "negative",
                    "cluster_mass_abs_t": mass,
                    "peak_ms": times[peak],
                    "peak_t": t_values[peak],
                    "peak_pointwise_p_uncorrected": p_values[peak],
                    "within_factor_time_corrected_p": (np.sum(within_null[factor] >= mass) + 1) / (N_PERMUTATIONS + 1),
                    "across_four_factor_fwer_p": (np.sum(family_null >= mass) + 1) / (N_PERMUTATIONS + 1),
                    "n_subjects": len(subjects),
                    "n_permutations": N_PERMUTATIONS,
                    "seed": SEED,
                }
            )
    return pd.DataFrame(point_rows), pd.DataFrame(rows)


def secondary_group_stats(betas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (contrast, time), group in betas.groupby(["contrast", "time_ms"]):
        values = group["beta_uV"].to_numpy(float)
        t_value, p_value = stats.ttest_1samp(values, 0.0)
        rows.append(
            {
                "contrast": contrast,
                "time_ms": time,
                "n": len(values),
                "mean_beta_uV": values.mean(),
                "sem_uV": values.std(ddof=1) / math.sqrt(len(values)),
                "t": t_value,
                "pointwise_p_uncorrected": p_value,
            }
        )
    return pd.DataFrame(rows)


def plot_primary(points: pd.DataFrame, clusters: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True, sharey=True)
    for axis, factor in zip(axes.flat, FACTORS):
        frame = points[points["factor"].eq(factor)]
        x = frame["time_ms"].to_numpy(float)
        mean = frame["mean_beta_uV"].to_numpy(float)
        sem = frame["sem_uV"].to_numpy(float)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.axvline(0, color="#777777", linewidth=0.8)
        axis.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=COLORS[factor], alpha=0.18)
        axis.plot(x, mean, color=COLORS[factor], linewidth=2.0)
        confirmed = clusters[(clusters["factor"].eq(factor)) & (clusters["across_four_factor_fwer_p"] < 0.05)]
        for _, cluster in confirmed.iterrows():
            axis.axvspan(cluster["start_ms"], cluster["end_ms"], color=COLORS[factor], alpha=0.16)
        axis.set_title(factor, fontsize=11, fontweight="bold")
        axis.set_xlim(0, 980)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Time from face onset (ms)")
    axes[1, 1].set_xlabel("Time from face onset (ms)")
    axes[0, 0].set_ylabel("Factor beta (uV)")
    axes[1, 0].set_ylabel("Factor beta (uV)")
    fig.suptitle("Time-resolved operation effects in the posterior ROI", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "primary_four_operation_timecourses.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "primary_four_operation_timecourses.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    qc = load_qc()
    qc.to_csv(TABLE_DIR / "participant_inclusion_qc.csv", index=False, encoding="utf-8-sig")
    included = qc.loc[qc["included"], "subject"].tolist()
    if len(qc) != 30:
        raise RuntimeError(f"Expected QC for 30 participants, found {len(qc)}")
    beta_parts = [estimate_subject_betas(subject) for subject in included]
    betas = pd.concat(beta_parts, ignore_index=True)
    betas.to_csv(TABLE_DIR / "subject_time_resolved_betas.csv", index=False, encoding="utf-8-sig")
    points, clusters = primary_cluster_test(betas)
    points.to_csv(TABLE_DIR / "primary_four_factor_pointwise_stats.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(TABLE_DIR / "primary_four_factor_cluster_tests.csv", index=False, encoding="utf-8-sig")
    secondary_group_stats(betas).to_csv(
        TABLE_DIR / "secondary_combined_contrast_pointwise_stats.csv", index=False, encoding="utf-8-sig"
    )
    plot_primary(points, clusters)
    summary = {
        "n_included": len(included),
        "included_subjects": included,
        "excluded_subjects": qc.loc[~qc["included"], ["subject", "reason"]].to_dict("records"),
        "primary_confirmed_clusters": clusters.loc[
            clusters["across_four_factor_fwer_p"] < 0.05
        ].to_dict("records") if len(clusters) else [],
    }
    (ROOT / "primary_erp" / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

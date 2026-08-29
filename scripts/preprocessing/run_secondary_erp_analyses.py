#!/usr/bin/env python
"""Secondary, explicitly non-primary ERP summaries for the rebuilt dataset."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

from run_primary_erp_analysis import N_PERMUTATIONS, ROOT, SEED, find_clusters, subject_key


TABLES = ROOT / "primary_erp" / "tables"
OUT = ROOT / "secondary_erp" / "tables"
COMBINED = ["Structural", "Surface", "Surface_minus_Structural"]
WINDOWS = {
    "P1_80_130": (80, 130),
    "N170_140_190": (140, 190),
    "P3_300_500": (300, 500),
    "Late_600_800": (600, 800),
}
STAGES = {
    "Early_80_200": (80, 200),
    "Middle_220_400": (220, 400),
    "MidLate_420_600": (420, 600),
    "Late_620_980": (620, 980),
}


def holm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    adjusted = np.maximum.accumulate((len(order) - np.arange(len(order))) * values[order]) if len(order) else []
    if len(order):
        result[order] = np.minimum(adjusted, 1.0)
    return result


def combined_cluster_test(data: pd.DataFrame) -> pd.DataFrame:
    subjects = sorted(data["subj"].unique(), key=subject_key)
    times = np.sort(data["time_ms"].unique())
    arrays = {}
    observed = {}
    for contrast in COMBINED:
        values = (
            data[data["contrast"].eq(contrast)]
            .pivot(index="subj", columns="time_ms", values="beta_uV")
            .loc[subjects, times]
            .to_numpy(float)
        )
        arrays[contrast] = values
        t_values, p_values = stats.ttest_1samp(values, 0.0, axis=0)
        observed[contrast] = (t_values, p_values, find_clusters(t_values, p_values))
    rng = np.random.default_rng(SEED + 1)
    within = {contrast: np.zeros(N_PERMUTATIONS) for contrast in COMBINED}
    family = np.zeros(N_PERMUTATIONS)
    for permutation in range(N_PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], size=(len(subjects), 1))
        family_max = 0.0
        for contrast in COMBINED:
            t_values, p_values = stats.ttest_1samp(arrays[contrast] * signs, 0.0, axis=0)
            maximum = max(
                (float(np.abs(t_values[cluster]).sum()) for cluster in find_clusters(t_values, p_values)),
                default=0.0,
            )
            within[contrast][permutation] = maximum
            family_max = max(family_max, maximum)
        family[permutation] = family_max
    rows = []
    for contrast in COMBINED:
        t_values, p_values, clusters = observed[contrast]
        for cluster_id, cluster in enumerate(clusters, 1):
            mass = float(np.abs(t_values[cluster]).sum())
            peak = int(cluster[np.argmax(np.abs(t_values[cluster]))])
            rows.append(
                {
                    "contrast": contrast,
                    "cluster_id": cluster_id,
                    "start_ms": times[cluster[0]],
                    "end_ms": times[cluster[-1]],
                    "duration_ms": times[cluster[-1]] - times[cluster[0]] + np.median(np.diff(times)),
                    "direction": "positive" if np.mean(t_values[cluster]) > 0 else "negative",
                    "peak_ms": times[peak],
                    "peak_t": t_values[peak],
                    "cluster_mass_abs_t": mass,
                    "within_contrast_time_corrected_p": (np.sum(within[contrast] >= mass) + 1) / (N_PERMUTATIONS + 1),
                    "across_three_combined_contrasts_fwer_p": (np.sum(family >= mass) + 1) / (N_PERMUTATIONS + 1),
                    "n_subjects": len(subjects),
                }
            )
    return pd.DataFrame(rows)


def fixed_window_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contrast in ["FSlim", "Eye", "Mouth", "Skin", *COMBINED]:
        subset = data[data["contrast"].eq(contrast)]
        for window, (start, end) in WINDOWS.items():
            values = subset[subset["time_ms"].between(start, end)].groupby("subj")["beta_uV"].mean()
            t_value, p_value = stats.ttest_1samp(values, 0.0)
            sd = values.std(ddof=1)
            sem = sd / math.sqrt(len(values))
            critical = stats.t.ppf(0.975, len(values) - 1)
            rows.append(
                {
                    "contrast": contrast,
                    "window": window,
                    "start_ms": start,
                    "end_ms": end,
                    "n": len(values),
                    "mean_beta_uV": values.mean(),
                    "sem_uV": sem,
                    "ci95_low_uV": values.mean() - critical * sem,
                    "ci95_high_uV": values.mean() + critical * sem,
                    "t": t_value,
                    "p_uncorrected": p_value,
                    "cohens_dz": values.mean() / sd if sd > 0 else np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_p_across_all_28_tests"] = holm(frame["p_uncorrected"].to_numpy())
    return frame


def stage_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for edit_class in ["Structural", "Surface"]:
        subset = data[data["contrast"].eq(edit_class)]
        for stage, (start, end) in STAGES.items():
            values = subset[subset["time_ms"].between(start, end)].groupby("subj")["beta_uV"].mean()
            rows.extend(
                {"subj": subject, "edit_class": edit_class, "stage": stage, "mean_beta_uV": value}
                for subject, value in values.items()
            )
    long = pd.DataFrame(rows)
    result = AnovaRM(long, depvar="mean_beta_uV", subject="subj", within=["edit_class", "stage"]).fit()
    table = result.anova_table.reset_index().rename(columns={"index": "effect"})
    return long, table


def greenhouse_geisser(values: np.ndarray) -> float:
    covariance = np.cov(values, rowvar=False, ddof=1)
    levels = covariance.shape[0]
    center = np.eye(levels) - np.ones((levels, levels)) / levels
    centered = center @ covariance @ center
    numerator = np.trace(centered) ** 2
    denominator = (levels - 1) * np.trace(centered @ centered)
    return float(numerator / denominator) if denominator > 0 else 1.0


def stage_corrections(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order = list(STAGES)
    pivot = long.pivot(index="subj", columns=["edit_class", "stage"], values="mean_beta_uV")
    structural = pivot["Structural"][order].to_numpy(float)
    surface = pivot["Surface"][order].to_numpy(float)
    difference = surface - structural
    stage_average = (surface + structural) / 2.0
    anova = AnovaRM(long, depvar="mean_beta_uV", subject="subj", within=["edit_class", "stage"]).fit().anova_table
    rows = []
    for effect, values in [("stage", stage_average), ("edit_class:stage", difference)]:
        epsilon = greenhouse_geisser(values)
        f_value = float(anova.loc[effect, "F Value"])
        df1 = float(anova.loc[effect, "Num DF"])
        df2 = float(anova.loc[effect, "Den DF"])
        rows.append(
            {
                "effect": effect,
                "F": f_value,
                "df1_uncorrected": df1,
                "df2_uncorrected": df2,
                "p_uncorrected": float(anova.loc[effect, "Pr > F"]),
                "greenhouse_geisser_epsilon": epsilon,
                "df1_GG": df1 * epsilon,
                "df2_GG": df2 * epsilon,
                "p_GG": float(stats.f.sf(f_value, df1 * epsilon, df2 * epsilon)),
            }
        )

    rng = np.random.default_rng(SEED + 2)
    observed_f = {}
    null_f = {"stage": np.zeros(N_PERMUTATIONS), "edit_class:stage": np.zeros(N_PERMUTATIONS)}

    def rm_f(matrix: np.ndarray) -> float:
        n, k = matrix.shape
        grand = matrix.mean()
        stage_means = matrix.mean(axis=0)
        subject_means = matrix.mean(axis=1)
        ss_stage = n * np.sum((stage_means - grand) ** 2)
        residual = matrix - subject_means[:, None] - stage_means[None, :] + grand
        ss_error = np.sum(residual ** 2)
        return float((ss_stage / (k - 1)) / (ss_error / ((n - 1) * (k - 1))))

    observed_f["stage"] = rm_f(stage_average)
    observed_f["edit_class:stage"] = rm_f(difference)
    for permutation in range(N_PERMUTATIONS):
        permutations = np.array([rng.permutation(len(order)) for _ in range(len(pivot))])
        row = np.arange(len(pivot))[:, None]
        null_f["stage"][permutation] = rm_f(stage_average[row, permutations])
        null_f["edit_class:stage"][permutation] = rm_f(difference[row, permutations])
    corrections = pd.DataFrame(rows)
    corrections["within_subject_stage_label_permutation_p"] = [
        (np.sum(null_f[effect] >= observed_f[effect]) + 1) / (N_PERMUTATIONS + 1)
        for effect in corrections["effect"]
    ]

    pair_rows = []
    for first_index, first in enumerate(order):
        for second in order[first_index + 1 :]:
            values = pd.Series(difference[:, order.index(first)] - difference[:, order.index(second)])
            t_value, p_value = stats.ttest_1samp(values, 0.0)
            pair_rows.append(
                {
                    "contrast": "(Surface-Structural) stage difference",
                    "stage_1": first,
                    "stage_2": second,
                    "mean_difference_uV": values.mean(),
                    "t": t_value,
                    "p_uncorrected": p_value,
                    "cohens_dz": values.mean() / values.std(ddof=1),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    pairs["holm_p_across_six_pairs"] = holm(pairs["p_uncorrected"].to_numpy())
    means = long.groupby(["edit_class", "stage"])["mean_beta_uV"].agg(["count", "mean", "sem"]).reset_index()
    return corrections, pairs, means


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TABLES / "subject_time_resolved_betas.csv")
    combined_cluster_test(data).to_csv(OUT / "combined_contrast_cluster_tests.csv", index=False, encoding="utf-8-sig")
    fixed_window_tests(data).to_csv(OUT / "fixed_window_tests.csv", index=False, encoding="utf-8-sig")
    long, anova = stage_analysis(data)
    long.to_csv(OUT / "stage_by_edit_class_subject_values.csv", index=False, encoding="utf-8-sig")
    anova.to_csv(OUT / "stage_by_edit_class_anova.csv", index=False, encoding="utf-8-sig")
    corrections, pairs, means = stage_corrections(long)
    corrections.to_csv(OUT / "stage_by_edit_class_corrected_tests.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(OUT / "stage_interaction_pairwise_tests.csv", index=False, encoding="utf-8-sig")
    means.to_csv(OUT / "stage_by_edit_class_descriptives.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

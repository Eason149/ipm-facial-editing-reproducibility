"""Predict behavior from LPP component amplitudes and facial factors.

This script replaces subject-level correlation tests with cross-validated
prediction. Targets are subject-centered naturalness/beauty ratings over the
16 edited conditions. Models are evaluated with leave-one-subject-out CV and
within-subject permutation tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
OUTPUTS = PROJECT / "outputs"
FIGURES = PROJECT / "figures"
LOGS = PROJECT / "logs"
BEHAVIOR_OUTPUTS = ROOT / "RSA_time_resolved_analysis" / "results_behavior_rsa" / "outputs"

TARGETS = ("Naturalness_choice", "Beauty_choice")
EEG_COLUMNS = {
    "pz": "LPP_Pz_mean_amplitude_RDM",
    "midline": "LPP_midline_mean_amplitude_RDM",
    "centroparietal": "LPP_centroparietal_mean_amplitude_RDM",
    "shared55": "LPP_shared_montage_mean_amplitude_RDM",
}
FACTOR_COLUMNS = ("FSlim", "Eye", "Mouth", "Skin")
RIDGE_ALPHA = 1.0
N_PERMUTATIONS = 1000
RANDOM_SEED = 20260612


def load_design() -> pd.DataFrame:
    behavior = pd.read_csv(BEHAVIOR_OUTPUTS / "behavior_condition_choices.csv")
    eeg = pd.read_csv(OUTPUTS / "comprehensive_component_condition_values.csv")
    eeg = eeg[eeg["analysis"].isin(EEG_COLUMNS.values())].copy()
    eeg = eeg[eeg["raw_CondID"].between(2, 17)].copy()
    eeg["CondID"] = eeg["raw_CondID"].astype(int) - 1
    eeg_wide = eeg.pivot_table(
        index=["subj", "CondID"],
        columns="analysis",
        values="component_value",
        aggfunc="first",
    ).reset_index()
    eeg_wide = eeg_wide.rename(columns={value: key for key, value in EEG_COLUMNS.items()})
    factors = pd.read_csv(OUTPUTS / "comprehensive_17_condition_table.csv")
    factors = factors[factors["raw_CondID"].between(2, 17)].copy()
    factors["CondID"] = factors["raw_CondID"].astype(int) - 1
    factors = factors[["CondID", *FACTOR_COLUMNS]]
    merged = behavior.merge(eeg_wide, on=["subj", "CondID"], how="inner").merge(factors, on="CondID", how="left")
    for target in TARGETS:
        merged[f"{target}_centered"] = merged[target] - merged.groupby("subj")[target].transform("mean")
    for column in EEG_COLUMNS:
        merged[f"{column}_z_within"] = (
            merged[column] - merged.groupby("subj")[column].transform("mean")
        ) / merged.groupby("subj")[column].transform("std").replace(0, np.nan)
    return merged.dropna().reset_index(drop=True)


def feature_matrix(data: pd.DataFrame, feature_set: str) -> np.ndarray:
    pz = data[["pz_z_within"]].to_numpy(dtype=float)
    lpp = data[[f"{name}_z_within" for name in EEG_COLUMNS]].to_numpy(dtype=float)
    factors = data[list(FACTOR_COLUMNS)].to_numpy(dtype=float)
    if feature_set == "Pz_only":
        return pz
    if feature_set == "LPP_ROIs":
        return lpp
    if feature_set == "Factors":
        return factors
    if feature_set == "Factors_plus_Pz":
        return np.column_stack([factors, pz])
    if feature_set == "Factors_plus_LPP":
        return np.column_stack([factors, lpp])
    if feature_set == "Factors_Pz_interactions":
        interactions = factors * pz
        return np.column_stack([factors, pz, interactions])
    if feature_set == "Factors_LPP_interactions":
        interactions = np.column_stack([factors * lpp[:, [index]] for index in range(lpp.shape[1])])
        return np.column_stack([factors, lpp, interactions])
    if feature_set == "All_factor_interactions_plus_Pz":
        terms = []
        for mask in range(1, 2 ** factors.shape[1]):
            columns = [index for index in range(factors.shape[1]) if mask & (1 << index)]
            terms.append(np.prod(factors[:, columns], axis=1))
        factor_terms = np.column_stack(terms)
        return np.column_stack([factor_terms, pz, factor_terms * pz])
    raise ValueError(f"Unknown feature set: {feature_set}")


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x_design = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(x_design.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(x_design.T @ x_design + alpha * penalty) @ x_design.T @ y_train
    return coef[1:], float(coef[0])


def predict_ridge(x: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return x @ coef + intercept


def cv_predict(data: pd.DataFrame, target: str, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    x = feature_matrix(data, feature_set)
    y = data[f"{target}_centered"].to_numpy(dtype=float)
    groups = data["subj"].to_numpy()
    predictions = np.full(len(y), np.nan)
    for group in np.unique(groups):
        test_index = np.where(groups == group)[0]
        train_index = np.where(groups != group)[0]
        x_train, x_test = standardize_train_test(x[train_index], x[test_index])
        y_train = y[train_index]
        coef, intercept = fit_ridge(x_train, y_train, RIDGE_ALPHA)
        predictions[test_index] = predict_ridge(x_test, coef, intercept)
    return y, predictions


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[valid]
    pred = y_pred[valid]
    if len(y) < 3:
        return {"r2": np.nan, "pearson_r": np.nan, "rmse": np.nan}
    pearson = stats.pearsonr(y, pred).statistic if np.std(pred) > 0 else np.nan
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "r2": float(r2),
        "pearson_r": float(pearson),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
    }


def permute_within_subject(data: pd.DataFrame, target: str, rng: np.random.Generator) -> pd.DataFrame:
    permuted = data.copy()
    centered = f"{target}_centered"
    values = []
    for _, subset in data.groupby("subj", sort=False):
        shuffled = subset[centered].to_numpy(dtype=float).copy()
        rng.shuffle(shuffled)
        values.append(pd.Series(shuffled, index=subset.index))
    permuted[centered] = pd.concat(values).sort_index().to_numpy()
    return permuted


def run_prediction() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_design()
    feature_sets = (
        "Pz_only",
        "LPP_ROIs",
        "Factors",
        "Factors_plus_Pz",
        "Factors_plus_LPP",
        "Factors_Pz_interactions",
        "Factors_LPP_interactions",
        "All_factor_interactions_plus_Pz",
    )
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    observed_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
    for target in TARGETS:
        for feature_set in feature_sets:
            y, pred = cv_predict(data, target, feature_set)
            metrics = evaluate_predictions(y, pred)
            observed_cache[(target, feature_set)] = (y, pred, metrics)
            permutation_r2 = []
            permutation_r = []
            for _ in range(N_PERMUTATIONS):
                permuted = permute_within_subject(data, target, rng)
                y_perm, pred_perm = cv_predict(permuted, target, feature_set)
                perm_metrics = evaluate_predictions(y_perm, pred_perm)
                permutation_r2.append(perm_metrics["r2"])
                permutation_r.append(perm_metrics["pearson_r"])
            permutation_r2 = np.asarray(permutation_r2, dtype=float)
            permutation_r = np.asarray(permutation_r, dtype=float)
            rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "n_observations": int(len(y)),
                    "n_subjects": int(data["subj"].nunique()),
                    "cv_r2": metrics["r2"],
                    "cv_pearson_r": metrics["pearson_r"],
                    "cv_rmse": metrics["rmse"],
                    "p_perm_r2_greater": float((np.sum(permutation_r2 >= metrics["r2"]) + 1) / (N_PERMUTATIONS + 1)),
                    "p_perm_r_greater": float((np.sum(permutation_r >= metrics["pearson_r"]) + 1) / (N_PERMUTATIONS + 1)),
                    "perm_r2_mean": float(np.nanmean(permutation_r2)),
                    "perm_r_mean": float(np.nanmean(permutation_r)),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "subj": data["subj"],
                        "CondID": data["CondID"],
                        "target": target,
                        "feature_set": feature_set,
                        "y_true_centered": y,
                        "y_pred_centered": pred,
                    }
                )
            )
    results = pd.DataFrame(rows)
    results["q_perm_r2_within_target"] = results.groupby("target")["p_perm_r2_greater"].transform(
        lambda values: fdr_bh(values.to_numpy(dtype=float))
    )
    results["q_perm_r_within_target"] = results.groupby("target")["p_perm_r_greater"].transform(
        lambda values: fdr_bh(values.to_numpy(dtype=float))
    )
    return results.sort_values(["target", "p_perm_r_greater"]).reset_index(drop=True), pd.concat(prediction_rows, ignore_index=True)


def fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    p_array = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_array.shape, np.nan, dtype=float)
    valid = np.isfinite(p_array)
    if not valid.any():
        return adjusted
    valid_p = p_array[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.minimum(corrected, 1.0)
    restored = np.empty_like(corrected)
    restored[order] = corrected
    adjusted[valid] = restored
    return adjusted


def plot_prediction_results(results: pd.DataFrame, path: Path) -> None:
    feature_order = [
        "Pz_only",
        "LPP_ROIs",
        "Factors",
        "Factors_plus_Pz",
        "Factors_plus_LPP",
        "Factors_Pz_interactions",
        "Factors_LPP_interactions",
        "All_factor_interactions_plus_Pz",
    ]
    labels = [
        "Pz only",
        "LPP ROIs",
        "Factors",
        "Factors+Pz",
        "Factors+LPP",
        "Factors x Pz",
        "Factors x LPP",
        "All interactions+Pz",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharey=True, constrained_layout=True)
    colors = {"Naturalness_choice": "#2a7f62", "Beauty_choice": "#b85b36"}
    for axis, target in zip(axes, TARGETS):
        subset = results[results["target"] == target].set_index("feature_set").reindex(feature_order)
        x = np.arange(len(feature_order))
        bars = axis.bar(x, subset["cv_pearson_r"], color=colors[target], alpha=0.88)
        axis.axhline(0, color="#333", lw=0.8)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_title(target.replace("_choice", ""), fontweight="bold")
        axis.set_ylabel("Leave-one-subject-out prediction r")
        axis.grid(axis="y", color="#e9e9e9", lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        for index, (_, row) in enumerate(subset.iterrows()):
            marker = ""
            if float(row["q_perm_r_within_target"]) < 0.05:
                marker = "*"
            elif float(row["p_perm_r_greater"]) < 0.05:
                marker = "+"
            if marker:
                y = row["cv_pearson_r"] + (0.015 if row["cv_pearson_r"] >= 0 else -0.015)
                axis.text(index, y, marker, ha="center", va="bottom" if row["cv_pearson_r"] >= 0 else "top", fontsize=13)
        axis.text(0.01, 0.98, "* FDR q < .05; + permutation p < .05", transform=axis.transAxes, va="top", fontsize=9)
    fig.suptitle("Cross-validated behavior prediction from LPP amplitudes and facial factors", fontweight="bold")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_scatter(predictions: pd.DataFrame, results: pd.DataFrame, path: Path) -> None:
    best = results.sort_values(["target", "p_perm_r_greater"]).groupby("target").head(1)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    colors = {"Naturalness_choice": "#2a7f62", "Beauty_choice": "#b85b36"}
    for axis, (_, row) in zip(axes, best.iterrows()):
        subset = predictions[(predictions["target"] == row["target"]) & (predictions["feature_set"] == row["feature_set"])]
        axis.scatter(subset["y_true_centered"], subset["y_pred_centered"], s=22, alpha=0.36, color=colors[row["target"]], edgecolors="none")
        if subset["y_true_centered"].std() > 0:
            slope, intercept = np.polyfit(subset["y_true_centered"], subset["y_pred_centered"], 1)
            x_line = np.linspace(subset["y_true_centered"].min(), subset["y_true_centered"].max(), 100)
            axis.plot(x_line, intercept + slope * x_line, color="#222", lw=1.5)
        axis.axhline(0, color="#bbb", lw=0.8)
        axis.axvline(0, color="#bbb", lw=0.8)
        axis.set_title(f"{row['target'].replace('_choice', '')}: {row['feature_set']}", fontweight="bold")
        axis.set_xlabel("Observed centered rating")
        axis.set_ylabel("Predicted centered rating")
        axis.grid(color="#eeeeee", lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.02,
            0.98,
            f"r={row['cv_pearson_r']:.3f}, perm p={row['p_perm_r_greater']:.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
    fig.suptitle("Best cross-validated prediction per behavior", fontweight="bold")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in (OUTPUTS, FIGURES, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    results, predictions = run_prediction()
    results.to_csv(OUTPUTS / "lpp_behavior_prediction_results.csv", index=False)
    predictions.to_csv(OUTPUTS / "lpp_behavior_prediction_cv_predictions.csv", index=False)
    plot_prediction_results(results, FIGURES / "lpp_behavior_prediction_cv")
    plot_best_scatter(predictions, results, FIGURES / "lpp_behavior_prediction_best_scatter")
    log_lines = [
        "Cross-validated behavior prediction",
        "===================================",
        "Targets are subject-centered naturalness/beauty ratings over 16 edited conditions.",
        "Evaluation uses leave-one-subject-out CV and within-subject target permutation.",
    ]
    for _, row in results.sort_values(["target", "p_perm_r_greater"]).iterrows():
        log_lines.append(
            f"- {row['target']} / {row['feature_set']}: "
            f"r={row['cv_pearson_r']:.4f}, R2={row['cv_r2']:.4f}, "
            f"p_r={row['p_perm_r_greater']:.4g}, q_r={row['q_perm_r_within_target']:.4g}"
        )
    (LOGS / "lpp_behavior_prediction_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    metadata: Mapping[str, object] = {
        "method": "Ridge regression with leave-one-subject-out cross-validation and within-subject permutation tests.",
        "n_permutations": N_PERMUTATIONS,
        "targets": list(TARGETS),
        "feature_columns": {
            "EEG": EEG_COLUMNS,
            "factors": FACTOR_COLUMNS,
        },
    }
    (OUTPUTS / "lpp_behavior_prediction_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"LPP behavior prediction complete: {PROJECT}")


if __name__ == "__main__":
    main()

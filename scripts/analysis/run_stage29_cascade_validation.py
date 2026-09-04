#!/usr/bin/env python
"""Frozen Stage 2.9 representation -> EEG -> behavior cascade audit.

The analysis specification was written before this script was executed.  Source
data are read only; all outputs are written to results/stage_2_9.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("IPM_DATA_ROOT", REPO.parent)).resolve()
LOCAL_PACKAGES = DATA_ROOT / "RSA_time_resolved_analysis" / ".python-packages"
if LOCAL_PACKAGES.exists():
    sys.path.append(str(LOCAL_PACKAGES))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


OUT = REPO / "results" / "stage_2_9"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEED_GAI = 20260904
SEED_EEG_BEHAVIOR = 20260905
SEED_PREDICTION = 20260906
NPERM = 10_000
SUBJECTS = [f"s{i}" for i in range(1, 31) if i not in (5, 18)]
FACTORS = ["FSlim", "Eye", "Mouth", "Skin"]
METRICS = ["G_new", "A2", "I"]
WINDOWS = {
    "350-600 ms": "MiddleLate_350_600",
    "600-1000 ms": "Late_600_1000",
}
ROIS = ["Occipitotemporal", "Posterior", "Centroparietal", "Frontal"]


def max_t(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Participant sign-flip maximum-|t| correction across matrix columns."""
    matrix = np.asarray(matrix, float)
    n, tests = matrix.shape
    observed = stats.ttest_1samp(matrix, 0.0, axis=0).statistic
    rng = np.random.default_rng(seed)
    maxima = np.zeros(NPERM)
    for start in range(0, NPERM, 500):
        size = min(500, NPERM - start)
        signs = rng.choice([-1.0, 1.0], size=(size, n, 1))
        perm = matrix[None, :, :] * signs
        means = perm.mean(axis=1)
        sds = perm.std(axis=1, ddof=1)
        tvals = means / (sds / np.sqrt(n))
        maxima[start:start + size] = np.nanmax(np.abs(tvals), axis=1)
    corrected = np.array([(1 + np.sum(maxima >= abs(t))) / (NPERM + 1) for t in observed])
    return observed, corrected


def group_table(
    betas: pd.DataFrame,
    keys: list[str],
    value: str,
    seed: int,
    family_label: str,
) -> pd.DataFrame:
    labels, vectors = [], []
    for key, sub in betas.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        vec = sub.set_index("Subject").reindex(SUBJECTS)[value].to_numpy(float)
        if not np.isfinite(vec).all():
            raise RuntimeError(f"Missing participant value for {key}")
        labels.append(key)
        vectors.append(vec)
    matrix = np.column_stack(vectors)
    tvals, pmax = max_t(matrix, seed)
    rows = []
    for j, key in enumerate(labels):
        vals = matrix[:, j]
        mean = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
        row = {keys[k]: key[k] for k in range(len(keys))}
        row.update({
            "family": family_label,
            "n_participants": len(vals),
            "mean_beta": mean,
            "se": se,
            "ci95_low": mean - stats.t.ppf(.975, len(vals) - 1) * se,
            "ci95_high": mean + stats.t.ppf(.975, len(vals) - 1) * se,
            "t": float(tvals[j]),
            "p_uncorrected": float(2 * stats.t.sf(abs(tvals[j]), len(vals) - 1)),
            "p_maxT_familywise": float(pmax[j]),
            "cohen_dz": float(mean / vals.std(ddof=1)),
            "n_permutations": NPERM,
            "seed": seed,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def standardize_metrics(metric: pd.DataFrame) -> pd.DataFrame:
    out = metric.copy()
    for col in METRICS:
        out[col] = (out[col].astype(float) - out[col].astype(float).mean()) / out[col].astype(float).std(ddof=0)
    return out


def load_analysis_data() -> pd.DataFrame:
    master_path = DATA_ROOT / "CHB_multimodal_facial_editing" / "multimodal_trial_master.csv"
    metric_path = DATA_ROOT / "ipm_stage_2_6" / "appearance_metric_sensitivity.csv"
    geometry_path = DATA_ROOT / "ipm_stage_2_6" / "geometric_metric_reproduced_values.csv"
    roi_columns = [f"{roi}_{tag}_Mean" for roi in ROIS for tag in WINDOWS.values()]
    usecols = [
        "Subject", "Picture", "Identity", "IsOriginal", "IsAttentionCheck",
        "EpochAccepted", "ArtifactFlag", "TrialOrder", "Beauty", "Naturalness",
        *FACTORS, *roi_columns,
    ]
    data = pd.read_csv(master_path, usecols=usecols, low_memory=False)
    data = data[
        data.Subject.isin(SUBJECTS)
        & data.IsOriginal.eq(0)
        & data.IsAttentionCheck.eq(0)
        & data.EpochAccepted.eq(True)
        & data.ArtifactFlag.eq(False)
    ].copy()
    metric = pd.read_csv(metric_path)[["identity", "picture_filename", "A2", "I"]]
    geometry = pd.read_csv(geometry_path)[["identity", "picture_filename", "G_new"]]
    metric = metric.merge(geometry, on=["identity", "picture_filename"], validate="one_to_one")
    metric = standardize_metrics(metric)
    data = data.merge(
        metric,
        left_on=["Identity", "Picture"],
        right_on=["identity", "picture_filename"],
        how="left",
        validate="many_to_one",
    )
    required = ["TrialOrder", "Beauty", "Naturalness", *FACTORS, *METRICS, *roi_columns]
    data = data.dropna(subset=required).copy()
    counts = data.groupby("Subject").size().reindex(SUBJECTS)
    if counts.isna().any() or (counts < 100).any():
        raise RuntimeError(f"Unexpected participant trial counts: {counts.to_dict()}")
    if data[METRICS].isna().any().any():
        raise RuntimeError("Metric merge failed")
    return data


def design_frame(sub: pd.DataFrame, include_representation: bool = True) -> pd.DataFrame:
    parts = [pd.Series(1.0, index=sub.index, name="Intercept")]
    if include_representation:
        parts.append(sub[FACTORS + METRICS].astype(float))
    order = sub["TrialOrder"].astype(float)
    order = (order - order.mean()) / order.std(ddof=0)
    parts.append(order.rename("TrialOrder_z"))
    parts.append(pd.get_dummies(sub["Identity"].astype(str), prefix="Identity", drop_first=True, dtype=float))
    return pd.concat(parts, axis=1)


def subject_ols(sub: pd.DataFrame, outcome: str, predictors: pd.DataFrame) -> pd.Series:
    x = predictors.to_numpy(float)
    y = sub[outcome].to_numpy(float)
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise RuntimeError(f"Rank-deficient model for {sub.Subject.iloc[0]} / {outcome}")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return pd.Series(beta, index=predictors.columns)


def representation_to_eeg(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    roi_rows = []
    for subject in SUBJECTS:
        sub = data[data.Subject.eq(subject)].copy()
        x = design_frame(sub, include_representation=True)
        for roi in ROIS:
            for window, tag in WINDOWS.items():
                outcome = f"{roi}_{tag}_Mean"
                beta = subject_ols(sub, outcome, x)
                for predictor in FACTORS + METRICS:
                    roi_rows.append({
                        "Subject": subject, "ROI": roi, "window": window,
                        "predictor": predictor, "beta_uV": beta[predictor],
                    })
                    if roi == "Centroparietal" and predictor in METRICS:
                        rows.append({
                            "Subject": subject, "window": window,
                            "metric": predictor, "beta_uV": beta[predictor],
                        })
    primary = pd.DataFrame(rows)
    roi_betas = pd.DataFrame(roi_rows)
    primary_stats = group_table(
        primary, ["window", "metric"], "beta_uV", SEED_GAI,
        "6 tests: 3 continuous metrics x 2 windows",
    )
    roi_stats = group_table(
        roi_betas, ["ROI", "window", "predictor"], "beta_uV", SEED_GAI + 100,
        "56-test ROI audit: 4 ROIs x 2 windows x 7 predictors",
    )
    primary.to_csv(OUT / "gai_to_eeg_subject_betas.csv", index=False, encoding="utf-8-sig")
    primary_stats.to_csv(OUT / "gai_to_eeg_group_stats.csv", index=False, encoding="utf-8-sig")
    roi_betas.to_csv(OUT / "roi_audit_subject_betas.csv", index=False, encoding="utf-8-sig")
    roi_stats.to_csv(OUT / "roi_audit_group_stats.csv", index=False, encoding="utf-8-sig")
    return primary, primary_stats, roi_stats


def eeg_to_behavior(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adjusted_rows, simple_rows = [], []
    cp_cols = [f"Centroparietal_{tag}_Mean" for tag in WINDOWS.values()]
    for subject in SUBJECTS:
        sub = data[data.Subject.eq(subject)].copy()
        eeg = sub[cp_cols].astype(float)
        eeg_z = (eeg - eeg.mean()) / eeg.std(ddof=0)
        eeg_z.columns = [f"EEG_{window}" for window in WINDOWS]

        x_adjusted = pd.concat([design_frame(sub, include_representation=True), eeg_z], axis=1)
        x_simple = pd.concat([design_frame(sub, include_representation=False), eeg_z], axis=1)
        for outcome in ["Beauty", "Naturalness"]:
            adjusted = subject_ols(sub, outcome, x_adjusted)
            simple = subject_ols(sub, outcome, x_simple)
            for window in WINDOWS:
                term = f"EEG_{window}"
                adjusted_rows.append({
                    "Subject": subject, "outcome": outcome, "window": window,
                    "rating_beta_per_within_subject_eeg_sd": adjusted[term],
                    "model": "representation-adjusted",
                })
                simple_rows.append({
                    "Subject": subject, "outcome": outcome, "window": window,
                    "rating_beta_per_within_subject_eeg_sd": simple[term],
                    "model": "EEG-only plus order and identity",
                })
    adjusted_betas = pd.DataFrame(adjusted_rows)
    simple_betas = pd.DataFrame(simple_rows)
    adjusted_stats = group_table(
        adjusted_betas, ["outcome", "window"],
        "rating_beta_per_within_subject_eeg_sd", SEED_EEG_BEHAVIOR,
        "4 tests: 2 outcomes x 2 windows, representation-adjusted",
    )
    simple_stats = group_table(
        simple_betas, ["outcome", "window"],
        "rating_beta_per_within_subject_eeg_sd", SEED_EEG_BEHAVIOR + 100,
        "4 tests: 2 outcomes x 2 windows, descriptive simplified model",
    )
    adjusted_betas.to_csv(OUT / "eeg_to_behavior_adjusted_subject_betas.csv", index=False, encoding="utf-8-sig")
    adjusted_stats.to_csv(OUT / "eeg_to_behavior_adjusted_group_stats.csv", index=False, encoding="utf-8-sig")
    simple_betas.to_csv(OUT / "eeg_to_behavior_simple_subject_betas.csv", index=False, encoding="utf-8-sig")
    simple_stats.to_csv(OUT / "eeg_to_behavior_simple_group_stats.csv", index=False, encoding="utf-8-sig")
    return adjusted_betas, adjusted_stats, simple_betas, simple_stats


def prediction_matrix(data: pd.DataFrame, include_eeg: bool) -> tuple[np.ndarray, list[str]]:
    parts = [data[FACTORS + METRICS].astype(float).reset_index(drop=True)]
    order = data["TrialOrder"].astype(float)
    parts.append(order.reset_index(drop=True).rename("TrialOrder"))
    parts.append(pd.get_dummies(data["Identity"].astype(str), prefix="Identity", drop_first=True, dtype=float).reset_index(drop=True))
    if include_eeg:
        cp_cols = [f"Centroparietal_{tag}_Mean" for tag in WINDOWS.values()]
        eeg = data[cp_cols].astype(float).reset_index(drop=True)
        eeg.columns = [f"EEG_{window}" for window in WINDOWS]
        parts.append(eeg)
    frame = pd.concat(parts, axis=1)
    return frame.to_numpy(float), list(frame.columns)


def fit_ridge_inner_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
) -> tuple[StandardScaler, Ridge, float]:
    splitter = GroupKFold(n_splits=5)
    scores = []
    for alpha in alphas:
        fold_scores = []
        for train, valid in splitter.split(x, y, groups):
            scaler = StandardScaler().fit(x[train])
            model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
            pred = model.predict(scaler.transform(x[valid]))
            fold_scores.append(mean_absolute_error(y[valid], pred))
        scores.append(float(np.mean(fold_scores)))
    best_alpha = float(alphas[int(np.argmin(scores))])
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=best_alpha).fit(scaler.transform(x), y)
    return scaler, model, best_alpha


def incremental_prediction(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_x, base_names = prediction_matrix(data, include_eeg=False)
    eeg_x, eeg_names = prediction_matrix(data, include_eeg=True)
    groups = data.Subject.to_numpy(str)
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    rows = []
    for outcome in ["Beauty", "Naturalness"]:
        y = data[outcome].to_numpy(float)
        for held in SUBJECTS:
            train = groups != held
            test = groups == held
            for label, x, names in [
                ("representations-only", base_x, base_names),
                ("representations-plus-EEG", eeg_x, eeg_names),
            ]:
                scaler, model, alpha = fit_ridge_inner_cv(x[train], y[train], groups[train], alphas)
                pred = model.predict(scaler.transform(x[test]))
                rows.append({
                    "outcome": outcome, "held_out_subject": held, "model": label,
                    "MAE": mean_absolute_error(y[test], pred), "best_alpha": alpha,
                    "n_train_trials": int(train.sum()), "n_test_trials": int(test.sum()),
                    "n_features": len(names),
                })
    results = pd.DataFrame(rows)
    wide = results.pivot(index=["outcome", "held_out_subject"], columns="model", values="MAE").reset_index()
    wide["MAE_improvement_from_EEG"] = wide["representations-only"] - wide["representations-plus-EEG"]
    temp = wide.rename(columns={"held_out_subject": "Subject"})
    stats_table = group_table(
        temp, ["outcome"], "MAE_improvement_from_EEG", SEED_PREDICTION,
        "2 tests: Beauty and Naturalness held-out MAE improvement",
    )
    results.to_csv(OUT / "cascade_prediction_nested_lopo.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUT / "cascade_prediction_improvement_by_subject.csv", index=False, encoding="utf-8-sig")
    stats_table.to_csv(OUT / "cascade_prediction_group_stats.csv", index=False, encoding="utf-8-sig")
    return wide, stats_table


def plot_cascade(gai: pd.DataFrame, eeg_behavior: pd.DataFrame, pred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3))
    colors = {"G_new": "#4C78A8", "A2": "#E45756", "I": "#72B7B2"}
    gai = gai.copy()
    gai["label"] = gai["metric"] + " | " + gai["window"]
    y = np.arange(len(gai))
    axes[0].errorbar(gai.mean_beta, y, xerr=[gai.mean_beta-gai.ci95_low, gai.ci95_high-gai.mean_beta], fmt="none", ecolor="#555", capsize=3)
    axes[0].scatter(gai.mean_beta, y, c=[colors[x] for x in gai.metric], s=45)
    axes[0].axvline(0, color="#777", lw=.8)
    axes[0].set_yticks(y, gai.label, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Unique EEG coefficient (uV)")
    axes[0].set_title("A. G-A2-I to central-parietal EEG")
    for i, row in gai.reset_index(drop=True).iterrows():
        ptxt = f"{row.p_maxT_familywise:.4f}" if row.p_maxT_familywise < .001 else f"{row.p_maxT_familywise:.3f}"
        axes[0].text(row.ci95_high, i-.18, f"pFWE={ptxt}", fontsize=7)

    eb = eeg_behavior.copy()
    eb["label"] = eb["outcome"] + " | " + eb["window"]
    y = np.arange(len(eb))
    axes[1].errorbar(eb.mean_beta, y, xerr=[eb.mean_beta-eb.ci95_low, eb.ci95_high-eb.mean_beta], fmt="o", color="#7A5195", capsize=3)
    axes[1].axvline(0, color="#777", lw=.8)
    axes[1].set_yticks(y, eb.label, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Rating change per within-person EEG SD")
    axes[1].set_title("B. EEG to behavior, adjusted")
    for i, row in eb.reset_index(drop=True).iterrows():
        axes[1].text(row.ci95_high, i-.18, f"pFWE={row.p_maxT_familywise:.3f}", fontsize=7)

    x = np.arange(len(pred))
    axes[2].bar(x, pred.mean_beta, color=["#4C78A8", "#E45756"])
    axes[2].errorbar(x, pred.mean_beta, yerr=[pred.mean_beta-pred.ci95_low, pred.ci95_high-pred.mean_beta], fmt="none", ecolor="black", capsize=4)
    axes[2].axhline(0, color="#777", lw=.8)
    axes[2].set_xticks(x, pred.outcome)
    axes[2].set_ylabel("Held-out MAE improvement from EEG")
    axes[2].set_title("C. EEG incremental prediction")
    for i, row in pred.reset_index(drop=True).iterrows():
        axes[2].text(i, row.ci95_high, f"pFWE={row.p_maxT_familywise:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Testing the proposed representation -> EEG -> behavior cascade", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "Figure_7_cascade_validation.png", dpi=280, bbox_inches="tight")
    fig.savefig(FIG / "Figure_7_cascade_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_roi_audit(roi: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), sharey=True, constrained_layout=True)
    for ax, window in zip(axes, WINDOWS):
        sub = roi[roi.window.eq(window)].copy()
        pivot = sub.pivot(index="predictor", columns="ROI", values="cohen_dz").reindex(FACTORS + METRICS)[ROIS]
        im = ax.imshow(pivot.to_numpy(float), cmap="RdBu_r", vmin=-1.1, vmax=1.1, aspect="auto")
        ax.set_xticks(np.arange(len(ROIS)), ROIS, rotation=28, ha="right")
        ax.set_yticks(np.arange(len(pivot)), pivot.index)
        ax.set_title(window)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                row = sub[(sub.predictor.eq(pivot.index[i])) & (sub.ROI.eq(ROIS[j]))].iloc[0]
                marker = "*" if row.p_maxT_familywise < .05 else ""
                ax.text(
                    j, i, f"{pivot.iloc[i,j]:.2f}{marker}", ha="center", va="center",
                    fontsize=8, weight="bold" if marker else "normal",
                )
    fig.colorbar(im, ax=axes, shrink=.78, pad=.025, label="Cohen dz across participants")
    fig.suptitle("Complete four-ROI sensitivity audit (not used to select the primary ROI)", fontsize=12, weight="bold")
    fig.savefig(FIG / "Figure_S_ROI_sensitivity.png", dpi=280, bbox_inches="tight")
    fig.savefig(FIG / "Figure_S_ROI_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        vals = []
        for value in row:
            vals.append(f"{value:.5g}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(data: pd.DataFrame, gai: pd.DataFrame, eb: pd.DataFrame, simple: pd.DataFrame, pred: pd.DataFrame, roi: pd.DataFrame) -> None:
    gai_pass = gai[gai.p_maxT_familywise < .05]
    eb_pass = eb[eb.p_maxT_familywise < .05]
    pred_pass = pred[(pred.mean_beta > 0) & (pred.p_maxT_familywise < .05)]
    cascade = (not gai_pass.empty) and (not eb_pass.empty) and (not pred_pass.empty)
    lines = [
        "# Stage 2.9 Cascade Validation Report", "",
        "## Frozen design", "",
        f"N={data.Subject.nunique()} participants; {len(data)} accepted edited-image trials. The continuous representation model used G_new, A2 and I simultaneously while controlling the four operations, trial order and identity. All group tests use the participant as the inferential unit.", "",
        "## G-A2-I to central-parietal EEG", "",
        markdown_table(gai, ["window", "metric", "mean_beta", "ci95_low", "ci95_high", "t", "p_maxT_familywise", "cohen_dz"]), "",
        "## Central-parietal EEG to behavior (representation-adjusted)", "",
        markdown_table(eb, ["outcome", "window", "mean_beta", "ci95_low", "ci95_high", "t", "p_maxT_familywise", "cohen_dz"]), "",
        "## Simplified EEG-to-behavior model", "",
        markdown_table(simple, ["outcome", "window", "mean_beta", "ci95_low", "ci95_high", "t", "p_maxT_familywise", "cohen_dz"]), "",
        "## Held-out-participant incremental prediction", "",
        markdown_table(pred, ["outcome", "mean_beta", "ci95_low", "ci95_high", "t", "p_maxT_familywise", "cohen_dz"]), "",
        "## ROI audit", "",
        "The primary central-parietal ROI was literature-motivated. The complete four-ROI table is retained as a sensitivity audit and was not used to replace the primary ROI. The familywise correction covers all 56 ROI x window x predictor tests.", "",
        markdown_table(roi, ["ROI", "window", "predictor", "mean_beta", "t", "p_maxT_familywise", "cohen_dz"]), "",
        "## Decision", "",
        f"Full statistical cascade supported: **{cascade}**.", "",
        "This is an associational cascade audit, not a causal mediation analysis. A failed edge is a substantive boundary result and is not grounds for choosing another ROI, metric implementation or time window.",
    ]
    (OUT / "CASCADE_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_analysis_data()
    _, gai_stats, roi_stats = representation_to_eeg(data)
    _, eb_stats, _, simple_stats = eeg_to_behavior(data)
    _, pred_stats = incremental_prediction(data)
    plot_cascade(gai_stats, eb_stats, pred_stats)
    plot_roi_audit(roi_stats)
    write_report(data, gai_stats, eb_stats, simple_stats, pred_stats, roi_stats)
    manifest = {
        "status": "completed",
        "specification": "docs/CASCADE_ANALYSIS_SPECIFICATION_20260904.md",
        "data_root": "${IPM_DATA_ROOT}",
        "n_participants": int(data.Subject.nunique()),
        "n_trials": int(len(data)),
        "participants": SUBJECTS,
        "metrics": METRICS,
        "factors": FACTORS,
        "primary_roi_channels": ["CPz", "Pz", "CP1", "CP2", "P1", "P2"],
        "windows": list(WINDOWS),
        "n_permutations": NPERM,
        "seeds": [SEED_GAI, SEED_EEG_BEHAVIOR, SEED_PREDICTION],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed Stage 2.9: {OUT}")


if __name__ == "__main__":
    main()

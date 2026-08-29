#!/usr/bin/env python
"""Time-resolved ERP dynamics and persistence analysis for edited-face EEG.

This analysis estimates trial-level factor effects within each subject, then
tests group-level beta time courses with temporal cluster permutation. The
outputs are designed for a paper narrative about onset, duration, persistence,
and surface/local versus structural/configural edit effects.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_bundled_python = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "python" / "python.exe"
if sys.version_info[:2] != (3, 12) and _bundled_python.exists() and Path(sys.executable).resolve() != _bundled_python.resolve():
    raise SystemExit(subprocess.call([str(_bundled_python), *sys.argv]))

_project_guess = Path.cwd()
_pkg = _project_guess / "RSA_time_resolved_analysis" / ".python-packages"
if _pkg.exists() and str(_pkg) not in sys.path:
    sys.path.append(str(_pkg))

_mpl_cache = Path.cwd() / ".codex_matplotlib_cache"
_mpl_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_time_resolved_factor_decoding import (
    FACTORS,
    STRUCTURAL,
    SURFACE,
    Context,
    N170_LPP_CHANNELS,
    assign_analysis_channels,
    extract_subject_matrix,
    find_eeg_files,
    fdr_bh,
    load_subject,
    natural_subject_key,
    parse_eprime_condition_map,
    save_csv,
    setup_dirs,
    setup_logging,
)


CONTRASTS = ["Skin", "FSlim", "Eye", "Mouth", "Surface", "Structural", "Surface_minus_Structural"]
STAGES = {
    "N170": (140, 200),
    "P2_EPN": (200, 300),
    "EditIntegration": (300, 450),
    "LPP": (450, 800),
}

CHANNEL_POS = {
    "FP1": (-0.35, 0.95), "FPZ": (0.0, 1.0), "FP2": (0.35, 0.95),
    "AF3": (-0.28, 0.82), "AF4": (0.28, 0.82),
    "F7": (-0.78, 0.58), "F5": (-0.58, 0.62), "F3": (-0.42, 0.55), "F1": (-0.18, 0.50),
    "FZ": (0.0, 0.48), "F2": (0.18, 0.50), "F4": (0.42, 0.55), "F6": (0.58, 0.62), "F8": (0.78, 0.58),
    "FC5": (-0.62, 0.34), "FC3": (-0.42, 0.30), "FC1": (-0.20, 0.27), "FCZ": (0.0, 0.25),
    "FC2": (0.20, 0.27), "FC4": (0.42, 0.30), "FC6": (0.62, 0.34),
    "T7": (-0.9, 0.0), "C5": (-0.68, 0.05), "C3": (-0.45, 0.0), "C1": (-0.22, 0.0),
    "CZ": (0.0, 0.0), "C2": (0.22, 0.0), "C4": (0.45, 0.0), "C6": (0.68, 0.05), "T8": (0.9, 0.0),
    "TP7": (-0.78, -0.25), "CP5": (-0.62, -0.30), "CP3": (-0.42, -0.32), "CP1": (-0.20, -0.34),
    "CPZ": (0.0, -0.35), "CP2": (0.20, -0.34), "CP4": (0.42, -0.32), "CP6": (0.62, -0.30), "TP8": (0.78, -0.25),
    "P7": (-0.72, -0.55), "P5": (-0.55, -0.58), "P3": (-0.38, -0.60), "P1": (-0.18, -0.62),
    "PZ": (0.0, -0.64), "P2": (0.18, -0.62), "P4": (0.38, -0.60), "P6": (0.55, -0.58), "P8": (0.72, -0.55),
    "PO7": (-0.55, -0.78), "PO5": (-0.42, -0.80), "PO3": (-0.28, -0.82), "POZ": (0.0, -0.84),
    "PO4": (0.28, -0.82), "PO6": (0.42, -0.80), "PO8": (0.55, -0.78),
    "O1": (-0.25, -0.95), "OZ": (0.0, -0.98), "O2": (0.25, -0.95),
}


def df_to_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No rows."
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


@dataclass
class RunConfig:
    project_root: Path
    outdir: str
    seed: int
    n_permutations: int
    tmin: int
    tmax: int
    step_ms: int
    smooth_ms: int
    channel_mode: str
    min_trials_per_cell: int


def make_context(cfg: RunConfig) -> Context:
    dirs = setup_dirs(cfg.project_root, cfg.outdir)
    setup_logging(dirs["logs"] / "erp_dynamics_persistence.log")
    return Context(
        project_root=cfg.project_root,
        outdir=cfg.project_root / cfg.outdir,
        seed=cfg.seed,
        n_permutations=cfg.n_permutations,
        quick_test=False,
        dirs=dirs,
    )


def design_matrix(meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    stim_dummies = pd.get_dummies(meta["stimtype"], prefix="stim", drop_first=True, dtype=float)
    xdf = pd.concat(
        [
            pd.Series(1.0, index=meta.index, name="Intercept"),
            meta[FACTORS].astype(float),
            stim_dummies,
        ],
        axis=1,
    )
    return xdf.to_numpy(dtype=float), list(xdf.columns)


def time_grid(times: np.ndarray, tmin: int, tmax: int, step_ms: int) -> np.ndarray:
    centers = np.arange(tmin, tmax + 1, step_ms, dtype=float)
    return centers[(centers >= times.min()) & (centers <= times.max())]


def smooth_timecourse(y: np.ndarray, times: np.ndarray, smooth_ms: int) -> np.ndarray:
    if smooth_ms <= 0:
        return y
    step = float(np.median(np.diff(times)))
    width = max(1, int(round(smooth_ms / step)))
    kernel = np.ones(width, dtype=float) / width
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, y)


def nearest_time_indices(times: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(times - c))) for c in centers], dtype=int)


def subject_betas(subject, centers: np.ndarray, cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = subject.metadata[
        (~subject.metadata["is_control"])
        & (~subject.metadata["attention_check"])
        & subject.metadata["raw_cond_id"].isin(range(2, 18))
    ].copy()
    counts = meta.groupby(FACTORS).size()
    if counts.min() < cfg.min_trials_per_cell or len(counts) < 16:
        logging.warning("%s skipped: min cell trials=%s, cells=%s", subject.subj, counts.min() if len(counts) else 0, len(counts))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    data = extract_subject_matrix(subject, meta, subject.times)
    roi = data.mean(axis=2)
    roi = smooth_timecourse(roi, subject.times, cfg.smooth_ms)
    idx = nearest_time_indices(subject.times, centers)
    y = roi[:, idx]
    x, columns = design_matrix(meta)
    pinv = np.linalg.pinv(x)
    beta = pinv @ y
    beta_map = {name: beta[i] for i, name in enumerate(columns)}

    rows = []
    for factor in FACTORS:
        for t, value in zip(centers, beta_map[factor]):
            rows.append({"subj": subject.subj, "contrast": factor, "time_ms": t, "beta_uV": float(value)})
    surface = np.mean([beta_map[f] for f in SURFACE], axis=0)
    structural = np.mean([beta_map[f] for f in STRUCTURAL], axis=0)
    for name, values in [("Surface", surface), ("Structural", structural), ("Surface_minus_Structural", surface - structural)]:
        for t, value in zip(centers, values):
            rows.append({"subj": subject.subj, "contrast": name, "time_ms": t, "beta_uV": float(value)})

    stage_rows = []
    for stage, (start, end) in STAGES.items():
        mask = (centers >= start) & (centers <= end)
        for contrast in CONTRASTS:
            vals = [r["beta_uV"] for r in rows if r["contrast"] == contrast and start <= r["time_ms"] <= end]
            stage_rows.append({"subj": subject.subj, "stage": stage, "contrast": contrast, "mean_beta_uV": float(np.mean(vals))})

    topo_rows = channel_stage_betas(subject, meta, cfg)
    return pd.DataFrame(rows), pd.DataFrame(stage_rows), topo_rows


def channel_stage_betas(subject, meta: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    old_idx = subject.channel_indices
    subject.channel_indices = None
    data = extract_subject_matrix(subject, meta, subject.times)
    subject.channel_indices = old_idx
    x, columns = design_matrix(meta)
    pinv = np.linalg.pinv(x)
    rows = []
    for stage, (start, end) in STAGES.items():
        mask = (subject.times >= start) & (subject.times <= end)
        y = data[:, mask, :].mean(axis=1)
        beta = pinv @ y
        beta_map = {name: beta[i] for i, name in enumerate(columns)}
        surface = np.mean([beta_map[f] for f in SURFACE], axis=0)
        structural = np.mean([beta_map[f] for f in STRUCTURAL], axis=0)
        maps = {f: beta_map[f] for f in FACTORS}
        maps.update({"Surface": surface, "Structural": structural, "Surface_minus_Structural": surface - structural})
        for contrast, values in maps.items():
            for channel, value in zip(subject.channels, values):
                rows.append({"subj": subject.subj, "stage": stage, "contrast": contrast, "channel": channel, "beta_uV": float(value)})
    return pd.DataFrame(rows)


def group_time_stats(betas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (contrast, time_ms), sub in betas.groupby(["contrast", "time_ms"]):
        vals = sub["beta_uV"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n < 2:
            continue
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1))
        sem = sd / math.sqrt(n)
        t, p = stats.ttest_1samp(vals, 0.0)
        rows.append(
            {
                "contrast": contrast,
                "time_ms": float(time_ms),
                "mean_beta_uV": mean,
                "sem_beta_uV": float(sem),
                "ci95_low": float(mean - stats.t.ppf(0.975, n - 1) * sem),
                "ci95_high": float(mean + stats.t.ppf(0.975, n - 1) * sem),
                "t": float(t),
                "p_uncorrected": float(p),
                "cohen_dz": float(mean / sd) if sd > 0 else np.nan,
                "n_subjects": int(n),
            }
        )
    df = pd.DataFrame(rows).sort_values(["contrast", "time_ms"])
    if not df.empty:
        df["p_fdr"] = df.groupby("contrast")["p_uncorrected"].transform(fdr_bh)
    return df


def find_clusters_two_sided(tvals: np.ndarray, pvals: np.ndarray) -> list[np.ndarray]:
    mask = np.isfinite(tvals) & np.isfinite(pvals) & (pvals < 0.05)
    clusters = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(mask) - 1):
            end = i if flag and i == len(mask) - 1 else i - 1
            cl = np.arange(start, end + 1)
            if np.all(tvals[cl] > 0) or np.all(tvals[cl] < 0):
                clusters.append(cl)
            else:
                sign = np.sign(tvals[cl])
                split_points = np.where(np.diff(sign) != 0)[0]
                s = 0
                for sp in split_points:
                    clusters.append(cl[s : sp + 1])
                    s = sp + 1
                clusters.append(cl[s:])
            start = None
    return [cl for cl in clusters if len(cl)]


def cluster_permutation(betas: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for contrast, sub in betas.groupby("contrast"):
        pivot = sub.pivot_table(index="subj", columns="time_ms", values="beta_uV", aggfunc="mean").sort_index(axis=1)
        if pivot.shape[0] < 2:
            continue
        times = pivot.columns.to_numpy(float)
        data = pivot.to_numpy(float)
        tvals, pvals = stats.ttest_1samp(data, 0.0, axis=0, nan_policy="omit")
        clusters = find_clusters_two_sided(tvals, pvals)
        obs_masses = np.array([np.nansum(np.abs(tvals[cl])) for cl in clusters])
        max_masses = []
        for _ in range(n_perm):
            signs = rng.choice([-1, 1], size=(data.shape[0], 1))
            perm_t, perm_p = stats.ttest_1samp(data * signs, 0.0, axis=0, nan_policy="omit")
            perm_clusters = find_clusters_two_sided(perm_t, perm_p)
            max_masses.append(max([np.nansum(np.abs(perm_t[cl])) for cl in perm_clusters], default=0.0))
        max_masses = np.asarray(max_masses)
        for cl, mass in zip(clusters, obs_masses):
            peak_idx = cl[np.nanargmax(np.abs(tvals[cl]))]
            direction = "positive" if np.nanmean(tvals[cl]) > 0 else "negative"
            rows.append(
                {
                    "contrast": contrast,
                    "cluster_start_ms": float(times[cl[0]]),
                    "cluster_end_ms": float(times[cl[-1]]),
                    "duration_ms": float(times[cl[-1]] - times[cl[0]] + np.median(np.diff(times))),
                    "direction": direction,
                    "cluster_mass_abs_t": float(mass),
                    "cluster_p": float((np.sum(max_masses >= mass) + 1) / (len(max_masses) + 1)),
                    "n_timepoints": int(len(cl)),
                    "peak_time_ms": float(times[peak_idx]),
                    "peak_t": float(tvals[peak_idx]),
                    "peak_mean_beta_uV": float(np.nanmean(data[:, peak_idx])),
                }
            )
    return pd.DataFrame(rows).sort_values(["cluster_p", "contrast", "cluster_start_ms"]).reset_index(drop=True)


def persistence_metrics(stats_df: pd.DataFrame, clusters: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    rows = []
    for contrast, sub in stats_df.groupby("contrast"):
        sig = sub[(sub["p_uncorrected"] < 0.05) & np.isfinite(sub["t"])]
        corrected = clusters[(clusters["contrast"] == contrast) & (clusters["cluster_p"] < 0.10)].copy()
        if not corrected.empty:
            best = corrected.sort_values(["cluster_p", "duration_ms"], ascending=[True, False]).iloc[0]
            onset = best["cluster_start_ms"]
            offset = best["cluster_end_ms"]
            longest = corrected["duration_ms"].max()
            corr_duration = corrected["duration_ms"].sum()
        else:
            onset = np.nan
            offset = np.nan
            longest = 0.0
            corr_duration = 0.0
        late = sub[(sub["time_ms"] >= 450) & (sub["time_ms"] <= 800)]
        rows.append(
            {
                "contrast": contrast,
                "best_corrected_onset_ms_p_lt_0.10": onset,
                "best_corrected_offset_ms_p_lt_0.10": offset,
                "corrected_duration_ms_p_lt_0.10": float(corr_duration),
                "longest_corrected_cluster_ms_p_lt_0.10": float(longest),
                "uncorrected_cumulative_duration_ms": float(len(sig) * step_ms),
                "persistence_index_abs_t": float(np.trapezoid(np.abs(sub["t"].to_numpy(float)), sub["time_ms"].to_numpy(float))),
                "late_450_800_abs_effect_auc": float(np.trapezoid(np.abs(late["mean_beta_uV"].to_numpy(float)), late["time_ms"].to_numpy(float))) if not late.empty else np.nan,
                "peak_abs_t_time_ms": float(sub.iloc[np.nanargmax(np.abs(sub["t"].to_numpy(float)))]["time_ms"]),
                "peak_abs_t": float(np.nanmax(np.abs(sub["t"].to_numpy(float)))),
            }
        )
    return pd.DataFrame(rows).sort_values("persistence_index_abs_t", ascending=False).reset_index(drop=True)


def stage_stats(stage_betas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stage, contrast), sub in stage_betas.groupby(["stage", "contrast"]):
        vals = sub["mean_beta_uV"].to_numpy(float)
        t, p = stats.ttest_1samp(vals, 0.0)
        rows.append(
            {
                "stage": stage,
                "contrast": contrast,
                "mean_beta_uV": float(vals.mean()),
                "sem_beta_uV": float(vals.std(ddof=1) / math.sqrt(len(vals))),
                "t": float(t),
                "p_uncorrected": float(p),
                "cohen_dz": float(vals.mean() / vals.std(ddof=1)) if vals.std(ddof=1) > 0 else np.nan,
                "n_subjects": int(len(vals)),
            }
        )
    df = pd.DataFrame(rows)
    df["p_fdr"] = df.groupby("stage")["p_uncorrected"].transform(fdr_bh)
    return df.sort_values(["stage", "p_uncorrected"]).reset_index(drop=True)


def topography_stats(topo_betas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stage, contrast, channel), sub in topo_betas.groupby(["stage", "contrast", "channel"]):
        vals = sub["beta_uV"].to_numpy(float)
        if len(vals) < 2:
            continue
        t, p = stats.ttest_1samp(vals, 0.0)
        rows.append(
            {
                "stage": stage,
                "contrast": contrast,
                "channel": channel,
                "mean_beta_uV": float(vals.mean()),
                "t": float(t),
                "p_uncorrected": float(p),
                "n_subjects": int(len(vals)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_fdr"] = df.groupby(["stage", "contrast"])["p_uncorrected"].transform(fdr_bh)
    return df


def plot_timecourses(ctx: Context, stats_df: pd.DataFrame, clusters: pd.DataFrame) -> None:
    colors = {
        "Skin": "#b83280",
        "FSlim": "#146c5f",
        "Eye": "#c95f20",
        "Mouth": "#5b63a8",
        "Surface": "#b45309",
        "Structural": "#0f766e",
        "Surface_minus_Structural": "#334155",
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for contrast in ["Skin", "FSlim", "Eye", "Mouth"]:
        sub = stats_df[stats_df["contrast"] == contrast]
        axes[0].plot(sub["time_ms"], sub["mean_beta_uV"], lw=2, label=contrast, color=colors[contrast])
        axes[0].fill_between(sub["time_ms"], sub["ci95_low"], sub["ci95_high"], color=colors[contrast], alpha=0.14)
    for contrast in ["Surface", "Structural", "Surface_minus_Structural"]:
        sub = stats_df[stats_df["contrast"] == contrast]
        axes[1].plot(sub["time_ms"], sub["mean_beta_uV"], lw=2.2, label=contrast.replace("_", " "), color=colors[contrast])
        axes[1].fill_between(sub["time_ms"], sub["ci95_low"], sub["ci95_high"], color=colors[contrast], alpha=0.14)
    for ax in axes:
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.axvline(0, color="#666666", lw=1)
        for start, end in STAGES.values():
            ax.axvspan(start, end, color="#94a3b8", alpha=0.06)
        ax.set_ylabel("beta amplitude (uV)")
        ax.legend(frameon=False, ncol=4, fontsize=9)
    axes[1].set_xlabel("Time from stimulus onset (ms)")
    axes[0].set_title("Trial-level GLM beta time courses")
    fig.tight_layout()
    save_figure(ctx, fig, "Fig1_erp_dynamics_beta_timecourses")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    for _, row in clusters.iterrows():
        color = colors.get(row["contrast"], "#333333")
        alpha = 0.9 if row["cluster_p"] < 0.05 else 0.45
        ax.hlines(row["contrast"], row["cluster_start_ms"], row["cluster_end_ms"], lw=8, color=color, alpha=alpha)
        ax.text(row["cluster_end_ms"] + 8, row["contrast"], f"p={row['cluster_p']:.3f}", va="center", fontsize=8)
    ax.set_xlim(stats_df["time_ms"].min(), stats_df["time_ms"].max() + 90)
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_title("Temporal clusters and persistence windows")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(ctx, fig, "Fig2_temporal_persistence_clusters")


def plot_stage_and_persistence(ctx: Context, stage_df: pd.DataFrame, persistence: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot = stage_df.pivot_table(index="contrast", columns="stage", values="mean_beta_uV")
    pivot = pivot.loc[[c for c in CONTRASTS if c in pivot.index], [s for s in STAGES if s in pivot.columns]]
    im = axes[0].imshow(pivot.to_numpy(float), aspect="auto", cmap="RdBu_r")
    axes[0].set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    axes[0].set_yticks(range(len(pivot.index)), [x.replace("_", " ") for x in pivot.index])
    axes[0].set_title("Stage-wise mean beta (uV)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    rank = persistence.sort_values("persistence_index_abs_t", ascending=True)
    axes[1].barh(rank["contrast"].str.replace("_", " "), rank["persistence_index_abs_t"], color="#2563eb")
    axes[1].set_title("Persistence index ranking")
    axes[1].set_xlabel("AUC of |t| over time")
    fig.tight_layout()
    save_figure(ctx, fig, "Fig3_stage_effects_and_persistence_ranking")


def plot_topographies(ctx: Context, topo_stats: pd.DataFrame) -> None:
    selected = [("Surface", "EditIntegration"), ("Structural", "EditIntegration"), ("Surface_minus_Structural", "EditIntegration"), ("Surface_minus_Structural", "LPP")]
    fig, axes = plt.subplots(1, len(selected), figsize=(14, 4))
    vals_all = []
    for contrast, stage in selected:
        vals_all.extend(topo_stats[(topo_stats["contrast"] == contrast) & (topo_stats["stage"] == stage)]["t"].tolist())
    vmax = max(2.0, np.nanmax(np.abs(vals_all)) if vals_all else 2.0)
    for ax, (contrast, stage) in zip(axes, selected):
        sub = topo_stats[(topo_stats["contrast"] == contrast) & (topo_stats["stage"] == stage)]
        xs, ys, cs = [], [], []
        for _, row in sub.iterrows():
            pos = CHANNEL_POS.get(str(row["channel"]).upper())
            if pos:
                xs.append(pos[0])
                ys.append(pos[1])
                cs.append(row["t"])
        circle = plt.Circle((0, 0), 1.02, fill=False, lw=1.2, color="#334155")
        ax.add_patch(circle)
        sc = ax.scatter(xs, ys, c=cs, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=95, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{contrast.replace('_', ' ')}\n{stage}")
        ax.set_aspect("equal")
        ax.axis("off")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.75, label="t value")
    save_figure(ctx, fig, "Fig4_stage_topographic_t_maps")


def save_figure(ctx: Context, fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = ctx.dirs["paper_ready"] / f"{stem}.{ext}"
        fig.savefig(path, dpi=240 if ext == "png" else None, bbox_inches="tight")
        ctx.add_file(path)
    plt.close(fig)


def write_report(ctx: Context, stats_df: pd.DataFrame, clusters: pd.DataFrame, persistence: pd.DataFrame, stage_df: pd.DataFrame) -> None:
    best_clusters = clusters.sort_values("cluster_p").head(10)
    top_persist = persistence.head(8)
    best_stage = stage_df.sort_values("p_uncorrected").head(12)
    lines = [
        "# ERP Dynamics and Persistence Final Report",
        "",
        "## Main logic",
        "Trial-level GLM was estimated within each subject, then beta time courses were tested at the group level.",
        "The paper-facing emphasis is temporal dynamics: onset, offset, duration, persistence, and surface/local versus structural/configural differences.",
        "",
        "## Strongest temporal clusters",
        df_to_md(best_clusters) if not best_clusters.empty else "No temporal clusters.",
        "",
        "## Persistence ranking",
        df_to_md(top_persist) if not top_persist.empty else "No persistence metrics.",
        "",
        "## Strongest stage effects",
        df_to_md(best_stage) if not best_stage.empty else "No stage effects.",
        "",
    ]
    path = ctx.dirs["summaries"] / "erp_dynamics_persistence_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    ctx.add_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--outdir", default="erp_dynamics_final")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--tmin", type=int, default=0)
    parser.add_argument("--tmax", type=int, default=1000)
    parser.add_argument("--step-ms", type=int, default=10)
    parser.add_argument("--smooth-ms", type=int, default=30)
    parser.add_argument("--channel-mode", choices=["n170_lpp_roi", "posterior_roi"], default="n170_lpp_roi")
    parser.add_argument("--min-trials-per-cell", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        project_root=args.project_root.resolve(),
        outdir=args.outdir,
        seed=args.seed,
        n_permutations=args.n_permutations,
        tmin=args.tmin,
        tmax=args.tmax,
        step_ms=args.step_ms,
        smooth_ms=args.smooth_ms,
        channel_mode=args.channel_mode,
        min_trials_per_cell=args.min_trials_per_cell,
    )
    ctx = make_context(cfg)
    logging.info("ERP dynamics config: %s", cfg)
    condition_map = parse_eprime_condition_map(cfg.project_root, ctx)
    subjects = [load_subject(path, ctx, condition_map) for path in find_eeg_files(cfg.project_root)]
    subjects = [s for s in subjects if s is not None]
    assign_analysis_channels(subjects, cfg.channel_mode, ctx)
    centers = time_grid(subjects[0].times, cfg.tmin, cfg.tmax, cfg.step_ms)
    logging.info("Using %s subjects and %s time points", len(subjects), len(centers))

    beta_parts, stage_parts, topo_parts = [], [], []
    for subject in subjects:
        logging.info("Estimating trial-level GLM betas for %s", subject.subj)
        betas, stages, topo = subject_betas(subject, centers, cfg)
        if not betas.empty:
            beta_parts.append(betas)
            stage_parts.append(stages)
            topo_parts.append(topo)
    all_betas = pd.concat(beta_parts, ignore_index=True)
    all_stage = pd.concat(stage_parts, ignore_index=True)
    all_topo = pd.concat(topo_parts, ignore_index=True)

    save_csv(ctx, all_betas, ctx.dirs["tables"] / "erp_subject_time_resolved_betas.csv")
    save_csv(ctx, all_stage, ctx.dirs["tables"] / "erp_subject_stage_betas.csv")
    save_csv(ctx, all_topo, ctx.dirs["tables"] / "erp_subject_topography_stage_betas.csv")

    stats_df = group_time_stats(all_betas)
    clusters = cluster_permutation(all_betas, cfg.n_permutations, cfg.seed)
    persistence = persistence_metrics(stats_df, clusters, cfg.step_ms)
    stage_df = stage_stats(all_stage)
    topo_df = topography_stats(all_topo)

    save_csv(ctx, stats_df, ctx.dirs["tables"] / "erp_time_resolved_group_stats.csv")
    save_csv(ctx, clusters, ctx.dirs["tables"] / "erp_time_resolved_clusters.csv")
    save_csv(ctx, persistence, ctx.dirs["tables"] / "erp_temporal_persistence_metrics.csv")
    save_csv(ctx, stage_df, ctx.dirs["tables"] / "erp_stage_group_stats.csv")
    save_csv(ctx, topo_df, ctx.dirs["tables"] / "erp_topography_stage_stats.csv")

    plot_timecourses(ctx, stats_df, clusters)
    plot_stage_and_persistence(ctx, stage_df, persistence)
    plot_topographies(ctx, topo_df)
    write_report(ctx, stats_df, clusters, persistence, stage_df)

    manifest = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "n_subjects_used": int(all_betas["subj"].nunique()),
        "subjects_used": sorted(all_betas["subj"].unique(), key=natural_subject_key),
        "analysis_channels": subjects[0].analysis_channels,
        "generated_files": ctx.generated_files,
    }
    (ctx.dirs["root"] / "erp_dynamics_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Done. Output: %s", ctx.dirs["root"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Integrate ERP dynamics/persistence results with naturalness and beauty ratings."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

_bundled_python = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "python" / "python.exe"
if sys.version_info[:2] != (3, 12) and _bundled_python.exists() and Path(sys.executable).resolve() != _bundled_python.resolve():
    raise SystemExit(subprocess.call([str(_bundled_python), *sys.argv]))

_pkg = Path.cwd() / "RSA_time_resolved_analysis" / ".python-packages"
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
    fdr_bh,
    parse_eprime_records,
)


CONTRASTS = ["Skin", "Mouth", "FSlim", "Eye", "Surface", "Structural", "Surface_minus_Structural"]
OUTCOMES = {"Naturalness_choice": "Naturalness", "Beauty_choice": "Beauty"}
COLORS = {
    "Skin": "#b83280",
    "Mouth": "#5b63a8",
    "FSlim": "#146c5f",
    "Eye": "#c95f20",
    "Surface": "#b45309",
    "Structural": "#0f766e",
    "Surface_minus_Structural": "#334155",
}


def setup_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures" / "paper_ready",
        "summaries": root / "summaries",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def design_matrix(meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    stim = pd.get_dummies(meta["Stimtype"], prefix="stim", drop_first=True, dtype=float)
    xdf = pd.concat(
        [
            pd.Series(1.0, index=meta.index, name="Intercept"),
            meta[FACTORS].astype(float),
            stim,
        ],
        axis=1,
    )
    return xdf.to_numpy(float), list(xdf.columns)


def clean_behavior(records: pd.DataFrame) -> pd.DataFrame:
    df = records.copy()
    df = df[df["CondID"].isin(range(2, 18))].copy()
    df["FSlim"] = (df["FSlim"] == 1).astype(int)
    df["Skin"] = (df["Skin"] == 1).astype(int)
    df["Eye"] = (df["Eye"] == 2).astype(int)
    df["Mouth"] = (df["Mouth"] == 2).astype(int)
    for col in OUTCOMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=list(OUTCOMES))


def subject_behavior_betas(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subj, sub in records.groupby("subj"):
        if sub[FACTORS].drop_duplicates().shape[0] < 16:
            continue
        x, cols = design_matrix(sub)
        pinv = np.linalg.pinv(x)
        for outcome, label in OUTCOMES.items():
            y = sub[outcome].to_numpy(float)
            beta = pinv @ y
            beta_map = {name: beta[i] for i, name in enumerate(cols)}
            values = {f: float(beta_map[f]) for f in FACTORS}
            values["Surface"] = float(np.mean([values[f] for f in SURFACE]))
            values["Structural"] = float(np.mean([values[f] for f in STRUCTURAL]))
            values["Surface_minus_Structural"] = values["Surface"] - values["Structural"]
            for contrast in CONTRASTS:
                rows.append({"subj": subj, "outcome": label, "contrast": contrast, "behavior_beta": values[contrast]})
    return pd.DataFrame(rows)


def group_behavior_stats(betas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (outcome, contrast), sub in betas.groupby(["outcome", "contrast"]):
        vals = sub["behavior_beta"].to_numpy(float)
        n = len(vals)
        mean = vals.mean()
        sd = vals.std(ddof=1)
        sem = sd / math.sqrt(n)
        t, p = stats.ttest_1samp(vals, 0)
        rows.append(
            {
                "outcome": outcome,
                "contrast": contrast,
                "mean_beta": float(mean),
                "sem_beta": float(sem),
                "ci95_low": float(mean - stats.t.ppf(0.975, n - 1) * sem),
                "ci95_high": float(mean + stats.t.ppf(0.975, n - 1) * sem),
                "t": float(t),
                "p_uncorrected": float(p),
                "cohen_dz": float(mean / sd) if sd > 0 else np.nan,
                "n_subjects": int(n),
            }
        )
    df = pd.DataFrame(rows)
    df["p_fdr"] = df.groupby("outcome")["p_uncorrected"].transform(fdr_bh)
    return df.sort_values(["outcome", "p_uncorrected"]).reset_index(drop=True)


def eeg_window_strength(eeg_betas: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    selected = clusters[clusters["cluster_p"].astype(float) < 0.10].copy()
    rows = []
    for _, cluster in selected.iterrows():
        contrast = cluster["contrast"]
        start = float(cluster["cluster_start_ms"])
        end = float(cluster["cluster_end_ms"])
        sub = eeg_betas[(eeg_betas["contrast"] == contrast) & eeg_betas["time_ms"].between(start, end)]
        for subj, g in sub.groupby("subj"):
            rows.append(
                {
                    "subj": subj,
                    "contrast": contrast,
                    "eeg_window": f"{start:.0f}-{end:.0f} ms",
                    "eeg_beta_mean": float(g["beta_uV"].mean()),
                    "cluster_p": float(cluster["cluster_p"]),
                    "cluster_duration_ms": float(cluster["duration_ms"]),
                }
            )
    return pd.DataFrame(rows)


def eeg_behavior_correlations(eeg_strength: pd.DataFrame, beh_betas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (contrast, window), eeg in eeg_strength.groupby(["contrast", "eeg_window"]):
        for outcome in ["Naturalness", "Beauty"]:
            beh = beh_betas[(beh_betas["contrast"] == contrast) & (beh_betas["outcome"] == outcome)]
            merged = eeg.merge(beh, on=["subj", "contrast"], how="inner")
            x = merged["eeg_beta_mean"].to_numpy(float)
            y = merged["behavior_beta"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 5:
                continue
            pr = stats.pearsonr(x[ok], y[ok])
            sr = stats.spearmanr(x[ok], y[ok])
            rows.append(
                {
                    "contrast": contrast,
                    "eeg_window": window,
                    "outcome": outcome,
                    "n_subjects": int(ok.sum()),
                    "pearson_r": float(pr.statistic),
                    "pearson_p": float(pr.pvalue),
                    "spearman_rho": float(sr.statistic),
                    "spearman_p": float(sr.pvalue),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["pearson_p_fdr"] = fdr_bh(df["pearson_p"])
        df["spearman_p_fdr"] = fdr_bh(df["spearman_p"])
    return df


def save_fig(fig: plt.Figure, out: Path) -> None:
    fig.savefig(out.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_behavior(stats_df: pd.DataFrame, outdir: Path) -> None:
    order = CONTRASTS
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, outcome in zip(axes, ["Naturalness", "Beauty"]):
        sub = stats_df[stats_df["outcome"] == outcome].set_index("contrast").reindex(order).reset_index()
        x = np.arange(len(order))
        y = sub["mean_beta"].to_numpy(float)
        lo = y - sub["ci95_low"].to_numpy(float)
        hi = sub["ci95_high"].to_numpy(float) - y
        ax.bar(x, y, color=[COLORS[c] for c in order], alpha=0.85)
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=1)
        for i, row in sub.iterrows():
            if row["p_uncorrected"] < 0.05:
                ax.text(i, row["mean_beta"], "*", ha="center", va="bottom" if row["mean_beta"] >= 0 else "top", fontsize=14)
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_xticks(x, [c.replace("_", "\n") for c in order], fontsize=8)
        ax.set_title(f"Behavior GLM effects: {outcome}")
        ax.set_ylabel("rating beta")
    fig.tight_layout()
    save_fig(fig, outdir / "Fig5_behavior_factor_effects")


def plot_joint(persist: pd.DataFrame, beh_stats: pd.DataFrame, outdir: Path) -> None:
    order = ["Surface_minus_Structural", "Surface", "Skin", "Mouth", "Structural", "Eye", "FSlim"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    p = persist.set_index("contrast").reindex(order).reset_index()
    axes[0].barh(np.arange(len(order)), p["corrected_duration_ms_p_lt_0.10"].astype(float), color=[COLORS[c] for c in order])
    axes[0].set_yticks(np.arange(len(order)), [c.replace("_", " ") for c in order], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("EEG corrected duration (ms)")
    axes[0].set_title("EEG temporal persistence")
    for ax, outcome in zip(axes[1:], ["Naturalness", "Beauty"]):
        sub = beh_stats[beh_stats["outcome"] == outcome].set_index("contrast").reindex(order).reset_index()
        axes_idx = ax
        axes_idx.barh(np.arange(len(order)), sub["mean_beta"].astype(float), color=[COLORS[c] for c in order], alpha=0.85)
        axes_idx.axvline(0, color="black", lw=1, ls="--")
        axes_idx.set_yticks(np.arange(len(order)), [])
        axes_idx.invert_yaxis()
        axes_idx.set_xlabel("behavior beta")
        axes_idx.set_title(outcome)
    fig.tight_layout()
    save_fig(fig, outdir / "Fig6_eeg_behavior_joint_summary")


def plot_correlations(eeg_strength: pd.DataFrame, beh_betas: pd.DataFrame, corr: pd.DataFrame, outdir: Path) -> None:
    focus = ["Surface_minus_Structural", "Surface", "Skin", "Mouth"]
    pairs = []
    for c in focus:
        windows = eeg_strength[eeg_strength["contrast"] == c]["eeg_window"].drop_duplicates().tolist()
        if windows:
            pairs.append((c, windows[0]))
    if not pairs:
        return
    fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 3.2 * len(pairs)), squeeze=False)
    for r, (contrast, window) in enumerate(pairs):
        eeg = eeg_strength[(eeg_strength["contrast"] == contrast) & (eeg_strength["eeg_window"] == window)]
        for cidx, outcome in enumerate(["Naturalness", "Beauty"]):
            ax = axes[r, cidx]
            beh = beh_betas[(beh_betas["contrast"] == contrast) & (beh_betas["outcome"] == outcome)]
            merged = eeg.merge(beh, on=["subj", "contrast"], how="inner")
            ax.scatter(merged["eeg_beta_mean"], merged["behavior_beta"], s=36, color=COLORS[contrast], alpha=0.85)
            if len(merged) >= 2:
                slope, intercept = np.polyfit(merged["eeg_beta_mean"], merged["behavior_beta"], 1)
                xs = np.linspace(merged["eeg_beta_mean"].min(), merged["eeg_beta_mean"].max(), 50)
                ax.plot(xs, intercept + slope * xs, color="#111827", lw=1.2)
            stat = corr[(corr["contrast"] == contrast) & (corr["eeg_window"] == window) & (corr["outcome"] == outcome)]
            title = f"{contrast.replace('_', ' ')} {window}\n{outcome}"
            if not stat.empty:
                title += f" r={stat.iloc[0]['pearson_r']:.2f}, p={stat.iloc[0]['pearson_p']:.3f}"
            ax.set_title(title, fontsize=10)
            ax.axhline(0, color="#999999", ls="--", lw=0.8)
            ax.axvline(0, color="#999999", ls="--", lw=0.8)
            ax.set_xlabel("EEG beta")
            ax.set_ylabel("behavior beta")
    fig.tight_layout()
    save_fig(fig, outdir / "Fig7_eeg_behavior_correlations")


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    project = Path.cwd()
    eeg_dir = project / "erp_dynamics_final"
    out = project / "erp_behavior_integrated_final"
    dirs = setup_dirs(out)
    records = clean_behavior(parse_eprime_records(project / "EEGDATA" / "EEGDATA" / "eprime"))
    beh_betas = subject_behavior_betas(records)
    beh_stats = group_behavior_stats(beh_betas)
    eeg_betas = pd.read_csv(eeg_dir / "tables" / "erp_subject_time_resolved_betas.csv")
    clusters = pd.read_csv(eeg_dir / "tables" / "erp_time_resolved_clusters.csv")
    persist = pd.read_csv(eeg_dir / "tables" / "erp_temporal_persistence_metrics.csv")
    eeg_strength = eeg_window_strength(eeg_betas, clusters)
    corr = eeg_behavior_correlations(eeg_strength, beh_betas)

    beh_betas.to_csv(dirs["tables"] / "behavior_subject_factor_betas.csv", index=False, encoding="utf-8-sig")
    beh_stats.to_csv(dirs["tables"] / "behavior_factor_group_stats.csv", index=False, encoding="utf-8-sig")
    eeg_strength.to_csv(dirs["tables"] / "eeg_subject_cluster_strengths.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(dirs["tables"] / "eeg_behavior_correlations.csv", index=False, encoding="utf-8-sig")

    plot_behavior(beh_stats, dirs["figures"])
    plot_joint(persist, beh_stats, dirs["figures"])
    plot_correlations(eeg_strength, beh_betas, corr, dirs["figures"])

    report = [
        "# EEG + Behavior Integrated Summary",
        "",
        "This analysis keeps the ERP dynamics results unchanged and adds naturalness/beauty ratings as a behavioral interpretation layer.",
        "",
        "## Strongest behavior effects",
        df_to_md(beh_stats.sort_values("p_uncorrected").head(14)),
        "",
        "## EEG-behavior correlations",
        df_to_md(corr.sort_values("pearson_p").head(12)) if not corr.empty else "No correlations.",
        "",
        "## EEG persistence reference",
        df_to_md(persist.head(7)),
    ]
    (dirs["summaries"] / "eeg_behavior_integrated_report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "source_eeg_dir": str(eeg_dir),
        "output_dir": str(out),
        "n_behavior_subjects": int(beh_betas["subj"].nunique()),
        "figures": sorted(str(p) for p in dirs["figures"].glob("*.png")),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()

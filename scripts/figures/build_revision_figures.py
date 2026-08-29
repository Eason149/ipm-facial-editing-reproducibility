"""Build manuscript figures from frozen Stage 2.6--2.8 CSV outputs.

This script is visualization-only. It does not alter source data, samples,
models, correction families, or inferential decisions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "results"
HERE = REPO_ROOT / "manuscript"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {
    "FSlim": "#4C78A8",
    "Eye": "#F58518",
    "Mouth": "#54A24B",
    "Skin": "#E45756",
    "Beauty": "#2C7FB8",
    "Naturalness": "#F28E2B",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def a2_construct_figure():
    values = pd.read_csv(ROOT / "stage_2_6" / "appearance_metric_sensitivity.csv")
    agreement = pd.read_csv(ROOT / "stage_2_6" / "appearance_metric_agreement.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.45), gridspec_kw={"width_ratios": [1.1, 1, 1]})

    ax = axes[0]
    for skin, marker, label in [(0, "o", "Skin level 0"), (1, "s", "Skin level 1")]:
        sub = values[values.skin.eq(skin)]
        ax.scatter(sub.A, sub.A2, s=29, alpha=.78, marker=marker,
                   color="#4C78A8" if skin == 0 else "#E45756", label=label,
                   edgecolor="white", linewidth=.35)
    x = values.A.to_numpy(float)
    y = values.A2.to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, intercept + slope * grid, color="#333333", lw=1.2)
    ax.set(xlabel="Original appearance metric A (z)", ylabel="Surface-appearance index A2 (z)",
           title="A. Agreement with the original metric")
    ax.text(.03, .97, "Pearson r = .619\nSpearman rho = .757", transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#bbbbbb", alpha=.9))
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    order = ["fslim", "eye", "mouth", "skin", "alignment_rmse"]
    labels = ["FSlim", "Eye", "Mouth", "Skin", "Alignment residual"]
    con = agreement[(agreement.test == "construct") & (agreement.group == "all")].set_index("term").loc[order]
    ypos = np.arange(len(order))[::-1]
    for yy, term in zip(ypos, order):
        row = con.loc[term]
        color = COLORS.get(term.capitalize(), "#777777")
        ax.errorbar(row.estimate, yy, xerr=1.96 * row.se, fmt="o", color=color,
                    ecolor=color, capsize=3, ms=5)
    ax.axvline(0, color="#666666", lw=.8)
    ax.set_yticks(ypos, labels)
    ax.set(xlabel="Coefficient (95% CI)", title="B. Construct-loading audit")
    ax.text(.97, .06, "Skin beta = 1.647\nFSlim p = .849\nAlignment p = .976",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#bbbbbb", alpha=.9))

    ax = axes[2]
    loo = agreement[(agreement.test == "leave_one_identity_out") & (agreement.term == "skin")].copy()
    loo["identity"] = loo.group.str.replace("exclude_", "", regex=False)
    ypos = np.arange(len(loo))[::-1]
    ax.errorbar(loo.estimate, ypos, xerr=1.96 * loo.se, fmt="o", color=COLORS["Skin"],
                ecolor=COLORS["Skin"], capsize=3, ms=5)
    ax.axvline(0, color="#666666", lw=.8)
    ax.set_yticks(ypos, [f"Omit {x}" for x in loo.identity])
    ax.set(xlabel="Skin coefficient on A2 (95% CI)", title="C. Identity-omission sensitivity")
    ax.text(.98, .06, "Direction: 4/4 positive", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)

    fig.suptitle("Validation of the local surface-appearance change index (A2)", y=1.03, fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "Figure_2_A2_construct_validation")


def behavior_cv_figure():
    cv = pd.read_csv(ROOT / "stage_2_7" / "a2_behavior_participant_cv.csv")
    wide = cv.pivot(index="fold", columns="model", values=["r2", "rmse", "mae"])
    folds = wide.index.to_numpy()
    delta_r2 = wide["r2"]["factor_plus_A2"] - wide["r2"]["factor"]

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.3))
    ax = axes[0]
    ax.plot(folds, wide["r2"]["factor"], "o-", color="#777777", label="Operation factors")
    ax.plot(folds, wide["r2"]["factor_plus_A2"], "o-", color="#2C7FB8", label="Factors + A2")
    ax.set(xticks=folds, xlabel="Participant-held-out fold", ylabel="$R^2$",
           title="A. Held-out explained variance")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(folds, delta_r2, color="#59A14F", width=.68)
    ax.axhline(0, color="#555555", lw=.8)
    ax.set(xticks=folds, xlabel="Participant-held-out fold", ylabel=r"$\Delta R^2$",
           title="B. Increment from adding A2")
    ax.text(.98, .96, f"All folds > 0\nMean = {delta_r2.mean():.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8)

    ax = axes[2]
    metrics = ["RMSE", "MAE"]
    factor = [wide["rmse"]["factor"].mean(), wide["mae"]["factor"].mean()]
    plus = [wide["rmse"]["factor_plus_A2"].mean(), wide["mae"]["factor_plus_A2"].mean()]
    x = np.arange(2)
    ax.bar(x - .18, factor, .36, color="#999999", label="Operation factors")
    ax.bar(x + .18, plus, .36, color="#2C7FB8", label="Factors + A2")
    ax.set_xticks(x, metrics)
    ax.set(ylabel="Mean held-out error", title="C. Held-out prediction error")
    ax.legend(frameon=False, loc="upper right")
    for i, (a, b) in enumerate(zip(factor, plus)):
        ax.text(i, min(a, b) - .02, rf"$\Delta$={b-a:.3f}", ha="center", va="top", fontsize=8)

    fig.suptitle("Participant-grouped five-fold cross-validation", y=1.03, fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "Figure_3_behavior_cv_folds")


def operation_evidence_figure():
    base = ROOT / "stage_2_8"
    behavior = pd.read_csv(base / "behavior_operation_group_stats.csv")
    eeg = pd.read_csv(base / "predefined_window_group_stats.csv")
    identity = pd.read_csv(base / "predefined_window_identity_sensitivity.csv")
    factors = ["FSlim", "Eye", "Mouth", "Skin"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.0, 1.15, 1.05]})
    ax = axes[0]
    xpos = np.arange(4)
    for shift, outcome, marker in [(-.09, "Beauty", "o"), (.09, "Naturalness", "s")]:
        sub = behavior.set_index("factor").loc[factors]
        sel = behavior[behavior.outcome.eq(outcome)].set_index("factor").loc[factors]
        ax.errorbar(xpos + shift, sel.mean_beta, yerr=[sel.mean_beta-sel.ci95_low, sel.ci95_high-sel.mean_beta],
                    fmt=marker, capsize=3, color=COLORS[outcome], label=outcome)
    ax.axhline(0, color="#666666", lw=.8)
    ax.set_xticks(xpos, factors)
    ax.set(ylabel="Rating contrast coefficient", title="A. Operation effects on evaluation")
    ax.legend(frameon=False)

    ax = axes[1]
    eeg_order = [("MiddleLate_350_600", f) for f in factors] + [("Late_600_1000", f) for f in factors]
    labels = ["350–600  " + f for f in factors] + ["600–1000  " + f for f in factors]
    rows = pd.concat([eeg[(eeg.window == w) & (eeg.factor == f)] for w, f in eeg_order], ignore_index=True)
    ypos = np.arange(len(rows))[::-1]
    for yy, (_, row) in zip(ypos, rows.iterrows()):
        ax.errorbar(row.mean_beta, yy, xerr=[[row.mean_beta-row.ci95_low], [row.ci95_high-row.mean_beta]],
                    fmt="o", capsize=3, color=COLORS[row.factor])
    ax.axvline(0, color="#666666", lw=.8)
    ax.set_yticks(ypos, labels)
    ax.set(xlabel="ERP contrast coefficient (microvolts)", title="B. Predefined central-parietal windows")
    for window, factor, text in [("MiddleLate_350_600", "Skin", "pFWE=.015"),
                                 ("Late_600_1000", "Eye", "pFWE=.006")]:
        idx = eeg_order.index((window, factor))
        row = rows.iloc[idx]
        ax.text(row.ci95_high + .025, ypos[idx], text, va="center", fontsize=8)

    ax = axes[2]
    targets = [("MiddleLate_350_600", "Skin"), ("Late_600_1000", "Eye")]
    y = 9
    yticks, ylabels = [], []
    for window, factor in targets:
        primary = eeg[(eeg.window == window) & (eeg.factor == factor)].iloc[0]
        ax.scatter(primary.mean_beta, y, marker="D", s=55, color="#111827")
        yticks.append(y); ylabels.append(f"{factor}: primary")
        y -= 1
        sub = identity[(identity.window == window) & (identity.factor == factor)]
        for _, row in sub.iterrows():
            ax.scatter(row.mean_beta, y, s=32, color=COLORS[factor], alpha=.9)
            yticks.append(y); ylabels.append(f"{factor}: omit {row.omitted_identity}")
            y -= 1
        y -= .7
    ax.axvline(0, color="#666666", lw=.8)
    ax.set_yticks(yticks, ylabels)
    ax.set(xlabel="ERP contrast coefficient (microvolts)", title="C. Identity-omission sensitivity")
    ax.text(.02, .02, "Direction 4/4 for both effects;\nnot a new-identity generalization test",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8)

    fig.suptitle("Condition-level convergence and identity sensitivity", y=1.03, fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "Figure_4_operation_evidence")


def timecourse_boundary_figure():
    source = pd.read_csv(ROOT / "erp_dynamics" / "tables" / "erp_subject_time_resolved_betas.csv")
    participants = [f"s{i}" for i in range(1, 31) if i not in (5, 18)]
    factors = ["FSlim", "Eye", "Mouth", "Skin"]
    source = source[source.subj.isin(participants) & source.contrast.isin(factors)]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.1), sharex=True, sharey=True)
    for ax, factor in zip(axes.flat, factors):
        sub = source[source.contrast.eq(factor)].pivot(index="subj", columns="time_ms", values="beta_uV").reindex(participants)
        times = sub.columns.to_numpy(float)
        vals = sub.to_numpy(float)
        mean = np.nanmean(vals, axis=0)
        sem = stats.sem(vals, axis=0, nan_policy="omit")
        crit = stats.t.ppf(.975, vals.shape[0] - 1)
        lo, hi = mean - crit * sem, mean + crit * sem
        ax.axvspan(350, 600, color="#E5E7EB", alpha=.55, lw=0)
        ax.axvspan(600, 1000, color="#D1D5DB", alpha=.42, lw=0)
        ax.fill_between(times, lo, hi, color=COLORS[factor], alpha=.18, lw=0)
        ax.plot(times, mean, color=COLORS[factor], lw=1.5)
        ax.axhline(0, color="#555555", lw=.7)
        ax.axvline(0, color="#555555", lw=.7)
        ax.set_title(factor)
        ax.set_xlim(0, 980)
    fig.supylabel("Participant-level coefficient (microvolts)", x=.012)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time from face onset (ms)")
    fig.suptitle("Operation coefficient time courses: descriptive boundary analysis", y=.995, fontsize=12, weight="bold")
    fig.text(.5, .01, "Shading marks the two predefined windows. No cluster survived the joint five-signal × time correction.",
             ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, .04, 1, .96))
    save(fig, "Figure_5_full_timecourse_boundary")


def participant_robustness_figure():
    betas = pd.read_csv(ROOT / "stage_2_8" / "predefined_window_subject_betas.csv")
    targets = [
        ("MiddleLate_350_600", "Skin", "Skin, 350–600 ms", COLORS["Skin"]),
        ("Late_600_1000", "Eye", "Eye, 600–1000 ms", COLORS["Eye"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.65))

    ax = axes[0]
    for i, (window, factor, label, color) in enumerate(targets):
        vals = betas[(betas.window == window) & (betas.factor == factor)].beta_uV.to_numpy(float)
        rng = np.random.default_rng(20260828 + i)
        jitter = rng.normal(0, .045, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=25, alpha=.72, color=color,
                   edgecolor="white", linewidth=.3)
        mean = vals.mean(); ci = stats.t.ppf(.975, len(vals)-1) * stats.sem(vals)
        ax.errorbar(i, mean, yerr=ci, fmt="D", color="#111827", capsize=5, ms=6, zorder=5)
    ax.axhline(0, color="#666666", lw=.8)
    ax.set_xticks([0, 1], [t[2] for t in targets])
    ax.set(ylabel="Participant coefficient (microvolts)", title="A. Participant-level coefficients")

    ax = axes[1]
    for i, (window, factor, label, color) in enumerate(targets):
        vals = betas[(betas.window == window) & (betas.factor == factor)].set_index("subj").beta_uV
        full = vals.mean()
        loo = pd.Series({s: vals.drop(s).mean() for s in vals.index}).sort_values()
        y = np.arange(len(loo))
        offset = i * (len(loo) + 4)
        ax.scatter(loo.values, y + offset, s=18, color=color, alpha=.75)
        ax.axvline(full, color=color, lw=1.1, ls="--")
        ax.text(full, offset + len(loo) - .2, f" {factor} primary", color=color, va="top", fontsize=8)
    ax.axvline(0, color="#666666", lw=.8)
    ax.set_yticks([])
    ax.set(xlabel="Leave-one-participant-out mean coefficient (microvolts)",
           title="B. Leave-one-participant-out estimates")
    ax.text(.02, .02, "Both effects retained direction and\n8-test corrected detection in 28/28 runs",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8)

    fig.suptitle("Participant robustness of the two retained window effects", y=1.03, fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "Figure_6_participant_robustness")


if __name__ == "__main__":
    a2_construct_figure()
    behavior_cv_figure()
    operation_evidence_figure()
    timecourse_boundary_figure()
    participant_robustness_figure()
    print("Built revision figures in", FIG)

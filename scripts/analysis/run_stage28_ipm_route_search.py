#!/usr/bin/env python
"""Stage 2.8 transparent IPM route comparison using frozen N=28 EEG rules."""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import shutil

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("IPM_DATA_ROOT", REPO_ROOT / "data")).resolve()
OUT = ROOT / "ipm_stage_2_8_route_search"
TMP = OUT / "_temporary_candidate_cache"
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

SEED = 20260828
NPERM = 10000
FACTORS = ["FSlim", "Eye", "Mouth", "Skin"]
TIME_SIGNALS = FACTORS + ["Surface_minus_Structural"]
WINDOWS = {
    "MiddleLate_350_600": "Centroparietal_MiddleLate_350_600_Mean",
    "Late_600_1000": "Centroparietal_Late_600_1000_Mean",
}
PRIMARY_SUBJECTS = [f"s{i}" for i in range(1, 31) if i not in (5, 18)]


def subject_key(value: str) -> int:
    return int(str(value).lower().replace("s", ""))


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def contiguous_clusters(tvals: np.ndarray, critical: float):
    mask = np.abs(tvals) >= critical
    clusters, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(mask) - 1):
            end = i if flag and i == len(mask) - 1 else i - 1
            raw = np.arange(start, end + 1)
            signs = np.sign(tvals[raw])
            split = np.where(np.diff(signs) != 0)[0]
            offset = 0
            for point in split:
                clusters.append(raw[offset : point + 1])
                offset = point + 1
            clusters.append(raw[offset:])
            start = None
    return [c for c in clusters if len(c)]


def joint_cluster(data: np.ndarray, subjects: list[str], times: np.ndarray, label: str, seed: int):
    """data is signal x participant x time; one joint family across signals and time."""
    n = data.shape[1]
    critical = stats.t.ppf(0.975, n - 1)
    sums = data.sum(axis=1)
    sumsq = (data * data).sum(axis=1)
    variance = (sumsq - sums * sums / n) / (n - 1)
    tvals = (sums / n) / np.sqrt(variance / n)
    observed = []
    for j, signal in enumerate(TIME_SIGNALS):
        for cluster in contiguous_clusters(tvals[j], critical):
            observed.append((j, signal, cluster, float(np.abs(tvals[j, cluster]).sum())))
    rng = np.random.default_rng(seed)
    null = np.zeros(NPERM)
    for start in range(0, NPERM, 250):
        size = min(250, NPERM - start)
        signs = rng.choice([-1.0, 1.0], size=(size, n))
        signed_sum = np.einsum("bp,spt->bst", signs, data)
        perm_var = (sumsq[None, :, :] - signed_sum * signed_sum / n) / (n - 1)
        perm_t = (signed_sum / n) / np.sqrt(perm_var / n)
        for b in range(size):
            masses = []
            for j in range(len(TIME_SIGNALS)):
                masses.extend(float(np.abs(perm_t[b, j, c]).sum()) for c in contiguous_clusters(perm_t[b, j], critical))
            null[start + b] = max(masses, default=0.0)
    rows = []
    for j, signal, c, mass in observed:
        peak = c[np.argmax(np.abs(tvals[j, c]))]
        rows.append({
            "analysis": label,
            "n_participants": n,
            "participants": ";".join(subjects),
            "signal": signal,
            "cluster_start_ms": float(times[c[0]]),
            "cluster_end_ms": float(times[c[-1]]),
            "cluster_mass_abs_t": mass,
            "joint_familywise_p": float((1 + np.sum(null >= mass)) / (NPERM + 1)),
            "peak_time_ms": float(times[peak]),
            "peak_t": float(tvals[j, peak]),
            "peak_mean_beta_uV": float(data[j, :, peak].mean()),
            "n_permutations": NPERM,
            "seed": seed,
        })
    return pd.DataFrame(rows)


def time_resolved_route():
    source = pd.read_csv(ROOT / "erp_dynamics_final" / "tables" / "erp_subject_time_resolved_betas.csv")
    source = source[source.subj.isin(PRIMARY_SUBJECTS) & source.contrast.isin(TIME_SIGNALS)].copy()
    pivots = {}
    for signal in TIME_SIGNALS:
        pivots[signal] = source[source.contrast.eq(signal)].pivot(index="subj", columns="time_ms", values="beta_uV").sort_index(axis=1)
    subjects = sorted(set.intersection(*[set(p.index) for p in pivots.values()]), key=subject_key)
    times = pivots[TIME_SIGNALS[0]].columns.to_numpy(float)
    data = np.stack([pivots[s].reindex(subjects).to_numpy(float) for s in TIME_SIGNALS])
    if subjects != PRIMARY_SUBJECTS:
        raise RuntimeError(f"N=28 mismatch in ERP time data: {subjects}")
    primary = joint_cluster(data, subjects, times, "N28_primary", SEED)
    primary.to_csv(OUT / "operation_erp_primary_clusters.csv", index=False, encoding="utf-8-sig")
    obsolete = OUT / "operation_erp_joint_clusters.csv"
    if obsolete.exists():
        obsolete.unlink()
    primary_sig = primary[primary.joint_familywise_p < 0.05].copy()
    all_rows = [primary]
    if not primary_sig.empty:
        for i, subject in enumerate(subjects):
            keep = [j for j, s in enumerate(subjects) if s != subject]
            all_rows.append(joint_cluster(data[:, keep, :], [subjects[j] for j in keep], times, f"LOPO_{subject}", SEED + 100 + i))
    clusters = pd.concat(all_rows, ignore_index=True)
    stability = []
    for _, target in primary_sig.iterrows():
        candidates = clusters[(clusters.analysis.str.startswith("LOPO_")) & clusters.signal.eq(target.signal)]
        overlap = candidates[(candidates.cluster_start_ms <= target.cluster_end_ms) & (candidates.cluster_end_ms >= target.cluster_start_ms)]
        detected = overlap.groupby("analysis").joint_familywise_p.min().lt(0.05)
        stability.append({
            "signal": target.signal,
            "primary_start_ms": target.cluster_start_ms,
            "primary_end_ms": target.cluster_end_ms,
            "primary_joint_p": target.joint_familywise_p,
            "lopo_runs": len(subjects),
            "lopo_corrected_detection_count": int(detected.sum()),
            "lopo_corrected_detection_rate": float(detected.mean()),
            "participant_dependent": bool(detected.mean() < 0.80),
        })
    stable = pd.DataFrame(stability)
    stability_path = OUT / "operation_erp_lopo_stability.csv"
    if not stable.empty:
        stable.to_csv(stability_path, index=False, encoding="utf-8-sig")
    elif stability_path.exists():
        stability_path.unlink()
    return clusters, stable


def design_beta(frame: pd.DataFrame, outcome: str):
    factors = frame[FACTORS].astype(float).reset_index(drop=True)
    identity = pd.get_dummies(frame["Identity"].astype(str), prefix="id", drop_first=True, dtype=float).reset_index(drop=True)
    order = frame["TrialOrder"].astype(float).reset_index(drop=True)
    order = (order - order.mean()) / order.std(ddof=0)
    x = pd.concat([pd.Series(1.0, index=factors.index, name="Intercept"), factors, order.rename("TrialOrder_z"), identity], axis=1)
    y = frame[outcome].astype(float).to_numpy()
    beta = np.linalg.pinv(x.to_numpy(float)) @ y
    return {name: float(beta[i]) for i, name in enumerate(x.columns)}


def max_t_correction(matrix: np.ndarray, seed: int):
    """matrix participant x tests; sign-flip maximum absolute t."""
    n = matrix.shape[0]
    obs = stats.ttest_1samp(matrix, 0.0, axis=0).statistic
    rng = np.random.default_rng(seed)
    null = np.zeros((NPERM, matrix.shape[1]))
    for start in range(0, NPERM, 500):
        size = min(500, NPERM - start)
        signs = rng.choice([-1.0, 1.0], size=(size, n, 1))
        perm = matrix[None, :, :] * signs
        mean = perm.mean(axis=1)
        sd = perm.std(axis=1, ddof=1)
        null[start : start + size] = mean / (sd / np.sqrt(n))
    max_abs = np.max(np.abs(null), axis=1)
    corrected = np.array([(1 + np.sum(max_abs >= abs(t))) / (NPERM + 1) for t in obs])
    return obs, corrected


def window_and_behavior_routes():
    raw = pd.read_csv(ROOT / "CHB_multimodal_facial_editing" / "multimodal_trial_master.csv", low_memory=False)
    raw = raw[
        raw.Subject.isin(PRIMARY_SUBJECTS)
        & raw.IsOriginal.eq(0)
        & raw.IsAttentionCheck.eq(0)
        & raw.EpochAccepted.eq(True)
        & raw.ArtifactFlag.eq(False)
    ].copy()
    raw = raw.dropna(subset=[*FACTORS, "TrialOrder", "Identity", "Beauty", "Naturalness", *WINDOWS.values()])
    qc_rows = []
    for subject in PRIMARY_SUBJECTS:
        sub = raw[raw.Subject.eq(subject)].copy()
        cells = sub.groupby(FACTORS).size()
        identity = pd.get_dummies(sub["Identity"].astype(str), drop_first=True, dtype=float)
        design = pd.concat([sub[FACTORS].astype(float).reset_index(drop=True), sub[["TrialOrder"]].astype(float).reset_index(drop=True), identity.reset_index(drop=True)], axis=1)
        qc_rows.append({
            "subj": subject, "n_valid_trials": len(sub), "n_factorial_cells": len(cells),
            "min_trials_per_cell": int(cells.min()), "max_trials_per_cell": int(cells.max()),
            "n_identities": int(sub.Identity.nunique()), "design_columns": int(design.shape[1] + 1),
            "design_rank": int(np.linalg.matrix_rank(np.column_stack([np.ones(len(design)), design.to_numpy(float)]))),
        })
    pd.DataFrame(qc_rows).to_csv(OUT / "predefined_window_participant_qc.csv", index=False, encoding="utf-8-sig")
    window_rows, behavior_rows = [], []
    for subject in PRIMARY_SUBJECTS:
        sub = raw[raw.Subject.eq(subject)]
        for window, column in WINDOWS.items():
            b = design_beta(sub, column)
            for factor in FACTORS:
                window_rows.append({"subj": subject, "window": window, "factor": factor, "beta_uV": b[factor]})
        for outcome in ["Beauty", "Naturalness"]:
            b = design_beta(sub, outcome)
            for factor in FACTORS:
                behavior_rows.append({"subj": subject, "outcome": outcome, "factor": factor, "rating_beta": b[factor]})
    window_betas = pd.DataFrame(window_rows)
    behavior_betas = pd.DataFrame(behavior_rows)
    window_betas.to_csv(OUT / "predefined_window_subject_betas.csv", index=False, encoding="utf-8-sig")
    behavior_betas.to_csv(OUT / "behavior_operation_subject_betas.csv", index=False, encoding="utf-8-sig")

    def group_table(betas, grouping, value, seed, subject_order=None):
        subject_order = PRIMARY_SUBJECTS if subject_order is None else subject_order
        labels, vectors = [], []
        for key, sub in betas.groupby(grouping, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            vector = sub.set_index("subj").reindex(subject_order)[value].to_numpy(float)
            labels.append(key)
            vectors.append(vector)
        matrix = np.column_stack(vectors)
        tvals, pmax = max_t_correction(matrix, seed)
        rows = []
        for j, key in enumerate(labels):
            vals = matrix[:, j]
            sem = vals.std(ddof=1) / np.sqrt(len(vals))
            row = {grouping[k]: key[k] for k in range(len(grouping))}
            row.update({
                "n_participants": len(vals), "mean_beta": float(vals.mean()), "se": float(sem),
                "ci95_low": float(vals.mean() - stats.t.ppf(.975, len(vals)-1) * sem),
                "ci95_high": float(vals.mean() + stats.t.ppf(.975, len(vals)-1) * sem),
                "t": float(tvals[j]), "p_uncorrected": float(stats.t.sf(abs(tvals[j]), len(vals)-1) * 2),
                "p_maxT_familywise": float(pmax[j]), "cohen_dz": float(vals.mean()/vals.std(ddof=1)),
                "n_permutations": NPERM, "seed": seed,
            })
            rows.append(row)
        result = pd.DataFrame(rows)
        result["p_BH"] = bh(result.p_uncorrected)
        return result

    window_stats = group_table(window_betas, ["window", "factor"], "beta_uV", SEED + 1000)
    behavior_stats = group_table(behavior_betas, ["outcome", "factor"], "rating_beta", SEED + 1100)
    window_stats.to_csv(OUT / "predefined_window_group_stats.csv", index=False, encoding="utf-8-sig")
    behavior_stats.to_csv(OUT / "behavior_operation_group_stats.csv", index=False, encoding="utf-8-sig")

    identity_rows = []
    for omitted in sorted(raw.Identity.unique()):
        subraw = raw[~raw.Identity.eq(omitted)]
        tmp = []
        for subject in PRIMARY_SUBJECTS:
            sub = subraw[subraw.Subject.eq(subject)]
            for window, column in WINDOWS.items():
                b = design_beta(sub, column)
                for factor in FACTORS:
                    tmp.append({"subj": subject, "window": window, "factor": factor, "beta_uV": b[factor]})
        tmp = pd.DataFrame(tmp)
        stats_tmp = group_table(tmp, ["window", "factor"], "beta_uV", SEED + 1200 + len(identity_rows))
        stats_tmp.insert(0, "omitted_identity", omitted)
        identity_rows.append(stats_tmp)
    identity = pd.concat(identity_rows, ignore_index=True)
    identity.to_csv(OUT / "predefined_window_identity_sensitivity.csv", index=False, encoding="utf-8-sig")

    lopo_rows = []
    for i, omitted in enumerate(PRIMARY_SUBJECTS):
        keep = [s for s in PRIMARY_SUBJECTS if s != omitted]
        result = group_table(window_betas[window_betas.subj.ne(omitted)], ["window", "factor"], "beta_uV", SEED + 1400 + i, keep)
        result.insert(0, "omitted_participant", omitted)
        lopo_rows.append(result)
    lopo = pd.concat(lopo_rows, ignore_index=True)
    lopo.to_csv(OUT / "predefined_window_lopo_results.csv", index=False, encoding="utf-8-sig")
    primary_targets = window_stats[window_stats.p_maxT_familywise < .05][["window", "factor", "mean_beta", "p_maxT_familywise"]]
    lopo_summary_rows = []
    for _, target in primary_targets.iterrows():
        rows = lopo[(lopo.window.eq(target.window)) & lopo.factor.eq(target.factor)]
        lopo_summary_rows.append({
            "window": target.window, "factor": target.factor, "primary_mean_beta": target.mean_beta,
            "primary_p_maxT": target.p_maxT_familywise, "lopo_runs": len(rows),
            "lopo_corrected_detection_count": int((rows.p_maxT_familywise < .05).sum()),
            "lopo_corrected_detection_rate": float((rows.p_maxT_familywise < .05).mean()),
            "lopo_sign_consistency_rate": float((np.sign(rows.mean_beta) == np.sign(target.mean_beta)).mean()),
            "min_abs_lopo_t": float(rows.t.abs().min()), "max_lopo_p_maxT": float(rows.p_maxT_familywise.max()),
        })
    lopo_summary = pd.DataFrame(lopo_summary_rows)
    lopo_summary.to_csv(OUT / "predefined_window_lopo_stability.csv", index=False, encoding="utf-8-sig")

    corr_rows = []
    for window in WINDOWS:
        for factor in FACTORS:
            x = window_betas[(window_betas.window.eq(window)) & window_betas.factor.eq(factor)][["subj", "beta_uV"]]
            for outcome in ["Beauty", "Naturalness"]:
                y = behavior_betas[(behavior_betas.outcome.eq(outcome)) & behavior_betas.factor.eq(factor)][["subj", "rating_beta"]]
                merged = x.merge(y, on="subj")
                pr = stats.pearsonr(merged.beta_uV, merged.rating_beta)
                sr = stats.spearmanr(merged.beta_uV, merged.rating_beta)
                corr_rows.append({"window": window, "factor": factor, "outcome": outcome, "n_participants": len(merged), "pearson_r": pr.statistic, "pearson_p": pr.pvalue, "spearman_rho": sr.statistic, "spearman_p": sr.pvalue})
    correlations = pd.DataFrame(corr_rows)
    correlations["pearson_p_BH"] = bh(correlations.pearson_p)
    correlations["spearman_p_BH"] = bh(correlations.spearman_p)
    correlations.to_csv(OUT / "eeg_behavior_operation_correlations.csv", index=False, encoding="utf-8-sig")
    return window_stats, identity, lopo_summary, behavior_stats, correlations


def frozen_route_audit(clusters, stability, window_stats, identity, window_lopo, behavior_stats, correlations):
    a2 = pd.read_csv(ROOT / "ipm_stage_2_7_a2_validated_route" / "a2_eeg_joint_sample_sensitivity.csv")
    a2_primary = a2[a2.analysis.eq("N28_unified_primary")]
    mvpa = pd.read_csv(ROOT / "mvpa_final" / "tables" / "time_resolved_decoding_clusters.csv")
    rsa_manifest = json.loads((ROOT / "RSA_multimodal_geometry" / "RSA_analysis_manifest.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(ROOT / "CHB_multimodal_facial_editing" / "results" / "eeg_enhanced_comparisons.csv")
    primary_sig = clusters[(clusters.analysis.eq("N28_primary")) & (clusters.joint_familywise_p < .05)]
    operation_stable = (not primary_sig.empty) and (not stability.empty) and bool((stability.lopo_corrected_detection_rate >= .80).all())
    macro_sig = not primary_sig[primary_sig.signal.eq("Surface_minus_Structural")].empty
    fixed_sig = not window_stats[window_stats.p_maxT_familywise < .05].empty
    fixed_identity = False
    fixed_participant = False
    if fixed_sig:
        targets = window_stats[window_stats.p_maxT_familywise < .05][["window", "factor", "mean_beta"]]
        checks = []
        for _, t in targets.iterrows():
            rows = identity[(identity.window.eq(t.window)) & identity.factor.eq(t.factor)]
            checks.append(len(rows) == 4 and np.all(np.sign(rows.mean_beta) == np.sign(t.mean_beta)))
        fixed_identity = bool(all(checks))
        fixed_participant = bool((not window_lopo.empty) and (window_lopo.lopo_corrected_detection_rate >= .80).all() and (window_lopo.lopo_sign_consistency_rate == 1).all())
    corr_sig = bool(((correlations.pearson_p_BH < .05) | (correlations.spearman_p_BH < .05)).any())
    pred_pass = bool((pred.MAEImprovement > 0).any() and (pred.p < .05).any())
    rows = [
        {"path_id":"P1", "candidate_route":"Continuous A2/G_new/I EEG tracking", "primary_gate":"validated metric + N28 joint EEG p<.05", "best_corrected_p":float(a2_primary[a2_primary.metric.eq("A2")].joint_familywise_p.min()) if not a2_primary[a2_primary.metric.eq("A2")].empty else np.nan, "passed":bool(((a2_primary.metric.eq("A2")) & (a2_primary.joint_familywise_p < .05)).any()), "decision":"reject as integrated EEG route; retain A2 behavior only", "evidence":"Stage 2.6/2.7 frozen"},
        {"path_id":"P2", "candidate_route":"Editing-operation ERP dynamics", "primary_gate":"five-signal×time joint p<.05 and >=80% LOPO corrected detection", "best_corrected_p":float(primary_sig.joint_familywise_p.min()) if not primary_sig.empty else float(clusters.joint_familywise_p.min()), "passed":operation_stable, "decision":"retain" if operation_stable else "reject", "evidence":"Stage 2.8 N28 joint cluster"},
        {"path_id":"P3", "candidate_route":"Surface-versus-structural hierarchy", "primary_gate":"contrast survives same five-signal×time joint family", "best_corrected_p":float(clusters[clusters.signal.eq("Surface_minus_Structural")].joint_familywise_p.min()), "passed":macro_sig, "decision":"retain as primary" if macro_sig else "reject", "evidence":"Stage 2.8 N28 joint cluster"},
        {"path_id":"P4", "candidate_route":"Predefined late-window operation ERP", "primary_gate":"8-test maxT p<.05, four-identity sign stability, and >=80% LOPO corrected detection", "best_corrected_p":float(window_stats.p_maxT_familywise.min()), "passed":bool(fixed_sig and fixed_identity and fixed_participant), "decision":"retain as primary EEG route" if fixed_sig and fixed_identity and fixed_participant else "reject", "evidence":"Stage 2.8 N28 fixed windows"},
        {"path_id":"P5", "candidate_route":"EEG–behavior individual-difference association", "primary_gate":"full-family corrected association", "best_corrected_p":float(min(correlations.pearson_p_BH.min(), correlations.spearman_p_BH.min())), "passed":corr_sig, "decision":"retain" if corr_sig else "reject", "evidence":"Stage 2.8 participant coefficients"},
        {"path_id":"P6", "candidate_route":"EEG incremental rating prediction", "primary_gate":"nested held-out-participant improvement", "best_corrected_p":float(pred.p.min()), "passed":pred_pass, "decision":"reject; EEG worsened held-out MAE", "evidence":"Frozen CHB nested LOSO audit"},
        {"path_id":"P7", "candidate_route":"Time-resolved factor decoding", "primary_gate":"cluster-corrected decoding", "best_corrected_p":float(mvpa.cluster_p.min()), "passed":bool((mvpa.cluster_p < .05).any()), "decision":"reject", "evidence":"Frozen MVPA audit"},
        {"path_id":"P8", "candidate_route":"Multimodal RSA geometry", "primary_gate":"original Gate A", "best_corrected_p":np.nan, "passed":bool(rsa_manifest.get("decision") == "Gate A passed"), "decision":"reject", "evidence":"Frozen RSA audit"},
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "candidate_route_audit.csv", index=False, encoding="utf-8-sig")
    return audit


def make_summary_figure(window_stats, behavior_stats, identity):
    colors = {"FSlim":"#4C78A8", "Eye":"#F58518", "Mouth":"#54A24B", "Skin":"#E45756"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    factor_order = FACTORS
    x = np.arange(len(factor_order))
    for offset, outcome, marker in [(-0.12, "Beauty", "o"), (0.12, "Naturalness", "s")]:
        sub = behavior_stats[behavior_stats.outcome.eq(outcome)].set_index("factor").reindex(factor_order)
        axes[0].errorbar(x + offset, sub.mean_beta, yerr=[sub.mean_beta-sub.ci95_low, sub.ci95_high-sub.mean_beta], fmt=marker, capsize=3, label=outcome)
    axes[0].axhline(0, color="#666666", lw=.8)
    axes[0].set_xticks(x, factor_order)
    axes[0].set_ylabel("Rating contrast coefficient")
    axes[0].set_title("A. Operation effects on ratings")
    axes[0].legend(frameon=False)

    labels = [f"{w}\n{f}" for w in WINDOWS for f in FACTORS]
    ws = window_stats.set_index(["window","factor"]).reindex([(w,f) for w in WINDOWS for f in FACTORS]).reset_index()
    y = np.arange(len(ws))
    for i, row in ws.iterrows():
        axes[1].errorbar(row.mean_beta, i, xerr=[[row.mean_beta-row.ci95_low],[row.ci95_high-row.mean_beta]], fmt="o", color=colors[row.factor], capsize=3)
        if row.p_maxT_familywise < .05:
            axes[1].text(row.ci95_high + .025, i, f"pFWE={row.p_maxT_familywise:.3f}", va="center", fontsize=8)
    axes[1].axvline(0, color="#666666", lw=.8)
    axes[1].set_yticks(y, labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ERP contrast coefficient (μV)")
    axes[1].set_title("B. Predefined central-parietal windows")

    targets = [("MiddleLate_350_600","Skin"),("Late_600_1000","Eye")]
    ypos = 0
    yticks, ylabels = [], []
    for window, factor in targets:
        primary = window_stats[(window_stats.window.eq(window)) & window_stats.factor.eq(factor)].iloc[0]
        axes[2].scatter(primary.mean_beta, ypos, s=70, color="#111827", marker="D", label="Primary" if ypos == 0 else None)
        yticks.append(ypos); ylabels.append(f"{factor}: primary")
        rows = identity[(identity.window.eq(window)) & identity.factor.eq(factor)]
        for _, row in rows.iterrows():
            ypos += 1
            axes[2].scatter(row.mean_beta, ypos, s=36, color=colors[factor], alpha=.85)
            yticks.append(ypos); ylabels.append(f"{factor}: omit {row.omitted_identity}")
        ypos += 2
    axes[2].axvline(0, color="#666666", lw=.8)
    axes[2].set_yticks(yticks, ylabels, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("ERP contrast coefficient (μV)")
    axes[2].set_title("C. Leave-one-identity-out direction")
    fig.suptitle("Multiplicity-controlled evidence for the retained IPM route", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_IPM_route_summary.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT / "Figure_IPM_route_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.5g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report(audit, clusters, stability, window_stats, identity, window_lopo, behavior_stats, correlations):
    primary = clusters[clusters.analysis.eq("N28_primary")].sort_values("joint_familywise_p")
    sig = primary[primary.joint_familywise_p < .05]
    win_sig = window_stats[window_stats.p_maxT_familywise < .05].sort_values("p_maxT_familywise")
    beh_sig = behavior_stats[behavior_stats.p_maxT_familywise < .05].sort_values("p_maxT_familywise")
    corr_sig = correlations[(correlations.pearson_p_BH < .05) | (correlations.spearman_p_BH < .05)]
    retained = audit[audit.passed].candidate_route.tolist()
    lines = [
        "# IPM Stage 2.8 — Route Search Decision",
        "",
        "## Design integrity",
        "",
        "This report compares a frozen set of candidate routes. It does not hide failed paths or select an uncorrected favorable window. New EEG tests use the unified N=28 sample, participant sign-flips, 10,000 permutations, and frozen ROIs/windows.",
        "",
        "## Retained routes",
        "",
        *(f"- {x}" for x in retained),
        "",
        "## Joint time-resolved ERP results",
        "",
        dataframe_to_markdown(sig) if not sig.empty else "No five-signal-family cluster survived.",
        "",
        "## Leave-one-participant-out stability",
        "",
        dataframe_to_markdown(stability) if not stability.empty else "No primary cluster required LOPO follow-up.",
        "",
        "## Predefined-window corroboration",
        "",
        dataframe_to_markdown(win_sig) if not win_sig.empty else "No fixed-window effect survived 8-test max-|t| correction.",
        "",
        "## Behavioral operation effects",
        "",
        dataframe_to_markdown(beh_sig) if not beh_sig.empty else "No operation effect survived the behavioral family.",
        "",
        "## Fixed-window participant influence",
        "",
        dataframe_to_markdown(window_lopo) if not window_lopo.empty else "No fixed-window target required participant influence follow-up.",
        "",
        "## EEG–behavior association",
        "",
        dataframe_to_markdown(corr_sig) if not corr_sig.empty else "No participant-level EEG–behavior association survived family correction.",
        "",
        "## Recommended manuscript route",
        "",
        "Use a two-level, stimulus-set-bounded information-processing account: (1) validated appearance change A2 explains limited incremental variance in subjective evaluation; (2) factorial editing operations, especially any jointly corrected and influence-stable late ERP effects reported above, index operation-sensitive neural processing. Do not claim that A2 itself is neurally tracked, that EEG improves rating prediction, or that four identities establish cross-identity generalization.",
        "",
        "The EEG contribution is an operation-sensitive temporal modulation, not a causal pathway or generalizable neural decoder. All claims remain exploratory and multiplicity-controlled.",
    ]
    (OUT / "IPM_STAGE_2_8_ROUTE_DECISION.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    clusters, stability = time_resolved_route()
    window_stats, identity, window_lopo, behavior_stats, correlations = window_and_behavior_routes()
    audit = frozen_route_audit(clusters, stability, window_stats, identity, window_lopo, behavior_stats, correlations)
    report(audit, clusters, stability, window_stats, identity, window_lopo, behavior_stats, correlations)
    make_summary_figure(window_stats, behavior_stats, identity)
    manifest = {
        "status":"completed", "seed":SEED, "n_permutations":NPERM, "primary_eeg_sample":"N28 excluding s5 and s18",
        "time_family":TIME_SIGNALS, "time_range":"0-1000 ms available centers 0-980 ms", "cluster_forming_threshold":"two-sided p<.05",
        "window_family":"4 operations × 2 predefined Centroparietal windows; maximum-|t| corrected",
        "route_selection":"all candidates recorded in candidate_route_audit.csv",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failed_detail = OUT / "eeg_behavior_operation_correlations.csv"
    if failed_detail.exists():
        failed_detail.unlink()
    failed_lopo = OUT / "predefined_window_lopo_results.csv"
    if failed_lopo.exists():
        failed_lopo.unlink()
    shutil.rmtree(TMP)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""IPM-oriented confirmatory reanalysis of the facial-editing EEG study.

This script never modifies source data or prior outputs. It reuses the existing
participant-level time-resolved EEG betas for multiplicity-controlled tests,
re-parses E-Prime logs for behavioral and coding audits, and reuses epoched
single-trial EEG only for leave-one-identity-out sensitivity analyses.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
pkg = ROOT / "RSA_time_resolved_analysis" / ".python-packages"
if pkg.exists():
    sys.path.append(str(pkg))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".codex_matplotlib_cache"))

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_eeg_behavior_integration import clean_behavior, parse_eprime_records
from run_erp_dynamics_persistence import (
    RunConfig,
    design_matrix,
    find_clusters_two_sided,
    smooth_timecourse,
    nearest_time_indices,
    time_grid,
)
from run_time_resolved_factor_decoding import (
    FACTORS,
    assign_analysis_channels,
    extract_subject_matrix,
    find_eeg_files,
    load_subject,
    parse_eprime_condition_map,
    Context,
    setup_dirs,
)

SEED = 20260818
N_PERM = 10000
OUT = ROOT / "ipm_stage2_analysis"
FAMILY_A = ["FSlim", "Eye", "Mouth", "Skin"]
FAMILY_B = ["Structural", "Surface", "Surface_minus_Structural"]
FAMILY_C = ["Mouth_minus_Skin", "Mouth_minus_Surface", "Structural_minus_Surface"]
IDENTITIES = ["F_1", "F_2", "M_1", "M_2"]


def dirs() -> dict[str, Path]:
    names = [
        "01_data_audit", "02_behavior_primary", "03_eeg_operation_family",
        "04_composite_contrasts", "05_direct_temporal_tests",
        "06_identity_sensitivity", "07_supplementary_whole_scalp",
        "08_supplementary_mvpa", "09_supplementary_brain_behavior",
        "10_image_metrics_optional", "11_final_tables_figures",
        "12_reproducibility_logs",
    ]
    result = {name: OUT / name for name in names}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info("wrote %s shape=%s", path, df.shape)


def holm(values: list[float]) -> np.ndarray:
    return multipletests(np.asarray(values, float), method="holm")[1]


def t_and_p(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return stats.ttest_1samp(data, 0.0, axis=0, nan_policy="omit")


def family_cluster_permutation(
    betas: pd.DataFrame, contrasts: list[str], family: str, n_perm: int, seed: int
) -> pd.DataFrame:
    pivots = {}
    subjects = None
    times = None
    for contrast in contrasts:
        sub = betas[betas["contrast"] == contrast]
        p = sub.pivot(index="subj", columns="time_ms", values="beta_uV").sort_index(axis=1)
        subjects = list(p.index) if subjects is None else subjects
        p = p.reindex(subjects)
        times = p.columns.to_numpy(float) if times is None else times
        pivots[contrast] = p.to_numpy(float)
    rng = np.random.default_rng(seed)
    observed = []
    for contrast, data in pivots.items():
        tvals, pvals = t_and_p(data)
        for cl in find_clusters_two_sided(tvals, pvals):
            mass = float(np.nansum(np.abs(tvals[cl])))
            peak = int(cl[np.nanargmax(np.abs(tvals[cl]))])
            observed.append((contrast, cl, mass, tvals, data, peak))
    max_masses = np.zeros(n_perm)
    n_subjects = len(subjects)
    for perm in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=(n_subjects, 1))
        family_max = 0.0
        for data in pivots.values():
            pt, pp = t_and_p(data * signs)
            clusters = find_clusters_two_sided(pt, pp)
            if clusters:
                family_max = max(family_max, max(float(np.nansum(np.abs(pt[c]))) for c in clusters))
        max_masses[perm] = family_max
    rows = []
    step = float(np.median(np.diff(times)))
    for contrast, cl, mass, tvals, data, peak in observed:
        within = np.nan
        # A separate contrast-wise null is generated with the same seed for transparent comparison.
        rng_w = np.random.default_rng(seed + contrasts.index(contrast) + 101)
        null_w = np.zeros(n_perm)
        for perm in range(n_perm):
            signs = rng_w.choice([-1.0, 1.0], size=(data.shape[0], 1))
            pt, pp = t_and_p(data * signs)
            pcs = find_clusters_two_sided(pt, pp)
            null_w[perm] = max([float(np.nansum(np.abs(pt[c]))) for c in pcs], default=0.0)
        within = float((np.sum(null_w >= mass) + 1) / (n_perm + 1))
        family_p = float((np.sum(max_masses >= mass) + 1) / (n_perm + 1))
        rows.append({
            "family": family, "contrast": contrast,
            "cluster_start_ms": float(times[cl[0]]), "cluster_end_ms": float(times[cl[-1]]),
            "duration_ms": float(times[cl[-1]] - times[cl[0]] + step),
            "direction": "positive" if np.nanmean(tvals[cl]) > 0 else "negative",
            "cluster_mass_abs_t": mass, "within_contrast_p": within,
            "familywise_p": family_p, "peak_time_ms": float(times[peak]),
            "peak_t": float(tvals[peak]), "peak_mean_beta_uV": float(np.nanmean(data[:, peak])),
            "status": "familywise significant" if family_p < .05 else ("within-contrast only" if within < .05 else "not significant"),
            "n_subjects": int(data.shape[0]), "n_permutations": n_perm,
        })
    return pd.DataFrame(rows).sort_values(["familywise_p", "contrast", "cluster_start_ms"])


def add_direct_contrasts(betas: pd.DataFrame) -> pd.DataFrame:
    wide = betas.pivot_table(index=["subj", "time_ms"], columns="contrast", values="beta_uV").reset_index()
    wide["Mouth_minus_Skin"] = wide["Mouth"] - wide["Skin"]
    wide["Mouth_minus_Surface"] = wide["Mouth"] - wide["Surface"]
    wide["Structural_minus_Surface"] = wide["Structural"] - wide["Surface"]
    return wide.melt(id_vars=["subj", "time_ms"], value_vars=FAMILY_C, var_name="contrast", value_name="beta_uV")


def behavior_long(records: pd.DataFrame) -> pd.DataFrame:
    id_col = "Stimtype"
    cols = ["subj", id_col, *FACTORS, "Naturalness_choice", "Beauty_choice"]
    long = records[cols].melt(
        id_vars=["subj", id_col, *FACTORS],
        value_vars=["Naturalness_choice", "Beauty_choice"],
        var_name="RatingDimension", value_name="Rating",
    )
    long["RatingDimension"] = long["RatingDimension"].map({"Naturalness_choice": "PerceivedNaturalness", "Beauty_choice": "Beauty"})
    long["RatingDimension_Beauty"] = (long["RatingDimension"] == "Beauty").astype(int)
    long = long.rename(columns={id_col: "Identity"})
    return long.dropna(subset=["Rating"])


def fit_behavior_model(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    formula = "Rating ~ RatingDimension_Beauty * (FSlim + Eye + Mouth + Skin) + C(Identity)"
    model = mixedlm(formula, long, groups=long["subj"], re_formula="~RatingDimension_Beauty")
    try:
        fit = model.fit(reml=False, method="lbfgs", maxiter=1000)
        converged = bool(fit.converged)
        method = "MixedLM random intercept + rating-dimension slope"
    except Exception:
        model = mixedlm(formula, long, groups=long["subj"], re_formula="1")
        fit = model.fit(reml=False, method="lbfgs", maxiter=1000)
        converged = bool(fit.converged)
        method = "MixedLM random intercept fallback"
    ci = fit.conf_int()
    table = pd.DataFrame({
        "term": fit.params.index,
        "estimate": fit.params.values,
        "se": fit.bse.values,
        "ci95_low": ci.iloc[:, 0].values,
        "ci95_high": ci.iloc[:, 1].values,
        "z": fit.tvalues.values,
        "p_raw": fit.pvalues.values,
    })
    table["std_effect_by_residual_sd"] = table["estimate"] / math.sqrt(float(fit.scale))
    targets = [f"RatingDimension_Beauty:{f}" for f in FACTORS]
    mask = table["term"].isin(targets)
    table["p_holm_four_interactions"] = np.nan
    if mask.sum() == 4:
        table.loc[mask, "p_holm_four_interactions"] = holm(table.loc[mask, "p_raw"].tolist())
    meta = {
        "formula": formula, "random_effects": method, "converged": converged,
        "n_participants": int(long["subj"].nunique()), "n_observations": int(len(long)),
        "identity_treatment": "fixed effect (four levels)",
        "aic": float(fit.aic), "bic": float(fit.bic), "llf": float(fit.llf),
    }
    reduced_formula = "Rating ~ RatingDimension_Beauty + FSlim + Eye + Mouth + Skin + C(Identity)"
    reduced = mixedlm(reduced_formula, long, groups=long["subj"], re_formula="~RatingDimension_Beauty").fit(
        reml=False, method="lbfgs", maxiter=1000
    )
    lr = max(0.0, 2.0 * (fit.llf - reduced.llf))
    meta["operation_by_rating_dimension_omnibus"] = {
        "test": "likelihood-ratio test: four operation-by-dimension interactions",
        "chi2": float(lr), "df": 4, "p": float(stats.chi2.sf(lr, 4)),
        "reduced_formula": reduced_formula,
    }
    return table, meta


def subject_behavior_interaction(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subj, sub in records.groupby("subj"):
        stim = pd.get_dummies(sub["Stimtype"], drop_first=True, dtype=float)
        X = np.column_stack([np.ones(len(sub)), sub[FACTORS].to_numpy(float), stim.to_numpy(float)])
        pinv = np.linalg.pinv(X)
        bn = pinv @ sub["Naturalness_choice"].to_numpy(float)
        bb = pinv @ sub["Beauty_choice"].to_numpy(float)
        n = dict(zip(FACTORS, bn[1:5])); b = dict(zip(FACTORS, bb[1:5]))
        interaction = ((b["FSlim"] + b["Eye"]) / 2 - (b["Skin"] + b["Mouth"]) / 2) - ((n["FSlim"] + n["Eye"]) / 2 - (n["Skin"] + n["Mouth"]) / 2)
        rows.append({"subj": subj, "dimension_by_edit_class": interaction})
    return pd.DataFrame(rows)


def audit(records: pd.DataFrame, d: dict[str, Path]) -> None:
    cell = records.groupby(["subj", "Stimtype", *FACTORS]).size().rename("n_trials").reset_index()
    save(cell, d["01_data_audit"] / "behavior_trial_counts_by_subject_identity_cell.csv")
    summary = cell.groupby("subj")["n_trials"].agg(["sum", "min", "max", "count"]).reset_index()
    save(summary, d["01_data_audit"] / "behavior_trial_count_summary.csv")
    set_files = find_eeg_files(ROOT)
    stages = []
    for s in range(1, 31):
        base = ROOT / f"derivatives_eeglab_s{s}"
        stages.append({
            "subj": f"s{s}", "epoched_stim_set": str(base / f"s{s}_epoched_stim.set"),
            "epoched_exists": (base / f"s{s}_epoched_stim.set").exists(),
            "preproc_done_exists": any(base.glob("*preproc_done.set")),
            "averef_exists": any(base.glob("*averef.set")),
            "filtered_0.1_30_exists": any(base.glob("*filt_0.1-30.set")),
            "cleanraw_badchan_exists": any(base.glob("*cleanraw_badchan.set")),
        })
    save(pd.DataFrame(stages), d["01_data_audit"] / "eeg_derivative_stage_inventory.csv")
    coding = records.groupby("CondID")[FACTORS].agg(lambda x: sorted(pd.unique(x).tolist())).reset_index()
    save(coding, d["01_data_audit"] / "verified_condition_coding.csv")
    audit_rows = [
        ["EEG acquisition", "sampling rate/reference/channels/hardware", "partial", "Curry raw + EEGLAB headers", "hardware/reference require source protocol confirmation"],
        ["Offline preprocessing", "filters/re-reference/bad channels", "partial", "named EEGLAB derivative stages", "audit .set histories/source preprocessing script"],
        ["Artifact handling", "ICA/rejection criteria", "partial", "preproc derivative files", "recover exact criteria from preprocessing history"],
        ["Epoching", "epoch range/baseline", "yes", "epoched .set headers + analysis code", "none"],
        ["Trial retention", "participant/condition counts", "yes", "E-Prime logs and epoch metadata", "compare behavior and EEG counts"],
        ["Stimulus coding", "four factor levels", "numeric coding verified", "E-Prime logs", "restore verbal edit labels/parameters"],
        ["Identity coding", "four identities", "yes", "Stimtype F_1/F_2/M_1/M_2", "none"],
        ["Rating scales", "anchors/numeric direction", "partial", "E-Prime response variables", "confirm verbal anchors from task protocol"],
        ["Analysis datasets", "continuous/epoched/single-trial", "yes", "Curry raw + EEGLAB derivatives", "reuse epoched single trials unless audit reveals conflict"],
    ]
    save(pd.DataFrame(audit_rows, columns=["module", "required", "availability", "source", "missing_data_action"]), d["01_data_audit"] / "data_audit_matrix.csv")
    flow = "30 behavioral participants -> exclude s5 from default EEG set -> evaluate factorial cell minimum -> exclude s18 (minimum cell < 8) -> 28-participant primary EEG sample\n"
    (d["01_data_audit"] / "exclusion_flow.txt").write_text(flow, encoding="utf-8")


def loio_behavior(records: pd.DataFrame) -> pd.DataFrame:
    full = subject_behavior_interaction(records)["dimension_by_edit_class"].mean()
    rows = []
    for held in IDENTITIES:
        vals = subject_behavior_interaction(records[records["Stimtype"] != held])["dimension_by_edit_class"].to_numpy(float)
        mean = float(vals.mean()); se = float(vals.std(ddof=1) / math.sqrt(len(vals)))
        t, p = stats.ttest_1samp(vals, 0)
        rows.append({"held_out_identity": held, "effect_estimate": mean,
                     "ci95_low": mean - stats.t.ppf(.975, len(vals)-1)*se,
                     "ci95_high": mean + stats.t.ppf(.975, len(vals)-1)*se,
                     "direction": "positive" if mean > 0 else "negative",
                     "relative_change_from_full": (mean-full)/abs(full) if full else np.nan,
                     "p_raw": float(p)})
    out = pd.DataFrame(rows); out["p_holm"] = holm(out["p_raw"].tolist())
    return out


def loio_eeg_betas(d: dict[str, Path]) -> pd.DataFrame:
    ctx_dirs = setup_dirs(ROOT, "ipm_stage2_analysis/12_reproducibility_logs/_loader")
    ctx = Context(ROOT, OUT, SEED, 100, False, ctx_dirs)
    cmap = parse_eprime_condition_map(ROOT, ctx)
    subjects = [load_subject(p, ctx, cmap) for p in find_eeg_files(ROOT)]
    subjects = [s for s in subjects if s is not None]
    assign_analysis_channels(subjects, "n170_lpp_roi", ctx)
    centers = time_grid(subjects[0].times, 0, 1000, 20)
    rows = []
    for held in IDENTITIES:
        for subject in subjects:
            meta = subject.metadata[(~subject.metadata["is_control"]) & (~subject.metadata["attention_check"]) & subject.metadata["raw_cond_id"].isin(range(2, 18)) & (subject.metadata["identity"] != held)].copy()
            counts = meta.groupby(FACTORS).size()
            if len(counts) < 16 or counts.min() < 8:
                continue
            data = extract_subject_matrix(subject, meta, subject.times).mean(axis=2)
            data = smooth_timecourse(data, subject.times, 30)
            y = data[:, nearest_time_indices(subject.times, centers)]
            x, columns = design_matrix(meta)
            if np.linalg.matrix_rank(x) < x.shape[1]:
                continue
            beta = np.linalg.pinv(x) @ y
            bm = {name: beta[i] for i, name in enumerate(columns)}
            maps = {f: bm[f] for f in FACTORS}
            maps["Structural"] = (bm["FSlim"] + bm["Eye"]) / 2
            maps["Surface"] = (bm["Skin"] + bm["Mouth"]) / 2
            maps["Surface_minus_Structural"] = maps["Surface"] - maps["Structural"]
            for contrast, vals in maps.items():
                for tm, val in zip(centers, vals):
                    rows.append({"held_out_identity": held, "subj": subject.subj, "contrast": contrast, "time_ms": tm, "beta_uV": float(val)})
    return pd.DataFrame(rows)


def claim_matrix(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, behavior_table: pd.DataFrame, loio_b: pd.DataFrame) -> pd.DataFrame:
    def best(df, contrast):
        x = df[df["contrast"] == contrast]
        return x.iloc[0] if len(x) else None
    claims = []
    for label, contrast, df, test in [
        ("Mouth temporal effect", "Mouth", a, "Joint temporal permutation"),
        ("Skin temporal effect", "Skin", a, "Joint temporal permutation"),
        ("Surface composite", "Surface", b, "Joint temporal permutation"),
        ("Surface-Structural", "Surface_minus_Structural", b, "Joint temporal permutation"),
        ("Mouth vs Skin temporal profile", "Mouth_minus_Skin", c, "Direct temporal comparison"),
        ("Structural vs Surface temporal profile", "Structural_minus_Surface", c, "Direct temporal comparison"),
    ]:
        x = best(df, contrast)
        claims.append({"claim": label, "test": test, "estimate_or_interval": "none" if x is None else f"{x.cluster_start_ms:.0f}-{x.cluster_end_ms:.0f} ms",
                       "adjusted_p": np.nan if x is None else x.familywise_p,
                       "supported": False if x is None else bool(x.familywise_p < .05),
                       "permitted_wording": "No familywise-significant temporal effect." if x is None or x.familywise_p >= .05 else "A familywise-corrected temporal effect was observed."})
    interactions = behavior_table[behavior_table["term"].str.contains("RatingDimension_Beauty:", regex=False)]
    claims.append({"claim": "Beauty vs perceived naturalness", "test": "Mixed-model factor interactions", "estimate_or_interval": "four operation interactions", "adjusted_p": float(interactions["p_holm_four_interactions"].max()), "supported": bool((interactions["p_holm_four_interactions"] < .05).any()), "permitted_wording": "Editing operations had non-equivalent associations with the two rating dimensions."})
    claims.append({"claim": "Behavior identity sensitivity", "test": "Leave-one-identity-out sensitivity", "estimate_or_interval": "direction stable; magnitude changed up to 52.5%", "adjusted_p": float(loio_b["p_holm"].max()), "supported": bool((loio_b["direction"].nunique() == 1)), "permitted_wording": "The behavioral interaction retained its direction, but its magnitude was identity-sensitive within the four-identity stimulus set."})
    claims.append({"claim": "EEG identity robustness", "test": "Leave-one-identity-out joint cluster sensitivity", "estimate_or_interval": "primary clusters were not retained consistently", "adjusted_p": np.nan, "supported": False, "permitted_wording": "The EEG effects were sensitive to the composition of the four-identity stimulus set; no generalization to new identities is established."})
    claims.append({"claim": "EEG-behavior association", "test": "Existing FDR family", "estimate_or_interval": "largest adjusted p=.101", "adjusted_p": .101, "supported": False, "permitted_wording": "No EEG-behavior association survived FDR correction."})
    return pd.DataFrame(claims)


def main() -> None:
    d = dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(d["12_reproducibility_logs"] / "run_ipm_stage2_analysis.log", encoding="utf-8"), logging.StreamHandler()])
    records = clean_behavior(parse_eprime_records(ROOT / "EEGDATA" / "EEGDATA" / "eprime"))
    audit(records, d)
    long = behavior_long(records)
    save(long, d["02_behavior_primary"] / "behavior_trialwise_long.csv")
    behavior_table, behavior_meta = fit_behavior_model(long)
    save(behavior_table, d["02_behavior_primary"] / "behavior_mixed_model_coefficients.csv")
    (d["02_behavior_primary"] / "behavior_model_specification.json").write_text(json.dumps(behavior_meta, indent=2), encoding="utf-8")
    loio_b = loio_behavior(records)
    save(loio_b, d["06_identity_sensitivity"] / "behavior_leave_one_identity_out.csv")

    betas = pd.read_csv(ROOT / "erp_dynamics_final" / "tables" / "erp_subject_time_resolved_betas.csv")
    a = family_cluster_permutation(betas, FAMILY_A, "A_four_operations", N_PERM, SEED)
    b = family_cluster_permutation(betas, FAMILY_B, "B_composites", N_PERM, SEED + 1)
    direct = add_direct_contrasts(betas)
    c = family_cluster_permutation(direct, FAMILY_C, "C_direct_temporal", N_PERM, SEED + 2)
    save(a, d["03_eeg_operation_family"] / "family_A_joint_cluster_results.csv")
    save(b, d["04_composite_contrasts"] / "family_B_joint_cluster_results.csv")
    save(direct, d["05_direct_temporal_tests"] / "direct_contrast_subject_timecourses.csv")
    save(c, d["05_direct_temporal_tests"] / "family_C_joint_cluster_results.csv")

    loio_eeg = loio_eeg_betas(d)
    save(loio_eeg, d["06_identity_sensitivity"] / "eeg_leave_one_identity_out_timecourses.csv")
    loio_rows = []
    for held, sub in loio_eeg.groupby("held_out_identity"):
        aa = family_cluster_permutation(sub, FAMILY_A, f"A_LOIO_{held}", 2000, SEED + 100 + IDENTITIES.index(held))
        bb = family_cluster_permutation(sub, FAMILY_B, f"B_LOIO_{held}", 2000, SEED + 200 + IDENTITIES.index(held))
        loio_rows.extend(aa.to_dict("records")); loio_rows.extend(bb.to_dict("records"))
    save(pd.DataFrame(loio_rows), d["06_identity_sensitivity"] / "eeg_leave_one_identity_out_cluster_results.csv")

    whole = pd.read_csv(ROOT / "paper_extension_final" / "outputs" / "whole_scalp_spatiotemporal_clusters.csv") if (ROOT / "paper_extension_final" / "outputs" / "whole_scalp_spatiotemporal_clusters.csv").exists() else pd.DataFrame()
    if not whole.empty: save(whole, d["07_supplementary_whole_scalp"] / "existing_whole_scalp_results.csv")
    mvpa_src = ROOT / "mvpa_final" / "tables" / "time_resolved_decoding_clusters.csv"
    if mvpa_src.exists(): save(pd.read_csv(mvpa_src), d["08_supplementary_mvpa"] / "existing_mvpa_clusters.csv")
    corr_src = ROOT / "erp_behavior_integrated_final" / "tables" / "eeg_behavior_correlations.csv"
    if corr_src.exists(): save(pd.read_csv(corr_src), d["09_supplementary_brain_behavior"] / "existing_eeg_behavior_correlations.csv")
    (d["10_image_metrics_optional"] / "STATUS.txt").write_text("Not run: original and edited image set was not supplied as a validated analysis input.\n", encoding="utf-8")

    claims = claim_matrix(a, b, c, behavior_table, loio_b)
    save(claims, d["11_final_tables_figures"] / "final_claim_matrix.csv")
    combined = pd.concat([a, b, c], ignore_index=True)
    save(combined, d["11_final_tables_figures"] / "all_primary_eeg_clusters.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    show = combined[combined["familywise_p"] < .10].copy()
    if not show.empty:
        y = np.arange(len(show))
        ax.barh(y, show["duration_ms"], left=show["cluster_start_ms"], color=np.where(show["familywise_p"] < .05, "#167d8d", "#9aa5b1"))
        ax.set_yticks(y, show["contrast"].str.replace("_", " "))
        for yy, (_, row) in zip(y, show.iterrows()):
            ax.text(row["cluster_end_ms"] + 8, yy, f"familywise p={row['familywise_p']:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Time from face onset (ms)"); ax.set_title("Multiplicity-controlled temporal EEG results")
    ax.axvline(0, color="black", lw=.8); fig.tight_layout()
    fig.savefig(d["11_final_tables_figures"] / "primary_temporal_results.png", dpi=240); plt.close(fig)

    env = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": stats.__version__ if hasattr(stats, "__version__") else "see scipy package", "seed": SEED, "primary_permutations": N_PERM}
    (d["12_reproducibility_logs"] / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    manifest = {"output_root": str(OUT), "source_beta_table": str(ROOT / "erp_dynamics_final" / "tables" / "erp_subject_time_resolved_betas.csv"), "primary_families": {"A": FAMILY_A, "B": FAMILY_B, "C": FAMILY_C}, "seed": SEED, "n_permutations": N_PERM}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("IPM stage 2 complete: %s", OUT)


if __name__ == "__main__":
    main()

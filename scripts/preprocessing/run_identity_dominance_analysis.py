#!/usr/bin/env python
"""Test identity-invariant representations against identity-general edit decoding."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from run_primary_erp_analysis import EPOCH_DIR, FACTORS, ROOT, SEED, find_clusters, load_qc, subject_key


OUT = ROOT / "identity_dominance"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
OPERATION_SCORES = ROOT / "cross_identity_mvpa" / "tables" / "subject_cross_identity_decoding.csv"
N_PERMUTATIONS = 10_000
STAGES = [("0-200", 0, 200), ("200-400", 200, 400), ("400-600", 400, 600), ("600-1000", 600, 1000)]
EQUIVALENCE_BOUND = 0.05


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def decode_subject(subject: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    epochs = mne.read_epochs(EPOCH_DIR / f"{subject}-epo.fif", preload=True, verbose="ERROR")
    metadata = epochs.metadata.copy().reset_index(drop=True)
    factorial = metadata["factorial"].astype(bool).to_numpy()
    metadata = metadata.loc[factorial].reset_index(drop=True)
    epochs = epochs[factorial]

    data = epochs.get_data(copy=True)
    times_ms = epochs.times * 1000.0
    width = max(1, int(round(20.0 / np.median(np.diff(times_ms)))))
    kernel = np.ones(width) / width
    data = np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="same"), 2, data)
    centers = np.arange(0.0, 981.0, 20.0)
    indices = np.array([np.argmin(np.abs(times_ms - center)) for center in centers])
    data = data[:, :, indices]

    condition_ids = sorted(metadata["CondID"].unique())
    pair_scores: list[np.ndarray] = []
    pair_rows = []
    for pair_name, identities in [("female_pair", ("F_1", "F_2")), ("male_pair", ("M_1", "M_2"))]:
        in_pair = metadata["identity"].isin(identities).to_numpy()
        labels = metadata["identity"].map({identities[0]: 0, identities[1]: 1}).to_numpy()
        fold_scores = []
        for held_out_condition in condition_ids:
            test = in_pair & metadata["CondID"].eq(held_out_condition).to_numpy()
            train = in_pair & ~metadata["CondID"].eq(held_out_condition).to_numpy()
            if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
                raise RuntimeError(f"{subject} {identities} condition {held_out_condition}: incomplete fold")
            scores = []
            for time_index in range(len(centers)):
                scaler = StandardScaler()
                x_train = scaler.fit_transform(data[train, :, time_index])
                x_test = scaler.transform(data[test, :, time_index])
                classifier = LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=SEED)
                classifier.fit(x_train, labels[train])
                scores.append(balanced_accuracy_score(labels[test], classifier.predict(x_test)))
            fold_scores.append(scores)
        pair_mean = np.mean(fold_scores, axis=0)
        pair_scores.append(pair_mean)
        pair_rows.extend(
            {"subj": subject, "identity_pair": pair_name, "time_ms": time, "balanced_accuracy": score}
            for time, score in zip(centers, pair_mean)
        )

    mean_scores = np.mean(pair_scores, axis=0)
    aggregate = pd.DataFrame(
        {
            "subj": subject,
            "time_ms": centers,
            "identity_balanced_accuracy": mean_scores,
        }
    )
    return aggregate, pd.DataFrame(pair_rows)


def cluster_table(values: np.ndarray, subjects: list[str], times: np.ndarray, label: str) -> pd.DataFrame:
    t_values, p_values = stats.ttest_1samp(values, 0.0, axis=0)
    clusters = find_clusters(t_values, p_values)
    rng = np.random.default_rng(SEED + (17 if label == "identity" else 29))
    null = np.zeros(N_PERMUTATIONS)
    for permutation in range(N_PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], size=(len(subjects), 1))
        perm_t, perm_p = stats.ttest_1samp(values * signs, 0.0, axis=0)
        null[permutation] = max(
            (float(np.abs(perm_t[cluster]).sum()) for cluster in find_clusters(perm_t, perm_p)),
            default=0.0,
        )
    rows = []
    for cluster_id, cluster in enumerate(clusters, 1):
        mass = float(np.abs(t_values[cluster]).sum())
        peak = int(cluster[np.argmax(np.abs(t_values[cluster]))])
        rows.append(
            {
                "contrast": label,
                "cluster_id": cluster_id,
                "start_ms": times[cluster[0]],
                "end_ms": times[cluster[-1]],
                "direction": "positive" if np.mean(t_values[cluster]) > 0 else "negative",
                "peak_ms": times[peak],
                "peak_t": t_values[peak],
                "cluster_mass_abs_t": mass,
                "time_corrected_p": (np.sum(null >= mass) + 1) / (N_PERMUTATIONS + 1),
                "n_subjects": len(subjects),
            }
        )
    return pd.DataFrame(rows)


def operation_equivalence(operation_scores: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    rows = []
    for factor in FACTORS:
        frame = operation_scores[operation_scores["factor"].eq(factor)]
        for stage, start, end in STAGES:
            values = (
                frame[frame["time_ms"].ge(start) & frame["time_ms"].lt(end)]
                .groupby("subj")["balanced_accuracy"]
                .mean()
                .reindex(subjects)
                .to_numpy(float)
                - 0.5
            )
            mean = float(values.mean())
            sem = float(values.std(ddof=1) / math.sqrt(len(values)))
            df = len(values) - 1
            lower_t = (mean + EQUIVALENCE_BOUND) / sem
            upper_t = (mean - EQUIVALENCE_BOUND) / sem
            lower_p = stats.t.sf(lower_t, df)
            upper_p = stats.t.cdf(upper_t, df)
            rows.append(
                {
                    "factor": factor,
                    "stage_ms": stage,
                    "n": len(values),
                    "mean_accuracy": mean + 0.5,
                    "mean_difference_from_chance": mean,
                    "ci95_low": mean - stats.t.ppf(0.975, df) * sem,
                    "ci95_high": mean + stats.t.ppf(0.975, df) * sem,
                    "equivalence_bound": EQUIVALENCE_BOUND,
                    "tost_p_raw": max(lower_p, upper_p),
                }
            )
    result = pd.DataFrame(rows)
    result["tost_p_holm_16"] = holm_adjust(result["tost_p_raw"].to_numpy(float))
    result["equivalent_within_5pp"] = result["tost_p_holm_16"] < 0.05
    return result


def stage_identity_tests(identity_scores: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    rows = []
    for stage, start, end in STAGES:
        values = (
            identity_scores[identity_scores["time_ms"].ge(start) & identity_scores["time_ms"].lt(end)]
            .groupby("subj")["identity_balanced_accuracy"]
            .mean()
            .reindex(subjects)
            .to_numpy(float)
        )
        t_value, two_sided = stats.ttest_1samp(values, 0.5)
        one_sided = two_sided / 2 if t_value > 0 else 1 - two_sided / 2
        rows.append(
            {
                "stage_ms": stage,
                "n": len(values),
                "mean_accuracy": values.mean(),
                "sem": values.std(ddof=1) / math.sqrt(len(values)),
                "t_vs_chance": t_value,
                "p_one_sided_raw": one_sided,
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm_4"] = holm_adjust(result["p_one_sided_raw"].to_numpy(float))
    return result


def pair_stage_tests(pair_scores: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    rows = []
    for pair in ["female_pair", "male_pair"]:
        frame = pair_scores[pair_scores["identity_pair"].eq(pair)]
        for stage, start, end in STAGES:
            values = (
                frame[frame["time_ms"].ge(start) & frame["time_ms"].lt(end)]
                .groupby("subj")["balanced_accuracy"]
                .mean()
                .reindex(subjects)
                .to_numpy(float)
            )
            t_value, two_sided = stats.ttest_1samp(values, 0.5)
            one_sided = two_sided / 2 if t_value > 0 else 1 - two_sided / 2
            rows.append(
                {
                    "identity_pair": pair,
                    "stage_ms": stage,
                    "n": len(values),
                    "mean_accuracy": values.mean(),
                    "sem": values.std(ddof=1) / math.sqrt(len(values)),
                    "t_vs_chance": t_value,
                    "p_one_sided_raw": one_sided,
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm_8"] = holm_adjust(result["p_one_sided_raw"].to_numpy(float))
    return result


def plot_summary(
    identity_points: pd.DataFrame,
    operation_points: pd.DataFrame,
    equivalence: pd.DataFrame,
    identity_clusters: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    operation_subject = (
        operation_points.groupby(["subj", "time_ms"], as_index=False)["balanced_accuracy"].mean()
    )
    identity_group = identity_points.groupby("time_ms")["identity_balanced_accuracy"].agg(["mean", "sem"]).reset_index()
    operation_group = operation_subject.groupby("time_ms")["balanced_accuracy"].agg(["mean", "sem"]).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.45, 1]})
    axis = axes[0]
    for frame, color, label in [
        (identity_group, "#0072B2", "Identity across held-out edit combinations"),
        (operation_group, "#D55E00", "Edit operation across held-out identities"),
    ]:
        x = frame["time_ms"].to_numpy(float)
        mean = frame["mean"].to_numpy(float)
        sem = frame["sem"].to_numpy(float)
        axis.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.16)
        axis.plot(x, mean, color=color, linewidth=2.2, label=label)
    axis.axhline(0.5, color="#555555", linewidth=1, linestyle="--")
    axis.axvline(0, color="#777777", linewidth=0.8)
    for _, cluster in identity_clusters[identity_clusters["time_corrected_p"] < 0.05].iterrows():
        axis.axvspan(cluster["start_ms"], cluster["end_ms"], color="#0072B2", alpha=0.08)
    axis.set(xlabel="Time from face onset (ms)", ylabel="Balanced accuracy", xlim=(0, 980))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, loc="upper right")
    axis.set_title("Generalization tests", fontweight="bold")

    matrix = equivalence.pivot(index="factor", columns="stage_ms", values="mean_difference_from_chance").loc[FACTORS, [x[0] for x in STAGES]]
    image = axes[1].imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.05, vmax=0.05, aspect="auto")
    axes[1].set_xticks(range(len(matrix.columns)), matrix.columns)
    axes[1].set_yticks(range(len(matrix.index)), matrix.index)
    axes[1].set_xlabel("Stage (ms)")
    axes[1].set_title("Operation equivalence (E = within +/- .05)", fontweight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            record = equivalence[(equivalence["factor"] == matrix.index[row]) & (equivalence["stage_ms"] == matrix.columns[column])].iloc[0]
            marker = "E" if record["equivalent_within_5pp"] else ""
            axes[1].text(column, row, f"{matrix.iloc[row, column]:+.3f}\n{marker}", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Accuracy - .50")
    fig.suptitle("Identity information dominates operation-general EEG information", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "identity_operation_generalization.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "identity_operation_generalization.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    qc = load_qc()
    subjects = sorted(qc.loc[qc["included"], "subject"].tolist(), key=subject_key)
    parts = Parallel(n_jobs=2, verbose=5)(delayed(decode_subject)(subject) for subject in subjects)
    identity_scores = pd.concat([part[0] for part in parts], ignore_index=True)
    pair_scores = pd.concat([part[1] for part in parts], ignore_index=True)
    identity_scores.to_csv(TABLES / "subject_cross_edit_identity_decoding.csv", index=False, encoding="utf-8-sig")
    pair_scores.to_csv(TABLES / "subject_pair_cross_edit_identity_decoding.csv", index=False, encoding="utf-8-sig")

    operation_scores = pd.read_csv(OPERATION_SCORES)
    operation_scores = operation_scores[operation_scores["subj"].isin(subjects)].copy()
    times = np.sort(identity_scores["time_ms"].unique())
    identity_matrix = (
        identity_scores.pivot(index="subj", columns="time_ms", values="identity_balanced_accuracy")
        .loc[subjects, times]
        .to_numpy(float)
        - 0.5
    )
    operation_average = (
        operation_scores.groupby(["subj", "time_ms"])["balanced_accuracy"].mean().unstack("time_ms")
        .loc[subjects, times]
        .to_numpy(float)
        - 0.5
    )

    identity_clusters = cluster_table(identity_matrix, subjects, times, "identity_vs_chance")
    dominance_clusters = cluster_table(identity_matrix - operation_average, subjects, times, "identity_minus_operation")
    pd.concat([identity_clusters, dominance_clusters], ignore_index=True).to_csv(
        TABLES / "identity_dominance_cluster_tests.csv", index=False, encoding="utf-8-sig"
    )
    identity_stage = stage_identity_tests(identity_scores, subjects)
    identity_stage.to_csv(TABLES / "identity_stage_tests.csv", index=False, encoding="utf-8-sig")
    pair_stage = pair_stage_tests(pair_scores, subjects)
    pair_stage.to_csv(TABLES / "identity_pair_stage_tests.csv", index=False, encoding="utf-8-sig")
    equivalence = operation_equivalence(operation_scores, subjects)
    equivalence.to_csv(TABLES / "operation_decoding_equivalence.csv", index=False, encoding="utf-8-sig")
    plot_summary(identity_scores, operation_scores, equivalence, identity_clusters)

    summary = {
        "n_subjects": len(subjects),
        "identity_test": "within-sex identity decoding, leave-one-edit-combination-out",
        "operation_test": "binary operation decoding, leave-one-identity-out",
        "equivalence_bound_accuracy_points": EQUIVALENCE_BOUND,
        "identity_confirmed_clusters": identity_clusters.loc[identity_clusters["time_corrected_p"] < 0.05].to_dict("records"),
        "dominance_confirmed_clusters": dominance_clusters.loc[dominance_clusters["time_corrected_p"] < 0.05].to_dict("records"),
        "identity_pair_stage_tests": pair_stage.to_dict("records"),
        "equivalent_operation_stages": equivalence.loc[equivalence["equivalent_within_5pp"]].to_dict("records"),
    }
    (OUT / "identity_dominance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

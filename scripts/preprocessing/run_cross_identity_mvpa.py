#!/usr/bin/env python
"""Leave-one-identity-out time-resolved decoding on the rebuilt epochs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from run_primary_erp_analysis import (
    EPOCH_DIR,
    FACTORS,
    ROOT,
    SEED,
    find_clusters,
    load_qc,
    subject_key,
)


OUT = ROOT / "cross_identity_mvpa"
TABLES = OUT / "tables"
N_PERMUTATIONS = 10_000


def decode_subject(subject: str) -> pd.DataFrame:
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
    identities = sorted(metadata["identity"].unique())
    rows = []
    for factor in FACTORS:
        labels = metadata[factor].to_numpy(int)
        fold_scores = []
        for held_out in identities:
            test = metadata["identity"].eq(held_out).to_numpy()
            train = ~test
            if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
                raise RuntimeError(f"{subject} {factor} {held_out}: a fold lacks one class")
            scores = []
            for time_index in range(len(centers)):
                scaler = StandardScaler()
                x_train = scaler.fit_transform(data[train, :, time_index])
                x_test = scaler.transform(data[test, :, time_index])
                classifier = LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=SEED)
                classifier.fit(x_train, labels[train])
                scores.append(balanced_accuracy_score(labels[test], classifier.predict(x_test)))
            fold_scores.append(scores)
        mean_scores = np.mean(fold_scores, axis=0)
        rows.extend(
            {"subj": subject, "factor": factor, "time_ms": time, "balanced_accuracy": score}
            for time, score in zip(centers, mean_scores)
        )
    return pd.DataFrame(rows)


def cluster_test(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(scores["subj"].unique(), key=subject_key)
    times = np.sort(scores["time_ms"].unique())
    arrays = {}
    observed = {}
    point_rows = []
    for factor in FACTORS:
        values = (
            scores[scores["factor"].eq(factor)]
            .pivot(index="subj", columns="time_ms", values="balanced_accuracy")
            .loc[subjects, times]
            .to_numpy(float)
            - 0.5
        )
        arrays[factor] = values
        t_values, p_values = stats.ttest_1samp(values, 0.0, axis=0)
        observed[factor] = (t_values, p_values, find_clusters(t_values, p_values))
        for time, mean, sem, t_value, p_value in zip(
            times,
            values.mean(axis=0) + 0.5,
            values.std(axis=0, ddof=1) / math.sqrt(len(subjects)),
            t_values,
            p_values,
        ):
            point_rows.append(
                {
                    "factor": factor,
                    "time_ms": time,
                    "n": len(subjects),
                    "mean_balanced_accuracy": mean,
                    "sem": sem,
                    "t_vs_chance": t_value,
                    "pointwise_p_uncorrected": p_value,
                }
            )
    rng = np.random.default_rng(SEED + 4)
    within = {factor: np.zeros(N_PERMUTATIONS) for factor in FACTORS}
    family = np.zeros(N_PERMUTATIONS)
    for permutation in range(N_PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], size=(len(subjects), 1))
        maxima = []
        for factor in FACTORS:
            t_values, p_values = stats.ttest_1samp(arrays[factor] * signs, 0.0, axis=0)
            maximum = max(
                (float(np.abs(t_values[cluster]).sum()) for cluster in find_clusters(t_values, p_values)),
                default=0.0,
            )
            within[factor][permutation] = maximum
            maxima.append(maximum)
        family[permutation] = max(maxima)
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
                    "direction": "above chance" if np.mean(t_values[cluster]) > 0 else "below chance",
                    "peak_ms": times[peak],
                    "peak_t": t_values[peak],
                    "cluster_mass_abs_t": mass,
                    "within_factor_time_corrected_p": (np.sum(within[factor] >= mass) + 1) / (N_PERMUTATIONS + 1),
                    "across_four_factor_fwer_p": (np.sum(family >= mass) + 1) / (N_PERMUTATIONS + 1),
                    "n_subjects": len(subjects),
                }
            )
    return pd.DataFrame(point_rows), pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    qc = load_qc()
    subjects = sorted(qc.loc[qc["included"], "subject"].tolist(), key=subject_key)
    parts = Parallel(n_jobs=2, verbose=5)(delayed(decode_subject)(subject) for subject in subjects)
    scores = pd.concat(parts, ignore_index=True)
    scores.to_csv(TABLES / "subject_cross_identity_decoding.csv", index=False, encoding="utf-8-sig")
    points, clusters = cluster_test(scores)
    points.to_csv(TABLES / "cross_identity_pointwise_stats.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(TABLES / "cross_identity_cluster_tests.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_subjects": len(subjects),
        "subjects": subjects,
        "classifier": "LinearSVC(C=1, class_weight=balanced)",
        "cross_validation": "leave one of four face identities out",
        "metric": "balanced accuracy",
        "confirmed_clusters": clusters.loc[clusters["across_four_factor_fwer_p"] < 0.05].to_dict("records") if len(clusters) else [],
    }
    (OUT / "cross_identity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

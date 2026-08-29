#!/usr/bin/env python
"""Cross-identity decoding of edited versus original face-image status."""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from run_primary_erp_analysis import EPOCH_DIR, ROOT, SEED, find_clusters, load_qc, subject_key


OUT = ROOT / "edit_presence_mvpa"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
N_PERMUTATIONS = 10_000


def decode_subject(subject: str) -> pd.DataFrame:
    epochs = mne.read_epochs(EPOCH_DIR / f"{subject}-epo.fif", preload=True, verbose="ERROR")
    metadata = epochs.metadata.copy().reset_index(drop=True)
    usable = metadata["factorial"].astype(bool) | metadata["control_original"].astype(bool)
    metadata = metadata.loc[usable].reset_index(drop=True)
    epochs = epochs[usable.to_numpy()]
    labels = metadata["factorial"].astype(int).to_numpy()

    data = epochs.get_data(copy=True)
    times_ms = epochs.times * 1000.0
    width = max(1, int(round(20.0 / np.median(np.diff(times_ms)))))
    kernel = np.ones(width) / width
    data = np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="same"), 2, data)
    centers = np.arange(0.0, 981.0, 20.0)
    indices = np.array([np.argmin(np.abs(times_ms - center)) for center in centers])
    data = data[:, :, indices]

    fold_scores = []
    identities = sorted(metadata["identity"].unique())
    for held_out in identities:
        test = metadata["identity"].eq(held_out).to_numpy()
        train = ~test
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
            raise RuntimeError(f"{subject} {held_out}: edited/original class missing")
        scores = []
        for time_index in range(len(centers)):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(data[train, :, time_index])
            x_test = scaler.transform(data[test, :, time_index])
            classifier = LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=SEED)
            classifier.fit(x_train, labels[train])
            scores.append(balanced_accuracy_score(labels[test], classifier.predict(x_test)))
        fold_scores.append(scores)
    return pd.DataFrame(
        {
            "subj": subject,
            "time_ms": centers,
            "balanced_accuracy": np.mean(fold_scores, axis=0),
            "n_original": int((labels == 0).sum()),
            "n_edited": int((labels == 1).sum()),
        }
    )


def test_timecourse(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(scores["subj"].unique(), key=subject_key)
    times = np.sort(scores["time_ms"].unique())
    values = (
        scores.pivot(index="subj", columns="time_ms", values="balanced_accuracy")
        .loc[subjects, times]
        .to_numpy(float)
        - 0.5
    )
    t_values, p_values = stats.ttest_1samp(values, 0.0, axis=0)
    clusters = find_clusters(t_values, p_values)
    rng = np.random.default_rng(SEED + 41)
    null = np.zeros(N_PERMUTATIONS)
    for permutation in range(N_PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], size=(len(subjects), 1))
        perm_t, perm_p = stats.ttest_1samp(values * signs, 0.0, axis=0)
        null[permutation] = max(
            (float(np.abs(perm_t[cluster]).sum()) for cluster in find_clusters(perm_t, perm_p)),
            default=0.0,
        )
    point_rows = []
    for time, mean, sem, t_value, p_value in zip(
        times,
        values.mean(axis=0) + 0.5,
        values.std(axis=0, ddof=1) / math.sqrt(len(subjects)),
        t_values,
        p_values,
    ):
        point_rows.append(
            {
                "time_ms": time,
                "n": len(subjects),
                "mean_balanced_accuracy": mean,
                "sem": sem,
                "t_vs_chance": t_value,
                "pointwise_p_uncorrected": p_value,
            }
        )
    cluster_rows = []
    for cluster_id, cluster in enumerate(clusters, 1):
        mass = float(np.abs(t_values[cluster]).sum())
        peak = int(cluster[np.argmax(np.abs(t_values[cluster]))])
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "start_ms": times[cluster[0]],
                "end_ms": times[cluster[-1]],
                "direction": "above chance" if np.mean(t_values[cluster]) > 0 else "below chance",
                "peak_ms": times[peak],
                "peak_t": t_values[peak],
                "peak_accuracy": values[:, peak].mean() + 0.5,
                "cluster_mass_abs_t": mass,
                "time_corrected_p": (np.sum(null >= mass) + 1) / (N_PERMUTATIONS + 1),
                "n_subjects": len(subjects),
            }
        )
    cluster_columns = [
        "cluster_id", "start_ms", "end_ms", "direction", "peak_ms", "peak_t",
        "peak_accuracy", "cluster_mass_abs_t", "time_corrected_p", "n_subjects",
    ]
    return pd.DataFrame(point_rows), pd.DataFrame(cluster_rows, columns=cluster_columns)


def make_figure(points: pd.DataFrame, clusters: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    x = points["time_ms"].to_numpy(float)
    mean = points["mean_balanced_accuracy"].to_numpy(float)
    sem = points["sem"].to_numpy(float)
    fig, axis = plt.subplots(figsize=(7.3, 4.4))
    axis.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color="#009E73", alpha=0.18)
    axis.plot(x, mean, color="#009E73", linewidth=2.2)
    axis.axhline(0.5, color="#555555", linewidth=1, linestyle="--")
    axis.axvline(0, color="#777777", linewidth=0.8)
    for _, cluster in clusters[clusters["time_corrected_p"] < 0.05].iterrows():
        axis.axvspan(cluster["start_ms"], cluster["end_ms"], color="#009E73", alpha=0.14)
    axis.set(xlabel="Time from face onset (ms)", ylabel="Balanced accuracy", xlim=(0, 980))
    axis.set_title("Edited-versus-original status across held-out identities", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "cross_identity_edit_presence.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "cross_identity_edit_presence.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    qc = load_qc()
    subjects = sorted(qc.loc[qc["included"], "subject"].tolist(), key=subject_key)
    parts = Parallel(n_jobs=2, verbose=5)(delayed(decode_subject)(subject) for subject in subjects)
    scores = pd.concat(parts, ignore_index=True)
    scores.to_csv(TABLES / "subject_cross_identity_edit_presence.csv", index=False, encoding="utf-8-sig")
    points, clusters = test_timecourse(scores)
    points.to_csv(TABLES / "edit_presence_pointwise_stats.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(TABLES / "edit_presence_cluster_tests.csv", index=False, encoding="utf-8-sig")
    make_figure(points, clusters)
    summary = {
        "n_subjects": len(subjects),
        "contrast": "edited factorial images versus unedited original controls",
        "cross_validation": "leave one of four source identities out",
        "classifier": "LinearSVC(C=1, class_weight=balanced)",
        "confirmed_clusters": clusters.loc[clusters["time_corrected_p"] < 0.05].to_dict("records") if len(clusters) else [],
        "limitation": "Edited status may include low-level production differences; it is not interpreted as conscious edit detection.",
    }
    (OUT / "edit_presence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

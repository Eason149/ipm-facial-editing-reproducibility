#!/usr/bin/env python
"""Four-factor familywise whole-scalp spatiotemporal cluster audit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components

from run_primary_erp_analysis import (
    EPOCH_DIR,
    FACTORS,
    ROOT,
    SEED,
    design_matrix,
    load_qc,
    smooth_timecourse,
    subject_key,
)


OUT = ROOT / "whole_scalp"
TABLES = OUT / "tables"
N_PERMUTATIONS = 5_000


def subject_maps(subject: str, channels: list[str] | None) -> tuple[np.ndarray, np.ndarray, list[str], mne.Info]:
    epochs = mne.read_epochs(EPOCH_DIR / f"{subject}-epo.fif", preload=True, verbose="ERROR")
    metadata = epochs.metadata.copy().reset_index(drop=True)
    factorial = metadata["factorial"].astype(bool).to_numpy()
    metadata = metadata.loc[factorial].reset_index(drop=True)
    epochs = epochs[factorial]
    if channels is None:
        channels = list(epochs.ch_names)
    if set(channels) != set(epochs.ch_names):
        raise RuntimeError(f"{subject}: channel set differs from the first included participant")
    epochs.reorder_channels(channels)
    data = epochs.get_data(units="uV", copy=True)
    times_ms = epochs.times * 1000.0
    centers = np.arange(0.0, 981.0, 20.0)
    indices = np.array([np.argmin(np.abs(times_ms - center)) for center in centers])
    trial_channel_time = np.transpose(data, (0, 1, 2)).reshape(-1, data.shape[-1])
    trial_channel_time = smooth_timecourse(trial_channel_time, times_ms, 30.0)
    data = trial_channel_time.reshape(data.shape[0], data.shape[1], data.shape[2])[:, :, indices]
    x, columns = design_matrix(metadata)
    beta = np.einsum("pt,tcs->pcs", np.linalg.pinv(x), data)
    maps = np.stack([beta[columns.index(factor)].T for factor in FACTORS], axis=0)
    return maps, centers, channels, epochs.info


def one_sample_t(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    return np.divide(mean, sd / math.sqrt(values.shape[0]), out=np.zeros_like(mean), where=sd > 0)


def clusters_from_t(t_values: np.ndarray, threshold: float, adjacency: sparse.spmatrix) -> list[dict]:
    flat = t_values.ravel()
    clusters = []
    for direction, mask in [("positive", flat > threshold), ("negative", flat < -threshold)]:
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        subgraph = adjacency[indices][:, indices]
        n_components, labels = connected_components(subgraph, directed=False)
        for component in range(n_components):
            vertices = indices[labels == component]
            clusters.append(
                {
                    "vertices": vertices,
                    "direction": direction,
                    "mass": float(np.abs(flat[vertices]).sum()),
                    "size": int(len(vertices)),
                }
            )
    return clusters


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    qc = load_qc()
    subjects = sorted(qc.loc[qc["included"], "subject"].tolist(), key=subject_key)
    maps = []
    channels = None
    info = None
    times = None
    for subject in subjects:
        subject_map, times, channels, info = subject_maps(subject, channels)
        maps.append(subject_map)
    values = np.stack(maps, axis=1)  # factor x subject x time x channel
    np.savez_compressed(
        OUT / "subject_factor_spatiotemporal_maps.npz",
        values=values,
        factors=np.array(FACTORS),
        subjects=np.array(subjects),
        times_ms=times,
        channels=np.array(channels),
    )
    channel_adjacency, adjacency_channels = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    if list(adjacency_channels) != channels:
        order = [adjacency_channels.index(channel) for channel in channels]
        channel_adjacency = channel_adjacency[order][:, order]
    adjacency = mne.stats.combine_adjacency(len(times), channel_adjacency).tocsr()
    threshold = float(stats.t.ppf(0.975, len(subjects) - 1))
    observed_t = np.stack([one_sample_t(values[index]) for index in range(len(FACTORS))])
    observed = [clusters_from_t(observed_t[index], threshold, adjacency) for index in range(len(FACTORS))]

    rng = np.random.default_rng(SEED + 3)
    family_null = np.zeros(N_PERMUTATIONS)
    within_null = np.zeros((len(FACTORS), N_PERMUTATIONS))
    for permutation in range(N_PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], size=(len(subjects), 1, 1))
        maxima = []
        for factor_index in range(len(FACTORS)):
            t_values = one_sample_t(values[factor_index] * signs)
            clusters = clusters_from_t(t_values, threshold, adjacency)
            maximum = max((cluster["mass"] for cluster in clusters), default=0.0)
            within_null[factor_index, permutation] = maximum
            maxima.append(maximum)
        family_null[permutation] = max(maxima)

    rows = []
    n_channels = len(channels)
    for factor_index, factor in enumerate(FACTORS):
        flat_t = observed_t[factor_index].ravel()
        for cluster_id, cluster in enumerate(observed[factor_index], 1):
            vertices = cluster["vertices"]
            time_indices = vertices // n_channels
            channel_indices = vertices % n_channels
            peak_vertex = int(vertices[np.argmax(np.abs(flat_t[vertices]))])
            peak_time_index = peak_vertex // n_channels
            peak_channel_index = peak_vertex % n_channels
            unique_channels = sorted({channels[index] for index in channel_indices})
            rows.append(
                {
                    "factor": factor,
                    "cluster_id": cluster_id,
                    "direction": cluster["direction"],
                    "start_ms": float(times[time_indices.min()]),
                    "end_ms": float(times[time_indices.max()]),
                    "n_time_channel_vertices": cluster["size"],
                    "n_unique_channels": len(unique_channels),
                    "channels": ";".join(unique_channels),
                    "cluster_mass_abs_t": cluster["mass"],
                    "peak_ms": float(times[peak_time_index]),
                    "peak_channel": channels[peak_channel_index],
                    "peak_t": float(flat_t[peak_vertex]),
                    "within_factor_spatiotemporal_p": (np.sum(within_null[factor_index] >= cluster["mass"]) + 1) / (N_PERMUTATIONS + 1),
                    "across_four_factor_spatiotemporal_fwer_p": (np.sum(family_null >= cluster["mass"]) + 1) / (N_PERMUTATIONS + 1),
                    "n_subjects": len(subjects),
                    "n_permutations": N_PERMUTATIONS,
                    "cluster_forming_two_sided_p": 0.05,
                }
            )
    frame = pd.DataFrame(rows).sort_values("across_four_factor_spatiotemporal_fwer_p")
    frame.to_csv(TABLES / "whole_scalp_four_factor_clusters.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_subjects": len(subjects),
        "subjects": subjects,
        "n_channels": len(channels),
        "n_times": len(times),
        "n_permutations": N_PERMUTATIONS,
        "confirmed_clusters": frame.loc[
            frame["across_four_factor_spatiotemporal_fwer_p"] < 0.05
        ].to_dict("records"),
    }
    (OUT / "whole_scalp_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

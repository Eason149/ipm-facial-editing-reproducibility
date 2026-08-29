"""Expanded component-window EEG RSA with control condition and full model space.

This standalone script includes the unedited/control condition (raw CondID 1),
yielding 17x17 RDMs, and tests all 1-, 2-, 3-, and 4-factor combination
models for FSlim, Eye, Mouth, and Skin, plus a control-vs-edited model.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
OUTPUTS = PROJECT / "outputs"
FIGURES = PROJECT / "figures"
LOGS = PROJECT / "logs"

STIM_BASES = {"F_1": 100, "F_2": 120, "M_1": 140, "M_2": 160, "T": 180}
ANALYZED_STIMTYPES = ("F_1", "F_2", "M_1", "M_2")
FACTORS = ("FSlim", "Eye", "Mouth", "Skin")

N170_LATERAL_ROI = ("P7", "P8", "PO7", "PO8")
N170_LEFT_ROI = ("P7", "PO7")
N170_RIGHT_ROI = ("P8", "PO8")
N170_POSTERIOR_ROI = (
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "O1",
    "OZ",
    "O2",
)
LPP_CENTROPARIETAL_ROI = ("CP3", "CP1", "CPZ", "CP2", "CP4", "P3", "P1", "PZ", "P2", "P4")
LPP_MIDLINE_ROI = ("CPZ", "PZ")
LPP_PZ_ROI = ("PZ",)
LPP_POSTERIOR_ROI = ("P3", "P1", "PZ", "P2", "P4", "PO3", "POZ", "PO4", "O1", "OZ", "O2")


@dataclass
class SubjectData:
    subj: str
    times: np.ndarray
    channels: list[str]
    patterns: np.ndarray | None
    formal_epoch_counts: dict[int, int]
    epoch_count: int
    formal_epoch_count: int
    excluded_reason: str = ""


def natural_subject_key(subject: str) -> int:
    match = re.search(r"(\d+)", str(subject))
    return int(match.group(1)) if match else 10**9


def decode_hdf5_reference(dataset: h5py.File, ref: h5py.Reference) -> object:
    array = np.asarray(dataset[ref][()]).squeeze()
    if array.dtype.kind in "ui" and np.size(array) >= 1:
        text = "".join(chr(int(value)) for value in np.ravel(array, order="F") if int(value))
        if text:
            return text
    return np.asarray(array).squeeze().item()


def decode_endcode(code: object) -> tuple[str, int] | tuple[None, None]:
    try:
        integer = int(str(code).strip())
    except ValueError:
        return None, None
    for stimtype, base in STIM_BASES.items():
        condition = integer - base
        if 1 <= condition <= 17:
            return stimtype, condition
    return None, None


def read_set_header(set_path: Path) -> dict[str, object]:
    with h5py.File(set_path, "r") as dataset:
        n_channels = int(dataset["nbchan"][0, 0])
        n_points = int(dataset["pnts"][0, 0])
        n_trials = int(dataset["trials"][0, 0])
        times = np.asarray(dataset["times"][()]).ravel().astype(float)
        channels = [
            str(decode_hdf5_reference(dataset, ref))
            for ref in dataset["chanlocs"]["labels"][:, 0]
        ]
        endcodes = [
            decode_hdf5_reference(dataset, ref)
            for ref in dataset["epoch"]["endCode"][:, 0]
        ]
        datfile = str(
            "".join(chr(int(value)) for value in dataset["datfile"][()].ravel() if int(value))
        )
    return {
        "n_channels": n_channels,
        "n_points": n_points,
        "n_trials": n_trials,
        "times": times,
        "channels": channels,
        "endcodes": endcodes,
        "fdt_path": set_path.parent / datfile,
    }


def load_condition_table() -> pd.DataFrame:
    path = ROOT / "RSA_time_resolved_analysis" / "results_behavior_rsa" / "outputs" / "condition_table.csv"
    if not path.exists():
        path = ROOT / "RSA_time_resolved_analysis" / "results" / "outputs" / "condition_table.csv"
    table = pd.read_csv(path)
    for column in ["CondID", "raw_CondID", *FACTORS]:
        table[column] = pd.to_numeric(table[column], errors="raise")
    return table.sort_values("CondID").reset_index(drop=True)


def find_eeg_set_files() -> list[Path]:
    return sorted(
        ROOT.glob("derivatives_eeglab_s*/s*_epoched_stim.set"),
        key=lambda path: natural_subject_key(path.parent.name),
    )


def load_subject(set_path: Path, condition_table: pd.DataFrame) -> SubjectData:
    subj_match = re.search(r"s(\d+)", set_path.stem.lower())
    subj = f"s{subj_match.group(1)}" if subj_match else set_path.stem
    header = read_set_header(set_path)
    decoded = [decode_endcode(code) for code in header["endcodes"]]
    raw_conditions = condition_table["raw_CondID"].astype(int).tolist()
    indices_by_condition: dict[int, list[int]] = {condition: [] for condition in raw_conditions}
    for index, (stimtype, condition) in enumerate(decoded):
        if stimtype in ANALYZED_STIMTYPES and condition in indices_by_condition:
            indices_by_condition[condition].append(index)
    counts = {condition: len(indices) for condition, indices in indices_by_condition.items()}
    missing = [condition for condition, count in counts.items() if count == 0]
    formal_epoch_count = int(sum(counts.values()))
    if missing:
        return SubjectData(
            subj,
            np.asarray(header["times"], dtype=float),
            list(header["channels"]),
            None,
            counts,
            int(header["n_trials"]),
            formal_epoch_count,
            f"Missing raw CondID: {missing}",
        )
    data = np.memmap(
        Path(header["fdt_path"]),
        dtype="<f4",
        mode="r",
        shape=(int(header["n_trials"]), int(header["n_points"]), int(header["n_channels"])),
    )
    patterns = []
    for raw_condition in raw_conditions:
        epochs = np.asarray(data[indices_by_condition[raw_condition], :, :], dtype=np.float64)
        patterns.append(epochs.mean(axis=0))
    return SubjectData(
        subj,
        np.asarray(header["times"], dtype=float),
        list(header["channels"]),
        np.stack(patterns, axis=0),
        counts,
        int(header["n_trials"]),
        formal_epoch_count,
    )


def vectorize_rdm(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    p_array = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_array.shape, np.nan, dtype=float)
    valid = np.isfinite(p_array)
    if not valid.any():
        return adjusted
    valid_p = p_array[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.minimum(corrected, 1.0)
    restored = np.empty_like(corrected)
    restored[order] = corrected
    adjusted[valid] = restored
    return adjusted


def rank_rows(values: np.ndarray) -> np.ndarray:
    return rankdata(values, axis=1, method="average")


def pearson_rows(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered_values = values - values.mean(axis=1, keepdims=True)
    centered_target = target - target.mean()
    denominator = np.sqrt(np.sum(centered_values**2, axis=1) * np.sum(centered_target**2))
    return np.divide(
        centered_values @ centered_target,
        denominator,
        out=np.full(values.shape[0], np.nan),
        where=denominator > np.finfo(float).eps,
    )


def spearman_rsa(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return pearson_rows(rank_rows(np.atleast_2d(x)), rankdata(y, method="average"))


def component_features(
    patterns: np.ndarray,
    times: np.ndarray,
    channels: Sequence[str],
    roi: Sequence[str],
    window: tuple[float, float],
    mode: str,
) -> np.ndarray:
    channel_indices = [channels.index(channel) for channel in roi if channel in channels]
    if len(channel_indices) != len(roi):
        missing = sorted(set(roi) - set(channels))
        raise ValueError(f"Missing ROI channels: {missing}")
    time_mask = (times >= window[0]) & (times <= window[1])
    if not time_mask.any():
        raise ValueError(f"No samples in window {window}.")
    selected = patterns[:, time_mask, :][:, :, channel_indices]
    if mode == "mean_abs":
        return selected.mean(axis=(1, 2))[:, None]
    if mode == "channel_euclidean":
        return selected.mean(axis=1)
    raise ValueError(f"Unknown mode: {mode}")


def component_asymmetry_features(
    patterns: np.ndarray,
    times: np.ndarray,
    channels: Sequence[str],
    left_roi: Sequence[str],
    right_roi: Sequence[str],
    window: tuple[float, float],
) -> np.ndarray:
    left = component_features(patterns, times, channels, left_roi, window, "mean_abs")[:, 0]
    right = component_features(patterns, times, channels, right_roi, window, "mean_abs")[:, 0]
    return (right - left)[:, None]


def feature_rdm(features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean_abs":
        values = features[:, 0]
        return np.abs(values[:, None] - values[None, :])
    diffs = features[:, None, :] - features[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=-1))


def make_17_condition_table() -> pd.DataFrame:
    formal = load_condition_table().copy()
    formal["condition_label"] = formal["CondID"].map(lambda value: f"Edited_{int(value):02d}")
    control = {
        "CondID": 0,
        "raw_CondID": 1,
        "FSlim": 0,
        "Eye": 0,
        "Mouth": 0,
        "Skin": 0,
        "FSlim_x_Eye": 0,
        "FSlim_x_Eye_x_Skin": 0,
        "condition_label": "Control_unedited",
    }
    table = pd.concat([pd.DataFrame([control]), formal], ignore_index=True)
    return table.sort_values(["raw_CondID"]).reset_index(drop=True)


def model_name(family: str, subset: Sequence[str]) -> str:
    prefix = "H" if family == "hamming" else "J"
    return f"{prefix}{len(subset)}_" + "_".join(subset)


def make_comprehensive_model_rdms(condition_table: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    models: dict[str, np.ndarray] = {}
    metadata: list[dict[str, object]] = []
    is_control = condition_table["raw_CondID"].to_numpy(dtype=int) == 1
    control_model = np.logical_xor(is_control[:, None], is_control[None, :]).astype(float)
    models["Control_vs_edited"] = control_model
    metadata.append(
        {
            "model": "Control_vs_edited",
            "family": "control",
            "dimension": 0,
            "factors": "none",
            "description": "Unedited control differs from every edited condition.",
        }
    )
    for dimension in range(1, len(FACTORS) + 1):
        for subset in itertools.combinations(FACTORS, dimension):
            values = condition_table.loc[:, list(subset)].to_numpy(dtype=int)
            diff_count = (values[:, None, :] != values[None, :, :]).sum(axis=2).astype(float)
            hamming = model_name("hamming", subset)
            joint = model_name("joint", subset)
            models[hamming] = diff_count
            models[joint] = (diff_count > 0).astype(float)
            metadata.append(
                {
                    "model": hamming,
                    "family": "hamming",
                    "dimension": dimension,
                    "factors": " x ".join(subset),
                    "description": "Distance equals the number of differing factor levels.",
                }
            )
            metadata.append(
                {
                    "model": joint,
                    "family": "joint_binary",
                    "dimension": dimension,
                    "factors": " x ".join(subset),
                    "description": "Distance is 1 if any factor in the subset differs.",
                }
            )
    return models, pd.DataFrame(metadata)


def common_channels(subjects: Sequence[SubjectData]) -> list[str]:
    if not subjects:
        return []
    common = set(subjects[0].channels)
    for subject in subjects[1:]:
        common &= set(subject.channels)
    return [channel for channel in subjects[0].channels if channel in common]


def make_analyses(shared_channels: Sequence[str]) -> list[dict[str, object]]:
    return [
        {
            "name": "N170_lateral_mean_amplitude_RDM",
            "component": "N170",
            "window": (140.0, 190.0),
            "roi": N170_LATERAL_ROI,
            "mode": "mean_abs",
        },
        {
            "name": "N170_lateral_channel_euclidean_RDM",
            "component": "N170",
            "window": (140.0, 190.0),
            "roi": N170_LATERAL_ROI,
            "mode": "channel_euclidean",
        },
        {
            "name": "N170_left_mean_amplitude_RDM",
            "component": "N170",
            "window": (140.0, 190.0),
            "roi": N170_LEFT_ROI,
            "mode": "mean_abs",
        },
        {
            "name": "N170_right_mean_amplitude_RDM",
            "component": "N170",
            "window": (140.0, 190.0),
            "roi": N170_RIGHT_ROI,
            "mode": "mean_abs",
        },
        {
            "name": "N170_right_minus_left_asymmetry_RDM",
            "component": "N170",
            "window": (140.0, 190.0),
            "left_roi": N170_LEFT_ROI,
            "right_roi": N170_RIGHT_ROI,
            "mode": "asymmetry_abs",
        },
        {
            "name": "N170_posterior_mean_amplitude_RDM",
            "component": "N170",
            "window": (140.0, 190.0),
            "roi": N170_POSTERIOR_ROI,
            "mode": "mean_abs",
        },
        {
            "name": "N170_shared_montage_mean_amplitude_RDM",
            "component": "N170",
            "window": (140.0, 190.0),
            "roi": tuple(shared_channels),
            "mode": "mean_abs",
        },
        {
            "name": "N170_shared_montage_channel_euclidean_RDM",
            "component": "N170",
            "window": (140.0, 190.0),
            "roi": tuple(shared_channels),
            "mode": "channel_euclidean",
        },
        {
            "name": "LPP_centroparietal_mean_amplitude_RDM",
            "component": "LPP",
            "window": (300.0, 800.0),
            "roi": LPP_CENTROPARIETAL_ROI,
            "mode": "mean_abs",
        },
        {
            "name": "LPP_centroparietal_channel_euclidean_RDM",
            "component": "LPP",
            "window": (300.0, 800.0),
            "roi": LPP_CENTROPARIETAL_ROI,
            "mode": "channel_euclidean",
        },
        {
            "name": "LPP_midline_mean_amplitude_RDM",
            "component": "LPP",
            "window": (300.0, 800.0),
            "roi": LPP_MIDLINE_ROI,
            "mode": "mean_abs",
        },
        {
            "name": "LPP_Pz_mean_amplitude_RDM",
            "component": "LPP",
            "window": (300.0, 800.0),
            "roi": LPP_PZ_ROI,
            "mode": "mean_abs",
        },
        {
            "name": "LPP_posterior_mean_amplitude_RDM",
            "component": "LPP",
            "window": (300.0, 800.0),
            "roi": LPP_POSTERIOR_ROI,
            "mode": "mean_abs",
        },
        {
            "name": "LPP_shared_montage_mean_amplitude_RDM",
            "component": "LPP",
            "window": (300.0, 800.0),
            "roi": tuple(shared_channels),
            "mode": "mean_abs",
        },
        {
            "name": "LPP_shared_montage_channel_euclidean_RDM",
            "component": "LPP",
            "window": (300.0, 800.0),
            "roi": tuple(shared_channels),
            "mode": "channel_euclidean",
        },
    ]


def analysis_features(subject: SubjectData, analysis: Mapping[str, object]) -> tuple[np.ndarray, str]:
    mode = str(analysis["mode"])
    if mode == "asymmetry_abs":
        features = component_asymmetry_features(
            subject.patterns,
            subject.times,
            subject.channels,
            analysis["left_roi"],
            analysis["right_roi"],
            analysis["window"],
        )
        return features, "mean_abs"
    features = component_features(
        subject.patterns,
        subject.times,
        subject.channels,
        analysis["roi"],
        analysis["window"],
        mode,
    )
    return features, mode


def run_expanded_rsa(
    subjects: Sequence[SubjectData],
    analyses: Sequence[Mapping[str, object]],
    model_rdms: Mapping[str, np.ndarray],
    model_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_vectors = {name: vectorize_rdm(matrix) for name, matrix in model_rdms.items()}
    model_order = model_metadata["model"].tolist()
    model_meta = model_metadata.set_index("model").to_dict("index")
    rsa_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []
    rdm_rows: list[dict[str, object]] = []
    pair_i, pair_j = np.triu_indices(len(next(iter(model_rdms.values()))), 1)
    for subject in subjects:
        assert subject.patterns is not None
        for analysis in analyses:
            features, rdm_mode = analysis_features(subject, analysis)
            rdm = feature_rdm(features, rdm_mode)
            eeg_vector = vectorize_rdm(rdm)
            for condition_index, values in enumerate(features):
                value_rows.append(
                    {
                        "subj": subject.subj,
                        "analysis": analysis["name"],
                        "condition_index": condition_index,
                        "raw_CondID": condition_index + 1,
                        "component_value": float(values.mean()),
                    }
                )
            for i, j, distance in zip(pair_i, pair_j, eeg_vector):
                rdm_rows.append(
                    {
                        "subj": subject.subj,
                        "analysis": analysis["name"],
                        "condition_i_index": int(i),
                        "condition_j_index": int(j),
                        "distance": float(distance),
                    }
            )
            for model_index, model in enumerate(model_order):
                meta = model_meta[model]
                rsa_rows.append(
                    {
                        "subj": subject.subj,
                        "analysis": analysis["name"],
                        "component": analysis["component"],
                        "window_start_ms": analysis["window"][0],
                        "window_end_ms": analysis["window"][1],
                        "mode": analysis["mode"],
                        "model": model,
                        "model_family": meta["family"],
                        "dimension": meta["dimension"],
                        "factors": meta["factors"],
                        "rho": float(spearman_rsa(eeg_vector, model_vectors[model])[0]),
                    }
                )
    return pd.DataFrame(rsa_rows), pd.DataFrame(rdm_rows), pd.DataFrame(value_rows)


def summarize(subject_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["analysis", "model", "model_family", "dimension", "factors"]
    for keys, subset in subject_results.groupby(group_columns, sort=False):
        meta = subset.iloc[0]
        for metric in ("rho",):
            values = subset[metric].dropna().to_numpy(dtype=float)
            t_value, p_value = stats.ttest_1samp(values, popmean=0.0) if len(values) > 1 else (np.nan, np.nan)
            mean = float(np.mean(values)) if len(values) else np.nan
            sem = float(stats.sem(values)) if len(values) > 1 else np.nan
            critical = float(stats.t.ppf(0.975, len(values) - 1)) if len(values) > 1 else np.nan
            rows.append(
                {
                    "analysis": keys[0],
                    "component": meta["component"],
                    "model": keys[1],
                    "model_family": keys[2],
                    "dimension": int(keys[3]),
                    "factors": keys[4],
                    "metric": metric,
                    "n_subjects": len(values),
                    "mean_rho": mean,
                    "sem_rho": sem,
                    "ci95_low": mean - critical * sem if len(values) > 1 else np.nan,
                    "ci95_high": mean + critical * sem if len(values) > 1 else np.nan,
                    "t_value": float(t_value),
                    "p_value": float(p_value),
                    "cohens_d": float(mean / np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary["q_fdr_all_models"] = summary.groupby(["analysis", "metric"])["p_value"].transform(
        lambda values: fdr_bh(values.to_numpy(dtype=float))
    )
    summary["q_fdr_family_dimension"] = summary.groupby(
        ["analysis", "metric", "model_family", "dimension"]
    )["p_value"].transform(lambda values: fdr_bh(values.to_numpy(dtype=float)))
    return summary.sort_values(["metric", "component", "analysis", "q_fdr_all_models", "p_value"]).reset_index(drop=True)


def plot_top_results(summary: pd.DataFrame, path: Path) -> None:
    candidates = summary[(summary["metric"] == "rho") & (summary["component"] == "LPP")].copy()
    candidates = candidates[np.isfinite(candidates["mean_rho"]) & np.isfinite(candidates["p_value"])]
    candidates = candidates.sort_values(["q_fdr_family_dimension", "p_value"]).head(20)
    labels = candidates["analysis"].str.replace("_", " ", regex=False) + "\n" + candidates["model"]
    y = np.arange(len(candidates))
    colors = candidates["dimension"].map({0: "#6b7280", 1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"})
    fig, axis = plt.subplots(figsize=(10.6, max(6, 0.38 * len(candidates) + 1.5)))
    axis.barh(y, candidates["mean_rho"], color=colors, alpha=0.88)
    axis.errorbar(
        candidates["mean_rho"],
        y,
        xerr=np.vstack([candidates["mean_rho"] - candidates["ci95_low"], candidates["ci95_high"] - candidates["mean_rho"]]),
        fmt="none",
        ecolor="#222",
        capsize=3,
        lw=1,
    )
    axis.axvline(0, color="#333", lw=0.8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Group mean Spearman rho")
    axis.set_title("Expanded 17-condition component RSA: top LPP model-space results", fontweight="bold")
    axis.grid(axis="x", color="#e9e9e9", lw=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lpp_pz_model_space(summary: pd.DataFrame, path: Path) -> None:
    display = summary[
        (summary["analysis"] == "LPP_Pz_mean_amplitude_RDM")
        & (summary["metric"] == "rho")
    ].copy()
    display = display[np.isfinite(display["mean_rho"]) & np.isfinite(display["p_value"])]
    order = ["control", "hamming", "joint_binary"]
    display["family_order"] = display["model_family"].map({name: i for i, name in enumerate(order)})
    display = display.sort_values(["family_order", "dimension", "p_value"])
    labels = display["model"]
    x = np.arange(len(display))
    fig, axis = plt.subplots(figsize=(13.5, 5.4))
    bars = axis.bar(x, display["mean_rho"], color="#49759c", alpha=0.9, width=0.72)
    for bar, (_, row) in zip(bars, display.iterrows()):
        if float(row["q_fdr_all_models"]) < 0.05:
            marker = "*"
        elif float(row["q_fdr_family_dimension"]) < 0.05:
            marker = "+"
        else:
            marker = ""
        if marker:
            y = row["mean_rho"] + (0.01 if row["mean_rho"] >= 0 else -0.01)
            axis.text(bar.get_x() + bar.get_width() / 2, y, marker, ha="center", va="bottom" if row["mean_rho"] >= 0 else "top")
    axis.axhline(0, color="#333", lw=0.8)
    axis.set_xticks(x, labels, rotation=75, ha="right", fontsize=7.6)
    axis.set_ylabel("Group mean Spearman rho")
    axis.set_title("LPP Pz amplitude RDM across the full 17-condition model space", fontweight="bold")
    axis.text(
        0.01,
        0.98,
        "* all-model FDR q < .05; + family/dimension FDR q < .05",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
    )
    axis.grid(axis="y", color="#e9e9e9", lw=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roi_selection_rationale(summary: pd.DataFrame, path: Path) -> None:
    selected_models = [
        "Control_vs_edited",
        "H1_FSlim",
        "H1_Mouth",
        "H2_FSlim_Mouth",
        "J2_FSlim_Mouth",
        "H2_FSlim_Skin",
        "J2_FSlim_Skin",
        "H3_FSlim_Mouth_Skin",
        "J3_FSlim_Mouth_Skin",
        "H4_FSlim_Eye_Mouth_Skin",
    ]
    analysis_order = [
        "N170_lateral_mean_amplitude_RDM",
        "N170_lateral_channel_euclidean_RDM",
        "N170_left_mean_amplitude_RDM",
        "N170_right_mean_amplitude_RDM",
        "N170_right_minus_left_asymmetry_RDM",
        "N170_posterior_mean_amplitude_RDM",
        "N170_shared_montage_mean_amplitude_RDM",
        "N170_shared_montage_channel_euclidean_RDM",
        "LPP_centroparietal_mean_amplitude_RDM",
        "LPP_centroparietal_channel_euclidean_RDM",
        "LPP_midline_mean_amplitude_RDM",
        "LPP_Pz_mean_amplitude_RDM",
        "LPP_posterior_mean_amplitude_RDM",
        "LPP_shared_montage_mean_amplitude_RDM",
        "LPP_shared_montage_channel_euclidean_RDM",
    ]
    row_labels = [
        "N170 lateral mean",
        "N170 lateral channels",
        "N170 left mean",
        "N170 right mean",
        "N170 R-L asymmetry",
        "N170 posterior mean",
        "N170 shared montage mean",
        "N170 shared montage channels",
        "LPP centroparietal mean",
        "LPP centroparietal channels",
        "LPP midline mean",
        "LPP Pz mean",
        "LPP posterior mean",
        "LPP shared montage mean",
        "LPP shared montage channels",
    ]
    model_labels = [
        "Control",
        "FSlim",
        "Mouth",
        "FSlim+Mouth\ncount",
        "FSlim+Mouth\nany",
        "FSlim+Skin\ncount",
        "FSlim+Skin\nany",
        "FSlim+Mouth+Skin\ncount",
        "FSlim+Mouth+Skin\nany",
        "All 4 factors\ncount",
    ]
    display = summary[
        (summary["metric"] == "rho")
        & (summary["analysis"].isin(analysis_order))
        & (summary["model"].isin(selected_models))
    ].copy()
    matrix = np.full((len(analysis_order), len(selected_models)), np.nan)
    q_matrix = np.full_like(matrix, np.nan, dtype=float)
    for row_index, analysis in enumerate(analysis_order):
        for col_index, model in enumerate(selected_models):
            hit = display[(display["analysis"] == analysis) & (display["model"] == model)]
            if not hit.empty:
                matrix[row_index, col_index] = float(hit.iloc[0]["mean_rho"])
                q_matrix[row_index, col_index] = float(hit.iloc[0]["q_fdr_all_models"])

    best = (
        summary[
            (summary["metric"] == "rho")
            & (summary["analysis"].isin(analysis_order))
            & np.isfinite(summary["p_value"])
        ]
        .sort_values(["analysis", "q_fdr_all_models", "p_value"])
        .groupby("analysis", sort=False)
        .head(1)
        .set_index("analysis")
        .reindex(analysis_order)
    )
    best_q = best["q_fdr_all_models"].to_numpy(dtype=float)
    best_score = -np.log10(np.clip(best_q, 1e-12, 1.0))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.8, 7.2),
        gridspec_kw={"width_ratios": [2.5, 1.0]},
        constrained_layout=True,
    )
    vmax = np.nanmax(np.abs(matrix))
    image = axes[0].imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_xticks(np.arange(len(selected_models)), model_labels, rotation=45, ha="right", fontsize=8.5)
    axes[0].set_yticks(np.arange(len(analysis_order)), row_labels)
    axes[0].set_title("Same FSlim-related models across all component ROIs", fontweight="bold")
    axes[0].set_xlabel("Model RDM")
    axes[0].set_ylabel("Component analysis")
    for row_index in range(q_matrix.shape[0]):
        for col_index in range(q_matrix.shape[1]):
            q_value = q_matrix[row_index, col_index]
            if np.isfinite(q_value) and q_value < 0.05:
                axes[0].text(col_index, row_index, "*", ha="center", va="center", color="black", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.02)
    cbar.set_label("Group mean Spearman rho")

    y = np.arange(len(analysis_order))
    colors = ["#5b8db8" if analysis == "LPP_Pz_mean_amplitude_RDM" else "#b9c0c8" for analysis in analysis_order]
    axes[1].barh(y, best_score, color=colors)
    axes[1].axvline(-np.log10(0.05), color="#333", linestyle="--", lw=1)
    axes[1].set_yticks(y, row_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("-log10(best all-model FDR q)")
    axes[1].set_title("Best evidence per ROI", fontweight="bold")
    axes[1].grid(axis="x", color="#e9e9e9", lw=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    for index, row in enumerate(best.itertuples()):
        if isinstance(row.model, str):
            axes[1].text(best_score[index] + 0.05, index, row.model, va="center", fontsize=7.5)
    fig.suptitle(
        "Why Pz was selected as the main LPP RSA result\n"
        "Stars mark tests surviving all-model FDR q < .05 within each ROI/analysis.",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pz_vs_shared_montage(summary: pd.DataFrame, path: Path) -> None:
    selected_models = [
        "Control_vs_edited",
        "H1_FSlim",
        "J1_FSlim",
        "H2_FSlim_Mouth",
        "J2_FSlim_Mouth",
        "H2_FSlim_Skin",
        "J2_FSlim_Skin",
        "H3_FSlim_Mouth_Skin",
        "H4_FSlim_Eye_Mouth_Skin",
    ]
    analyses = [
        "LPP_shared_montage_mean_amplitude_RDM",
        "LPP_shared_montage_channel_euclidean_RDM",
        "LPP_centroparietal_mean_amplitude_RDM",
        "LPP_midline_mean_amplitude_RDM",
        "LPP_Pz_mean_amplitude_RDM",
    ]
    labels = [
        "Shared montage\nmean amplitude",
        "Shared montage\nchannel distance",
        "Centroparietal\nmean amplitude",
        "Midline\nmean amplitude",
        "Pz\nmean amplitude",
    ]
    model_labels = [
        "Control",
        "H1 FSlim",
        "J1 FSlim",
        "H2 FSlim+Mouth",
        "J2 FSlim+Mouth",
        "H2 FSlim+Skin",
        "J2 FSlim+Skin",
        "H3 FSlim+Mouth+Skin",
        "H4 all factors",
    ]
    display = summary[
        (summary["metric"] == "rho")
        & (summary["analysis"].isin(analyses))
        & (summary["model"].isin(selected_models))
    ].copy()
    matrix = np.full((len(analyses), len(selected_models)), np.nan)
    q_matrix = np.full_like(matrix, np.nan, dtype=float)
    for row_index, analysis in enumerate(analyses):
        for col_index, model in enumerate(selected_models):
            hit = display[(display["analysis"] == analysis) & (display["model"] == model)]
            if not hit.empty:
                matrix[row_index, col_index] = float(hit.iloc[0]["mean_rho"])
                q_matrix[row_index, col_index] = float(hit.iloc[0]["q_fdr_all_models"])

    fig, axis = plt.subplots(figsize=(12.6, 4.8), constrained_layout=True)
    vmax = np.nanmax(np.abs(matrix))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axis.set_xticks(np.arange(len(selected_models)), model_labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(analyses)), labels)
    axis.set_title("Pz-LPP result compared with shared-montage and broader LPP ROIs", fontweight="bold")
    axis.set_xlabel(
        "Model RDM\n"
        "Stars mark all-model FDR q < .05 within each analysis. "
        "Shared montage uses the 55 common channels across included subjects."
    )
    axis.set_ylabel("LPP analysis")
    for row_index in range(q_matrix.shape[0]):
        for col_index in range(q_matrix.shape[1]):
            q_value = q_matrix[row_index, col_index]
            if np.isfinite(q_value) and q_value < 0.05:
                axis.text(col_index, row_index, "*", ha="center", va="center", color="black", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    cbar.set_label("Group mean Spearman rho")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in (OUTPUTS, FIGURES, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    condition_table = make_17_condition_table()
    model_rdms, model_metadata = make_comprehensive_model_rdms(condition_table)
    subjects_data = [load_subject(path, condition_table) for path in find_eeg_set_files()]
    qc_rows = []
    for subject in subjects_data:
        qc_rows.append(
            {
                "subj": subject.subj,
                "total_epochs": subject.epoch_count,
                "analyzed_epochs": subject.formal_epoch_count,
                "saved_channel_count": len(subject.channels),
                "n_conditions_present": sum(count > 0 for count in subject.formal_epoch_counts.values()),
                "min_epochs_per_condition": min(subject.formal_epoch_counts.values()),
                "max_epochs_per_condition": max(subject.formal_epoch_counts.values()),
                "included": subject.patterns is not None,
                "exclusion_reason": subject.excluded_reason,
            }
        )
    qc = pd.DataFrame(qc_rows).sort_values("subj", key=lambda col: col.map(natural_subject_key))
    included = [subject for subject in subjects_data if subject.patterns is not None]
    shared_channels = common_channels(included)
    analyses = make_analyses(shared_channels)
    rsa, rdms, values = run_expanded_rsa(included, analyses, model_rdms, model_metadata)
    summary = summarize(rsa)
    condition_table.to_csv(OUTPUTS / "comprehensive_17_condition_table.csv", index=False, encoding="utf-8-sig")
    model_metadata.to_csv(OUTPUTS / "comprehensive_model_metadata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"channel": shared_channels}).to_csv(
        OUTPUTS / "comprehensive_shared_montage_channels.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({name: vectorize_rdm(rdm) for name, rdm in model_rdms.items()}).to_csv(
        OUTPUTS / "comprehensive_model_rdm_vectors.csv", index=False
    )
    qc.to_csv(OUTPUTS / "comprehensive_component_rsa_subject_quality_control.csv", index=False, encoding="utf-8-sig")
    rsa.to_csv(OUTPUTS / "comprehensive_component_rsa_subject_results.csv", index=False)
    summary.to_csv(OUTPUTS / "comprehensive_component_rsa_group_stats.csv", index=False)
    rdms.to_csv(OUTPUTS / "comprehensive_component_eeg_rdm_vectors.csv", index=False)
    values.to_csv(OUTPUTS / "comprehensive_component_condition_values.csv", index=False)
    plot_top_results(summary, FIGURES / "comprehensive_component_rsa_top_lpp_results")
    plot_lpp_pz_model_space(summary, FIGURES / "comprehensive_lpp_pz_model_space")
    plot_roi_selection_rationale(summary, FIGURES / "comprehensive_roi_selection_rationale")
    plot_pz_vs_shared_montage(summary, FIGURES / "comprehensive_pz_vs_shared_montage")
    valid_summary = summary[np.isfinite(summary["p_value"])].copy()
    significant_all = valid_summary[valid_summary["q_fdr_all_models"] < 0.05]
    significant_family = valid_summary[valid_summary["q_fdr_family_dimension"] < 0.05]
    metadata = {
        "n_conditions": int(len(condition_table)),
        "n_models": int(len(model_metadata)),
        "shared_montage_channel_count": int(len(shared_channels)),
        "shared_montage_channels": shared_channels,
        "included_subjects": [subject.subj for subject in included],
        "excluded_subjects": qc.loc[~qc["included"], "subj"].tolist(),
        "analyses": analyses,
    }
    (OUTPUTS / "comprehensive_component_rsa_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_lines = [
        "Comprehensive component amplitude EEG RSA",
        "=========================================",
        f"Conditions: {len(condition_table)} (control + 16 edited conditions)",
        f"Models: {len(model_metadata)} (control + all 1/2/3/4-factor hamming and joint-binary models)",
        f"Shared montage channels: {len(shared_channels)}",
        f"Subjects included: {len(included)} ({', '.join(subject.subj for subject in included)})",
        f"Subjects excluded: {', '.join(qc.loc[~qc['included'], 'subj'].astype(str)) or 'None'}",
        f"All-model FDR-significant tests: {len(significant_all)}",
        f"Family/dimension FDR-significant tests: {len(significant_family)}",
    ]
    for _, row in significant_family.sort_values(["q_fdr_family_dimension", "p_value"]).head(30).iterrows():
        log_lines.append(
            f"- {row['analysis']} / {row['metric']} / {row['model']}: "
            f"rho={row['mean_rho']:.4f}, p={row['p_value']:.4g}, "
            f"q_family={row['q_fdr_family_dimension']:.4g}, q_all={row['q_fdr_all_models']:.4g}"
        )
    (LOGS / "comprehensive_component_rsa_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Comprehensive component RSA complete: {PROJECT}")


if __name__ == "__main__":
    main()

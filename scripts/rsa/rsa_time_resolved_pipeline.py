"""Time-resolved EEG representational similarity analysis for face evaluation.

This pipeline uses the EEGLAB epoched stimulus data in this workspace rather
than ERP-window summary tables. It performs:

1. Shared-scalp and posterior-channel EEG RDM construction at every sample.
2. Partial Spearman RSA for four facial factors and two interactions.
3. One-sample group statistics and sign-flipping cluster permutation tests.
4. Publication-ready figures and processing logs.

The experimental event rule supplied with the data is:
    endCode = Stimtype base + CondID
where F_1/F_2/M_1/M_2/T use bases 100/120/140/160/180.
Only F_1, F_2, M_1, and M_2 formal-manipulation epochs are analyzed.
CondID 1 is the original/control face; CondID 2-17 are the 16 factorial
conditions.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy import stats
from scipy.stats import rankdata

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - enables archive-only reruns.
    h5py = None


STIM_BASES = {"F_1": 100, "F_2": 120, "M_1": 140, "M_2": 160, "T": 180}
ANALYZED_STIMTYPES = ("F_1", "F_2", "M_1", "M_2")
FORMAL_RAW_CONDITIONS = tuple(range(2, 18))
MODEL_ORDER = (
    "FSlim",
    "Eye",
    "Mouth",
    "Skin",
    "FSlim_x_Eye",
    "FSlim_x_Eye_x_Skin",
)
BEHAVIOR_ORDER = ("Naturalness_choice", "Beauty_choice")
MODEL_LABELS = {
    "FSlim": "FSlim",
    "Eye": "Eye",
    "Mouth": "Mouth",
    "Skin": "Skin",
    "FSlim_x_Eye": "FSlim x Eye",
    "FSlim_x_Eye_x_Skin": "FSlim x Eye x Skin",
    "Naturalness_choice": "Naturalness choice",
    "Beauty_choice": "Beauty choice",
}
POSTERIOR_CHANNELS = (
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
PLOT_COLORS = {
    "FSlim": "#1b9e77",
    "Eye": "#d95f02",
    "Mouth": "#7570b3",
    "Skin": "#e7298a",
    "FSlim_x_Eye": "#1f78b4",
    "FSlim_x_Eye_x_Skin": "#e31a1c",
    "Naturalness_choice": "#4c78a8",
    "Beauty_choice": "#f58518",
}
SHORT_MODEL_LABELS = {
    "FSlim": "FSlim",
    "Eye": "Eye",
    "Mouth": "Mouth",
    "Skin": "Skin",
    "FSlim_x_Eye": "FSlim\nx Eye",
    "FSlim_x_Eye_x_Skin": "FSlim x Eye\nx Skin",
    "Naturalness_choice": "Naturalness",
    "Beauty_choice": "Beauty",
}


@dataclass
class EegSubject:
    """Metadata and condition-averaged patterns from one EEGLAB dataset."""

    subj: str
    times: np.ndarray
    channels: list[str]
    patterns: np.ndarray | None
    formal_epoch_counts: dict[int, int]
    epoch_count: int
    formal_epoch_count: int
    excluded_reason: str = ""


def natural_subject_key(subject: str) -> int:
    """Return the integer component of identifiers such as ``s12``."""

    match = re.search(r"(\d+)", str(subject))
    return int(match.group(1)) if match else 10**9


def find_eprime_files(eprime_root: Path) -> list[Path]:
    """Locate E-Prime text exports recursively."""

    return sorted(eprime_root.rglob("*.txt"))


def read_text_with_fallback(path: Path) -> str:
    """Read a source text export using common encodings in this dataset."""

    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_eprime_records(eprime_root: Path) -> pd.DataFrame:
    """Extract trial-level factorial variables and ratings from E-Prime logs.

    E-Prime logs contain nested log frames, including incomplete parent frames.
    Only rows with a condition, all four facial factors, stimulus identity, and
    both ratings are retained.
    """

    keys = [
        "CondID",
        "FSlim",
        "Eye",
        "Mouth",
        "Skin",
        "Stimtype",
        "Gender",
        "Picture",
        "Original",
        "RealnessRating.RESP",
        "RealnessRating.RT",
        "LikingRating.RESP",
        "LikingRating.RT",
    ]
    rows: list[dict[str, object]] = []
    for path in find_eprime_files(eprime_root):
        text = read_text_with_fallback(path)
        subj_match = re.search(r"^Subject:\s*(\d+)\s*$", text, flags=re.MULTILINE)
        subject = f"s{subj_match.group(1)}" if subj_match else path.parent.name.lower()
        frames = re.findall(
            r"\*\*\* LogFrame Start \*\*\*(.*?)\*\*\* LogFrame End \*\*\*",
            text,
            flags=re.DOTALL,
        )
        for frame in frames:
            row: dict[str, object] = {"subj": subject, "source_file": str(path)}
            for key in keys:
                match = re.search(
                    rf"^\s*{re.escape(key)}:\s*(.*?)\s*$",
                    frame,
                    flags=re.MULTILINE,
                )
                row[key] = match.group(1).strip() if match else np.nan
            required = ["CondID", "FSlim", "Eye", "Mouth", "Skin", "Stimtype"]
            if any(pd.isna(row[key]) for key in required):
                continue
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No usable E-Prime trial rows found under {eprime_root}")
    records = pd.DataFrame(rows)
    numeric = [
        "CondID",
        "FSlim",
        "Eye",
        "Mouth",
        "Skin",
        "Original",
        "RealnessRating.RESP",
        "RealnessRating.RT",
        "LikingRating.RESP",
        "LikingRating.RT",
    ]
    for column in numeric:
        records[column] = pd.to_numeric(records[column], errors="coerce")
    records = records[
        records["Stimtype"].isin(ANALYZED_STIMTYPES)
        & records["CondID"].isin(FORMAL_RAW_CONDITIONS)
    ].copy()
    return records.reset_index(drop=True)


def standardize_factor_coding(records: pd.DataFrame) -> pd.DataFrame:
    """Effect-code the four two-level factors as -1 and +1."""

    coded = records.copy()
    maps = {
        "FSlim": {0: -1, 1: 1},
        "Skin": {0: -1, 1: 1},
        "Eye": {1: -1, 2: 1},
        "Mouth": {1: -1, 2: 1},
    }
    for factor, mapping in maps.items():
        unexpected = set(coded[factor].dropna().unique()) - set(mapping)
        if unexpected:
            raise ValueError(f"Unexpected {factor} levels: {sorted(unexpected)}")
        coded[factor] = coded[factor].map(mapping)
    coded["CondID"] = coded["CondID"].astype(int) - 1
    coded["raw_CondID"] = coded["CondID"] + 1
    return coded


def make_condition_table(records: pd.DataFrame) -> pd.DataFrame:
    """Create and validate the fixed 16-condition factorial design table."""

    columns = ["CondID", "raw_CondID", "FSlim", "Eye", "Mouth", "Skin"]
    table = records[columns].drop_duplicates().sort_values("CondID").reset_index(drop=True)
    if len(table) != 16 or table["CondID"].tolist() != list(range(1, 17)):
        raise ValueError("The E-Prime mapping does not yield exactly 16 formal conditions.")
    if table[["FSlim", "Eye", "Mouth", "Skin"]].duplicated().any():
        raise ValueError("Duplicate factor combinations found in the formal condition table.")
    table["FSlim_x_Eye"] = table["FSlim"] * table["Eye"]
    table["FSlim_x_Eye_x_Skin"] = table["FSlim"] * table["Eye"] * table["Skin"]
    return table


def decode_hdf5_reference(dataset: h5py.File, ref: h5py.Reference) -> object:
    """Decode a scalar/string MATLAB v7.3 object referenced in an EEGLAB file."""

    array = np.asarray(dataset[ref][()]).squeeze()
    if array.dtype.kind in "ui" and np.size(array) >= 1:
        text = "".join(chr(int(value)) for value in np.ravel(array, order="F") if int(value))
        if text:
            return text
    return np.asarray(array).squeeze().item()


def decode_endcode(code: object) -> tuple[str, int] | tuple[None, None]:
    """Decode an E-Prime endCode into stimulus identity and raw condition ID."""

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
    """Read EEGLAB metadata and epoch endCodes from a MATLAB v7.3 ``.set`` file."""

    with h5py.File(set_path, "r") as dataset:
        n_channels = int(dataset["nbchan"][0, 0])
        n_points = int(dataset["pnts"][0, 0])
        n_trials = int(dataset["trials"][0, 0])
        sample_rate = float(dataset["srate"][0, 0])
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
        "sample_rate": sample_rate,
        "times": times,
        "channels": channels,
        "endcodes": endcodes,
        "fdt_path": set_path.parent / datfile,
    }


def find_eeg_set_files(data_root: Path) -> list[Path]:
    """Return one epoched stimulus EEGLAB file for each subject."""

    files = list(data_root.glob("derivatives_eeglab_s*/s*_epoched_stim.set"))
    return sorted(files, key=lambda path: natural_subject_key(path.parent.name))


def load_eeg_data(
    set_path: Path,
    condition_table: pd.DataFrame,
) -> EegSubject:
    """Load one subject and compute condition-average full-scalp time patterns.

    Data are read from the EEGLAB external float32 ``.fdt`` array in MATLAB's
    channel-fast ordering, exposed here as epoch x time x channel. The saved
    preprocessed epochs are used as stored; no additional signal transform is
    introduced by the RSA pipeline.
    """

    subj_match = re.search(r"s(\d+)", set_path.stem.lower())
    subj = f"s{subj_match.group(1)}" if subj_match else set_path.stem
    header = read_set_header(set_path)
    decoded = [decode_endcode(code) for code in header["endcodes"]]
    raw_conditions = condition_table["raw_CondID"].tolist()
    indices_by_condition: dict[int, list[int]] = {condition: [] for condition in raw_conditions}
    for index, (stimtype, condition) in enumerate(decoded):
        if stimtype in ANALYZED_STIMTYPES and condition in indices_by_condition:
            indices_by_condition[condition].append(index)
    counts = {condition: len(indices) for condition, indices in indices_by_condition.items()}
    formal_epoch_count = int(sum(counts.values()))
    missing = [condition for condition, count in counts.items() if count == 0]
    if missing:
        return EegSubject(
            subj=subj,
            times=header["times"],
            channels=header["channels"],
            patterns=None,
            formal_epoch_counts=counts,
            epoch_count=int(header["n_trials"]),
            formal_epoch_count=formal_epoch_count,
            excluded_reason=f"Missing formal raw CondID: {missing}",
        )
    fdt_path = Path(header["fdt_path"])
    if not fdt_path.exists():
        raise FileNotFoundError(f"Missing FDT file for {subj}: {fdt_path}")
    data = np.memmap(
        fdt_path,
        dtype="<f4",
        mode="r",
        shape=(int(header["n_trials"]), int(header["n_points"]), int(header["n_channels"])),
    )
    times = np.asarray(header["times"], dtype=float)
    patterns: list[np.ndarray] = []
    for raw_condition in raw_conditions:
        epochs = np.asarray(data[indices_by_condition[raw_condition], :, :], dtype=np.float64)
        patterns.append(epochs.mean(axis=0))
    return EegSubject(
        subj=subj,
        times=times,
        channels=list(header["channels"]),
        patterns=np.stack(patterns, axis=0),
        formal_epoch_counts=counts,
        epoch_count=int(header["n_trials"]),
        formal_epoch_count=formal_epoch_count,
    )


def vectorize_rdm(rdm: np.ndarray) -> np.ndarray:
    """Extract the strict upper triangle of an RDM or RDM time series."""

    indices = np.triu_indices(rdm.shape[-1], k=1)
    return rdm[..., indices[0], indices[1]]


def compute_eeg_rdm(patterns: np.ndarray) -> np.ndarray:
    """Compute time-resolved EEG RDM vectors using 1 - Pearson correlation.

    Parameters
    ----------
    patterns:
        Array shaped ``conditions x time x channels``.
    """

    time_patterns = np.transpose(patterns, (1, 0, 2))
    centered = time_patterns - time_patterns.mean(axis=2, keepdims=True)
    norms = np.sqrt(np.sum(centered**2, axis=2, keepdims=True))
    normalized = np.divide(
        centered,
        norms,
        out=np.zeros_like(centered),
        where=norms > np.finfo(float).eps,
    )
    correlation = np.einsum("tic,tjc->tij", normalized, normalized)
    correlation = np.clip(correlation, -1.0, 1.0)
    return vectorize_rdm(1.0 - correlation)


def make_model_rdms(condition_table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Construct four factor and two interaction model RDM matrices."""

    models: dict[str, np.ndarray] = {}
    for model in MODEL_ORDER:
        values = condition_table[model].to_numpy()
        if model in {"FSlim", "Eye", "Mouth", "Skin"}:
            models[model] = (values[:, None] != values[None, :]).astype(float)
        else:
            models[model] = np.abs(values[:, None] - values[None, :]).astype(float)
    return models


def make_behavior_condition_choices(
    records: pd.DataFrame, subjects: Sequence[str]
) -> pd.DataFrame:
    """Average Naturalness and Beauty choices for every subject and condition.

    The E-Prime task stores Naturalness as ``RealnessRating.RESP`` and Beauty
    or liking as ``LikingRating.RESP``. Ratings are kept as the participant's
    actual button choices and averaged across repeated stimuli within each of
    the 16 formal conditions.
    """

    response_columns = {
        "RealnessRating.RESP": "Naturalness_choice",
        "LikingRating.RESP": "Beauty_choice",
    }
    available = records[records["subj"].isin(subjects)].copy()
    for source in response_columns:
        available[source] = pd.to_numeric(available[source], errors="coerce")
    grouped = (
        available.groupby(["subj", "CondID"], as_index=False)
        .agg(
            Naturalness_choice=("RealnessRating.RESP", "mean"),
            Beauty_choice=("LikingRating.RESP", "mean"),
            n_naturalness_choices=("RealnessRating.RESP", "count"),
            n_beauty_choices=("LikingRating.RESP", "count"),
        )
        .sort_values(["subj", "CondID"], key=lambda col: col.map(natural_subject_key) if col.name == "subj" else col)
    )
    return grouped.reset_index(drop=True)


def make_behavior_rdm_vectors(
    behavior_choices: pd.DataFrame, subjects: Sequence[str]
) -> dict[str, np.ndarray]:
    """Build subject-specific behavior RDM vectors from choice averages."""

    vectors: dict[str, list[np.ndarray]] = {behavior: [] for behavior in BEHAVIOR_ORDER}
    for subject in subjects:
        subject_table = (
            behavior_choices[behavior_choices["subj"] == subject]
            .set_index("CondID")
            .reindex(range(1, 17))
        )
        for behavior in BEHAVIOR_ORDER:
            values = subject_table[behavior].to_numpy(dtype=float)
            if np.isnan(values).any():
                vectors[behavior].append(np.full(120, np.nan, dtype=float))
            else:
                vectors[behavior].append(vectorize_rdm(np.abs(values[:, None] - values[None, :])))
    return {behavior: np.vstack(values) for behavior, values in vectors.items()}


def behavior_rdm_vectors_to_long(
    behavior_rdms: Mapping[str, np.ndarray], subjects: Sequence[str]
) -> pd.DataFrame:
    """Convert behavior RDM vectors to a condition-pair long table."""

    rows: list[dict[str, object]] = []
    pair_i, pair_j = np.triu_indices(16, 1)
    for behavior, values in behavior_rdms.items():
        for subject_index, subject in enumerate(subjects):
            for i, j, distance in zip(pair_i + 1, pair_j + 1, values[subject_index]):
                rows.append(
                    {
                        "subj": subject,
                        "behavior": behavior,
                        "condition_i": int(i),
                        "condition_j": int(j),
                        "distance": float(distance),
                    }
                )
    return pd.DataFrame(rows)


def run_behavior_model_rsa(
    behavior_rdms: Mapping[str, np.ndarray],
    subjects: Sequence[str],
    model_rdms: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """RSA between behavioral choice RDMs and the factorial model RDMs."""

    model_vectors = {name: vectorize_rdm(matrix) for name, matrix in model_rdms.items()}
    rows: list[dict[str, object]] = []
    for behavior, behavior_vectors in behavior_rdms.items():
        for subject_index, subject in enumerate(subjects):
            subject_rdm = behavior_vectors[subject_index]
            if np.isnan(subject_rdm).any():
                continue
            for model in MODEL_ORDER:
                target = model_vectors[model]
                other_models = [name for name in MODEL_ORDER if name != model]
                covariates = np.column_stack([model_vectors[name] for name in other_models])
                rows.append(
                    {
                        "subj": subject,
                        "behavior": behavior,
                        "model": model,
                        "rho": float(spearman_rsa(subject_rdm, target)[0]),
                        "partial_rho": float(partial_spearman(subject_rdm, target, covariates)[0]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_behavior_model_rsa(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize behavior-model RSA across subjects."""

    rows: list[dict[str, object]] = []
    for (behavior, model), subset in results.groupby(["behavior", "model"], sort=False):
        for value_column in ("rho", "partial_rho"):
            values = subset[value_column].dropna().to_numpy(dtype=float)
            t_value, p_value = stats.ttest_1samp(values, popmean=0.0) if len(values) > 1 else (np.nan, np.nan)
            mean = float(np.mean(values)) if len(values) else np.nan
            sem = float(stats.sem(values)) if len(values) > 1 else np.nan
            critical = float(stats.t.ppf(0.975, len(values) - 1)) if len(values) > 1 else np.nan
            rows.append(
                {
                    "behavior": behavior,
                    "model": model,
                    "metric": value_column,
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
    summary["p_fdr_six_models"] = summary.groupby(["behavior", "metric"])["p_value"].transform(
        lambda values: fdr_bh(values.to_numpy(dtype=float))
    )
    return summary.sort_values(["behavior", "metric", "model"]).reset_index(drop=True)


def run_eeg_behavior_rsa(
    rdm_vectors: np.ndarray,
    subjects: Sequence[str],
    times: np.ndarray,
    behavior_rdms: Mapping[str, np.ndarray],
    analysis: str,
) -> pd.DataFrame:
    """Time-resolved RSA between EEG RDMs and subject-specific behavior RDMs."""

    curves: dict[str, list[np.ndarray]] = {behavior: [] for behavior in BEHAVIOR_ORDER}
    for subject_index, _subject in enumerate(subjects):
        subject_eeg_rdm = rdm_vectors[subject_index]
        for behavior in BEHAVIOR_ORDER:
            target = behavior_rdms[behavior][subject_index]
            if np.isnan(target).any():
                curves[behavior].append(np.full(len(times), np.nan, dtype=float))
            else:
                curves[behavior].append(spearman_rsa(subject_eeg_rdm, target))
    arrays = {behavior: np.vstack(values) for behavior, values in curves.items()}
    return curves_to_long(subjects, times, arrays, "rho", analysis)


def rank_rows(values: np.ndarray) -> np.ndarray:
    """Rank-transform each row of a two-dimensional matrix."""

    return rankdata(values, axis=1, method="average")


def pearson_rows(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation between each row and one target vector."""

    centered_values = values - values.mean(axis=1, keepdims=True)
    centered_target = target - target.mean()
    denominator = np.sqrt(np.sum(centered_values**2, axis=1) * np.sum(centered_target**2))
    return np.divide(
        centered_values @ centered_target,
        denominator,
        out=np.full(values.shape[0], np.nan),
        where=denominator > np.finfo(float).eps,
    )


def partial_spearman(
    x: np.ndarray, y: np.ndarray, covariates: np.ndarray | None = None
) -> np.ndarray:
    """Compute partial Spearman rho for one or multiple RDM vectors.

    ``x`` may contain a row for every time point. The function rank-transforms
    x, y, and all covariates; residualizes x and y on the ranked covariates;
    and computes Pearson correlations between residuals.
    """

    x_matrix = np.atleast_2d(np.asarray(x, dtype=float))
    y_vector = rankdata(np.asarray(y, dtype=float), method="average")
    x_ranked = rank_rows(x_matrix)
    if covariates is None or np.asarray(covariates).size == 0:
        return pearson_rows(x_ranked, y_vector)
    covariate_matrix = np.asarray(covariates, dtype=float)
    if covariate_matrix.ndim == 1:
        covariate_matrix = covariate_matrix[:, None]
    ranked_covariates = np.column_stack(
        [rankdata(covariate_matrix[:, index], method="average") for index in range(covariate_matrix.shape[1])]
    )
    design = np.column_stack([np.ones(len(y_vector)), ranked_covariates])
    projection = design @ np.linalg.pinv(design)
    y_residual = y_vector - projection @ y_vector
    x_residual = x_ranked - (projection @ x_ranked.T).T
    return pearson_rows(x_residual, y_residual)


def spearman_rsa(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Spearman RSA rho for every time row in ``x``."""

    return pearson_rows(rank_rows(np.atleast_2d(x)), rankdata(y, method="average"))


def curves_to_long(
    subjects: Sequence[str],
    times: np.ndarray,
    curves: Mapping[str, np.ndarray],
    value_column: str,
    analysis: str,
) -> pd.DataFrame:
    """Convert model-by-subject time-course matrices to a tidy table."""

    frames: list[pd.DataFrame] = []
    for model, values in curves.items():
        frame = pd.DataFrame(values, index=subjects, columns=times)
        frame.index.name = "subj"
        long = frame.reset_index().melt(id_vars="subj", var_name="time_ms", value_name=value_column)
        long["model"] = model
        long["analysis"] = analysis
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def run_subject_level_rsa(
    rdm_vectors: np.ndarray,
    subjects: Sequence[str],
    times: np.ndarray,
    model_rdms: Mapping[str, np.ndarray],
    analysis: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ordinary and partial model-RDM RSA for each subject and sample."""

    model_vectors = {name: vectorize_rdm(matrix) for name, matrix in model_rdms.items()}
    ordinary: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_ORDER}
    partial: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_ORDER}
    for subject_index, _subject in enumerate(subjects):
        subject_rdm = rdm_vectors[subject_index]
        for model in MODEL_ORDER:
            target = model_vectors[model]
            other_models = [name for name in MODEL_ORDER if name != model]
            covariates = np.column_stack([model_vectors[name] for name in other_models])
            ordinary[model].append(spearman_rsa(subject_rdm, target))
            partial[model].append(partial_spearman(subject_rdm, target, covariates))
    ordinary_arrays = {name: np.vstack(values) for name, values in ordinary.items()}
    partial_arrays = {name: np.vstack(values) for name, values in partial.items()}
    return (
        curves_to_long(subjects, times, ordinary_arrays, "rho", analysis),
        curves_to_long(subjects, times, partial_arrays, "partial_rho", analysis),
    )


def run_group_statistics(results: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Run one-sample t-tests against zero for each model and sample."""

    rows: list[dict[str, object]] = []
    for (analysis, model, time_ms), subset in results.groupby(["analysis", "model", "time_ms"], sort=False):
        values = subset[value_column].dropna().to_numpy(dtype=float)
        t_value, p_value = stats.ttest_1samp(values, popmean=0.0) if len(values) > 1 else (np.nan, np.nan)
        rows.append(
            {
                "analysis": analysis,
                "model": model,
                "time_ms": float(time_ms),
                "n_subjects": len(values),
                "mean_rho": float(np.nanmean(values)) if len(values) else np.nan,
                "sem_rho": float(stats.sem(values)) if len(values) > 1 else np.nan,
                "t_value": float(t_value),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis", "model", "time_ms"]).reset_index(drop=True)


def fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted p values."""

    p_array = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_array.shape, np.nan, dtype=float)
    valid = np.isfinite(p_array)
    if not valid.any():
        return adjusted
    valid_p = p_array[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    n_values = len(ranked)
    corrected = ranked * n_values / np.arange(1, n_values + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.minimum(corrected, 1.0)
    restored = np.empty_like(corrected)
    restored[order] = corrected
    adjusted[valid] = restored
    return adjusted


def summarize_n170_window(
    results: pd.DataFrame,
    value_column: str = "partial_rho",
    window: tuple[float, float] = (140.0, 190.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize participant-level mean RSA within the predefined N170 window."""

    selected = results[results["time_ms"].between(window[0], window[1])].copy()
    participant_means = (
        selected.groupby(["analysis", "model", "subj"], as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "n170_mean_partial_rho"})
    )
    rows: list[dict[str, object]] = []
    for (analysis, model), subset in participant_means.groupby(["analysis", "model"], sort=False):
        values = subset["n170_mean_partial_rho"].dropna().to_numpy(dtype=float)
        t_value, p_value = stats.ttest_1samp(values, popmean=0.0) if len(values) > 1 else (np.nan, np.nan)
        sem = float(stats.sem(values)) if len(values) > 1 else np.nan
        critical = float(stats.t.ppf(0.975, len(values) - 1)) if len(values) > 1 else np.nan
        mean = float(np.mean(values)) if len(values) else np.nan
        rows.append(
            {
                "analysis": analysis,
                "model": model,
                "window_start_ms": window[0],
                "window_end_ms": window[1],
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
    summary["p_fdr_six_models"] = summary.groupby("analysis")["p_value"].transform(
        lambda values: fdr_bh(values.to_numpy(dtype=float))
    )
    return participant_means, summary.sort_values(["analysis", "model"]).reset_index(drop=True)


def summarize_fslim_targeted_n170(
    results: pd.DataFrame,
    value_column: str,
    metric: str,
    window: tuple[float, float] = (140.0, 190.0),
) -> pd.DataFrame:
    """Summarize the theory-driven FSlim N170 test for one RSA metric."""

    selected = results[
        (results["model"] == "FSlim") & results["time_ms"].between(window[0], window[1])
    ].copy()
    participant_means = (
        selected.groupby(["analysis", "subj"], as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "n170_mean_rho"})
    )
    rows: list[dict[str, object]] = []
    for analysis, subset in participant_means.groupby("analysis", sort=False):
        values = subset["n170_mean_rho"].dropna().to_numpy(dtype=float)
        t_value, p_two_sided = stats.ttest_1samp(values, popmean=0.0) if len(values) > 1 else (np.nan, np.nan)
        mean = float(np.mean(values)) if len(values) else np.nan
        sem = float(stats.sem(values)) if len(values) > 1 else np.nan
        critical = float(stats.t.ppf(0.975, len(values) - 1)) if len(values) > 1 else np.nan
        if np.isfinite(t_value):
            p_one_tailed_positive = float(p_two_sided / 2.0) if t_value > 0 else float(1.0 - p_two_sided / 2.0)
        else:
            p_one_tailed_positive = np.nan
        rows.append(
            {
                "analysis": analysis,
                "model": "FSlim",
                "metric": metric,
                "window_start_ms": window[0],
                "window_end_ms": window[1],
                "n_subjects": len(values),
                "mean_rho": mean,
                "sem_rho": sem,
                "ci95_low": mean - critical * sem if len(values) > 1 else np.nan,
                "ci95_high": mean + critical * sem if len(values) > 1 else np.nan,
                "t_value": float(t_value),
                "p_two_sided": float(p_two_sided),
                "p_one_tailed_positive": p_one_tailed_positive,
                "cohens_d": float(mean / np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_component_window(
    results: pd.DataFrame,
    value_column: str,
    window_name: str,
    window: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize any participant-level RSA time course within a named window."""

    selected = results[results["time_ms"].between(window[0], window[1])].copy()
    participant_means = (
        selected.groupby(["analysis", "model", "subj"], as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "window_mean_rho"})
    )
    participant_means["window"] = window_name
    participant_means["window_start_ms"] = window[0]
    participant_means["window_end_ms"] = window[1]
    rows: list[dict[str, object]] = []
    for (analysis, model), subset in participant_means.groupby(["analysis", "model"], sort=False):
        values = subset["window_mean_rho"].dropna().to_numpy(dtype=float)
        t_value, p_value = stats.ttest_1samp(values, popmean=0.0) if len(values) > 1 else (np.nan, np.nan)
        mean = float(np.mean(values)) if len(values) else np.nan
        sem = float(stats.sem(values)) if len(values) > 1 else np.nan
        critical = float(stats.t.ppf(0.975, len(values) - 1)) if len(values) > 1 else np.nan
        rows.append(
            {
                "analysis": analysis,
                "model": model,
                "window": window_name,
                "window_start_ms": window[0],
                "window_end_ms": window[1],
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
    summary["p_fdr_within_analysis"] = summary.groupby(["analysis", "window"])["p_value"].transform(
        lambda values: fdr_bh(values.to_numpy(dtype=float))
    )
    return (
        participant_means.sort_values(["analysis", "model", "subj"]).reset_index(drop=True),
        summary.sort_values(["analysis", "model"]).reset_index(drop=True),
    )


def _one_sample_t(values: np.ndarray) -> np.ndarray:
    """Vectorized one-sample t statistic for subject x time data."""

    n_subjects = values.shape[0]
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=1)
    denominator = std / np.sqrt(n_subjects)
    return np.divide(mean, denominator, out=np.zeros_like(mean), where=denominator > 0)


def _clusters_from_t(t_values: np.ndarray, threshold: float) -> list[tuple[int, int, float, str]]:
    """Identify same-direction contiguous supra-threshold clusters."""

    clusters: list[tuple[int, int, float, str]] = []
    for sign, direction in ((1, "positive"), (-1, "negative")):
        mask = sign * t_values > threshold
        padded = np.concatenate([[False], mask, [False]])
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
        for start, end in zip(starts, ends):
            mass = float(np.sum(np.abs(t_values[start : end + 1])))
            clusters.append((int(start), int(end), mass, direction))
    return sorted(clusters, key=lambda item: item[0])


def cluster_permutation_1d(
    data: np.ndarray,
    times: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run a two-sided sign-flipping cluster permutation test.

    A single random sign is assigned to each participant's complete time
    course on every permutation. Clusters consist of adjacent points exceeding
    the two-sided uncorrected ``alpha`` threshold in the same direction.
    Cluster mass is the sum of absolute t statistics.
    """

    values = np.asarray(data, dtype=float)
    valid_subjects = ~np.isnan(values).any(axis=1)
    values = values[valid_subjects]
    if values.shape[0] < 2:
        return pd.DataFrame()
    threshold = float(stats.t.ppf(1.0 - alpha / 2.0, df=values.shape[0] - 1))
    observed_t = _one_sample_t(values)
    observed_clusters = _clusters_from_t(observed_t, threshold)
    if not observed_clusters:
        return pd.DataFrame(
            columns=[
                "start_time",
                "end_time",
                "cluster_mass",
                "direction",
                "cluster_p",
                "significant",
                "threshold_t",
                "n_subjects",
                "n_permutations",
            ]
        )
    rng = np.random.default_rng(seed)
    null_maxima = np.zeros(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=(values.shape[0], 1))
        permuted_t = _one_sample_t(values * signs)
        clusters = _clusters_from_t(permuted_t, threshold)
        null_maxima[permutation] = max((cluster[2] for cluster in clusters), default=0.0)
    rows: list[dict[str, object]] = []
    for start, end, mass, direction in observed_clusters:
        p_value = (1.0 + float(np.sum(null_maxima >= mass))) / (n_permutations + 1.0)
        rows.append(
            {
                "start_time": float(times[start]),
                "end_time": float(times[end]),
                "cluster_mass": mass,
                "direction": direction,
                "cluster_p": p_value,
                "significant": bool(p_value < alpha),
                "threshold_t": threshold,
                "n_subjects": int(values.shape[0]),
                "n_permutations": int(n_permutations),
            }
        )
    return pd.DataFrame(rows)


def run_cluster_tests(
    results: pd.DataFrame,
    value_column: str,
    inference_window: tuple[float, float],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Run cluster inference for every analysis/model time course."""

    rows: list[pd.DataFrame] = []
    for index, ((analysis, model), subset) in enumerate(results.groupby(["analysis", "model"], sort=False)):
        pivot = subset.pivot(index="subj", columns="time_ms", values=value_column).sort_index(axis=1)
        selected_times = pivot.columns.to_numpy(dtype=float)
        mask = (selected_times >= inference_window[0]) & (selected_times <= inference_window[1])
        clusters = cluster_permutation_1d(
            pivot.to_numpy(dtype=float)[:, mask],
            selected_times[mask],
            n_permutations=n_permutations,
            seed=seed + index,
        )
        if clusters.empty:
            clusters = pd.DataFrame(
                [
                    {
                        "start_time": np.nan,
                        "end_time": np.nan,
                        "cluster_mass": np.nan,
                        "direction": "none",
                        "cluster_p": np.nan,
                        "significant": False,
                        "threshold_t": np.nan,
                        "n_subjects": pivot.shape[0],
                        "n_permutations": n_permutations,
                    }
                ]
            )
        clusters.insert(0, "model", model)
        clusters.insert(0, "analysis", analysis)
        rows.append(clusters)
    return pd.concat(rows, ignore_index=True)


def plot_model_rdms(model_rdms: Mapping[str, np.ndarray], output_dir: Path) -> None:
    """Plot the six factorial model RDM matrices."""

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), constrained_layout=True)
    vmax = max(float(np.max(model_rdms[model])) for model in MODEL_ORDER)
    for axis, model in zip(axes.ravel(), MODEL_ORDER):
        image = axis.imshow(model_rdms[model], cmap="viridis", norm=Normalize(0, vmax))
        axis.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        axis.set_xlabel("Condition")
        axis.set_ylabel("Condition")
        axis.set_xticks([0, 4, 8, 12, 15], [1, 5, 9, 13, 16])
        axis.set_yticks([0, 4, 8, 12, 15], [1, 5, 9, 13, 16])
    fig.colorbar(image, ax=axes, shrink=0.85, label="Model distance")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"model_rdms.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def vector_to_rdm(vector: np.ndarray, size: int = 16) -> np.ndarray:
    """Restore a symmetric RDM matrix from an upper-triangle vector."""

    matrix = np.zeros((size, size), dtype=float)
    indices = np.triu_indices(size, 1)
    matrix[indices] = vector
    matrix[(indices[1], indices[0])] = vector
    return matrix


def plot_behavior_rdms(behavior_rdms: Mapping[str, np.ndarray], output_dir: Path) -> None:
    """Plot group-average Naturalness and Beauty choice RDMs."""

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    matrices = {
        behavior: vector_to_rdm(np.nanmean(values, axis=0))
        for behavior, values in behavior_rdms.items()
    }
    vmax = max(float(np.nanmax(matrix)) for matrix in matrices.values())
    image = None
    for axis, behavior in zip(axes, BEHAVIOR_ORDER):
        image = axis.imshow(matrices[behavior], cmap="magma", norm=Normalize(0, vmax))
        axis.set_title(MODEL_LABELS[behavior], fontsize=11, fontweight="bold")
        axis.set_xlabel("Condition")
        axis.set_ylabel("Condition")
        axis.set_xticks([0, 4, 8, 12, 15], [1, 5, 9, 13, 16])
        axis.set_yticks([0, 4, 8, 12, 15], [1, 5, 9, 13, 16])
    fig.colorbar(image, ax=axes, shrink=0.82, label="Mean absolute choice difference")
    fig.suptitle("Behavioral choice RDMs", fontsize=13, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"behavior_choice_rdms.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_behavior_model_rsa(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot partial RSA between behavioral choice RDMs and factorial models."""

    display = summary[summary["metric"] == "partial_rho"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True, constrained_layout=True)
    for axis, behavior in zip(axes, BEHAVIOR_ORDER):
        section = display[display["behavior"] == behavior].set_index("model").reindex(MODEL_ORDER)
        x = np.arange(len(MODEL_ORDER))
        means = section["mean_rho"].to_numpy(dtype=float)
        errors_low = means - section["ci95_low"].to_numpy(dtype=float)
        errors_high = section["ci95_high"].to_numpy(dtype=float) - means
        colors = [PLOT_COLORS[model] for model in MODEL_ORDER]
        axis.bar(x, means, color=colors, alpha=0.84, width=0.68)
        axis.errorbar(
            x,
            means,
            yerr=np.vstack([errors_low, errors_high]),
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3,
        )
        for index, (model, row) in enumerate(section.iterrows()):
            q_value = float(row["p_fdr_six_models"])
            if q_value < 0.001:
                marker = "***"
            elif q_value < 0.01:
                marker = "**"
            elif q_value < 0.05:
                marker = "*"
            else:
                marker = ""
            if marker:
                offset = 0.018 if means[index] >= 0 else -0.026
                va = "bottom" if means[index] >= 0 else "top"
                axis.text(index, means[index] + offset, marker, ha="center", va=va, fontsize=12)
        axis.axhline(0, color="#333333", lw=0.8)
        axis.set_xticks(x, [SHORT_MODEL_LABELS[model] for model in MODEL_ORDER], rotation=0)
        axis.set_title(MODEL_LABELS[behavior], fontsize=12, fontweight="bold")
        axis.set_xlabel("Factorial model RDM")
        axis.grid(axis="y", color="#ededed", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Partial Spearman rho")
    fig.suptitle(
        "Behavioral choice RDMs align with facial-factor model RDMs\n"
        "Error bars show 95% CI; stars mark six-model FDR q < .05",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"behavior_model_partial_rsa.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_curves(
    stats_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    models: Sequence[str],
    title: str,
    output_stem: Path,
    x_limits: tuple[float, float] = (-100.0, 600.0),
    y_label: str = "Partial Spearman rho",
    window_span: tuple[float, float] | None = (140.0, 190.0),
    window_label: str = "N170 (140-190 ms)",
) -> None:
    """Render a paper-style time-course figure with cluster annotations."""

    fig, axis = plt.subplots(figsize=(10.2, 5.5))
    if window_span is not None:
        axis.axvspan(window_span[0], window_span[1], color="#d9d9d9", alpha=0.36, label=window_label)
    axis.axhline(0, color="#333333", lw=0.8)
    display = stats_table[
        stats_table["model"].isin(models)
        & stats_table["time_ms"].between(x_limits[0], x_limits[1])
    ]
    for model in models:
        curve = display[display["model"] == model].sort_values("time_ms")
        if curve.empty:
            continue
        time = curve["time_ms"].to_numpy()
        mean = curve["mean_rho"].to_numpy()
        sem = curve["sem_rho"].to_numpy()
        color = PLOT_COLORS.get(model, "#333333")
        axis.plot(time, mean, color=color, lw=1.8, label=MODEL_LABELS[model])
        axis.fill_between(time, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0)
    visible_y = display["mean_rho"].to_numpy()
    spread = float(np.nanmax(visible_y) - np.nanmin(visible_y)) if visible_y.size else 0.1
    line_step = max(spread * 0.065, 0.003)
    base = float(np.nanmin(visible_y)) - line_step * 1.6 if visible_y.size else -0.05
    for row_index, model in enumerate(models):
        clusters = cluster_table[
            (cluster_table["model"] == model) & (cluster_table["significant"] == True)  # noqa: E712
        ]
        y = base - row_index * line_step
        for _, cluster in clusters.iterrows():
            axis.plot(
                [cluster["start_time"], cluster["end_time"]],
                [y, y],
                color=PLOT_COLORS.get(model, "#333333"),
                lw=3.0,
                solid_capstyle="butt",
            )
    if not cluster_table[cluster_table["significant"] == True].empty:  # noqa: E712
        axis.text(x_limits[0] + 5, base + line_step * 0.35, "cluster p < .05", fontsize=8)
    axis.set_xlim(*x_limits)
    axis.set_xlabel("Time from stimulus onset (ms)")
    axis.set_ylabel(y_label)
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#ededed", linewidth=0.7)
    axis.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_time_resolved_rsa(
    stats_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    title_prefix: str,
) -> None:
    """Plot factor, interaction, and key-factor partial RSA curves."""

    _plot_curves(
        stats_table,
        cluster_table,
        ["FSlim", "Eye", "Mouth", "Skin"],
        f"{title_prefix}: facial factor representations",
        output_dir / f"{prefix}_factor_partial_rsa",
    )
    _plot_curves(
        stats_table,
        cluster_table,
        ["FSlim_x_Eye", "FSlim_x_Eye_x_Skin"],
        f"{title_prefix}: interaction representations",
        output_dir / f"{prefix}_interaction_partial_rsa",
    )
    _plot_curves(
        stats_table,
        cluster_table,
        ["FSlim", "FSlim_x_Eye", "FSlim_x_Eye_x_Skin"],
        f"{title_prefix}: key hypotheses",
        output_dir / f"{prefix}_key_partial_rsa",
    )
    _plot_curves(
        stats_table,
        cluster_table,
        ["FSlim", "Eye", "Mouth", "Skin"],
        f"{title_prefix}: N170 facial factors",
        output_dir / f"{prefix}_n170_factor_partial_rsa",
        x_limits=(120.0, 215.0),
    )
    _plot_curves(
        stats_table,
        cluster_table,
        ["FSlim_x_Eye", "FSlim_x_Eye_x_Skin"],
        f"{title_prefix}: N170 interactions",
        output_dir / f"{prefix}_n170_interaction_partial_rsa",
        x_limits=(120.0, 215.0),
    )


def plot_eeg_behavior_rsa(
    stats_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    title_prefix: str,
) -> None:
    """Plot EEG-to-behavior time-resolved RSA curves."""

    _plot_curves(
        stats_table,
        cluster_table,
        BEHAVIOR_ORDER,
        f"{title_prefix}: EEG-behavior representational alignment",
        output_dir / f"{prefix}_eeg_behavior_rsa",
        y_label="Spearman rho",
    )


def plot_lpp_eeg_behavior_rsa(
    stats_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    title_prefix: str,
) -> None:
    """Plot EEG-to-behavior RSA focused on the LPP window."""

    _plot_curves(
        stats_table,
        cluster_table,
        BEHAVIOR_ORDER,
        f"{title_prefix}: LPP-window EEG-behavior RSA",
        output_dir / f"{prefix}_lpp_eeg_behavior_rsa",
        x_limits=(300.0, 800.0),
        y_label="Spearman rho",
        window_span=(300.0, 800.0),
        window_label="LPP (300-800 ms)",
    )


def plot_t_statistics_heatmap(stats_table: pd.DataFrame, output_dir: Path) -> None:
    """Plot inferential time courses as model-by-time t statistic heatmaps."""

    titles = {
        "shared_scalp_factor": "Shared scalp montage (55 channels)",
        "posterior_roi_factor": "Posterior ROI sensitivity (19 channels)",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 5.9), sharex=True, constrained_layout=True)
    selected = stats_table[stats_table["time_ms"].between(0, 600)]
    largest = float(selected["t_value"].abs().max())
    vmax = max(3.5, np.ceil(largest * 2) / 2)
    image = None
    for axis, analysis in zip(axes, ("shared_scalp_factor", "posterior_roi_factor")):
        section = selected[selected["analysis"] == analysis]
        times = np.sort(section["time_ms"].unique())
        matrix = (
            section.pivot(index="model", columns="time_ms", values="t_value")
            .reindex(MODEL_ORDER)
            .reindex(columns=times)
            .to_numpy(dtype=float)
        )
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="none",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[times[0], times[-1], len(MODEL_ORDER) - 0.5, -0.5],
        )
        threshold = float(
            section.loc[section["model"] == MODEL_ORDER[0], "n_subjects"]
            .map(lambda n: stats.t.ppf(0.975, int(n) - 1))
            .iloc[0]
        )
        uncorrected = np.abs(matrix) > threshold
        axis.contour(
            times,
            np.arange(len(MODEL_ORDER)),
            uncorrected.astype(float),
            levels=[0.5],
            colors="#202020",
            linewidths=0.6,
        )
        axis.axvspan(140, 190, edgecolor="#111111", facecolor="none", lw=1.0, linestyle="--")
        axis.set_yticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[model] for model in MODEL_ORDER])
        axis.set_title(titles[analysis], loc="left", fontsize=11, fontweight="bold")
    axes[-1].set_xlabel("Time from stimulus onset (ms)")
    fig.supylabel("Representational model")
    colorbar = fig.colorbar(image, ax=axes, pad=0.02, shrink=0.88)
    colorbar.set_label("One-sample t value")
    fig.suptitle(
        "Partial RSA group statistics across time\n"
        "Dashed box: N170 window; thin contours: pointwise p < .05 (uncorrected)",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"partial_rsa_t_statistics_heatmap.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_n170_distributions(
    participant_means: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, seed: int = 42
) -> None:
    """Plot participant distributions and 95% CIs for N170-averaged RSA."""

    rng = np.random.default_rng(seed)
    titles = {
        "shared_scalp_factor": "Shared scalp montage (55 channels)",
        "posterior_roi_factor": "Posterior ROI sensitivity (19 channels)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.8), sharey=True, constrained_layout=True)
    for axis, analysis in zip(axes, ("shared_scalp_factor", "posterior_roi_factor")):
        subset = participant_means[participant_means["analysis"] == analysis]
        summary_section = summary[summary["analysis"] == analysis].set_index("model")
        for index, model in enumerate(MODEL_ORDER):
            values = subset.loc[subset["model"] == model, "n170_mean_partial_rho"].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            parts = axis.violinplot(
                values,
                positions=[index],
                widths=0.68,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(PLOT_COLORS[model])
                body.set_edgecolor("none")
                body.set_alpha(0.17)
            jitter = rng.uniform(-0.16, 0.16, size=len(values))
            axis.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=14,
                alpha=0.4,
                color=PLOT_COLORS[model],
                edgecolors="none",
            )
            row = summary_section.loc[model]
            axis.errorbar(
                index,
                row["mean_rho"],
                yerr=[
                    [row["mean_rho"] - row["ci95_low"]],
                    [row["ci95_high"] - row["mean_rho"]],
                ],
                fmt="o",
                color=PLOT_COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=6,
                lw=1.6,
                capsize=3,
                zorder=5,
            )
        axis.axhline(0, color="#333333", lw=0.9)
        axis.set_xticks(np.arange(len(MODEL_ORDER)), [SHORT_MODEL_LABELS[model] for model in MODEL_ORDER])
        axis.tick_params(axis="x", labelsize=8)
        axis.set_title(titles[analysis], fontsize=11, fontweight="bold")
        axis.grid(axis="y", color="#ececec", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean partial Spearman rho (140-190 ms)")
    fig.suptitle("N170 window summary (140-190 ms)", fontsize=14, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"n170_window_partial_rsa_distributions.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_quality_control(
    condition_counts: pd.DataFrame, qc_table: pd.DataFrame, output_dir: Path
) -> None:
    """Plot retained epoch coverage and saved-channel counts by participant."""

    ordered_subjects = qc_table["subj"].tolist()
    heatmap = (
        condition_counts.pivot(index="subj", columns="CondID", values="n_epochs")
        .reindex(index=ordered_subjects, columns=range(1, 17))
        .fillna(0)
        .to_numpy(dtype=float)
    )
    included = qc_table["included"].to_numpy(dtype=bool)
    fig, (heat_axis, channel_axis) = plt.subplots(
        1, 2, figsize=(13.0, 8.0), gridspec_kw={"width_ratios": [1.8, 1.0]}, constrained_layout=True
    )
    image = heat_axis.imshow(heatmap, cmap="viridis", vmin=0, vmax=np.max(heatmap), aspect="auto")
    heat_axis.set_xticks(np.arange(16), np.arange(1, 17))
    heat_axis.set_yticks(np.arange(len(ordered_subjects)), ordered_subjects)
    heat_axis.set_xlabel("Analysis condition ID")
    heat_axis.set_ylabel("Participant")
    heat_axis.set_title("Valid formal epochs per condition", fontsize=11, fontweight="bold")
    excluded_index = np.flatnonzero(~included)
    for index in excluded_index:
        heat_axis.add_patch(
            plt.Rectangle((-0.5, index - 0.5), 16, 1, fill=False, edgecolor="#d62728", lw=2.0)
        )
    colorbar = fig.colorbar(image, ax=heat_axis, shrink=0.7, pad=0.02)
    colorbar.set_label("Epoch count")
    colors = np.where(included, "#497aa7", "#d62728")
    channel_axis.barh(np.arange(len(ordered_subjects)), qc_table["saved_channel_count"], color=colors)
    channel_axis.axvline(55, color="#111111", lw=1.2, linestyle="--", label="Shared montage = 55")
    channel_axis.set_yticks(np.arange(len(ordered_subjects)), ordered_subjects)
    channel_axis.invert_yaxis()
    channel_axis.set_xlim(0, max(66, int(qc_table["saved_channel_count"].max()) + 2))
    channel_axis.set_xlabel("Channels retained in epoched file")
    channel_axis.set_title("Channel availability", fontsize=11, fontweight="bold")
    channel_axis.legend(frameon=False, loc="lower right")
    channel_axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "EEG data quality control\nRed outline/bar indicates excluded participant (incomplete formal conditions)",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"eeg_data_quality_control.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rdm_archive(
    path: Path, rdm_vectors: np.ndarray, subjects: Sequence[str], times: np.ndarray
) -> None:
    """Save compact reusable EEG RDM vectors as a compressed NumPy archive."""

    np.savez_compressed(
        path,
        eeg_rdm_vectors=rdm_vectors.astype(np.float32),
        subjects=np.asarray(subjects),
        time_ms=np.asarray(times, dtype=np.float32),
        pair_cond_i=np.triu_indices(16, 1)[0] + 1,
        pair_cond_j=np.triu_indices(16, 1)[1] + 1,
    )


def load_rdm_archive(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load reusable EEG RDM vectors saved by ``save_rdm_archive``."""

    with np.load(path, allow_pickle=False) as archive:
        return (
            archive["eeg_rdm_vectors"].astype(float),
            [str(subject) for subject in archive["subjects"].tolist()],
            archive["time_ms"].astype(float),
        )


def write_log(path: Path, lines: Iterable[str]) -> None:
    """Write a UTF-8 plain-text processing log."""

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Command-line entry point for the full RSA workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-start-ms", type=float, default=0.0)
    parser.add_argument("--inference-end-ms", type=float, default=600.0)
    parser.add_argument(
        "--reuse-rdm-archives",
        action="store_true",
        help="Reuse existing eeg_rdm_vectors_*.npz archives instead of rereading EEGLAB files.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    outputs = output_root / "outputs"
    figures = output_root / "figures"
    logs = output_root / "logs"
    for directory in (outputs, figures, logs):
        directory.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.random.seed(args.seed)

    eprime_root = args.data_root / "EEGDATA" / "EEGDATA" / "eprime"
    records = standardize_factor_coding(parse_eprime_records(eprime_root))
    condition_table = make_condition_table(records)
    model_rdms = make_model_rdms(condition_table)
    condition_table.to_csv(outputs / "condition_table.csv", index=False, encoding="utf-8-sig")
    model_vector_rows = []
    pair_i, pair_j = np.triu_indices(16, 1)
    for model, rdm in model_rdms.items():
        for i, j, value in zip(pair_i + 1, pair_j + 1, vectorize_rdm(rdm)):
            model_vector_rows.append(
                {"model": model, "condition_i": i, "condition_j": j, "distance": value}
            )
    pd.DataFrame(model_vector_rows).to_csv(outputs / "factor_model_rdm_vectors.csv", index=False)
    plot_model_rdms(model_rdms, figures)

    eeg_files = find_eeg_set_files(args.data_root)
    reuse_archives = args.reuse_rdm_archives or h5py is None
    metadata_path = outputs / "analysis_metadata.json"
    shared_archive = outputs / "eeg_rdm_vectors_shared_scalp.npz"
    posterior_archive = outputs / "eeg_rdm_vectors_posterior_roi.npz"
    if reuse_archives:
        if not shared_archive.exists() or not posterior_archive.exists():
            raise FileNotFoundError(
                "RDM archive reuse was requested, but eeg_rdm_vectors_*.npz files are missing."
            )
        shared_rdms, subjects, times = load_rdm_archive(shared_archive)
        posterior_rdms, posterior_subjects, posterior_times = load_rdm_archive(posterior_archive)
        if subjects != posterior_subjects or not np.array_equal(times, posterior_times):
            raise ValueError("Shared-scalp and posterior RDM archives have inconsistent subjects or times.")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            channels = list(metadata.get("shared_scalp_channels", []))
            posterior_channels = list(metadata.get("posterior_channels", POSTERIOR_CHANNELS))
        else:
            channels = []
            posterior_channels = list(POSTERIOR_CHANNELS)
        qc_path = outputs / "eeg_subject_quality_control.csv"
        if qc_path.exists():
            qc_table = pd.read_csv(qc_path)
        else:
            qc_table = pd.DataFrame(
                {
                    "subj": subjects,
                    "included": True,
                    "exclusion_reason": "",
                }
            )
    else:
        if not eeg_files:
            raise FileNotFoundError("No derivatives_eeglab_s*/s*_epoched_stim.set files were found.")
        subjects_data = [load_eeg_data(path, condition_table) for path in eeg_files]
        qc_rows = []
        for subject in subjects_data:
            row: dict[str, object] = {
                "subj": subject.subj,
                "total_epochs": subject.epoch_count,
                "formal_epochs": subject.formal_epoch_count,
                "saved_channel_count": len(subject.channels),
                "n_formal_conditions_present": sum(count > 0 for count in subject.formal_epoch_counts.values()),
                "min_formal_epochs_per_condition": min(subject.formal_epoch_counts.values()),
                "max_formal_epochs_per_condition": max(subject.formal_epoch_counts.values()),
                "included": subject.patterns is not None,
                "exclusion_reason": subject.excluded_reason,
            }
            qc_rows.append(row)
        qc_table = pd.DataFrame(qc_rows).sort_values("subj", key=lambda col: col.map(natural_subject_key))
        qc_table.to_csv(outputs / "eeg_subject_quality_control.csv", index=False, encoding="utf-8-sig")
        condition_count_rows = []
        for subject in subjects_data:
            for raw_condition, count in subject.formal_epoch_counts.items():
                condition_count_rows.append(
                    {
                        "subj": subject.subj,
                        "CondID": raw_condition - 1,
                        "raw_CondID": raw_condition,
                        "n_epochs": count,
                        "included": subject.patterns is not None,
                    }
                )
        condition_counts = pd.DataFrame(condition_count_rows).sort_values(
            ["subj", "CondID"], key=lambda col: col.map(natural_subject_key) if col.name == "subj" else col
        )
        condition_counts.to_csv(outputs / "eeg_condition_epoch_counts.csv", index=False, encoding="utf-8-sig")
        plot_quality_control(condition_counts, qc_table, figures)

        included = [subject for subject in subjects_data if subject.patterns is not None]
        subjects = [subject.subj for subject in included]
        times = included[0].times
        for subject in included:
            if not np.array_equal(subject.times, times):
                raise ValueError(f"Time axis differs for {subject.subj}.")
        common_set = set(included[0].channels)
        for subject in included[1:]:
            common_set &= set(subject.channels)
        channels = [channel for channel in included[0].channels if channel in common_set]
        posterior_channels = [channel for channel in POSTERIOR_CHANNELS if channel in common_set]
        if len(channels) < 2:
            raise ValueError("Fewer than two channels are shared by included subjects.")
        if len(posterior_channels) != len(POSTERIOR_CHANNELS):
            raise ValueError("Not all configured posterior sensitivity channels exist in the data.")
        shared_rdms = np.stack(
            [
                compute_eeg_rdm(
                    subject.patterns[:, :, [subject.channels.index(channel) for channel in channels]]
                )
                for subject in included
            ]
        )
        posterior_rdms = np.stack(
            [
                compute_eeg_rdm(
                    subject.patterns[
                        :, :, [subject.channels.index(channel) for channel in posterior_channels]
                    ]
                )
                for subject in included
            ]
        )
        save_rdm_archive(outputs / "eeg_rdm_vectors_shared_scalp.npz", shared_rdms, subjects, times)
        save_rdm_archive(outputs / "eeg_rdm_vectors_posterior_roi.npz", posterior_rdms, subjects, times)

    behavior_choices = make_behavior_condition_choices(records, subjects)
    behavior_choices.to_csv(outputs / "behavior_condition_choices.csv", index=False, encoding="utf-8-sig")
    behavior_rdms = make_behavior_rdm_vectors(behavior_choices, subjects)
    behavior_rdm_vectors_to_long(behavior_rdms, subjects).to_csv(
        outputs / "behavior_choice_rdm_vectors.csv", index=False
    )
    plot_behavior_rdms(behavior_rdms, figures)
    behavior_model_rsa = run_behavior_model_rsa(behavior_rdms, subjects, model_rdms)
    behavior_model_rsa.to_csv(outputs / "behavior_model_rsa_subject_results.csv", index=False)
    behavior_model_stats = summarize_behavior_model_rsa(behavior_model_rsa)
    behavior_model_stats.to_csv(outputs / "behavior_model_rsa_group_stats.csv", index=False)
    plot_behavior_model_rsa(behavior_model_stats, figures)

    shared_ordinary, shared_partial = run_subject_level_rsa(
        shared_rdms, subjects, times, model_rdms, "shared_scalp_factor"
    )
    posterior_ordinary, posterior_partial = run_subject_level_rsa(
        posterior_rdms, subjects, times, model_rdms, "posterior_roi_factor"
    )
    factor_ordinary = pd.concat([shared_ordinary, posterior_ordinary], ignore_index=True)
    factor_partial = pd.concat([shared_partial, posterior_partial], ignore_index=True)
    factor_ordinary.to_csv(outputs / "eeg_factor_rsa_spearman_results.csv", index=False)
    factor_partial.to_csv(outputs / "eeg_factor_rsa_partial_results.csv", index=False)
    fslim_targeted = pd.concat(
        [
            summarize_fslim_targeted_n170(factor_ordinary, "rho", "ordinary_spearman"),
            summarize_fslim_targeted_n170(factor_partial, "partial_rho", "partial_spearman"),
        ],
        ignore_index=True,
    )
    fslim_targeted.to_csv(outputs / "eeg_fslim_targeted_n170_stats.csv", index=False)
    factor_stats = run_group_statistics(factor_partial, "partial_rho")
    factor_stats.to_csv(outputs / "eeg_factor_rsa_group_stats.csv", index=False)
    n170_participant_means, n170_stats = summarize_n170_window(factor_partial)
    n170_participant_means.to_csv(outputs / "eeg_n170_window_subject_means.csv", index=False)
    n170_stats.to_csv(outputs / "eeg_n170_window_group_stats.csv", index=False)
    inference_window = (args.inference_start_ms, args.inference_end_ms)
    factor_clusters = run_cluster_tests(
        factor_partial,
        "partial_rho",
        inference_window,
        args.n_permutations,
        args.seed,
    )
    factor_clusters.to_csv(outputs / "eeg_factor_rsa_cluster_results.csv", index=False)
    lpp_window = (300.0, 800.0)
    lpp_factor_subject_means, lpp_factor_stats = summarize_component_window(
        factor_partial, "partial_rho", "LPP", lpp_window
    )
    lpp_factor_subject_means.to_csv(outputs / "eeg_lpp_window_factor_subject_means.csv", index=False)
    lpp_factor_stats.to_csv(outputs / "eeg_lpp_window_factor_group_stats.csv", index=False)
    lpp_factor_clusters = run_cluster_tests(
        factor_partial,
        "partial_rho",
        lpp_window,
        args.n_permutations,
        args.seed + 2000,
    )
    lpp_factor_clusters.to_csv(outputs / "eeg_lpp_factor_cluster_results.csv", index=False)
    plot_t_statistics_heatmap(factor_stats, figures)
    plot_n170_distributions(n170_participant_means, n170_stats, figures, args.seed)

    shared_eeg_behavior = run_eeg_behavior_rsa(
        shared_rdms, subjects, times, behavior_rdms, "shared_scalp_eeg_behavior"
    )
    posterior_eeg_behavior = run_eeg_behavior_rsa(
        posterior_rdms, subjects, times, behavior_rdms, "posterior_roi_eeg_behavior"
    )
    eeg_behavior = pd.concat([shared_eeg_behavior, posterior_eeg_behavior], ignore_index=True)
    eeg_behavior.to_csv(outputs / "eeg_behavior_rsa_results.csv", index=False)
    eeg_behavior_stats = run_group_statistics(eeg_behavior, "rho")
    eeg_behavior_stats.to_csv(outputs / "eeg_behavior_rsa_group_stats.csv", index=False)
    eeg_behavior_clusters = run_cluster_tests(
        eeg_behavior,
        "rho",
        inference_window,
        args.n_permutations,
        args.seed + 1000,
    )
    eeg_behavior_clusters.to_csv(outputs / "eeg_behavior_rsa_cluster_results.csv", index=False)
    lpp_behavior_subject_means, lpp_behavior_stats = summarize_component_window(
        eeg_behavior, "rho", "LPP", lpp_window
    )
    lpp_behavior_subject_means.to_csv(outputs / "eeg_lpp_window_behavior_subject_means.csv", index=False)
    lpp_behavior_stats.to_csv(outputs / "eeg_lpp_window_behavior_group_stats.csv", index=False)
    lpp_behavior_clusters = run_cluster_tests(
        eeg_behavior,
        "rho",
        lpp_window,
        args.n_permutations,
        args.seed + 3000,
    )
    lpp_behavior_clusters.to_csv(outputs / "eeg_lpp_behavior_cluster_results.csv", index=False)

    for analysis, prefix, label in (
        ("shared_scalp_factor", "shared_scalp", f"Shared scalp montage ({len(channels)} channels)"),
        ("posterior_roi_factor", "posterior_roi", "Posterior ROI sensitivity"),
    ):
        plot_time_resolved_rsa(
            factor_stats[factor_stats["analysis"] == analysis],
            factor_clusters[factor_clusters["analysis"] == analysis],
            figures,
            prefix,
            label,
        )

    for analysis, prefix, label in (
        ("shared_scalp_eeg_behavior", "shared_scalp", f"Shared scalp montage ({len(channels)} channels)"),
        ("posterior_roi_eeg_behavior", "posterior_roi", "Posterior ROI sensitivity"),
    ):
        plot_eeg_behavior_rsa(
            eeg_behavior_stats[eeg_behavior_stats["analysis"] == analysis],
            eeg_behavior_clusters[eeg_behavior_clusters["analysis"] == analysis],
            figures,
            prefix,
            label,
        )

    plot_lpp_eeg_behavior_rsa(
        eeg_behavior_stats[eeg_behavior_stats["analysis"] == "shared_scalp_eeg_behavior"],
        lpp_behavior_clusters[lpp_behavior_clusters["analysis"] == "shared_scalp_eeg_behavior"],
        figures,
        "shared_scalp",
        f"Shared scalp montage ({len(channels)} channels)",
    )

    significant_factor = factor_clusters[factor_clusters["significant"] == True]  # noqa: E712
    significant_eeg_behavior = eeg_behavior_clusters[eeg_behavior_clusters["significant"] == True]  # noqa: E712
    significant_lpp_behavior = lpp_behavior_clusters[lpp_behavior_clusters["significant"] == True]  # noqa: E712
    significant_n170 = n170_stats[n170_stats["p_fdr_six_models"] < 0.05]
    significant_behavior_model = behavior_model_stats[behavior_model_stats["p_fdr_six_models"] < 0.05]
    log_lines = [
        "Time-resolved EEG RSA processing log",
        "===================================",
        f"Data root: {args.data_root.resolve()}",
        "Event code rule: endCode = Stimtype base + raw CondID; bases F_1/F_2/M_1/M_2/T = 100/120/140/160/180.",
        "Analyzed stimuli: F_1, F_2, M_1, M_2. T trials are not included in the formal-factor RSA.",
        "Condition rule: raw CondID 1 is original/control; raw CondID 2-17 are recoded to analysis CondID 1-16.",
        f"EEGLAB epoch files found: {len(eeg_files)}",
        f"Subjects included in EEG RSA: {len(subjects)} ({', '.join(subjects)})",
        f"Subjects excluded: {', '.join(qc_table.loc[~qc_table['included'], 'subj']) or 'None'}",
        f"Channels in main analysis: {len(channels)} shared scalp channels ({', '.join(channels)})",
        "Channel QC decision: epoched files retain different post-cleaning channel subsets; RSA uses only channels common to every included participant rather than mixing nonmatching scalp locations.",
        f"Channels in sensitivity analysis: {len(posterior_channels)} ({', '.join(posterior_channels)})",
        f"Sampling/time range: {times[0]:.0f} to {times[-1]:.0f} ms, {len(times)} samples",
        "Preprocessing at analysis stage: none added; saved preprocessed epoched signals are used as stored.",
        "EEG distance: 1 - Pearson correlation across multi-channel condition-average patterns.",
        "Factor RSA: partial Spearman; each model controls the other five factor/interaction RDMs.",
        "Behavioral RDMs: subject-specific absolute condition-pair differences in RealnessRating.RESP (Naturalness) and LikingRating.RESP (Beauty).",
        "Behavior model RSA: behavioral choice RDMs are correlated with the same six factorial model RDMs.",
        "EEG-behavior RSA: EEG RDMs are correlated over time with subject-specific Naturalness and Beauty choice RDMs.",
        "FSlim targeted N170 output: ordinary and partial Spearman rho averaged from 140-190 ms for the FSlim model only, intended as a theory-driven follow-up to prior FSlim-N170 evidence.",
        f"Cluster test: two-sided sign-flipping, {args.n_permutations} permutations, inference interval {inference_window[0]:.0f}-{inference_window[1]:.0f} ms, seed {args.seed}.",
        f"Significant factor clusters after correction: {len(significant_factor)}",
        f"Significant EEG-behavior clusters after correction: {len(significant_eeg_behavior)}",
        f"Significant LPP-window EEG-behavior clusters after correction: {len(significant_lpp_behavior)}",
        f"N170 window summary: participant rho averaged from 140-190 ms; FDR adjusted across six models within each montage; significant tests: {len(significant_n170)}.",
        f"Significant behavior-model RSA tests after six-model FDR: {len(significant_behavior_model)}.",
        "Scope: EEG factor/interaction RSA plus behavioral-choice RDM RSA. Eye-tracking RDM RSA was not run in this analysis.",
        "Processing status: completed.",
    ]
    write_log(logs / "rsa_processing_log.txt", log_lines)
    metadata = {
        "included_subjects": subjects,
        "excluded_subjects": qc_table.loc[~qc_table["included"], "subj"].tolist(),
        "shared_scalp_channels": channels,
        "posterior_channels": posterior_channels,
        "time_ms": [float(times[0]), float(times[-1])],
        "n_permutations": args.n_permutations,
        "inference_window_ms": list(inference_window),
        "seed": args.seed,
        "behavior_choices": {
            "Naturalness_choice": "RealnessRating.RESP",
            "Beauty_choice": "LikingRating.RESP",
        },
    }
    (outputs / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RSA analysis complete. Results written to: {output_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Time-resolved factor decoding and temporal generalization for edited-face EEG.

The script reads trial-level EEGLAB epoched stimulus files from the current
project, builds single-trial metadata from the epoch endCode rule, and runs
within-subject MVPA for the four edited-face factors.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_bundled_python = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "python" / "python.exe"
if sys.version_info[:2] != (3, 12) and _bundled_python.exists() and Path(sys.executable).resolve() != _bundled_python.resolve():
    raise SystemExit(subprocess.call([str(_bundled_python), *sys.argv]))

_project_guess = Path.cwd()
_pkg = _project_guess / "RSA_time_resolved_analysis" / ".python-packages"
if _pkg.exists() and str(_pkg) not in sys.path:
    sys.path.append(str(_pkg))

_mpl_cache = Path.cwd() / ".codex_matplotlib_cache"
_mpl_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import h5py
import matplotlib
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy import ndimage, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for MVPA decoding. Install it for the bundled "
        "Python, e.g. python -m pip install scikit-learn, then rerun."
    ) from exc


FACTORS = ["FSlim", "Eye", "Mouth", "Skin"]
STRUCTURAL = ["FSlim", "Eye"]
SURFACE = ["Skin", "Mouth"]
STIM_BASES = {"F_1": 100, "F_2": 120, "M_1": 140, "M_2": 160, "T": 180}
ANALYZED_STIMTYPES = {"F_1", "F_2", "M_1", "M_2"}
FORMAL_RAW_CONDITIONS = set(range(2, 18))
DEFAULT_SUBJECTS = {f"s{i}" for i in list(range(1, 5)) + list(range(6, 31))}
WINDOW_TAGS = {"P1": (80, 130), "N170": (140, 190), "post-N170": (200, 300), "late": (300, 800)}
COLORS = {"FSlim": "#146c5f", "Eye": "#c95f20", "Mouth": "#5b63a8", "Skin": "#b83280"}
POSTERIOR_CHANNELS = [
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
]
N170_LPP_CHANNELS = [
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
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
]


@dataclass
class Context:
    project_root: Path
    outdir: Path
    seed: int
    n_permutations: int
    quick_test: bool
    dirs: dict[str, Path]
    warnings: list[str] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        logging.warning(message)

    def add_file(self, path: Path) -> None:
        self.generated_files.append(str(path.resolve()))


@dataclass
class SubjectEpochs:
    subj: str
    set_path: Path
    fdt_path: Path
    times: np.ndarray
    channels: list[str]
    sample_rate: float
    data: np.memmap
    metadata: pd.DataFrame
    baseline_applied: bool
    channel_indices: np.ndarray | None = None
    analysis_channels: list[str] = field(default_factory=list)


def setup_dirs(project_root: Path, outdir: str) -> dict[str, Path]:
    root = project_root / outdir
    dirs = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "paper_ready": root / "figures" / "paper_ready",
        "logs": root / "logs",
        "summaries": root / "summaries",
        "intermediate": root / "intermediate",
        "tgm_matrices": root / "intermediate" / "temporal_generalization_subject_matrices",
        "models": root / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(path: Path) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def save_csv(ctx: Context, df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    ctx.add_file(path)
    logging.info("Wrote %s shape=%s", path, df.shape)


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def save_text(ctx: Context, text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")
    ctx.add_file(path)
    logging.info("Wrote %s", path)


def save_json(ctx: Context, obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    ctx.add_file(path)
    logging.info("Wrote %s", path)


def df_to_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No rows."
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append("" if pd.isna(value) else f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if pd.isna(obj):
        return None
    return str(obj)


def natural_subject_key(value: str | Path) -> int:
    match = re.search(r"s?(\d+)", str(value).lower())
    return int(match.group(1)) if match else 10**9


def decode_hdf5_reference(dataset: h5py.File, ref: h5py.Reference) -> object:
    array = np.asarray(dataset[ref][()]).squeeze()
    if array.dtype.kind in "ui" and np.size(array) >= 1:
        text = "".join(chr(int(value)) for value in np.ravel(array, order="F") if int(value))
        if text:
            return text
    return np.asarray(array).squeeze().item()


def read_set_header(set_path: Path) -> dict[str, Any]:
    with h5py.File(set_path, "r") as dataset:
        n_channels = int(dataset["nbchan"][0, 0])
        n_points = int(dataset["pnts"][0, 0])
        n_trials = int(dataset["trials"][0, 0])
        sample_rate = float(dataset["srate"][0, 0])
        times = np.asarray(dataset["times"][()]).ravel().astype(float)
        channels = [str(decode_hdf5_reference(dataset, ref)) for ref in dataset["chanlocs"]["labels"][:, 0]]
        endcodes = [decode_hdf5_reference(dataset, ref) for ref in dataset["epoch"]["endCode"][:, 0]]
        datfile = "".join(chr(int(value)) for value in dataset["datfile"][()].ravel() if int(value))
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


def decode_endcode(code: object) -> tuple[str | None, int | None]:
    try:
        integer = int(str(code).strip())
    except ValueError:
        return None, None
    for stimtype, base in STIM_BASES.items():
        condition = integer - base
        if 1 <= condition <= 17:
            return stimtype, condition
    return None, None


def condition_factors(raw_cond_id: int, condition_map: dict[int, dict[str, int]] | None = None) -> dict[str, int]:
    if condition_map and raw_cond_id in condition_map:
        return condition_map[raw_cond_id].copy()
    cond_id = raw_cond_id - 1
    if not 1 <= cond_id <= 16:
        return {factor: 0 for factor in FACTORS}
    # Fallback matching the verified E-Prime condition order:
    # Skin fastest, FSlim second, Mouth third, Eye slowest.
    bits = [(cond_id - 1) >> shift & 1 for shift in (1, 3, 2, 0)]
    return dict(zip(FACTORS, bits))


def read_text_with_fallback(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_eprime_condition_map(project_root: Path, ctx: Context) -> dict[int, dict[str, int]]:
    eprime_root = project_root / "EEGDATA" / "EEGDATA" / "eprime"
    rows: list[dict[str, Any]] = []
    if not eprime_root.exists():
        ctx.warn("E-Prime directory not found; using verified fallback CondID factor order.")
        return {}
    for path in sorted(eprime_root.rglob("*.txt")):
        text = read_text_with_fallback(path)
        frames = re.findall(r"\*\*\* LogFrame Start \*\*\*(.*?)\*\*\* LogFrame End \*\*\*", text, flags=re.DOTALL)
        for frame in frames:
            row: dict[str, Any] = {}
            for key in ["CondID", "FSlim", "Eye", "Mouth", "Skin", "Stimtype"]:
                match = re.search(rf"^\s*{re.escape(key)}:\s*(.*?)\s*$", frame, flags=re.MULTILINE)
                row[key] = match.group(1).strip() if match else np.nan
            if any(pd.isna(row[k]) for k in ["CondID", "FSlim", "Eye", "Mouth", "Skin", "Stimtype"]):
                continue
            rows.append(row)
    if not rows:
        ctx.warn("No usable E-Prime condition rows found; using verified fallback CondID factor order.")
        return {}
    df = pd.DataFrame(rows)
    for col in ["CondID", "FSlim", "Eye", "Mouth", "Skin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["CondID"].isin(range(2, 18)) & df["Stimtype"].isin(ANALYZED_STIMTYPES)].copy()
    if df.empty:
        ctx.warn("E-Prime condition rows did not contain formal edited trials; using verified fallback CondID factor order.")
        return {}
    maps = {
        "FSlim": {0: 0, 1: 1},
        "Skin": {0: 0, 1: 1},
        "Eye": {1: 0, 2: 1},
        "Mouth": {1: 0, 2: 1},
    }
    for factor, mapping in maps.items():
        df[factor] = df[factor].map(mapping)
    table = df[["CondID", *FACTORS]].drop_duplicates().sort_values("CondID")
    if len(table) != 16 or table[FACTORS].duplicated().any():
        ctx.warn("E-Prime condition map failed 16-condition validation; using verified fallback CondID factor order.")
        return {}
    condition_map = {int(row["CondID"]): {factor: int(row[factor]) for factor in FACTORS} for _, row in table.iterrows()}
    logging.info("Loaded verified E-Prime condition map:\n%s", table.to_string(index=False))
    return condition_map


def find_eeg_files(project_root: Path) -> list[Path]:
    files = list(project_root.glob("derivatives_eeglab_s*/s*_epoched_stim.set"))
    if not files:
        files = list(project_root.rglob("*_epoched_stim.set"))
    return sorted(files, key=lambda path: natural_subject_key(path.name))


def load_subject(set_path: Path, ctx: Context, condition_map: dict[int, dict[str, int]] | None = None) -> SubjectEpochs | None:
    subj_match = re.search(r"s(\d+)", set_path.stem.lower())
    subj = f"s{subj_match.group(1)}" if subj_match else set_path.stem.lower()
    if subj not in DEFAULT_SUBJECTS:
        ctx.warn(f"{subj}: excluded by default subject list.")
        return None
    header = read_set_header(set_path)
    fdt_path = Path(header["fdt_path"])
    if not fdt_path.exists():
        ctx.warn(f"{subj}: missing FDT file {fdt_path}; skipped.")
        return None
    rows = []
    for trial_index, code in enumerate(header["endcodes"]):
        stimtype, raw_cond_id = decode_endcode(code)
        if stimtype is None or raw_cond_id is None:
            continue
        facs = condition_factors(raw_cond_id, condition_map)
        rows.append(
            {
                "subj": subj,
                "trial_id": trial_index + 1,
                "epoch_index": trial_index,
                "stimtype": stimtype,
                "raw_cond_id": raw_cond_id,
                "cond_id": raw_cond_id - 1,
                "is_control": raw_cond_id == 1,
                "attention_check": stimtype == "T",
                "identity": stimtype,
                "bad_trial": False,
                **facs,
            }
        )
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        ctx.warn(f"{subj}: no decodable epoch endCode rows; skipped.")
        return None
    data = np.memmap(
        fdt_path,
        dtype="<f4",
        mode="r",
        shape=(int(header["n_trials"]), int(header["n_points"]), int(header["n_channels"])),
    )
    times = np.asarray(header["times"], dtype=float)
    baseline_applied = bool(np.nanmean(np.abs(data[: min(10, data.shape[0]), (times >= -200) & (times <= 0), :])) < 1e-6)
    return SubjectEpochs(
        subj=subj,
        set_path=set_path,
        fdt_path=fdt_path,
        times=times,
        channels=list(header["channels"]),
        sample_rate=float(header["sample_rate"]),
        data=data,
        metadata=metadata,
        baseline_applied=baseline_applied,
    )


def extract_subject_matrix(subject: SubjectEpochs, metadata: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    data = np.asarray(subject.data[metadata["epoch_index"].to_numpy(dtype=int), :, :], dtype=np.float32)
    if subject.channel_indices is not None:
        data = data[:, :, subject.channel_indices]
    baseline_mask = (times >= -200) & (times <= 0)
    if baseline_mask.any():
        data = data - data[:, baseline_mask, :].mean(axis=1, keepdims=True)
    return data


def make_windows(times: np.ndarray, window_ms: int, step_ms: int, tmin: int, tmax: int) -> pd.DataFrame:
    starts = np.arange(tmin, tmax - window_ms + 0.1, step_ms)
    rows = []
    for start in starts:
        end = start + window_ms
        mask = (times >= start) & (times < end)
        if mask.any():
            rows.append(
                {
                    "window_start_ms": float(start),
                    "window_end_ms": float(end),
                    "window_center_ms": float((start + end) / 2),
                    "time_indices": np.where(mask)[0],
                }
            )
    return pd.DataFrame(rows)


def feature_for_window(data: np.ndarray, indices: np.ndarray, mode: str) -> np.ndarray:
    chunk = data[:, indices, :]
    if mode == "window_mean":
        return chunk.mean(axis=1)
    return chunk.reshape(chunk.shape[0], -1)


def make_pseudotrials(
    data: np.ndarray,
    metadata: pd.DataFrame,
    factor: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    if size <= 1 or metadata.empty:
        return data, metadata
    group_cols = [factor, *nuisance_columns(factor)]
    pseudo_data = []
    pseudo_rows = []
    for _, cell in metadata.groupby(group_cols, dropna=False):
        idx = cell.index.to_numpy().copy()
        rng.shuffle(idx)
        for start in range(0, len(idx), size):
            chunk = idx[start : start + size]
            if len(chunk) < max(2, size // 2):
                continue
            pseudo_data.append(data[chunk].mean(axis=0))
            row = cell.loc[chunk[0]].copy()
            row["trial_id"] = f"pseudo_{len(pseudo_rows) + 1}"
            row["n_raw_trials_in_pseudotrial"] = int(len(chunk))
            pseudo_rows.append(row)
    if not pseudo_data:
        return data, metadata
    return np.stack(pseudo_data, axis=0).astype(np.float32), pd.DataFrame(pseudo_rows).reset_index(drop=True)


def assign_analysis_channels(subjects: list[SubjectEpochs], mode: str, ctx: Context) -> None:
    if not subjects:
        return
    common = set(subjects[0].channels)
    for subject in subjects[1:]:
        common &= set(subject.channels)
    if mode == "all":
        for subject in subjects:
            subject.channel_indices = None
            subject.analysis_channels = list(subject.channels)
        ctx.warn("Channel mode all selected: using each subject's retained channels; channel sets may differ.")
        return
    if mode == "posterior_roi":
        selected = [channel for channel in POSTERIOR_CHANNELS if channel in common]
    elif mode == "n170_lpp_roi":
        selected = [channel for channel in N170_LPP_CHANNELS if channel in common]
    else:
        selected = [channel for channel in subjects[0].channels if channel in common]
    if not selected:
        raise ValueError(f"No channels selected for channel mode {mode}.")
    for subject in subjects:
        lookup = {channel: i for i, channel in enumerate(subject.channels)}
        subject.analysis_channels = selected
        subject.channel_indices = np.array([lookup[channel] for channel in selected], dtype=int)
    logging.info("Channel mode %s selected %s channels: %s", mode, len(selected), ", ".join(selected))


def make_classifier(kind: str, seed: int) -> Pipeline:
    if kind == "linearsvm":
        clf = LinearSVC(class_weight="balanced", max_iter=10000, random_state=seed)
    else:
        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def scores_from_estimator(estimator: Pipeline, x_test: np.ndarray) -> np.ndarray:
    clf = estimator.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return estimator.predict_proba(x_test)[:, 1]
    return estimator.decision_function(x_test)


def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return np.nan


def nuisance_columns(factor: str) -> list[str]:
    return [col for col in FACTORS if col != factor]


def selected_factors(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "factors", "all")
    if not raw or str(raw).lower() == "all":
        return FACTORS
    factors = [item.strip() for item in str(raw).split(",") if item.strip()]
    bad = [factor for factor in factors if factor not in FACTORS]
    if bad:
        raise ValueError(f"Unknown factors in --factors: {bad}. Valid: {FACTORS}")
    return factors


def balanced_indices(metadata: pd.DataFrame, factor: str, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    warnings_out = []
    keep: list[int] = []
    nuis = nuisance_columns(factor)
    for _, cell in metadata.groupby(nuis, dropna=False):
        idx0 = cell.index[cell[factor] == 0].to_numpy()
        idx1 = cell.index[cell[factor] == 1].to_numpy()
        take = min(len(idx0), len(idx1))
        if take == 0:
            warnings_out.append(f"{factor}: skipped nuisance cell {cell[nuis].iloc[0].to_dict()} because one class was empty.")
            continue
        keep.extend(rng.choice(idx0, take, replace=False).tolist())
        keep.extend(rng.choice(idx1, take, replace=False).tolist())
    return np.array(sorted(keep), dtype=int), warnings_out


def split_generator(metadata: pd.DataFrame, factor: str, cv_mode: str, seed: int, n_splits_default: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    y = metadata[factor].to_numpy(dtype=int)
    if cv_mode == "condition_generalization":
        splits = []
        nuis = nuisance_columns(factor)
        combos = metadata[nuis].astype(str).agg("_".join, axis=1).to_numpy()
        for combo in sorted(set(combos)):
            test = np.where(combos == combo)[0]
            train = np.where(combos != combo)[0]
            if len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2:
                splits.append((train, test))
        return splits
    if cv_mode == "leave_one_identity_out" and "identity" in metadata.columns:
        splits = []
        groups = metadata["identity"].astype(str).to_numpy()
        for group in sorted(set(groups)):
            test = np.where(groups == group)[0]
            train = np.where(groups != group)[0]
            if len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2:
                splits.append((train, test))
        return splits
    min_class = int(np.bincount(y, minlength=2).min())
    n_splits = min(n_splits_default, min_class)
    if n_splits < 3 and min_class >= 2:
        n_splits = 2
    if n_splits < 2:
        return []
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def run_subject_time_resolved(
    subject: SubjectEpochs,
    ctx: Context,
    args: argparse.Namespace,
    windows: pd.DataFrame,
    decoding_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(ctx.seed + natural_subject_key(subject.subj))
    metadata = subject.metadata.copy()
    metadata = metadata[
        metadata["raw_cond_id"].isin(FORMAL_RAW_CONDITIONS)
        & metadata["stimtype"].isin(ANALYZED_STIMTYPES)
        & ~metadata["attention_check"]
        & ~metadata["bad_trial"]
    ].reset_index(drop=True)
    if metadata.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = extract_subject_matrix(subject, metadata, subject.times)
    rows, count_rows = [], []
    for factor in selected_factors(args):
        work_meta = metadata.copy()
        if decoding_version == "nuisance_balanced":
            loc, cell_warnings = balanced_indices(work_meta, factor, rng)
            for warning in cell_warnings[:3]:
                ctx.warn(f"{subject.subj}: {warning}")
            work_meta = work_meta.loc[loc].reset_index(drop=True)
            work_data = data[loc]
        else:
            work_data = data
        work_data, work_meta = make_pseudotrials(work_data, work_meta, factor, args.pseudotrial_size, rng)
        counts = work_meta[factor].value_counts().to_dict()
        n0, n1 = int(counts.get(0, 0)), int(counts.get(1, 0))
        skip_reason = ""
        if min(n0, n1) < 20:
            skip_reason = f"fewer than 20 trials per class after {decoding_version}"
        n_splits_default = 3 if args.quick_test else 5
        splits = [] if skip_reason else split_generator(work_meta, factor, args.cv_mode, ctx.seed, n_splits_default=n_splits_default)
        if not splits and not skip_reason:
            skip_reason = f"no valid CV splits for {args.cv_mode}"
        count_rows.append(
            {
                "subj": subject.subj,
                "factor": factor,
                "decoding_version": decoding_version,
                "n_trials_used": len(work_meta),
                "n_class0": n0,
                "n_class1": n1,
                "n_folds": len(splits),
                "skipped": bool(skip_reason),
                "skip_reason": skip_reason,
            }
        )
        if skip_reason:
            ctx.warn(f"{subject.subj} {factor} {decoding_version}: {skip_reason}")
            continue
        y = work_meta[factor].to_numpy(dtype=int)
        for _, win in windows.iterrows():
            x = feature_for_window(work_data, win["time_indices"], args.feature_mode)
            fold_metrics = []
            for train, test in splits:
                estimator = make_classifier(args.classifier, ctx.seed)
                estimator.fit(x[train], y[train])
                pred = estimator.predict(x[test])
                score = scores_from_estimator(estimator, x[test])
                fold_metrics.append(
                    (
                        auc_safe(y[test], score),
                        balanced_accuracy_score(y[test], pred),
                        accuracy_score(y[test], pred),
                        len(train),
                        len(test),
                    )
                )
            arr = np.asarray(fold_metrics, dtype=float)
            rows.append(
                {
                    "subj": subject.subj,
                    "factor": factor,
                    "decoding_version": decoding_version,
                    "classifier": args.classifier,
                    "feature_mode": args.feature_mode,
                    "pseudotrial_size": int(args.pseudotrial_size),
                    "window_start_ms": win["window_start_ms"],
                    "window_end_ms": win["window_end_ms"],
                    "window_center_ms": win["window_center_ms"],
                    "AUC": float(np.nanmean(arr[:, 0])),
                    "balanced_accuracy": float(np.nanmean(arr[:, 1])),
                    "accuracy": float(np.nanmean(arr[:, 2])),
                    "n_trials_used": len(work_meta),
                    "n_class0": n0,
                    "n_class1": n1,
                    "n_folds": len(splits),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(count_rows)


def fdr_bh(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    vals = p[ok]
    order = np.argsort(vals)
    ranks = np.arange(1, len(vals) + 1)
    ranked = vals[order] * len(vals) / ranks
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(vals)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def one_sample_stats(values: np.ndarray, chance: float = 0.5) -> dict[str, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return dict(mean_AUC=np.nan, median_AUC=np.nan, sd_AUC=np.nan, sem_AUC=np.nan, ci95_low=np.nan, ci95_high=np.nan, t=np.nan, df=n - 1, p_uncorrected=np.nan, Cohen_dz=np.nan, n_subjects=n)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    sem = sd / math.sqrt(n)
    t_value, p_value = stats.ttest_1samp(values, chance)
    crit = stats.t.ppf(0.975, n - 1)
    return {
        "mean_AUC": mean,
        "median_AUC": float(np.median(values)),
        "sd_AUC": sd,
        "sem_AUC": float(sem),
        "ci95_low": float(mean - crit * sem),
        "ci95_high": float(mean + crit * sem),
        "t": float(t_value),
        "df": int(n - 1),
        "p_uncorrected": float(p_value),
        "Cohen_dz": float((mean - chance) / sd) if sd > 0 else np.nan,
        "n_subjects": int(n),
    }


def group_stats_time(subject_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in subject_results.groupby(["factor", "decoding_version", "classifier", "feature_mode", "window_start_ms", "window_end_ms", "window_center_ms"], dropna=False):
        row = dict(zip(["factor", "decoding_version", "classifier", "feature_mode", "window_start_ms", "window_end_ms", "window_center_ms"], keys))
        row.update(one_sample_stats(sub["AUC"].to_numpy(dtype=float), 0.5))
        rows.append(row)
    stats_df = pd.DataFrame(rows).sort_values(["factor", "decoding_version", "window_center_ms"])
    stats_df["p_fdr"] = stats_df.groupby(["factor", "decoding_version"])["p_uncorrected"].transform(fdr_bh)
    return stats_df.reset_index(drop=True)


def find_1d_clusters(tvals: np.ndarray, pvals: np.ndarray, centers: np.ndarray) -> list[np.ndarray]:
    mask = np.isfinite(tvals) & np.isfinite(pvals) & (pvals < 0.05) & (tvals > 0)
    clusters = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(mask) - 1):
            end = i if flag and i == len(mask) - 1 else i - 1
            clusters.append(np.arange(start, end + 1))
            start = None
    return clusters


def cluster_permutation_1d(pivot: pd.DataFrame, n_perm: int, seed: int, chance: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pivot.columns.to_numpy(dtype=float)
    data = pivot.to_numpy(dtype=float)
    observed = data - chance
    tvals, pvals = stats.ttest_1samp(data, chance, axis=0, nan_policy="omit")
    clusters = find_1d_clusters(tvals, pvals, times)
    obs_masses = np.array([np.nansum(tvals[cluster]) for cluster in clusters], dtype=float)
    max_masses = []
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=(observed.shape[0], 1))
        perm_data = chance + observed * signs
        perm_t, perm_p = stats.ttest_1samp(perm_data, chance, axis=0, nan_policy="omit")
        perm_clusters = find_1d_clusters(perm_t, perm_p, times)
        max_masses.append(max([np.nansum(perm_t[cl]) for cl in perm_clusters], default=0.0))
    max_masses = np.asarray(max_masses)
    rows = []
    for cluster, mass in zip(clusters, obs_masses):
        peak_idx = cluster[np.nanargmax(tvals[cluster])]
        rows.append(
            {
                "cluster_start_ms": float(times[cluster[0]]),
                "cluster_end_ms": float(times[cluster[-1]]),
                "cluster_mass": float(mass),
                "cluster_p": float((np.sum(max_masses >= mass) + 1) / (len(max_masses) + 1)),
                "n_time_windows": int(len(cluster)),
                "peak_time_ms": float(times[peak_idx]),
                "peak_t": float(tvals[peak_idx]),
                "peak_mean_AUC": float(np.nanmean(data[:, peak_idx])),
            }
        )
    return pd.DataFrame(rows)


def cluster_table_time(subject_results: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    for (factor, version), sub in subject_results.groupby(["factor", "decoding_version"]):
        pivot = sub.pivot_table(index="subj", columns="window_center_ms", values="AUC", aggfunc="mean").sort_index(axis=1)
        if pivot.shape[0] < 2:
            continue
        clusters = cluster_permutation_1d(pivot, n_perm, seed + natural_subject_key(factor + version))
        for _, row in clusters.iterrows():
            rows.append({"factor": factor, "decoding_version": version, **row.to_dict()})
    return pd.DataFrame(rows)


def run_temporal_generalization(
    subjects: list[SubjectEpochs],
    ctx: Context,
    args: argparse.Namespace,
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    factors = selected_factors(args)
    if args.quick_test:
        factors = [factor for factor in ["FSlim", "Eye"] if factor in factors]
    summaries = []
    matrices_by_factor: dict[str, list[np.ndarray]] = {factor: [] for factor in factors}
    subjects_by_factor: dict[str, list[str]] = {factor: [] for factor in factors}
    for subject in subjects:
        rng = np.random.default_rng(ctx.seed + natural_subject_key(subject.subj))
        metadata = subject.metadata[
            subject.metadata["raw_cond_id"].isin(FORMAL_RAW_CONDITIONS)
            & subject.metadata["stimtype"].isin(ANALYZED_STIMTYPES)
            & ~subject.metadata["attention_check"]
        ].reset_index(drop=True)
        if metadata.empty:
            continue
        data = extract_subject_matrix(subject, metadata, subject.times)
        for factor in factors:
            existing = ctx.dirs["tgm_matrices"] / f"{subject.subj}_{factor}.npy"
            if existing.exists():
                matrix = np.load(existing)
                times = windows["window_center_ms"].to_numpy(dtype=float)
                diag = np.diag(matrix)
                early = (times >= 140) & (times <= 190)
                late = (times >= 300) & (times <= 800)
                offdiag = ~np.eye(len(times), dtype=bool)
                summaries.append(
                    {
                        "subj": subject.subj,
                        "factor": factor,
                        "mean_diag_AUC": float(np.nanmean(diag)),
                        "peak_diag_AUC": float(np.nanmax(diag)),
                        "peak_diag_time_ms": float(times[np.nanargmax(diag)]),
                        "mean_offdiag_AUC": float(np.nanmean(matrix[offdiag])),
                        "mean_early_to_late_AUC": float(np.nanmean(matrix[np.ix_(early, late)])) if early.any() and late.any() else np.nan,
                        "mean_late_to_late_AUC": float(np.nanmean(matrix[np.ix_(late, late)])) if late.any() else np.nan,
                    }
                )
                matrices_by_factor[factor].append(matrix)
                subjects_by_factor[factor].append(subject.subj)
                continue
            loc, _ = balanced_indices(metadata, factor, rng)
            work_meta = metadata.loc[loc].reset_index(drop=True)
            if work_meta[factor].value_counts().min() < 20:
                continue
            n_splits_default = 3 if args.quick_test else 5
            splits = split_generator(work_meta, factor, args.cv_mode, ctx.seed, n_splits_default=n_splits_default)
            if not splits:
                continue
            work_data = data[loc]
            y = work_meta[factor].to_numpy(dtype=int)
            features = [feature_for_window(work_data, win["time_indices"], "window_mean") for _, win in windows.iterrows()]
            matrix = np.zeros((len(windows), len(windows)), dtype=float)
            for ti, x_train_time in enumerate(features):
                fold_mats = []
                for train, test in splits:
                    estimator = make_classifier(args.classifier, ctx.seed)
                    estimator.fit(x_train_time[train], y[train])
                    fold_scores = []
                    for x_test_time in features:
                        score = scores_from_estimator(estimator, x_test_time[test])
                        fold_scores.append(auc_safe(y[test], score))
                    fold_mats.append(fold_scores)
                matrix[ti, :] = np.nanmean(np.asarray(fold_mats, dtype=float), axis=0)
            out = ctx.dirs["tgm_matrices"] / f"{subject.subj}_{factor}.npy"
            np.save(out, matrix.astype(np.float32))
            ctx.add_file(out)
            times = windows["window_center_ms"].to_numpy(dtype=float)
            diag = np.diag(matrix)
            early = (times >= 140) & (times <= 190)
            late = (times >= 300) & (times <= 800)
            offdiag = ~np.eye(len(times), dtype=bool)
            summaries.append(
                {
                    "subj": subject.subj,
                    "factor": factor,
                    "mean_diag_AUC": float(np.nanmean(diag)),
                    "peak_diag_AUC": float(np.nanmax(diag)),
                    "peak_diag_time_ms": float(times[np.nanargmax(diag)]),
                    "mean_offdiag_AUC": float(np.nanmean(matrix[offdiag])),
                    "mean_early_to_late_AUC": float(np.nanmean(matrix[np.ix_(early, late)])) if early.any() and late.any() else np.nan,
                    "mean_late_to_late_AUC": float(np.nanmean(matrix[np.ix_(late, late)])) if late.any() else np.nan,
                }
            )
            matrices_by_factor[factor].append(matrix)
            subjects_by_factor[factor].append(subject.subj)
    summary = pd.DataFrame(summaries)
    group_stats, group_mats = temporal_group_stats(matrices_by_factor, subjects_by_factor, windows)
    return summary, group_stats, group_mats


def temporal_group_stats(matrices_by_factor: dict[str, list[np.ndarray]], subjects_by_factor: dict[str, list[str]], windows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    group_mats = {}
    times = windows["window_center_ms"].to_numpy(dtype=float)
    for factor, mats in matrices_by_factor.items():
        if not mats:
            continue
        arr = np.stack(mats, axis=0)
        group_mats[factor] = np.nanmean(arr, axis=0)
        tvals, pvals = stats.ttest_1samp(arr, 0.5, axis=0, nan_policy="omit")
        for i, train_time in enumerate(times):
            for j, test_time in enumerate(times):
                rows.append(
                    {
                        "factor": factor,
                        "train_time_ms": train_time,
                        "test_time_ms": test_time,
                        "mean_AUC": float(np.nanmean(arr[:, i, j])),
                        "t": float(tvals[i, j]),
                        "p_uncorrected": float(pvals[i, j]),
                        "n_subjects": int(np.isfinite(arr[:, i, j]).sum()),
                    }
                )
    return pd.DataFrame(rows), group_mats


def cluster_permutation_2d(arr: np.ndarray, times: np.ndarray, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tvals, pvals = stats.ttest_1samp(arr, 0.5, axis=0, nan_policy="omit")
    mask = np.isfinite(tvals) & (pvals < 0.05) & (tvals > 0)
    labels, n_lab = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    masses = [float(np.nansum(tvals[labels == lab])) for lab in range(1, n_lab + 1)]
    centered = arr - 0.5
    max_masses = []
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=(arr.shape[0], 1, 1))
        perm = 0.5 + centered * signs
        pt, pp = stats.ttest_1samp(perm, 0.5, axis=0, nan_policy="omit")
        pmask = np.isfinite(pt) & (pp < 0.05) & (pt > 0)
        plabels, pn = ndimage.label(pmask, structure=np.ones((3, 3), dtype=int))
        max_masses.append(max([float(np.nansum(pt[plabels == lab])) for lab in range(1, pn + 1)], default=0.0))
    max_masses = np.asarray(max_masses)
    rows = []
    mean_mat = np.nanmean(arr, axis=0)
    for lab, mass in zip(range(1, n_lab + 1), masses):
        coords = np.argwhere(labels == lab)
        peak_coord = coords[np.nanargmax(tvals[labels == lab])]
        tag = interpret_tgm_cluster(times, coords)
        rows.append(
            {
                "cluster_train_start_ms": float(times[coords[:, 0].min()]),
                "cluster_train_end_ms": float(times[coords[:, 0].max()]),
                "cluster_test_start_ms": float(times[coords[:, 1].min()]),
                "cluster_test_end_ms": float(times[coords[:, 1].max()]),
                "cluster_mass": float(mass),
                "cluster_p": float((np.sum(max_masses >= mass) + 1) / (len(max_masses) + 1)),
                "peak_train_time_ms": float(times[peak_coord[0]]),
                "peak_test_time_ms": float(times[peak_coord[1]]),
                "peak_t": float(tvals[peak_coord[0], peak_coord[1]]),
                "peak_mean_AUC": float(mean_mat[peak_coord[0], peak_coord[1]]),
                "interpretation_tag": tag,
            }
        )
    return pd.DataFrame(rows)


def interpret_tgm_cluster(times: np.ndarray, coords: np.ndarray) -> str:
    train = times[coords[:, 0]]
    test = times[coords[:, 1]]
    diag_dist = np.abs(train - test)
    if np.nanmedian(diag_dist) <= 60 and np.nanpercentile(diag_dist, 75) <= 120:
        return "diagonal_transient"
    if ((train >= 140) & (train <= 190)).mean() > 0.25 and ((test >= 300) & (test <= 800)).mean() > 0.25:
        return "early_to_late_generalization"
    if ((train >= 300) & (train <= 800)).mean() > 0.5 and ((test >= 300) & (test <= 800)).mean() > 0.5:
        return "late_sustained"
    if (train.max() - train.min() > 300) and (test.max() - test.min() > 300):
        return "broad_generalization"
    return "unsupported"


def temporal_clusters_from_saved(ctx: Context, factors: list[str], windows: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    times = windows["window_center_ms"].to_numpy(dtype=float)
    for factor in factors:
        files = sorted(ctx.dirs["tgm_matrices"].glob(f"*_{factor}.npy"))
        if len(files) < 2:
            continue
        arr = np.stack([np.load(path) for path in files], axis=0)
        clusters = cluster_permutation_2d(arr, times, n_perm, seed + natural_subject_key(factor))
        for _, row in clusters.iterrows():
            rows.append({"factor": factor, **row.to_dict()})
    return pd.DataFrame(rows)


def structural_surface_time(subject_results: pd.DataFrame, ctx: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_version = preferred_decoding_version(subject_results)
    primary = subject_results[subject_results["decoding_version"] == primary_version]
    pivot = primary.pivot_table(index=["subj", "window_center_ms"], columns="factor", values="AUC").reset_index()
    if not set(FACTORS).issubset(pivot.columns):
        return pd.DataFrame(), pd.DataFrame()
    pivot["structural_AUC"] = pivot[STRUCTURAL].mean(axis=1)
    pivot["surface_AUC"] = pivot[SURFACE].mean(axis=1)
    pivot["structural_minus_surface"] = pivot["structural_AUC"] - pivot["surface_AUC"]
    rows = []
    for time_ms, sub in pivot.groupby("window_center_ms"):
        vals = sub["structural_minus_surface"].to_numpy(dtype=float)
        stat = one_sample_stats(vals + 0.5, 0.5)
        rows.append({"window_center_ms": time_ms, "mean_difference": float(np.nanmean(vals)), "t": stat["t"], "p_uncorrected": stat["p_uncorrected"], "n_subjects": stat["n_subjects"]})
    stats_df = pd.DataFrame(rows)
    stats_df["p_fdr"] = fdr_bh(stats_df["p_uncorrected"])
    diff_pivot = pivot.pivot(index="subj", columns="window_center_ms", values="structural_minus_surface").sort_index(axis=1) + 0.5
    clusters = cluster_permutation_1d(diff_pivot, ctx.n_permutations, ctx.seed + 9001, 0.5)
    return stats_df, clusters


def behavior_link(ctx: Context, subject_results: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    eprime_root = ctx.project_root / "EEGDATA" / "EEGDATA" / "eprime"
    if not eprime_root.exists():
        ctx.warn("Behavior link skipped: E-Prime behavior directory not found.")
        return pd.DataFrame()
    try:
        records = parse_eprime_records(eprime_root)
    except Exception as exc:
        ctx.warn(f"Behavior link skipped: {exc}")
        return pd.DataFrame()
    primary_version = preferred_decoding_version(subject_results)
    primary = subject_results[subject_results["decoding_version"] == primary_version]
    rows = []
    for factor in FACTORS:
        sig = clusters[(clusters["factor"] == factor) & (clusters["cluster_p"] < 0.05)] if not clusters.empty else pd.DataFrame()
        if not sig.empty:
            first = sig.sort_values("cluster_p").iloc[0]
            mask = (primary["factor"] == factor) & (primary["window_center_ms"].between(first["cluster_start_ms"], first["cluster_end_ms"]))
            strength = primary[mask].groupby("subj")["AUC"].mean()
            window_label = f"cluster_{first['cluster_start_ms']:.0f}_{first['cluster_end_ms']:.0f}"
        else:
            mask = (primary["factor"] == factor) & (primary["window_center_ms"].between(300, 800))
            strength = primary[mask].groupby("subj")["AUC"].mean()
            window_label = "theoretical_late_300_800"
        effects = behavior_factor_effects(records, factor)
        merged = effects.merge(strength.rename("decoding_strength"), left_on="subj", right_index=True, how="inner")
        for outcome in ["Beauty_effect_factor", "Naturalness_effect_factor", "Dissociation_effect_factor"]:
            x = merged["decoding_strength"].to_numpy(dtype=float)
            y = merged[outcome].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 4:
                continue
            pearson = stats.pearsonr(x[ok], y[ok])
            spearman = stats.spearmanr(x[ok], y[ok])
            rows.append(
                {
                    "factor": factor,
                    "strength_window": window_label,
                    "behavior_effect": outcome,
                    "n_subjects": int(ok.sum()),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["pearson_p_fdr"] = fdr_bh(df["pearson_p"])
        df["spearman_p_fdr"] = fdr_bh(df["spearman_p"])
    return df


def parse_eprime_records(eprime_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(eprime_root.rglob("*.txt")):
        raw = path.read_bytes()
        text = None
        for enc in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            continue
        subj_match = re.search(r"^Subject:\s*(\d+)\s*$", text, flags=re.MULTILINE)
        subj = f"s{subj_match.group(1)}" if subj_match else path.parent.name.lower()
        frames = re.findall(r"\*\*\* LogFrame Start \*\*\*(.*?)\*\*\* LogFrame End \*\*\*", text, flags=re.DOTALL)
        for frame in frames:
            row = {"subj": subj}
            for key in ["CondID", "FSlim", "Eye", "Mouth", "Skin", "Stimtype", "RealnessRating.RESP", "LikingRating.RESP"]:
                match = re.search(rf"^\s*{re.escape(key)}:\s*(.*?)\s*$", frame, flags=re.MULTILINE)
                row[key] = match.group(1).strip() if match else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("No E-Prime text rows found.")
    for col in ["CondID", "FSlim", "Eye", "Mouth", "Skin", "RealnessRating.RESP", "LikingRating.RESP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["CondID"].isin(range(2, 18)) & df["Stimtype"].isin(ANALYZED_STIMTYPES)].copy()
    df["Beauty_choice"] = df["LikingRating.RESP"]
    df["Naturalness_choice"] = df["RealnessRating.RESP"]
    return df


def behavior_factor_effects(records: pd.DataFrame, factor: str) -> pd.DataFrame:
    work = records.copy()
    if factor == "Eye":
        work[factor] = (work[factor] == 2).astype(int)
    elif factor == "Mouth":
        work[factor] = (work[factor] == 2).astype(int)
    else:
        work[factor] = (work[factor] == 1).astype(int)
    rows = []
    for subj, sub in work.groupby("subj"):
        means = sub.groupby(factor)[["Beauty_choice", "Naturalness_choice"]].mean()
        if {0, 1}.issubset(means.index):
            beauty = float(means.loc[1, "Beauty_choice"] - means.loc[0, "Beauty_choice"])
            natural = float(means.loc[1, "Naturalness_choice"] - means.loc[0, "Naturalness_choice"])
            rows.append({"subj": subj, "Beauty_effect_factor": beauty, "Naturalness_effect_factor": natural, "Dissociation_effect_factor": beauty - natural})
    return pd.DataFrame(rows)


def plot_time_overview(ctx: Context, subject_results: pd.DataFrame, group_stats: pd.DataFrame, clusters: pd.DataFrame) -> None:
    primary_version = preferred_decoding_version(subject_results)
    primary = subject_results[subject_results["decoding_version"] == primary_version]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, factor in zip(axes.ravel(), FACTORS):
        draw_timecourse(ax, primary[primary["factor"] == factor], factor, clusters)
    fig.suptitle("Time-resolved factor decoding", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(ctx, fig, "Fig1_time_resolved_decoding_overview")


def draw_timecourse(ax: plt.Axes, data: pd.DataFrame, factor: str, clusters: pd.DataFrame) -> None:
    for label, (a, b) in WINDOW_TAGS.items():
        ax.axvspan(a, b, color="#d8d4c6", alpha=0.22, lw=0)
    pivot = data.pivot_table(index="subj", columns="window_center_ms", values="AUC").sort_index(axis=1)
    times = pivot.columns.to_numpy(dtype=float)
    mean = pivot.mean(axis=0).to_numpy()
    sem = pivot.sem(axis=0).to_numpy()
    ci = 1.96 * sem
    ax.plot(times, mean, color=COLORS[factor], lw=2.2, label=factor)
    ax.fill_between(times, mean - ci, mean + ci, color=COLORS[factor], alpha=0.18)
    ax.axhline(0.5, color="black", lw=1, ls="--")
    ax.axvline(0, color="#555555", lw=0.8)
    if not clusters.empty:
        version = str(data["decoding_version"].dropna().iloc[0]) if "decoding_version" in data.columns and not data.empty else "nuisance_balanced"
        sig = clusters[(clusters["factor"] == factor) & (clusters["decoding_version"] == version) & (clusters["cluster_p"] < 0.05)]
        for _, row in sig.iterrows():
            ax.hlines(0.485, row["cluster_start_ms"], row["cluster_end_ms"], color=COLORS[factor], lw=4)
    if len(mean):
        peak = int(np.nanargmax(mean))
        clabel = "n.s."
        if not clusters.empty:
            version = str(data["decoding_version"].dropna().iloc[0]) if "decoding_version" in data.columns and not data.empty else "nuisance_balanced"
            subc = clusters[(clusters["factor"] == factor) & (clusters["decoding_version"] == version)]
            if not subc.empty:
                clabel = f"cluster p={subc['cluster_p'].min():.3f}"
        ax.text(0.02, 0.94, f"peak {mean[peak]:.3f} @ {times[peak]:.0f} ms\n{clabel}", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title(factor, fontweight="bold")
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.46, max(0.62, np.nanmax(mean + ci) + 0.02 if len(mean) else 0.62))
    ax.spines[["top", "right"]].set_visible(False)


def preferred_decoding_version(subject_results: pd.DataFrame) -> str:
    if subject_results.empty or "decoding_version" not in subject_results.columns:
        return "nuisance_balanced"
    versions = subject_results["decoding_version"].dropna().astype(str)
    if versions.empty:
        return "nuisance_balanced"
    if (versions == "nuisance_balanced").any():
        return "nuisance_balanced"
    return str(versions.iloc[0])


def plot_all_factors(ctx: Context, subject_results: pd.DataFrame, clusters: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    primary_version = preferred_decoding_version(subject_results)
    primary = subject_results[subject_results["decoding_version"] == primary_version]
    for factor in FACTORS:
        data = primary[primary["factor"] == factor]
        pivot = data.pivot_table(index="subj", columns="window_center_ms", values="AUC").sort_index(axis=1)
        times = pivot.columns.to_numpy(dtype=float)
        mean = pivot.mean(axis=0).to_numpy()
        ci = 1.96 * pivot.sem(axis=0).to_numpy()
        ax.plot(times, mean, color=COLORS[factor], lw=2.1, label=factor)
        ax.fill_between(times, mean - ci, mean + ci, color=COLORS[factor], alpha=0.08)
        if not clusters.empty:
            sig = clusters[(clusters["factor"] == factor) & (clusters["decoding_version"] == primary_version) & (clusters["cluster_p"] < 0.05)]
            ybar = 0.475 - FACTORS.index(factor) * 0.006
            for _, row in sig.iterrows():
                ax.hlines(ybar, row["cluster_start_ms"], row["cluster_end_ms"], color=COLORS[factor], lw=4)
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_title("All-factor decoding comparison", fontweight="bold")
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("AUC")
    ax.legend(frameon=False, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(ctx, fig, "Fig2_all_factor_decoding_comparison")


def plot_tgm(ctx: Context, group_mats: dict[str, np.ndarray], tgm_clusters: pd.DataFrame, windows: pd.DataFrame) -> None:
    if not group_mats:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    times = windows["window_center_ms"].to_numpy(dtype=float)
    extent = [times.min(), times.max(), times.max(), times.min()]
    for ax, factor in zip(axes.ravel(), FACTORS):
        mat = group_mats.get(factor)
        if mat is None:
            ax.axis("off")
            continue
        im = ax.imshow(mat, extent=extent, aspect="auto", cmap="RdBu_r", vmin=0.44, vmax=0.56)
        ax.plot([times.min(), times.max()], [times.min(), times.max()], color="black", lw=0.8, alpha=0.5)
        ax.axhline(165, color="black", lw=0.6, ls=":")
        ax.axvline(165, color="black", lw=0.6, ls=":")
        ax.axhline(300, color="black", lw=0.6, ls=":")
        ax.axvline(300, color="black", lw=0.6, ls=":")
        tag = "unsupported"
        if not tgm_clusters.empty:
            sig = tgm_clusters[(tgm_clusters["factor"] == factor) & (tgm_clusters["cluster_p"] < 0.05)]
            if not sig.empty:
                tag = sig.sort_values("cluster_p").iloc[0]["interpretation_tag"]
                for _, row in sig.iterrows():
                    rect = plt.Rectangle(
                        (row["cluster_test_start_ms"], row["cluster_train_start_ms"]),
                        row["cluster_test_end_ms"] - row["cluster_test_start_ms"],
                        row["cluster_train_end_ms"] - row["cluster_train_start_ms"],
                        fill=False,
                        ec="black",
                        lw=1.3,
                    )
                    ax.add_patch(rect)
        ax.set_title(f"{factor}: {tag}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Test time (ms)")
        ax.set_ylabel("Train time (ms)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Mean AUC")
    fig.suptitle("Temporal generalization", fontsize=15, fontweight="bold")
    save_fig(ctx, fig, "Fig3_temporal_generalization_panel")


def plot_structural_surface(ctx: Context, subject_results: pd.DataFrame, clusters: pd.DataFrame) -> None:
    primary_version = preferred_decoding_version(subject_results)
    primary = subject_results[subject_results["decoding_version"] == primary_version]
    pivot = primary.pivot_table(index=["subj", "window_center_ms"], columns="factor", values="AUC").reset_index()
    if not set(FACTORS).issubset(pivot.columns):
        return
    pivot["structural"] = pivot[STRUCTURAL].mean(axis=1)
    pivot["surface"] = pivot[SURFACE].mean(axis=1)
    pivot["diff"] = pivot["structural"] - pivot["surface"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for label, color in [("structural", "#0f766e"), ("surface", "#a85528")]:
        mat = pivot.pivot(index="subj", columns="window_center_ms", values=label).sort_index(axis=1)
        times = mat.columns.to_numpy(dtype=float)
        mean = mat.mean(axis=0).to_numpy()
        ci = 1.96 * mat.sem(axis=0).to_numpy()
        ax1.plot(times, mean, label=label, color=color, lw=2.2)
        ax1.fill_between(times, mean - ci, mean + ci, color=color, alpha=0.12)
    diff_mat = pivot.pivot(index="subj", columns="window_center_ms", values="diff").sort_index(axis=1)
    times = diff_mat.columns.to_numpy(dtype=float)
    mean = diff_mat.mean(axis=0).to_numpy()
    ci = 1.96 * diff_mat.sem(axis=0).to_numpy()
    ax2.plot(times, mean, color="#334155", lw=2.2)
    ax2.fill_between(times, mean - ci, mean + ci, color="#334155", alpha=0.14)
    ax2.axhline(0, color="black", ls="--", lw=1)
    if not clusters.empty:
        for _, row in clusters[clusters["cluster_p"] < 0.05].iterrows():
            ax2.hlines(np.nanmin(mean - ci) - 0.004, row["cluster_start_ms"], row["cluster_end_ms"], color="#334155", lw=4)
    for ax in [ax1, ax2]:
        ax.axvline(0, color="#555555", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.axhline(0.5, color="black", ls="--", lw=1)
    ax1.set_ylabel("AUC")
    ax1.set_title("Structural versus surface/local decoding", fontweight="bold")
    ax1.legend(frameon=False)
    ax2.set_ylabel("Structural - surface/local AUC")
    ax2.set_xlabel("Time from stimulus onset (ms)")
    fig.tight_layout()
    save_fig(ctx, fig, "Fig4_structural_vs_surface_comparison")


def plot_cluster_summary(ctx: Context, subject_results: pd.DataFrame, clusters: pd.DataFrame) -> None:
    primary_version = preferred_decoding_version(subject_results)
    primary = subject_results[subject_results["decoding_version"] == primary_version]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(FACTORS))
    for i, factor in enumerate(FACTORS):
        data = primary[primary["factor"] == factor]
        pivot = data.pivot_table(index="subj", columns="window_center_ms", values="AUC").sort_index(axis=1)
        mean = pivot.mean(axis=0)
        peak_time = float(mean.idxmax()) if len(mean) else np.nan
        peak_auc = float(mean.max()) if len(mean) else np.nan
        sig = clusters[(clusters["factor"] == factor) & (clusters["decoding_version"] == primary_version)] if not clusters.empty else pd.DataFrame()
        if not sig.empty:
            best = sig.sort_values("cluster_p").iloc[0]
            if best["cluster_p"] < 0.05:
                ax.hlines(i, best["cluster_start_ms"], best["cluster_end_ms"], color=COLORS[factor], lw=8, alpha=0.8)
                label = f"peak {peak_auc:.3f} @ {peak_time:.0f} ms, p={best['cluster_p']:.3f}"
            else:
                ax.plot(peak_time, i, "o", color=COLORS[factor])
                label = f"peak {peak_auc:.3f} @ {peak_time:.0f} ms, n.s."
        else:
            ax.plot(peak_time, i, "o", color=COLORS[factor])
            label = f"peak {peak_auc:.3f} @ {peak_time:.0f} ms, n.s."
        ax.text(1010, i, label, va="center", fontsize=9)
    ax.set_yticks(y, FACTORS)
    ax.set_xlim(-220, 1220)
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_title("Decoding cluster summary", fontweight="bold")
    ax.axvline(0, color="#555555", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(ctx, fig, "Fig5_decoding_cluster_summary")


def save_fig(ctx: Context, fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        path = ctx.dirs["paper_ready"] / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        ctx.add_file(path)
    plt.close(fig)


def write_summaries(
    ctx: Context,
    subjects: list[SubjectEpochs],
    overview: pd.DataFrame,
    subject_results: pd.DataFrame,
    clusters: pd.DataFrame,
    tgm_clusters: pd.DataFrame,
    ss_clusters: pd.DataFrame,
    behavior_stats: pd.DataFrame,
) -> None:
    if subject_results.empty:
        primary = subject_results
        primary_version = "nuisance_balanced"
    elif (subject_results["decoding_version"] == "nuisance_balanced").any():
        primary_version = "nuisance_balanced"
        primary = subject_results[subject_results["decoding_version"] == primary_version]
    else:
        primary_version = str(subject_results["decoding_version"].dropna().iloc[0])
        primary = subject_results[subject_results["decoding_version"] == primary_version]
    rows = []
    for factor in FACTORS:
        data = primary[primary["factor"] == factor]
        mean = data.pivot_table(index="subj", columns="window_center_ms", values="AUC").mean(axis=0)
        peak_auc = float(mean.max()) if len(mean) else np.nan
        peak_time = float(mean.idxmax()) if len(mean) else np.nan
        sig = clusters[(clusters["factor"] == factor) & (clusters["decoding_version"] == primary_version)] if not clusters.empty else pd.DataFrame()
        best = sig.sort_values("cluster_p").iloc[0].to_dict() if not sig.empty else {}
        tgm = tgm_clusters[tgm_clusters["factor"] == factor].sort_values("cluster_p").iloc[0].to_dict() if not tgm_clusters.empty and (tgm_clusters["factor"] == factor).any() else {}
        p = best.get("cluster_p", np.nan)
        level = "Strong" if np.isfinite(p) and p < 0.05 else ("Moderate" if np.isfinite(p) and p < 0.10 else "Suggestive" if len(mean) and float(mean.max()) > 0.52 else "Unsupported")
        rows.append(
            {
                "factor": factor,
                "peak AUC": f"{peak_auc:.3f}" if np.isfinite(peak_auc) else "NA",
                "peak time": f"{peak_time:.0f} ms" if np.isfinite(peak_time) else "NA",
                "significant cluster": f"{best.get('cluster_start_ms', np.nan):.0f}-{best.get('cluster_end_ms', np.nan):.0f} ms" if np.isfinite(p) and p < 0.05 else "n.s.",
                "cluster p": f"{p:.3f}" if np.isfinite(p) else "NA",
                "temporal generalization": tgm.get("interpretation_tag", "unsupported"),
                "interpretation": "cluster-corrected decoding" if level == "Strong" else "no stable cluster-corrected evidence",
                "paper-use level": level,
            }
        )
    summary_table = pd.DataFrame(rows)
    fig_paths = sorted(str(p.resolve()) for p in ctx.dirs["paper_ready"].glob("Fig*.png"))
    md = [
        "# Time-resolved MVPA Decoding Results",
        "",
        f"- Data read: {'yes' if subjects else 'no'}",
        f"- Included subjects: {len(subjects)}",
        f"- Total edited formal trials in metadata: {int(overview['edited_trials'].sum()) if not overview.empty else 0}",
        f"- Channels: {int(overview['n_channels'].median()) if not overview.empty else 'NA'}",
        f"- Time range: {subjects[0].times.min():.0f} to {subjects[0].times.max():.0f} ms" if subjects else "- Time range: NA",
        "",
        "## Factor Overview",
        df_to_md(summary_table),
        "",
        "## Interpretation",
        recommended_statement(summary_table),
        "",
        "## Structural vs Surface/Local",
        "Cluster-corrected structural versus surface/local differences were detected." if not ss_clusters.empty and (ss_clusters["cluster_p"] < 0.05).any() else "No cluster-corrected structural versus surface/local time-resolved difference was detected.",
        "",
        "## EEG-Behavior Link",
        "Exploratory behavior-link correlations were computed; treat them as non-causal exploratory results." if not behavior_stats.empty else "Behavior-link analysis was skipped or produced no stable rows.",
        "",
        "## Paper-ready Figures",
        *[f"- {path}" for path in fig_paths],
        "",
        "## Warnings",
        *[f"- {w}" for w in ctx.warnings[:80]],
    ]
    md_path = ctx.dirs["summaries"] / "DECODING_FINAL_RESULTS_OVERVIEW.md"
    save_text(ctx, "\n".join(md), md_path)
    js = {
        "n_subjects": len(subjects),
        "n_trials": int(overview["edited_trials"].sum()) if not overview.empty else 0,
        "n_channels": int(overview["n_channels"].median()) if not overview.empty else None,
        "time_range": [float(subjects[0].times.min()), float(subjects[0].times.max())] if subjects else None,
        "factor_results": summary_table.to_dict("records"),
        "time_resolved_clusters": clusters.to_dict("records") if not clusters.empty else [],
        "temporal_generalization_clusters": tgm_clusters.to_dict("records") if not tgm_clusters.empty else [],
        "structural_surface_results": ss_clusters.to_dict("records") if not ss_clusters.empty else [],
        "behavior_link_results": behavior_stats.to_dict("records") if not behavior_stats.empty else [],
        "paper_ready_figures": fig_paths,
        "warnings": ctx.warnings,
        "generated_files": ctx.generated_files,
    }
    save_json(ctx, js, ctx.dirs["summaries"] / "DECODING_FINAL_RESULTS_OVERVIEW.json")


def recommended_statement(summary_table: pd.DataFrame) -> str:
    strong = summary_table.loc[summary_table["paper-use level"] == "Strong", "factor"].tolist()
    if strong:
        return (
            "Time-resolved MVPA tested whether controlled facial-editing factors were represented in distributed EEG patterns. "
            f"{', '.join(strong)} showed cluster-corrected decoding in the time-resolved analysis. Temporal generalization results "
            "indicate whether these neural codes were transient, sustained, or generalized across evaluation stages. These results "
            "support treating digital facial edits as distributed and temporally evolving EEG patterns rather than isolated single-electrode ERP effects."
        )
    return (
        "Time-resolved MVPA did not identify cluster-corrected factor decoding in the completed runs. The curves and peaks may be useful "
        "for exploratory planning, but the current evidence should not be phrased as stable EEG decoding of the facial-editing factors."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--outdir", default="mvpa_outputs")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--run-time-resolved", action="store_true")
    parser.add_argument("--run-temporal-generalization", action="store_true")
    parser.add_argument("--run-structural-surface", action="store_true")
    parser.add_argument("--run-behavior-link", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--window-ms", type=int, default=40)
    parser.add_argument("--step-ms", type=int, default=10)
    parser.add_argument("--tmin", type=int, default=-200)
    parser.add_argument("--tmax", type=int, default=1000)
    parser.add_argument("--classifier", choices=["logreg", "linearsvm"], default="logreg")
    parser.add_argument("--feature-mode", choices=["window_mean", "flattened"], default="window_mean")
    parser.add_argument("--channel-mode", choices=["shared_scalp", "posterior_roi", "n170_lpp_roi", "all"], default="shared_scalp")
    parser.add_argument("--factors", default="all", help="Comma-separated factors to decode, or all.")
    parser.add_argument("--pseudotrial-size", type=int, default=1, help="Average trials within target+nuisance cells before decoding.")
    parser.add_argument("--cv-mode", choices=["nuisance_balanced", "stratified_trial", "condition_generalization", "leave_one_identity_out"], default="nuisance_balanced")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    if args.run_all:
        args.run_time_resolved = args.run_temporal_generalization = args.run_structural_surface = args.run_behavior_link = True
    if not any([args.run_time_resolved, args.run_temporal_generalization, args.run_structural_surface, args.run_behavior_link]):
        args.run_time_resolved = True
    if args.quick_test:
        args.n_permutations = min(args.n_permutations, 100)
        args.window_ms = max(args.window_ms, 80)
        args.step_ms = max(args.step_ms, 20)

    project_root = args.project_root.resolve()
    dirs = setup_dirs(project_root, args.outdir)
    setup_logging(dirs["logs"] / "run_time_resolved_factor_decoding.log")
    ctx = Context(project_root, dirs["root"], args.seed, args.n_permutations, args.quick_test, dirs)
    logging.info("Starting MVPA decoding: %s", vars(args))

    condition_map = parse_eprime_condition_map(project_root, ctx)
    if condition_map:
        cond_rows = [{"raw_cond_id": raw, "cond_id": raw - 1, **vals} for raw, vals in sorted(condition_map.items())]
        save_csv(ctx, pd.DataFrame(cond_rows), dirs["tables"] / "mvpa_verified_condition_factor_map.csv")

    files = find_eeg_files(project_root)
    if not files:
        ctx.warn("No trial-level EEG epoch files found. Expected derivatives_eeglab_s*/s*_epoched_stim.set with matching .fdt.")
        write_summaries(ctx, [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return
    subjects = [sub for path in files if (sub := load_subject(path, ctx, condition_map)) is not None]
    if not subjects:
        ctx.warn("No subjects could be loaded after filtering.")
        write_summaries(ctx, [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return
    assign_analysis_channels(subjects, args.channel_mode, ctx)

    missing_expected = sorted(DEFAULT_SUBJECTS - {s.subj for s in subjects}, key=natural_subject_key)
    if missing_expected:
        ctx.warn(f"Expected subjects missing from loaded dataset: {missing_expected}")
    overview, all_meta = data_overview(subjects)
    save_csv(ctx, overview, dirs["tables"] / "mvpa_data_overview.csv")
    windows = make_windows(subjects[0].times, args.window_ms, args.step_ms, args.tmin, args.tmax)
    subject_results = pd.DataFrame()
    count_rows = []

    if args.run_time_resolved:
        result_chunks = []
        versions = ["nuisance_balanced"]
        if args.cv_mode == "stratified_trial":
            versions = ["stratified_trial"]
        elif args.cv_mode in {"condition_generalization", "leave_one_identity_out"}:
            versions = ["nuisance_balanced"]
        elif not args.quick_test:
            versions.append("stratified_trial")
        for subject in subjects:
            for version in versions:
                res, counts = run_subject_time_resolved(subject, ctx, args, windows, version)
                if not res.empty:
                    result_chunks.append(res)
                if not counts.empty:
                    count_rows.append(counts)
        subject_results = pd.concat(result_chunks, ignore_index=True) if result_chunks else pd.DataFrame()
        trial_counts = pd.concat(count_rows, ignore_index=True) if count_rows else pd.DataFrame()
        save_csv(ctx, trial_counts, dirs["tables"] / "mvpa_subject_trial_counts.csv")
        skipped = trial_counts[trial_counts["skipped"]] if not trial_counts.empty else pd.DataFrame()
        save_csv(ctx, skipped, dirs["tables"] / "mvpa_missingness_and_skipped_subjects.csv")
        save_text(ctx, preprocessing_summary(subjects, overview, skipped), dirs["summaries"] / "mvpa_data_preprocessing_summary.md")
        save_csv(ctx, subject_results, dirs["tables"] / "time_resolved_decoding_subject_results.csv")
        group_stats = group_stats_time(subject_results) if not subject_results.empty else pd.DataFrame()
        save_csv(ctx, group_stats, dirs["tables"] / "time_resolved_decoding_group_stats.csv")
        clusters = cluster_table_time(subject_results, ctx.n_permutations, ctx.seed) if not subject_results.empty else pd.DataFrame()
        save_csv(ctx, clusters, dirs["tables"] / "time_resolved_decoding_clusters.csv")
    else:
        path = dirs["tables"] / "time_resolved_decoding_subject_results.csv"
        if path.exists():
            subject_results = read_csv_safe(path)
            clusters = read_csv_safe(dirs["tables"] / "time_resolved_decoding_clusters.csv")
            group_stats = read_csv_safe(dirs["tables"] / "time_resolved_decoding_group_stats.csv")
        else:
            clusters = group_stats = pd.DataFrame()

    tgm_clusters = pd.DataFrame()
    group_mats: dict[str, np.ndarray] = {}
    if args.run_temporal_generalization:
        tgm_summary, tgm_stats, group_mats = run_temporal_generalization(subjects, ctx, args, windows)
        save_csv(ctx, tgm_summary, dirs["tables"] / "temporal_generalization_subject_summary.csv")
        save_csv(ctx, tgm_stats, dirs["tables"] / "temporal_generalization_group_stats.csv")
        tgm_factors = ["FSlim", "Eye"] if args.quick_test else FACTORS
        tgm_clusters = temporal_clusters_from_saved(ctx, tgm_factors, windows, ctx.n_permutations, ctx.seed)
        save_csv(ctx, tgm_clusters, dirs["tables"] / "temporal_generalization_clusters.csv")

    ss_stats = ss_clusters = pd.DataFrame()
    if args.run_structural_surface and not subject_results.empty:
        ss_stats, ss_clusters = structural_surface_time(subject_results, ctx)
        save_csv(ctx, ss_stats, dirs["tables"] / "structural_vs_surface_time_resolved_stats.csv")
        save_csv(ctx, ss_clusters, dirs["tables"] / "structural_vs_surface_time_resolved_clusters.csv")
        save_csv(ctx, pd.DataFrame(), dirs["tables"] / "structural_vs_surface_temporal_generalization_stats.csv")
        save_csv(ctx, pd.DataFrame(), dirs["tables"] / "structural_vs_surface_temporal_generalization_clusters.csv")

    behavior_stats = pd.DataFrame()
    if args.run_behavior_link and not subject_results.empty:
        behavior_stats = behavior_link(ctx, subject_results, clusters)
        if not behavior_stats.empty:
            save_csv(ctx, behavior_stats, dirs["tables"] / "decoding_behavior_link_stats.csv")
        else:
            ctx.warn("Behavior link skipped or empty; no decoding_behavior_link_stats.csv written.")

    if not subject_results.empty:
        plot_time_overview(ctx, subject_results, group_stats, clusters)
        plot_all_factors(ctx, subject_results, clusters)
        plot_structural_surface(ctx, subject_results, ss_clusters)
        plot_cluster_summary(ctx, subject_results, clusters)
    if group_mats:
        plot_tgm(ctx, group_mats, tgm_clusters, windows)

    write_summaries(ctx, subjects, overview, subject_results, clusters, tgm_clusters, ss_clusters, behavior_stats)
    print("\nGenerated files:")
    for path in ctx.generated_files:
        print(path)


def data_overview(subjects: list[SubjectEpochs]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, metas = [], []
    for subject in subjects:
        meta = subject.metadata.copy()
        formal = meta[meta["raw_cond_id"].isin(FORMAL_RAW_CONDITIONS) & meta["stimtype"].isin(ANALYZED_STIMTYPES)]
        row = {
            "subj": subject.subj,
            "set_path": str(subject.set_path),
            "total_epoch_rows": len(meta),
            "edited_trials": len(formal),
            "n_channels": len(subject.analysis_channels) if subject.analysis_channels else len(subject.channels),
            "raw_retained_channels": len(subject.channels),
            "sample_rate": subject.sample_rate,
            "time_min_ms": float(subject.times.min()),
            "time_max_ms": float(subject.times.max()),
            "baseline_corrected_in_script": True,
        }
        for factor in FACTORS:
            counts = formal[factor].value_counts().to_dict()
            row[f"{factor}_class0"] = int(counts.get(0, 0))
            row[f"{factor}_class1"] = int(counts.get(1, 0))
        rows.append(row)
        metas.append(formal)
    return pd.DataFrame(rows), pd.concat(metas, ignore_index=True)


def preprocessing_summary(subjects: list[SubjectEpochs], overview: pd.DataFrame, skipped: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# MVPA Data Preprocessing Summary",
            "",
            f"- Loaded subjects: {len(subjects)}",
            f"- Edited formal trials: {int(overview['edited_trials'].sum()) if not overview.empty else 0}",
            f"- Median channels: {int(overview['n_channels'].median()) if not overview.empty else 'NA'}",
            "- Original/control and attention-check trials were excluded.",
            "- Baseline correction used -200 to 0 ms inside the script.",
            "- Existing preprocessed epochs were used as stored; no filtering or interpolation was rerun.",
            "",
            "## Skipped Subject/Task Rows",
            df_to_md(skipped) if skipped is not None and not skipped.empty else "No skipped subject-factor rows.",
        ]
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()

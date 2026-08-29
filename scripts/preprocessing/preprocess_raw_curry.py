#!/usr/bin/env python
"""Rebuild trial-level EEG epochs from the original Curry recordings.

The pipeline aligns Curry trigger pairs to each participant's E-Prime log
before preprocessing. It never concatenates interrupted acquisitions blindly.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("IPM_DATA_ROOT", REPO_ROOT / "data")).resolve()
RAW_ROOT = ROOT / "EEGDATA" / "EEGDATA" / "curry8" / "Daya-zqy26"
EPRIME_ROOT = ROOT / "EEGDATA" / "EEGDATA" / "eprime"
OUT_ROOT = ROOT / "paper_extension_final" / "reanalysis_30"
FACTORS = ["FSlim", "Eye", "Mouth", "Skin"]
IDENTITY_BASE = {"F_1": 100, "F_2": 120, "M_1": 140, "M_2": 160, "T": 180}


@dataclass
class Alignment:
    subject: str
    acquisition: str
    raw_pairs: int
    expected_trials: int
    matched_trials: int
    matched_formal: int
    matched_attention: int
    unmatched_raw_pairs: int
    unmatched_log_trials: int
    longest_match: int


def subject_number(value: str) -> int:
    match = re.fullmatch(r"s(\d+)", value.lower())
    if not match:
        raise ValueError(f"Invalid subject label: {value}")
    return int(match.group(1))


def parse_eprime(path: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in path.read_text(encoding="utf-16", errors="ignore").splitlines():
        line = raw_line.strip()
        if line == "*** LogFrame Start ***":
            current = {}
        elif line == "*** LogFrame End ***":
            if current and current.get("Procedure") == "expProc":
                records.append(current)
            current = None
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()

    frame = pd.DataFrame(records)
    if len(frame) != 468:
        raise RuntimeError(f"Expected 468 E-Prime trials in {path}, found {len(frame)}")
    for column in ["CondID", "IsBack"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for column in FACTORS:
        values = frame[column].replace({"N": 0, "Y": 1, "No": 0, "Yes": 1})
        frame[column] = pd.to_numeric(values, errors="raise").astype(int)
    frame["expected_code"] = [IDENTITY_BASE[s] + c for s, c in zip(frame["Stimtype"], frame["CondID"])]
    frame["log_trial"] = np.arange(1, len(frame) + 1)
    frame["identity"] = frame["Stimtype"]
    frame["attention_check"] = frame["Stimtype"].eq("T") | frame["IsBack"].eq(1)
    frame["control_original"] = (~frame["attention_check"]) & frame["CondID"].eq(1)
    frame["factorial"] = (~frame["attention_check"]) & frame["CondID"].between(2, 17)
    return frame


def read_raw(path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_curry(path, preload=False, verbose="ERROR")
    if "Trigger" in raw.ch_names:
        raw.set_channel_types({"Trigger": "stim"})
    if "VEOG" in raw.ch_names:
        raw.set_channel_types({"VEOG": "eog"})
    rename = {}
    if "CB1" in raw.ch_names:
        rename["CB1"] = "PO9"
    if "CB2" in raw.ch_names:
        rename["CB2"] = "PO10"
    if rename:
        raw.rename_channels(rename)
    montage = mne.channels.make_standard_montage("standard_1005")
    canonical = {name.upper(): name for name in montage.ch_names}
    case_rename = {
        name: canonical[name.upper()]
        for name in raw.ch_names
        if name.upper() in canonical and name != canonical[name.upper()]
    }
    if case_rename:
        raw.rename_channels(case_rename)
    raw.set_montage(montage, on_missing="warn")
    return raw


def trigger_pairs(raw: mne.io.BaseRaw) -> pd.DataFrame:
    descriptions = [str(value).strip() for value in raw.annotations.description]
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    rows = []
    starts = [idx for idx, value in enumerate(descriptions) if value == "10"]
    for position, idx in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else len(descriptions)
        end_idx = next(
            (j for j in range(idx + 1, stop) if descriptions[j].isdigit() and 101 <= int(descriptions[j]) <= 197),
            None,
        )
        if end_idx is not None:
            rows.append(
                {
                    "raw_pair": len(rows) + 1,
                    "onset_s": float(onsets[idx]),
                    "end_onset_s": float(onsets[end_idx]),
                    "observed_code": int(descriptions[end_idx]),
                }
            )
    return pd.DataFrame(rows)


def align_pairs(pairs: pd.DataFrame, expected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    observed_codes = pairs["observed_code"].tolist()
    expected_codes = expected["expected_code"].tolist()
    matcher = difflib.SequenceMatcher(a=observed_codes, b=expected_codes, autojunk=False)
    matches = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            matches.append((block.a + offset, block.b + offset))
    if not matches:
        raise RuntimeError("No Curry trigger pairs matched the E-Prime trial sequence")
    raw_idx, log_idx = zip(*matches)
    aligned = pairs.iloc[list(raw_idx)].reset_index(drop=True).join(
        expected.iloc[list(log_idx)].reset_index(drop=True), rsuffix="_log"
    )
    if not np.array_equal(aligned["observed_code"], aligned["expected_code"]):
        raise AssertionError("Internal event-alignment failure")
    metrics = {
        "matched": len(aligned),
        "unmatched_raw": len(pairs) - len(aligned),
        "unmatched_log": len(expected) - len(aligned),
        "longest_match": max((block.size for block in matcher.get_matching_blocks()), default=0),
    }
    return aligned, metrics


def find_best_acquisition(subject: str, expected: pd.DataFrame) -> tuple[Path, mne.io.BaseRaw, pd.DataFrame, Alignment]:
    candidates = []
    for path in sorted((RAW_ROOT / subject).glob("*.cdt")):
        raw = read_raw(path)
        pairs = trigger_pairs(raw)
        aligned, metrics = align_pairs(pairs, expected)
        candidates.append((metrics["matched"], metrics["longest_match"], path, raw, pairs, aligned, metrics))
    if not candidates:
        raise FileNotFoundError(f"No Curry acquisition found for {subject}")
    _, _, path, raw, pairs, aligned, metrics = max(candidates, key=lambda item: (item[0], item[1]))
    audit = Alignment(
        subject=subject,
        acquisition=path.name,
        raw_pairs=len(pairs),
        expected_trials=len(expected),
        matched_trials=len(aligned),
        matched_formal=int((~aligned["attention_check"]).sum()),
        matched_attention=int(aligned["attention_check"].sum()),
        unmatched_raw_pairs=metrics["unmatched_raw"],
        unmatched_log_trials=metrics["unmatched_log"],
        longest_match=metrics["longest_match"],
    )
    return path, raw, aligned, audit


def eprime_file(subject: str) -> Path:
    number = subject_number(subject)
    files = sorted(EPRIME_ROOT.glob(f"S*/*FaceExperiment-{number}-*.txt"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one E-Prime text log for {subject}, found {len(files)}")
    return files[0]


def detect_bad_channels(raw: mne.io.BaseRaw, seed: int) -> tuple[list[str], dict[str, list[str]]]:
    qc = raw.copy().pick("eeg").filter(1.0, 30.0, verbose="ERROR").resample(250, verbose="ERROR")
    noisy = NoisyChannels(qc, do_detrend=True, random_state=seed, matlab_strict=False)
    noisy.find_all_bads(ransac=True, correlation=True)
    categories = {
        "bad_by_nan": list(noisy.bad_by_nan),
        "bad_by_flat": list(noisy.bad_by_flat),
        "bad_by_deviation": list(noisy.bad_by_deviation),
        "bad_by_hf_noise": list(noisy.bad_by_hf_noise),
        "bad_by_correlation": list(noisy.bad_by_correlation),
        "bad_by_ransac": list(noisy.bad_by_ransac),
    }
    bads = sorted(set().union(*categories.values()))
    return bads, categories


def fit_and_apply_ica(raw: mne.io.BaseRaw, event_samples: np.ndarray, seed: int) -> tuple[ICA, dict]:
    fit_raw = raw.copy().filter(1.0, 30.0, verbose="ERROR")
    fit_events = np.column_stack([event_samples, np.zeros(len(event_samples), int), np.ones(len(event_samples), int)])
    fit_epochs = mne.Epochs(
        fit_raw,
        fit_events,
        event_id={"face": 1},
        tmin=-0.2,
        tmax=1.0,
        baseline=None,
        picks="eeg",
        preload=True,
        reject={"eeg": 1_000e-6},
        reject_by_annotation=True,
        verbose="ERROR",
    )
    ica = ICA(n_components=0.99, method="infomax", fit_params={"extended": True}, random_state=seed, max_iter="auto")
    ica.fit(fit_epochs, decim=4, verbose="ERROR")
    labels = label_components(fit_epochs, ica, method="iclabel")
    eye_labels = {"eye blink", "eye movement"}
    exclude = [
        idx
        for idx, (label, probability) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
        if label in eye_labels and float(probability) >= 0.90
    ]
    ica.exclude = exclude
    ica.apply(raw, verbose="ERROR")
    audit = {
        "n_components": int(ica.n_components_),
        "excluded_components": exclude,
        "excluded_labels": [labels["labels"][idx] for idx in exclude],
        "excluded_probabilities": [float(labels["y_pred_proba"][idx]) for idx in exclude],
        "all_labels": list(labels["labels"]),
        "all_label_probabilities": [float(value) for value in labels["y_pred_proba"]],
        "ica_fit_epochs": len(fit_epochs),
    }
    return ica, audit


def preprocess_subject(
    subject: str,
    overwrite: bool,
    seed: int,
    n_jobs: int,
    artifact_method: str,
) -> dict:
    epoch_dir = OUT_ROOT / "epochs"
    qc_dir = OUT_ROOT / "qc"
    alignment_dir = OUT_ROOT / "alignment"
    for directory in (epoch_dir, qc_dir, alignment_dir):
        directory.mkdir(parents=True, exist_ok=True)
    epoch_path = epoch_dir / f"{subject}-epo.fif"
    qc_path = qc_dir / f"{subject}_qc.json"
    if epoch_path.exists() and qc_path.exists() and not overwrite:
        return json.loads(qc_path.read_text(encoding="utf-8"))

    expected = parse_eprime(eprime_file(subject))
    acquisition, raw, aligned, alignment = find_best_acquisition(subject, expected)
    aligned.to_csv(alignment_dir / f"{subject}_event_alignment.csv", index=False, encoding="utf-8-sig")
    formal = aligned.loc[~aligned["attention_check"]].copy().reset_index(drop=True)
    if len(formal) < 400:
        logging.warning("%s has only %d aligned formal trials before EEG cleaning", subject, len(formal))

    crop_start = max(0.0, float(formal["onset_s"].min()) - 5.0)
    crop_stop = min(raw.times[-1], float(formal["onset_s"].max()) + 5.0)
    raw.crop(crop_start, crop_stop).load_data(verbose="ERROR")
    formal["onset_cropped_s"] = formal["onset_s"] - crop_start
    event_samples = raw.first_samp + np.rint(
        formal["onset_cropped_s"].to_numpy() * raw.info["sfreq"]
    ).astype(int)

    bads, bad_categories = detect_bad_channels(raw, seed)
    raw.info["bads"] = bads
    n_eeg = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
    bad_fraction = len(bads) / n_eeg
    if bad_fraction > 0.25:
        logging.warning("%s has %d/%d globally bad EEG channels", subject, len(bads), n_eeg)
    raw.filter(0.1, 30.0, picks="eeg", verbose="ERROR")
    raw.notch_filter([50.0], picks="eeg", verbose="ERROR")
    if bads:
        raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    ica, ica_audit = fit_and_apply_ica(raw, event_samples, seed)

    events = np.column_stack([event_samples, np.zeros(len(event_samples), int), np.arange(1, len(event_samples) + 1)])
    metadata_columns = [
        "log_trial",
        "identity",
        "Stimtype",
        "CondID",
        "Picture",
        *FACTORS,
        "control_original",
        "factorial",
        "observed_code",
        "raw_pair",
    ]
    metadata = formal[metadata_columns].copy()
    epochs = mne.Epochs(
        raw,
        events,
        event_id=None,
        tmin=-0.2,
        tmax=1.0,
        baseline=(-0.2, 0.0),
        picks="eeg",
        preload=True,
        metadata=metadata,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    epochs.resample(250, npad="auto", verbose="ERROR")
    epoch_construction_drop_reasons = {}
    for reasons in epochs.drop_log:
        for reason in reasons:
            epoch_construction_drop_reasons[reason] = epoch_construction_drop_reasons.get(reason, 0) + 1
    before_artifact_rejection = len(epochs)
    if artifact_method == "fixed":
        peak_absolute = np.max(np.abs(epochs.get_data(copy=False)), axis=(1, 2))
        bad_epoch_mask = peak_absolute > 120e-6
        epochs_clean = epochs.copy().drop(np.flatnonzero(bad_epoch_mask), reason="ABS_120_UV")
        rejected = int(bad_epoch_mask.sum())
        locally_interpolated = 0
        artifact_parameters = {"absolute_voltage_threshold_uV": 120.0}
    else:
        ar = AutoReject(
            n_interpolate=[1, 4, 8],
            consensus=[0.2, 0.4, 0.6, 0.8],
            cv=5,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=False,
        )
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
        rejected = int(reject_log.bad_epochs.sum())
        locally_interpolated = int((reject_log.labels == 2).any(axis=1).sum())
        artifact_parameters = {
            "consensus": float(ar.consensus_["eeg"]),
            "n_interpolate": int(ar.n_interpolate_["eeg"]),
        }
    epochs_clean.save(epoch_path, overwrite=True, verbose="ERROR")
    ica.save(qc_dir / f"{subject}-ica.fif", overwrite=True, verbose="ERROR")

    factorial = epochs_clean.metadata.loc[epochs_clean.metadata["factorial"].astype(bool)].copy()
    cell_counts = factorial.groupby(FACTORS, observed=True).size()
    qc = {
        "subject": subject,
        "source_acquisition": str(acquisition),
        "alignment": asdict(alignment),
        "crop_start_s": crop_start,
        "crop_stop_s": crop_stop,
        "n_eeg_channels": n_eeg,
        "global_bad_channels": bads,
        "global_bad_fraction": bad_fraction,
        "bad_channel_categories": bad_categories,
        "ica": ica_audit,
        "formal_trials_aligned": len(formal),
        "epoch_construction_drop_reasons": epoch_construction_drop_reasons,
        "artifact_method": artifact_method,
        "artifact_parameters": artifact_parameters,
        "epochs_before_artifact_rejection": before_artifact_rejection,
        "epochs_after_artifact_rejection": len(epochs_clean),
        "epochs_rejected": rejected,
        "epochs_with_local_interpolation": locally_interpolated,
        "factorial_trials_after_cleaning": len(factorial),
        "factorial_cells_present": int(len(cell_counts)),
        "minimum_trials_per_factorial_cell": int(cell_counts.min()) if len(cell_counts) else 0,
        "maximum_trials_per_factorial_cell": int(cell_counts.max()) if len(cell_counts) else 0,
        "eligible_min8_all16": bool(len(cell_counts) == 16 and cell_counts.min() >= 8),
        "sampling_rate_hz": float(epochs_clean.info["sfreq"]),
        "epoch_tmin_s": float(epochs_clean.tmin),
        "epoch_tmax_s": float(epochs_clean.tmax),
    }
    qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    return qc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=[f"s{i}" for i in range(1, 31)])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--artifact-method", choices=["fixed", "autoreject"], default="fixed")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(OUT_ROOT / "preprocessing.log", encoding="utf-8"), logging.StreamHandler()],
    )
    results = []
    for subject in sorted(args.subjects, key=subject_number):
        logging.info("Starting %s", subject)
        try:
            results.append(
                preprocess_subject(
                    subject,
                    args.overwrite,
                    args.seed + subject_number(subject),
                    args.n_jobs,
                    args.artifact_method,
                )
            )
        except Exception as exc:
            logging.exception("Failed %s", subject)
            results.append({"subject": subject, "error": repr(exc)})
        pd.json_normalize(results, sep=".").to_csv(
            OUT_ROOT / "preprocessing_qc.csv", index=False, encoding="utf-8-sig"
        )
    failures = [row for row in results if "error" in row]
    if failures:
        raise SystemExit(f"Preprocessing failed for {len(failures)} subject(s)")


if __name__ == "__main__":
    main()

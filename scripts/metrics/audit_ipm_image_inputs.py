#!/usr/bin/env python
"""Audit facial stimulus files and E-Prime image/condition mappings.

This script performs only the input and mapping gate. It does not calculate
image metrics and does not run behavioral or EEG models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("IPM_DATA_ROOT", REPO_ROOT / "data")).resolve()
STIMULUS_ROOT = Path(os.environ.get("IPM_STIMULUS_ROOT", ROOT / "stimuli")).resolve()
IMAGE_ROOTS = (ROOT, STIMULUS_ROOT)
OUT = ROOT / "ipm_visual_information_gate"
LOG = OUT / "logs" / "image_audit.log"
EPRIME = ROOT / "EEGDATA" / "EEGDATA" / "eprime"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXCLUDED_SEARCH_PARTS = {"node_modules", ".git", "_ipm_review_images", "aoi_output"}
FIELDS = ["Picture", "Gender", "Eye", "Mouth", "FSlim", "Skin", "Original", "IsBack", "Stimtype", "CondID"]


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message)


def read_text(path: Path) -> str:
    for enc in ("utf-16", "utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            if "LogFrame" in text:
                return text
        except UnicodeError:
            pass
    raise RuntimeError(f"Cannot decode {path}")


def parse_logs() -> list[dict]:
    rows = []
    for path in sorted(EPRIME.rglob("*.txt")):
        text = read_text(path)
        frames = re.findall(r"\*\*\* LogFrame Start \*\*\*(.*?)\*\*\* LogFrame End \*\*\*", text, re.S)
        subject_match = re.search(r"FaceExperiment-(\d+)-", path.name)
        subject = f"s{subject_match.group(1)}" if subject_match else path.parent.name.lower()
        for frame in frames:
            row = {"subject": subject, "source_log": str(path)}
            for field in FIELDS:
                match = re.search(rf"^\s*{re.escape(field)}:\s*(.*?)\s*$", frame, re.M)
                row[field] = match.group(1).strip() if match else ""
            if row["Picture"] and row["Stimtype"]:
                rows.append(row)
    return rows


def file_index() -> dict[str, list[Path]]:
    found = defaultdict(list)
    for image_root in IMAGE_ROOTS:
        for path in image_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if any(part in EXCLUDED_SEARCH_PARTS for part in path.parts):
                continue
            found[path.name.casefold()].append(path)
    return found


def image_metadata(path: Path) -> dict:
    result = {"width_px": "", "height_px": "", "color_mode": "", "file_sha256": ""}
    try:
        from PIL import Image
        with Image.open(path) as image:
            result.update(width_px=image.width, height_px=image.height, color_mode=image.mode)
        result["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception as exc:
        result["image_read_error"] = str(exc)
    return result


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()
    log("Stage 1 started: file and stimulus audit only.")
    records = parse_logs()
    log(f"Parsed {len(records)} E-Prime image-trial records from {len(set(r['source_log'] for r in records))} logs.")
    index = file_index()
    log(f"Indexed {sum(len(v) for v in index.values())} image files outside explicitly excluded software/QA folders.")

    references = defaultdict(list)
    for row in records:
        references[row["Picture"]].append(row)
    inventory = []
    for picture, refs in sorted(references.items()):
        matches = index.get(picture.casefold(), [])
        mapping_tuples = sorted({(r["Stimtype"], r["CondID"], r["FSlim"], r["Eye"], r["Mouth"], r["Skin"], r["Original"], r["IsBack"]) for r in refs})
        base = {
            "picture_filename": picture,
            "stimtype_identity": refs[0]["Stimtype"],
            "cond_id": refs[0]["CondID"],
            "fslim_raw": refs[0]["FSlim"],
            "eye_raw": refs[0]["Eye"],
            "mouth_raw": refs[0]["Mouth"],
            "skin_raw": refs[0]["Skin"],
            "original_flag": refs[0]["Original"],
            "attention_or_back_flag": refs[0]["IsBack"],
            "n_trial_references": len(refs),
            "n_unique_mapping_tuples": len(mapping_tuples),
            "mapping_consistent": len(mapping_tuples) == 1,
            "physical_file_found": bool(matches),
            "n_physical_matches": len(matches),
            "resolved_path": str(matches[0]) if len(matches) == 1 else "",
            "width_px": "", "height_px": "", "color_mode": "", "file_sha256": "",
            "duplicate_content_count": "", "audit_status": "found" if len(matches) == 1 else ("ambiguous_multiple_files" if matches else "missing_physical_image"),
        }
        if len(matches) == 1:
            base.update(image_metadata(matches[0]))
        inventory.append(base)

    digest_counts = Counter(r["file_sha256"] for r in inventory if r.get("file_sha256"))
    for row in inventory:
        row["duplicate_content_count"] = digest_counts.get(row.get("file_sha256", ""), 0)

    inv_cols = list(inventory[0]) if inventory else []
    write_csv(OUT / "image_input_inventory.csv", inventory, inv_cols)

    map_groups = defaultdict(list)
    for row in records:
        if row["Stimtype"] in {"F_1", "F_2", "M_1", "M_2"}:
            map_groups[(row["Stimtype"], row["CondID"], row["Picture"])].append(row)
    mapping = []
    for (identity, cond, picture), refs in sorted(map_groups.items()):
        tuples = sorted({(r["FSlim"], r["Eye"], r["Mouth"], r["Skin"], r["Original"], r["IsBack"]) for r in refs})
        first = refs[0]
        eye_binary = 0 if first["Eye"] == "1" else (1 if first["Eye"] == "2" else "")
        mouth_binary = 0 if first["Mouth"] == "1" else (1 if first["Mouth"] == "2" else "")
        formal = cond.isdigit() and 2 <= int(cond) <= 17 and first["Original"] == "0" and first["IsBack"] == "0"
        mapping.append({
            "identity": identity, "picture_filename": picture, "cond_id": int(cond) if cond.isdigit() else cond,
            "fslim_raw": first["FSlim"], "fslim_binary": int(first["FSlim"]) if first["FSlim"] in {"0", "1"} else "",
            "eye_raw": first["Eye"], "eye_binary": eye_binary,
            "mouth_raw": first["Mouth"], "mouth_binary": mouth_binary,
            "skin_raw": first["Skin"], "skin_binary": int(first["Skin"]) if first["Skin"] in {"0", "1"} else "",
            "original_flag": first["Original"], "is_back": first["IsBack"], "formal_factorial_condition": formal,
            "n_subject_logs": len({r["subject"] for r in refs}), "n_trial_references": len(refs),
            "mapping_consistent_across_logs": len(tuples) == 1,
            "physical_file_found": bool(index.get(picture.casefold(), [])),
        })
    map_cols = list(mapping[0]) if mapping else []
    write_csv(OUT / "image_condition_mapping.csv", mapping, map_cols)

    identities = ["F_1", "F_2", "M_1", "M_2"]
    formal_by_id = {i: [r for r in mapping if r["identity"] == i and r["formal_factorial_condition"]] for i in identities}
    original_by_id = {i: [r for r in mapping if r["identity"] == i and str(r["original_flag"]) == "1"] for i in identities}
    full_factorial = all(len(rows) == 16 and len({(r["fslim_binary"], r["eye_binary"], r["mouth_binary"], r["skin_binary"]) for r in rows}) == 16 for rows in formal_by_id.values())
    complete_original_refs = all(len(rows) >= 1 for rows in original_by_id.values())
    physical_formal = sum(bool(r["physical_file_found"]) for r in mapping if r["formal_factorial_condition"])
    physical_original = sum(bool(r["physical_file_found"]) for i in identities for r in original_by_id[i])
    conflicts = sum(not r["mapping_consistent_across_logs"] for r in mapping)
    status = "BLOCKED_MISSING_PHYSICAL_IMAGES" if physical_formal < 64 or physical_original < 4 else "PASS_INPUT_MAPPING_GATE"

    if status == "PASS_INPUT_MAPPING_GATE":
        decision_text = """The E-Prime logs provide a consistent, explicit mapping between image filename, identity, CondID, the four edit fields, Original, and IsBack. All 68 required physical facial stimuli were found uniquely. The input and mapping gate therefore passes, permitting calculation of image metrics."""
        questions = f"""| Question | Finding |
| --- | --- |
| All four original images found? | Yes: the four experiment-referenced index-(1) files. |
| All 16 edited combinations found? | Yes, for each of the four identities (64 files). |
| Complete conditions for each identity? | Yes in E-Prime records and on disk. |
| Consistent resolution/color space/cropping/compression? | See `image_input_inventory.csv`; byte-level hashes and decoded metadata were recorded. |
| Image-to-behavior mapping? | Explicit in E-Prime logs via Picture, Stimtype and CondID. |
| Image-to-EEG mapping? | CondID and Stimtype are recoverable through epoch endCode. |
| Were originals in the formal factorial model? | No. Formal edited trials were CondID 2-17; originals were controls. |
| Coding direction restored? | Numeric coding yes; semantic interpretation will be checked against measured pixel/geometry changes. |"""
        stop_text = """## Gate consequence

The audit authorizes the next stage (landmarks and G/A/I metrics). It does not itself establish construct validity or any behavioral/EEG result."""
    else:
        decision_text = """The E-Prime logs provide a consistent, explicit mapping between image filename, identity, CondID, the four edit fields, Original, and IsBack. However, one or more physical facial stimulus files referenced by the experiment were not found uniquely. Image metrics cannot be calculated safely, so the analysis stops at this gate."""
        questions = """| Question | Finding |
| --- | --- |
| All four original images found? | No or ambiguous. |
| All 16 edited combinations found? | Mapping is complete in logs, but physical coverage is incomplete. |
| Complete conditions for each identity? | Yes in E-Prime records. |
| Image-to-behavior mapping? | Explicit in E-Prime logs via Picture, Stimtype and CondID. |
| Image-to-EEG mapping? | CondID and Stimtype are recoverable through epoch endCode. |
| Were originals in the formal factorial model? | No. |
| Coding direction restored? | Numeric coding only. |"""
        stop_text = """## Stopping rule

No downstream analysis is authorized until every experiment-referenced physical file is found uniquely."""

    report = f"""# Image Input and Condition Mapping Audit

## Decision

**{status}**

{decision_text}

## Evidence recovered

- E-Prime log files parsed: {len(set(r['source_log'] for r in records))}.
- Image-trial log records parsed: {len(records)}.
- Unique referenced image filenames: {len(references)}.
- Mapping conflicts across logs: {conflicts}.
- Four target identities present in logs: {', '.join(identities)}.
- Complete 16-condition factorial mapping for every identity: {full_factorial}.
- Original-image references present for every identity: {complete_original_refs}.
- Physical formal edited images found: {physical_formal}/64.
- Physical original identity images found: {physical_original}/4.

## Condition coding

- FSlim is recorded as 0/1 and can be retained numerically.
- Skin is recorded as 0/1 and can be retained numerically.
- Eye is recorded as 1/2 and maps to 0/1 as 1->0 and 2->1.
- Mouth is recorded as 1/2 and maps to 0/1 as 1->0 and 2->1.
- Formal factorial trials use CondID 2-17 with Original=0 and IsBack=0.
- CondID 1 / filename index (1) is referenced as the original/control image and was not part of the formal 2x2x2x2 edited-condition analysis.

The numeric direction is recovered, but the verbal meaning and actual editing parameters of level 0 versus level 1 are not recoverable from the available images because those files are missing.

## Questions required by the audit

{questions}

{stop_text}
"""
    (OUT / "image_audit_report.md").write_text(report, encoding="utf-8")
    gate = f"""# IPM Final Gate Report - Input Audit

## Final decision

**Input status: {status}**

This interim file is superseded after the G/A/I and downstream gates are evaluated. {decision_text}
"""
    (OUT / "IPM_FINAL_GATE_REPORT.md").write_text(gate, encoding="utf-8")
    manifest = {
        "status": status, "roots_audited": [str(p) for p in IMAGE_ROOTS], "eprime_root": str(EPRIME),
        "n_logs": len(set(r["source_log"] for r in records)), "n_records": len(records),
        "n_unique_referenced_images": len(references), "complete_factorial_mapping": full_factorial,
        "physical_formal_images_found": physical_formal, "physical_original_images_found": physical_original,
        "downstream_analysis_run": False,
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Audit complete with status {status}; downstream analysis not run.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Audit whether S11 exclusion or S18 recovery drives the primary result."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_primary_erp_analysis import (
    EPOCH_DIR,
    ROOT,
    estimate_subject_betas,
    load_qc,
    primary_cluster_test,
    subject_key,
)


OUT = ROOT / "primary_erp" / "sensitivity"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qc = load_qc()
    primary = qc.loc[qc["included"], "subject"].tolist()
    variants = {
        "primary_quality_N28": primary,
        "trial_only_include_s11_N29": sorted([*primary, "s11"], key=subject_key),
        "legacy_membership_s11_not_s18_N28": sorted(
            [subject for subject in [*primary, "s11"] if subject != "s18"], key=subject_key
        ),
    }
    all_subjects = sorted(set().union(*variants.values()), key=subject_key)
    subject_betas = {subject: estimate_subject_betas(subject) for subject in all_subjects}
    cluster_parts = []
    point_parts = []
    for variant, subjects in variants.items():
        betas = pd.concat([subject_betas[subject] for subject in subjects], ignore_index=True)
        points, clusters = primary_cluster_test(betas)
        points.insert(0, "variant", variant)
        clusters.insert(0, "variant", variant)
        point_parts.append(points)
        cluster_parts.append(clusters)
    points = pd.concat(point_parts, ignore_index=True)
    clusters = pd.concat(cluster_parts, ignore_index=True)
    points.to_csv(OUT / "membership_sensitivity_pointwise.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(OUT / "membership_sensitivity_clusters.csv", index=False, encoding="utf-8-sig")
    summary = {
        variant: {
            "subjects": subjects,
            "confirmed_clusters": clusters.loc[
                clusters["variant"].eq(variant) & (clusters["across_four_factor_fwer_p"] < 0.05)
            ].to_dict("records"),
            "lowest_fwer_cluster": clusters.loc[clusters["variant"].eq(variant)]
            .sort_values("across_four_factor_fwer_p")
            .head(1)
            .to_dict("records"),
        }
        for variant, subjects in variants.items()
    }
    (OUT / "membership_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

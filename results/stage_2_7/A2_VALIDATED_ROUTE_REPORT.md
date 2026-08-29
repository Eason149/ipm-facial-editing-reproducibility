# Validated A2 Route — Frozen Follow-up

## Scope

This follow-up implements the Stage 2.6 decision without changing A2, the N=28 quality rule, ROI, time range, cluster-forming threshold, joint G_new/A2/I family, or random seed. It does not select a favorable EEG window.

## Behavioral incremental value

- Nested factor+A2 versus factor likelihood-ratio p=9.026e-153; ΔAIC=-696.19; ΔBIC=-680.10.
- Frozen factor+A2 coefficients: [{'term': 'A2', 'estimate': 0.29774031266584405, 'p': 5.0576918457675254e-46}, {'term': 'RatingDimension_Beauty:A2', 'estimate': -0.6896278881722651, 'p': 7.75553490003647e-155}].
- Participant-grouped CV is in `a2_behavior_cv_summary.csv`; it assesses held-out participants, not new stimuli.

## EEG

- N=28 jointly corrected A2 clusters: none.
- N=29, N=28, and all leave-one-participant-out versions use 10,000 G_new/A2/I × time maximum-cluster permutations.
- Leave-one-identity-out results remain descriptive and do not establish cross-identity generalization.

## Decision

A2 is retained as a validated stimulus-level appearance dimension with limited behavioral incremental value. It is **not** supported as a stable EEG-tracked dimension. The defensible IPM route is therefore behavioral/information-oriented and stimulus-set-bounded; neural-tracking language remains prohibited.

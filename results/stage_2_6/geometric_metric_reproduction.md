# Geometric Metric Reproduction

## Environment and frozen formula

A clean environment was locked to Python 3.9, MediaPipe 0.10.14, NumPy 1.26.4 and OpenCV 4.11.0. All 68 core images were rerun. The seven pre-existing G components, RMS scaling and equal-weight mean were retained without reference to behavioral or EEG outcomes.

## Detection

- Successful full 468-point detections: 68/68.
- Failures: 0.
- Face-detection confidence range: 0.8581–0.9429.
- Raw per-image coordinates and detection status are in `landmarks_468/`.

## Agreement with Stage 2 G

- Pearson r=0.6660.
- Spearman rho=0.6170.
- Mean absolute standardized difference=0.6630.
- Mean difference=-0.0000; 95% limits of agreement [-1.6147, 1.6147].

Identity- and operation-specific agreement is reported in `geometric_metric_agreement.csv`. This reproduction is algorithmically independent of the behavioral and EEG outcomes but uses the frozen Stage 2 formula.

## Frozen-model reruns

G construct validation, behavior coefficients and N=28 joint EEG reruns are in `geometric_construct_validation.csv`, `behavior_frozen_metric_models.csv`, and `frozen_metric_joint_eeg_clusters.csv`. Significant corrected clusters: [{'model': 'original_A', 'metric': 'G_new', 'n_participants': 28, 'cluster_start_ms': 420.0, 'cluster_end_ms': 580.0, 'cluster_mass_abs_t': 25.970127912920326, 'joint_familywise_p': 0.030096990300969902, 'peak_time_ms': 460.0, 'peak_t': 3.6639696750307005, 'n_permutations': 10000}, {'model': 'original_A', 'metric': 'A', 'n_participants': 28, 'cluster_start_ms': 440.0, 'cluster_end_ms': 580.0, 'cluster_mass_abs_t': 22.822353809122337, 'joint_familywise_p': 0.044495550444955505, 'peak_time_ms': 580.0, 'peak_t': -3.381576536105139, 'n_permutations': 10000}]

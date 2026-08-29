# Appearance Metric Replication

## Frozen independent A2

Before examining A2 outcomes, A2 was fixed as the equal-weight RMS-scaled composite of three quantities measured only in the two cheek AOIs after five-landmark similarity alignment: (1) Lab a/b histogram Jensen–Shannon divergence, (2) absolute L* change, and (3) four-orientation Gabor texture-response change. It does not reuse the Stage 2 A skin mask or its SSIM/LBP/edge components. No alternative A2 variants were screened.

## Agreement and construct checks

- A–A2 Pearson r=0.6189; Spearman rho=0.7573.
- Skin effect on A2: beta=1.6466, p=8.403e-36.
- FSlim cross-loading on A2: beta=0.0319, p=0.8492.
- Residual geometric-alignment association: beta=0.0059, p=0.9763.
- A2 correlations with G and I: r=-0.0521 and r=0.0720.

Leave-one-identity-out coefficients and p values are in `appearance_metric_agreement.csv`. Because only four identities exist, stability is descriptive and does not establish new-stimulus generalization.

## Frozen-model reruns

Original A, A2, and the prespecified robust composite AR=(standardized A+A2) were each entered in a full three-predictor joint model with G_new and I. Results are in `behavior_frozen_metric_models.csv`, `frozen_metric_joint_eeg_clusters.csv`, and `frozen_metric_identity_sensitivity.csv`.

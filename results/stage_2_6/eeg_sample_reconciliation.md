# EEG Sample Reconciliation

## Frozen primary rule

The unified primary EEG sample is **N=28**, determined by the pre-existing data-quality rule: exclude s5 by the default EEG subject list and require at least 8 effective trials in every factorial cell after pooling the four identities. s18 fails this rule. N=29 is retained only as a sensitivity analysis; significance was not used to choose the primary sample.

## Participant lists

- N=29: s1, s2, s3, s4, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30.
- N=28: s1, s2, s3, s4, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30.
- Difference: s18 only (s5 is absent from both).

## Why s18 entered the continuous model

The continuous G/A/I script inherited `DEFAULT_SUBJECTS`, which excludes s5 but includes s18, and did not reapply the factorial-cell minimum. Continuous regression can remain full-rank with incomplete factorial-cell coverage because G/A/I are trial-level continuous predictors; nevertheless, using a different quality rule for the same EEG dataset is not defensible. This was a pipeline omission, not a result-selected choice.

## s18 coverage

- Formal effective trials: 102.
- Factorial cells present after pooling identities: 16/16.
- Minimum/median/maximum trials per cell: 3/6.5/8.
- Cells below 8 trials: 13.

Full counts are in `s18_condition_counts.csv`.

## Frozen analysis specification

All variants use the already-frozen G/A/I betas, posterior 25-channel ROI, 0–1000 ms in 20-ms steps, two-sided pointwise cluster-forming threshold p<.05, absolute-t cluster mass, participant-level sign flip, and 10,000 joint maximum-cluster permutations across G/A/I × time.

## Sensitivity scope

The CSV contains N=29, N=28, an explicitly labeled leave-s18-out duplicate check, and all 29 leave-one-participant-out versions. Significant cluster rows across LOPO versions: 29. These analyses assess influence only and do not redefine the primary sample.

## Timing of rules

The N=28 rule is documented in the prior Stage 2 exclusion-flow artifact before Stage 2.6. The N=29 continuous script used an earlier default loader rule but omitted the cell check. There is no evidence that either rule was chosen after inspecting the Stage 2.6 sensitivity results; however, the continuous analysis's inconsistent implementation means only the N=28 rule is retained prospectively here.

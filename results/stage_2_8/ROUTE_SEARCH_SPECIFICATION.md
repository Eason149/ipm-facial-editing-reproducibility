# IPM Stage 2.8 — Frozen Route Search Specification

Timestamp: 2026-08-28 (before Stage 2.8 model execution)

## Purpose

Compare several defensible IPM routes without choosing a result solely because it is significant. All new EEG analyses use the Stage 2.6 unified N=28 quality sample (s5 and s18 excluded), participant-level inference, two-sided tests, seed 20260828, and 10,000 sign-flip permutations.

## Candidate routes and gates

1. **Continuous visual-metric tracking (A2/G_new/I).** Reuse the frozen Stage 2.6/2.7 results. Passes as an integrated route only if a validated metric has behavioral incremental value and an N=28 EEG cluster survives the frozen joint G_new/A2/I × time correction.
2. **Editing-operation ERP dynamics.** Jointly test FSlim, Eye, Mouth, and Skin time courses together with the prespecified Surface-minus-Structural contrast in one five-signal × 0–1000 ms maximum-cluster family. Passes if at least one cluster has joint familywise p < .05 and is not participant-dependent.
3. **Surface-versus-structural hierarchy.** The contrast is fixed as mean(Skin, Mouth) minus mean(FSlim, Eye) and is included in the same five-signal family above. It is not tested in a favorable window selected after inspection.
4. **Predefined late-window ERP effects.** Independently summarize Centroparietal 350–600 ms and 600–1000 ms single-trial means. Estimate all four operation coefficients within participant while controlling TrialOrder and identity. Correct all 4 operations × 2 windows together with a maximum-|t| sign-flip test. This is corroborative, not an independent replication of the time-resolved analysis.
5. **EEG–behavior individual-difference association.** Correlate matched participant-level operation coefficients across all four operations and both rating outcomes; correct the full family. Passes only if corrected evidence is stable, not because one uncorrected correlation is favorable.
6. **EEG incremental rating prediction, MVPA decoding, and RSA.** Audit their already frozen, leakage-controlled or family-corrected outputs. They pass only under their original gates; no model/window retuning is permitted in Stage 2.8.

## Stability rules

- For significant time clusters: leave-one-participant-out influence analysis with the same five-signal family and 10,000 permutations.
- For predefined windows: leave-one-identity-out estimates are reported for all four identities; identities are sensitivity strata, not a random sample of a stimulus population.
- A route is not called generalizable if it is supported only by four identities.
- Failure outputs are recorded in a compact route audit. Stage 2.8 temporary caches and abandoned candidate-specific files are deleted; pre-existing user data and prior audit outputs are not deleted.

## Allowed claims

Any retained route remains exploratory, multiplicity-controlled, participant-generalizable only, and stimulus-set-bounded. No causal pathway, universal mechanism, cross-identity generalization, or stable neural tracking claim is allowed.

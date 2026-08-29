# Stimulus-Level Inference Boundary

- Participants are the resampling and sign-flip units in the behavioral/EEG inferential models.
- The same 64 edited images are repeatedly evaluated by multiple participants; trial rows are therefore not independent stimulus samples.
- Participant-grouped cross-validation tests prediction for held-out participants exposed to the same stimulus set. It does not test prediction for new images or identities.
- The stimulus structure is four identities × sixteen edits. The effective identity-level sample size is four, not 64 identities and not the number of participant trials.
- Identity-specific G_new/A2/I behavioral coefficients and confidence intervals are provided in `identity_level_behavior_coefficients.csv` and `identity_level_coefficient_forest.pdf`.
- Leave-one-identity-out analyses are sensitivity checks with only three identities remaining. They cannot establish population-level cross-identity generalization.

All permitted claims must therefore be described as **stimulus-set-bounded conditional associations**.

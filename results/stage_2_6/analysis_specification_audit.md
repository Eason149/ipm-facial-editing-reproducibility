# Analysis Specification Audit

## Classification

**Exploratory multiplicity-controlled.**

The posterior ROI code predates the G/A analysis, but no timestamped preregistration or locked secondary-analysis protocol for G/A/I was found. The results therefore cannot be called confirmatory or preregistered secondary.

## Frozen specification

| Item | Finding |
| --- | --- |
| Posterior ROI | 25 common posterior channels: P7/P5/P3/P1/Pz/P2/P4/P6/P8, PO7/PO5/PO3/POz/PO4/PO6/PO8, O1/Oz/O2, CP5/CP3/CP1/CPz/CP2/CP4. |
| ROI timing | Defined in `run_time_resolved_factor_decoding.py` on 12 July 2026, before the 18 August G/A scripts. |
| ROI independence | Based on the study's earlier N170/LPP analysis framework and posterior theory; temporally prior to G/A, but not independent of this dataset. |
| Time range | 0–1000 ms, inherited from the prior time-resolved ERP analysis and fixed before the Stage 2.6 reruns. |
| Sampling of coefficients | 20-ms centers after 30-ms smoothing. |
| Cluster-forming threshold | Two-sided pointwise p<.05. |
| Cluster mass | Sum of absolute one-sample t values across contiguous time points. |
| Permutation family | Maximum cluster mass jointly across all three simultaneously fitted information metrics and all time points. |
| Sign-flip unit | Participant: one sign is applied to the participant's complete coefficient time course within each permutation. |
| First-level predictors | G/A/I (or frozen G_new/A2/I alternative) enter simultaneously, plus intercept and identity dummy controls. |
| Nuisance controls | Identity fixed effects; no trial order, rating, reaction time, or low-level global-image covariates. |
| Random seed | 20260818, with documented offsets for sensitivity variants. |
| Missing trials | Available formal trials are retained; continuous regression does not mathematically require all 16 cells, but the unified primary sample additionally enforces the prior minimum of 8 trials per pooled factorial cell. |
| Primary sample | N=28, excluding s5 and s18 by the pre-existing quality rules. |

## Specification consequence

The N=29 results are sensitivity results. The frozen N=28 results must be reported regardless of which version crosses .05. Likewise, original A, A2, and robust-composite models must be presented as specification checks rather than selecting the most favorable result.

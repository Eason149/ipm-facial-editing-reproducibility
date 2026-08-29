# IPM Stage 2.8 — Route Search Decision

## Design integrity

This report compares a frozen set of candidate routes. It does not hide failed paths or select an uncorrected favorable window. New EEG tests use the unified N=28 sample, participant sign-flips, 10,000 permutations, and frozen ROIs/windows.

## Retained routes

- Predefined late-window operation ERP

## Joint time-resolved ERP results

No five-signal-family cluster survived.

## Leave-one-participant-out stability

No primary cluster required LOPO follow-up.

## Predefined-window corroboration

| window | factor | n_participants | mean_beta | se | ci95_low | ci95_high | t | p_uncorrected | p_maxT_familywise | cohen_dz | n_permutations | seed | p_BH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Late_600_1000 | Eye | 28 | 0.28841 | 0.078664 | 0.12701 | 0.44982 | 3.6664 | 0.0010622 | 0.0062994 | 0.69288 | 10000 | 20261828 | 0.0084978 |
| MiddleLate_350_600 | Skin | 28 | -0.27891 | 0.083745 | -0.45074 | -0.10708 | -3.3304 | 0.0025189 | 0.015398 | -0.62939 | 10000 | 20261828 | 0.010076 |

## Behavioral operation effects

| outcome | factor | n_participants | mean_beta | se | ci95_low | ci95_high | t | p_uncorrected | p_maxT_familywise | cohen_dz | n_permutations | seed | p_BH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Naturalness | Eye | 28 | 0.71567 | 0.12034 | 0.46874 | 0.96259 | 5.9469 | 2.4338e-06 | 9.999e-05 | 1.1239 | 10000 | 20261928 | 1.6384e-05 |
| Beauty | Eye | 28 | -0.44285 | 0.086678 | -0.6207 | -0.265 | -5.1091 | 2.2718e-05 | 0.00019998 | -0.96553 | 10000 | 20261928 | 4.5437e-05 |
| Naturalness | FSlim | 28 | 0.22828 | 0.040781 | 0.14461 | 0.31196 | 5.5978 | 6.1439e-06 | 0.00019998 | 1.0579 | 10000 | 20261928 | 1.6384e-05 |
| Naturalness | Skin | 28 | 0.40931 | 0.072762 | 0.26001 | 0.5586 | 5.6253 | 5.7093e-06 | 0.00019998 | 1.0631 | 10000 | 20261928 | 1.6384e-05 |
| Beauty | FSlim | 28 | -0.18353 | 0.040263 | -0.26614 | -0.10092 | -4.5583 | 9.9787e-05 | 0.00029997 | -0.86144 | 10000 | 20261928 | 0.00015966 |
| Beauty | Skin | 28 | -0.16583 | 0.038484 | -0.2448 | -0.086873 | -4.3092 | 0.00019444 | 0.00069993 | -0.81437 | 10000 | 20261928 | 0.00025926 |

## Fixed-window participant influence

| window | factor | primary_mean_beta | primary_p_maxT | lopo_runs | lopo_corrected_detection_count | lopo_corrected_detection_rate | lopo_sign_consistency_rate | min_abs_lopo_t | max_lopo_p_maxT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiddleLate_350_600 | Skin | -0.27891 | 0.015398 | 28 | 28 | 1 | 1 | 3.0615 | 0.033697 |
| Late_600_1000 | Eye | 0.28841 | 0.0062994 | 28 | 28 | 1 | 1 | 3.3992 | 0.014299 |

## EEG–behavior association

No participant-level EEG–behavior association survived family correction.

## Recommended manuscript route

Use a two-level, stimulus-set-bounded information-processing account: (1) validated appearance change A2 explains limited incremental variance in subjective evaluation; (2) factorial editing operations, especially any jointly corrected and influence-stable late ERP effects reported above, index operation-sensitive neural processing. Do not claim that A2 itself is neurally tracked, that EEG improves rating prediction, or that four identities establish cross-identity generalization.

The EEG contribution is an operation-sensitive temporal modulation, not a causal pathway or generalizable neural decoder. All claims remain exploratory and multiplicity-controlled.
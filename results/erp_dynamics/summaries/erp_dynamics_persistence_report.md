# ERP Dynamics and Persistence Final Report

## Main logic
Trial-level GLM was estimated within each subject, then beta time courses were tested at the group level.
The paper-facing emphasis is temporal dynamics: onset, offset, duration, persistence, and surface/local versus structural/configural differences.

## Strongest temporal clusters
| contrast | cluster_start_ms | cluster_end_ms | duration_ms | direction | cluster_mass_abs_t | cluster_p | n_timepoints | peak_time_ms | peak_t | peak_mean_beta_uV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Surface_minus_Structural | 420 | 580 | 180 | negative | 25.55 | 0.01099 | 9 | 520 | -3.521 | -0.1879 |
| Surface | 420 | 580 | 180 | negative | 25.07 | 0.01699 | 9 | 560 | -3.255 | -0.2374 |
| Skin | 440 | 580 | 160 | negative | 21.27 | 0.02599 | 8 | 580 | -3.371 | -0.3011 |
| Mouth | 300 | 380 | 100 | negative | 13.41 | 0.03948 | 5 | 340 | -3.14 | -0.268 |
| Structural | 920 | 980 | 80 | positive | 11.78 | 0.06347 | 4 | 980 | 3.273 | 0.1566 |
| Eye | 680 | 740 | 80 | positive | 9.929 | 0.1099 | 4 | 700 | 2.661 | 0.2115 |
| Eye | 940 | 980 | 60 | positive | 7.673 | 0.1514 | 3 | 980 | 2.575 | 0.1847 |
| Surface | 320 | 340 | 40 | negative | 4.894 | 0.2154 | 2 | 320 | -2.723 | -0.1828 |
| Surface_minus_Structural | 880 | 900 | 40 | negative | 4.868 | 0.2189 | 2 | 880 | -2.519 | -0.1672 |
| Structural | 680 | 700 | 40 | positive | 5.123 | 0.2364 | 2 | 700 | 2.616 | 0.1279 |

## Persistence ranking
| contrast | best_corrected_onset_ms_p_lt_0.10 | best_corrected_offset_ms_p_lt_0.10 | corrected_duration_ms_p_lt_0.10 | longest_corrected_cluster_ms_p_lt_0.10 | uncorrected_cumulative_duration_ms | persistence_index_abs_t | late_450_800_abs_effect_auc | peak_abs_t_time_ms | peak_abs_t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Surface_minus_Structural | 420 | 580 | 180 | 180 | 240 | 1472 | 46.65 | 520 | 3.521 |
| Surface | 420 | 580 | 180 | 180 | 240 | 1377 | 33.27 | 560 | 3.255 |
| Skin | 440 | 580 | 160 | 160 | 180 | 1302 | 47.37 | 580 | 3.371 |
| Eye | nan | nan | 0 | 0 | 180 | 1199 | 37.89 | 700 | 2.661 |
| Mouth | 300 | 380 | 100 | 100 | 140 | 1161 | 28.35 | 340 | 3.14 |
| Structural | 920 | 980 | 80 | 80 | 120 | 996.5 | 15.4 | 980 | 3.273 |
| FSlim | nan | nan | 0 | 0 | 20 | 877.4 | 23.7 | 780 | 2.135 |

## Strongest stage effects
| stage | contrast | mean_beta_uV | sem_beta_uV | t | p_uncorrected | cohen_dz | n_subjects | p_fdr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EditIntegration | Mouth | -0.1966 | 0.06794 | -2.894 | 0.007435 | -0.547 | 28 | 0.05204 |
| EditIntegration | Surface | -0.1552 | 0.06242 | -2.487 | 0.01938 | -0.4699 | 28 | 0.06783 |
| LPP | Surface_minus_Structural | -0.1298 | 0.05748 | -2.258 | 0.03222 | -0.4268 | 28 | 0.1956 |
| LPP | Skin | -0.1305 | 0.06798 | -1.92 | 0.0655 | -0.3628 | 28 | 0.1956 |
| LPP | Surface | -0.09133 | 0.05088 | -1.795 | 0.08384 | -0.3392 | 28 | 0.1956 |
| EditIntegration | Surface_minus_Structural | -0.1308 | 0.07413 | -1.764 | 0.08899 | -0.3334 | 28 | 0.2077 |
| N170 | Surface | -0.1073 | 0.06227 | -1.723 | 0.09637 | -0.3256 | 28 | 0.4184 |
| N170 | Mouth | -0.1478 | 0.09222 | -1.603 | 0.1206 | -0.3029 | 28 | 0.4184 |
| EditIntegration | Skin | -0.1138 | 0.07701 | -1.477 | 0.1511 | -0.2792 | 28 | 0.2645 |
| P2_EPN | Mouth | -0.1628 | 0.1137 | -1.432 | 0.1637 | -0.2706 | 28 | 0.4853 |
| N170 | Surface_minus_Structural | -0.07708 | 0.05591 | -1.379 | 0.1793 | -0.2605 | 28 | 0.4184 |
| P2_EPN | FSlim | -0.09144 | 0.07049 | -1.297 | 0.2056 | -0.2451 | 28 | 0.4853 |

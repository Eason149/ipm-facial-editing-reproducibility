# EEG Preprocessing Protocol Recovery

## Recovered processing chain

The original Curry recordings and the MATLAB preprocessing/epoch scripts were located. The recoverable chain is:

1. Import Curry `.cdt` using EEGLAB `loadcurry`; stored histories show trigger retention and Curry locations disabled.
2. EEGLAB FIR filtering with `pop_eegfiltnew`, 0.1–30 Hz.
3. Conditional CleanLine at 50 Hz. The code is explicit, but actual plugin availability/execution is not preserved for all participants.
4. Bad-channel handling: preferred `clean_rawdata` (flatline 5 s, channel correlation .8, line-noise criterion 4; burst/window rejection disabled), with kurtosis threshold 5 fallback.
5. Average reference.
6. Extended runica on continuous data.
7. ICLabel; remove Eye components only when Eye probability ≥.90.
8. Pair start code 10 with the next valid 101–197 end code; discard missing pairs and 0-back codes 181–197.
9. Epoch −200 to 1000 ms, baseline −200 to 0 ms.
10. Reject epochs exceeding ±120 µV over any retained channel.

## Direct derivative checks

Final EEGLAB datasets contain average-reference metadata, non-empty ICA matrices, ICLabel classification probabilities, clean-channel masks, 1000-Hz sampling, and −200 to approximately 1000-ms epochs. Thirty `epoched_stim` derivatives are present.

## Evidence limits

- EEGLAB histories contain the original Curry import but are not cumulative for later processing commands.
- CleanLine is conditional in the code and cannot be verified as executed for every participant.
- The code allows two bad-channel branches; the exact branch is not recoverable for every participant.
- Rejected IC and epoch IDs were not saved as complete participant-level audit tables.
- No interpolation step exists in the recovered workflow; removed channels were not restored.

The workflow is reportable at the protocol level, with these qualifications. It is not legitimate to state that CleanLine ran for every participant or that all participants used the same bad-channel branch.

## Full reconstruction plan if stricter provenance is required

If the journal requires participant-level executable provenance, rebuild from the Curry files in a version-locked EEGLAB environment: freeze plugin versions; remove conditional branches; record channel removals, ICA rank/convergence, ICLabel probabilities and rejected IC IDs; record rejected epoch indices/reasons; save cumulative histories and checksums; compare rebuilt and existing derivative ERPs before substituting them. This plan has not been executed in Stage 2.6.

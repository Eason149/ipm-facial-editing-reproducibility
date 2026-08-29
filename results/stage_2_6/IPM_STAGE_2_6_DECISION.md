# IPM Stage 2.6 Decision

## Final decision

**B. Ready only with one validated visual-information dimension**

The only visual dimension with a clean independent construct replication is appearance change as operationalized by the frozen A2 metric. This decision does not authorize a neural-tracking claim for A2: A2 did not produce a jointly corrected EEG cluster in the unified N=28 model.

## Decision against the four alternatives

- **Not A:** the reproduced G did not agree highly with the Stage 2 G (Pearson r=.666; Spearman rho=.617; standardized mean absolute difference=.663). EEG significance also depended on whether original A, A2 or the robust A/A2 composite was used. Identity sensitivity remained substantial.
- **B selected:** A2 showed strong Skin construct validity, no FSlim cross-loading, no residual association with geometric alignment error, moderate agreement with A, and descriptive leave-one-identity stability. Behavioral associations remained under the frozen A2 model.
- **Not C:** the original preprocessing and epoch scripts were recovered and the main steps are now reportable. CleanLine execution and participant-specific branch provenance remain qualified, but preprocessing reconstruction is no longer the single blocking issue.
- **Not D:** metric effects were not wholly non-reproducible. In the unified N=28 model using reproduced G with original A and I, G_new showed a 420–580 ms jointly corrected cluster (p=.0301) and original A showed 440–580 ms (p=.0445). These effects are specification-sensitive and cannot support a general neural-tracking claim.

## Minimum-A criteria audit

| Criterion | Result |
| --- | --- |
| 1. Unified primary EEG rule | Met: N=28; s5 and s18 excluded by pre-existing quality rules. |
| 2. New G highly agrees with old G | **Not met.** |
| 3. A or A2 has reasonable construct validity | Met for A2; original A retains FSlim cross-loading. |
| 4. At least one frozen G/A EEG result survives joint correction | Technically met only in the original-A specification with G_new; not robust to A2/robust-composite specification. |
| 5. Not determined by s18 or one metric implementation | s18 is not determinative, but **metric specification remains determinative**. |
| 6. EEG preprocessing is reportable | Substantially met with explicit qualifications; CleanLine execution remains unavailable. |

Because criteria 2 and 5 fail, A is prohibited.

## EEG sample reconciliation

- N=29 contained s1–s4 and s6–s30, including s18.
- N=28 removes s18 in addition to the pre-existing exclusion of s5.
- s18 had 102 formal trials; all 16 pooled factorial cells were represented, but the minimum cell count was 3 and 13/16 cells had fewer than 8 trials.
- The N=28 old-G/A/I analysis retained A at 440–580 ms (p≈.045) but old G fell to p≈.077. Inclusion of s18 therefore affected the threshold crossing for old G.
- Leave-one-participant-out results crossed .05 inconsistently; no participant was used to redefine the sample.

## Frozen metric findings

### Geometry

- Full 468-point rerun: 68/68 successful; detection confidence .858–.943.
- New G construct effects: FSlim positive and strong; Mouth also changed G; Eye remained unsupported; Skin was null.
- New/old G agreement was not high enough for criterion A.

### Appearance

- A2 was fixed before outcome testing as aligned cheek-region Lab chroma divergence, luminance change and Gabor texture change.
- A–A2 agreement: Pearson r=.619; Spearman rho=.757.
- Skin→A2: β=1.647, p<10⁻³⁵.
- FSlim→A2: p=.849; alignment error→A2: p=.976.
- A2 correlations with old G and I were −.052 and .072.

### Unified N=28 EEG specification checks

| Model | Jointly corrected finding |
| --- | --- |
| G_new + original A + I | G_new 420–580 ms, p=.0301; A 440–580 ms, p=.0445 |
| G_new + A2 + I | No corrected metric cluster; G_new candidate p=.0820 |
| G_new + robust A/A2 composite + I | No corrected metric cluster; G_new candidate p=.0507 |

The favorable original-A version must not be selected while suppressing the A2 and robust-composite results.

## Permitted IPM route

If manuscript work resumes, the defensible route is limited to independently validated **appearance information A2** as a stimulus-level construct associated with behavioral ratings in this four-identity set. The EEG result may be reported only as a specification-sensitive exploratory result involving the original A implementation, not as validation of A2 or a stable neural mechanism.

Required wording:

- exploratory;
- multiplicity-controlled;
- stimulus-set-bounded;
- conditional association;
- limited incremental value.

Prohibited wording:

- confirmatory mechanism;
- generalizable neural tracking;
- causal pathway;
- universal face-processing mechanism;
- cross-identity generalization.

## Remaining boundary

Participants are the inferential unit. The 64 edited images are repeated observations from four identities, not 64 independent identity samples. Participant-grouped cross-validation tests new-participant prediction only. External computational validation is not feasible without the missing original editing program and exact parameter provenance.

# Algorithmic Face Editing: Behavioral and EEG Reproducibility Repository

This repository contains the complete analysis code, frozen derivative statistics, audit reports, manuscript figures, and Chinese LaTeX draft for the project provisionally titled:

> **Output-State Change and Editing Provenance in Algorithmic Face Editing: Behavioral and EEG Evidence**

中文说明：本仓库保存论文主路线的数据处理脚本、冻结结果表、可重复性审计、绘图代码与LaTeX稿件。原始EEG、E-Prime记录和可识别人脸图像因许可、隐私和文件规模限制不在仓库中。

## Scope

The repository distinguishes two complementary forms of visual information:

1. **Output-state information**: local cheek-region surface-appearance change, operationalized by the frozen A2 metric.
2. **Transformation-provenance information**: the editing operation and affected region (FSlim, Eye, Mouth, Skin).

The principal inferential sample is fixed at 28 EEG participants (s5 and s18 excluded by the unified data-quality rule). Behavioral models use 30 participants. No analysis in this repository should be interpreted as prospectively preregistered or as supporting causal mediation, universal neural tracking, or new-identity generalization.

## Repository structure

```text
data/                         protected inputs are placed here locally only
docs/                         locked results and reproducibility documentation
manuscript/                   Chinese LaTeX draft, references, and final figures
results/
  visual_gate/                initial visual-information gate outputs
  stage_2_6/                  sample, metric, preprocessing, and specification audits
  stage_2_7/                  frozen A2 route and participant-grouped CV
  stage_2_8/                  operation-level behavioral and EEG route
  erp_dynamics/               full-time-course descriptive boundary outputs
scripts/
  preprocessing/              Curry-to-epoch and ERP reconstruction scripts
  metrics/                    image audit, G, A, A2, and identity metrics
  analysis/                   behavioral, EEG, permutation, sensitivity, and route analyses
  rsa/                        supplementary RSA boundary analyses
  figures/                    manuscript figure regeneration
```

## Reproducibility levels

- **Directly reproducible from included derivatives**: manuscript figures, tables, sample reconciliation summaries, A2 construct checks, grouped-participant CV summaries, fixed-window EEG summaries, and sensitivity displays.
- **Executable with protected inputs supplied locally**: image metrics, trial-level behavioral models, Curry EEG reconstruction, continuous EEG models, MVPA, and RSA.
- **Not distributed**: raw Curry EEG, E-Prime logs, raw/edited face images, trial-level identifiable data, 468-point facial landmark coordinate files, model binaries, and virtual environments.

See [DATA_ACCESS.md](DATA_ACCESS.md) for the required local input layout and exclusions.

## Environment

Create an isolated Python environment and install the declared dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The landmark analysis was originally frozen in a separate MediaPipe environment; its recorded package metadata is retained in `results/stage_2_6/landmark_environment.json`. The historical EEG preprocessing workflow could only be recovered at protocol level for some participant-specific steps; see `results/stage_2_6/eeg_preprocessing_protocol.md` and `eeg_preprocessing_evidence_table.csv`.

## Configure protected data paths

The scripts no longer contain machine-specific absolute paths. Set the following environment variables before running analyses:

```powershell
$env:IPM_DATA_ROOT = "D:\path\to\protected_project_data"
$env:IPM_STIMULUS_ROOT = "D:\path\to\protected_face_images"
```

If unset, scripts look for protected inputs under the repository's ignored `data/` directory.

## Main processing order

```powershell
python scripts/metrics/audit_ipm_image_inputs.py
python scripts/metrics/compute_ipm_visual_metrics.py
python scripts/analysis/run_ipm_visual_gate_models.py

python scripts/analysis/run_stage26_eeg_sample_validation.py
python scripts/metrics/run_stage26_landmark_reproduction.py
python scripts/metrics/run_stage26_appearance_replication.py
python scripts/analysis/run_stage26_frozen_metric_models.py

python scripts/analysis/run_stage27_a2_validated_route.py
python scripts/analysis/run_stage28_ipm_route_search.py
python scripts/figures/build_revision_figures.py
```

The raw Curry reconstruction is a separate, computationally intensive recovery route:

```powershell
python scripts/preprocessing/preprocess_raw_curry.py --subjects s1 s2 --artifact-method fixed
python scripts/preprocessing/run_primary_erp_analysis.py
```

Run `--help` before computationally intensive scripts. Seeds and permutation counts are stored in the scripts and output manifests.

## Frozen headline results

- A2--A agreement: Pearson $r=.6189$; Spearman $\rho=.7573$.
- Participant-grouped behavioral CV: Factor $R^2=.1446$; Factor+A2 $R^2=.1657$; $\Delta R^2=.0211$.
- Skin, 350--600 ms: $\beta=-.2789\,\mu V$, familywise $p=.0154$, $d_z=-.629$.
- Eye, 600--1000 ms: $\beta=.2884\,\mu V$, familywise $p=.0063$, $d_z=.693$.
- Both fixed-window EEG results survived all 28 leave-one-participant-out reruns; identity-omission directions were consistent across four identities.
- Full-time-course joint cluster tests, continuous A2 EEG, EEG--behavior association, and EEG incremental rating prediction were not supported.

The complete locked result statement is in `docs/FINAL_RESULTS_LOCK_CN.md`.

## Data and inference boundaries

- Participants are the inferential unit; trial count is not the participant sample size.
- The 64 edited images are factorial variants of four identities, not 64 independent identity stimuli.
- Participant-grouped cross-validation evaluates prediction for unseen participants only.
- Identity-omission analyses are sensitivity checks, not tests of population-level cross-identity generalization.
- Statistical significance, construct validity, held-out utility, and temporal correspondence are reported as distinct evidence criteria.

## License and citation

No license is granted for the protected human-participant data or face stimuli. A software license has intentionally not been selected pending author and institutional approval. Until then, reuse requires permission from the repository owner. Citation metadata is provided in `CITATION.cff` and should be updated when the manuscript receives a DOI.


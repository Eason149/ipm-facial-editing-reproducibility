# Protected data access and expected layout

Raw human-participant and facial stimulus data are intentionally excluded from GitHub. This repository is complete at the level of analysis code, frozen derivative statistics, audit documentation, and manuscript outputs, but protected-input analyses require locally authorized files.

## Expected data root

Set `IPM_DATA_ROOT` to a directory containing the original project layout used by the scripts. The principal required paths are:

```text
${IPM_DATA_ROOT}/
  EEGDATA/EEGDATA/curry8/Daya-zqy26/sXX/*.cdt
  EEGDATA/EEGDATA/eprime/
  CHB_multimodal_facial_editing/
  paper_extension_final/reanalysis_30/
  erp_dynamics_final/
  ipm_visual_information_gate/
  ipm_stage_2_6/
  ipm_stage_2_7_a2_validated_route/
  mvpa_final/
  RSA_multimodal_geometry/
```

Set `IPM_STIMULUS_ROOT` to the authorized directory containing the 68 core images. Image filenames must match `results/visual_gate/image_condition_mapping.csv`.

## Files intentionally excluded

- Curry recordings and processed epoch arrays;
- E-Prime logs and unrestricted trial-level behavioral records;
- raw and edited facial images;
- 468-point per-image landmark coordinate files;
- face-recognition/detection model binaries;
- virtual environments, caches, and temporary candidate-route outputs.

These exclusions prevent accidental publication of potentially identifiable or licensed material and avoid GitHub size limits. They do not alter the included frozen numerical results.

## Integrity checks

The repository retains image and model hashes where permitted. Local file paths were removed or replaced with environment-variable placeholders before version control. Any authorized reproducibility run should compare regenerated manifests and hashes with the stored audit tables before interpreting downstream differences.


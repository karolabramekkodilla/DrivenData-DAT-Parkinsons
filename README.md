# DrivenData DaTSCAN Parkinson's Challenge

This repository contains my work for the DrivenData DaTSCAN Parkinson's disease classification challenge. The task was to predict whether a DaTSCAN NIfTI volume is pathologic and submit probabilities optimized for log loss.

The most important part of the repository is `code_execution/`, because the competition required a self-contained inference package. The original medical image data and trained model weights are not included in this public repository.

## Repository structure

| Path | Purpose |
| --- | --- |
| `code_execution/` | Final competition-style inference package. |
| `code_execution/main.py` | Loads NIfTI scans, preprocesses volumes, runs model ensembles, and writes `submission.csv`. |
| `code_execution/functions/` | Image preprocessing, cropping, centering, rotation, and profile extraction utilities. |
| `code_execution/model/` | PyTorch model definitions used by the inference script. |
| `src/notebooks/` | Exploratory notebooks from the research phase. |
| `docs/oof_stacking_plan.md` | Notes on the validation leakage found after submission and the planned out-of-fold fix. |

## Problem

The input data consists of 3D DaTSCAN images in NIfTI format. The output is a probability in the `is_pathologic` column for each `uid`.

The main challenges were:

- different image shapes and voxel spacing;
- head position differences between scans;
- small dataset size for 3D deep learning;
- avoiding data leakage while comparing model families;
- keeping inference runnable inside the competition environment.

## Pipeline

The final inference pipeline in `code_execution/main.py` follows this high-level flow:

1. Load each `.nii.gz` scan listed in `submission_format.csv`.
2. Standardize intensity with a custom NIfTI normalization function. The volume is divided 
   by the mean value of the brightest voxels, selected with the `percent_top` parameter.
3. Resample the scan to a common voxel size of `2.46 x 2.46 x 2.46`.
4. Estimate the scan center from 1D intensity profiles calculated along the x, y, and z axes.
5. Estimate rotation, tilt, and pitch errors by comparing image-derived profiles with template 
   profiles using a similarity score.
6. Rotate and crop the volume around the detected brain region.
7. Run several PyTorch model families:
   - CNN models trained on the rotated/cropped 3D volumes;
   - an additional 3D CNN variant trained as a late ensemble candidate;
   - a profile-based MLP.
8. Combine predictions with a saved scikit-learn meta-classifier and a weighted blend.
9. Clip final probabilities to avoid exact `0` or `1` for log loss stability.

Centering and rotation correction were based on custom 1D profile heuristics. 
The profiles were generated with directional filters/aggregations along the volume axes. 
This was a practical preprocessing approach under the competition deadline, but the 
filter design itself could be improved, for example by selecting profile variants through 
a similarity-based validation procedure.

ANTsPyX registration was also tested as a preprocessing variant. It was available 
in the development environment, but it did not improve the final validation/submission 
behavior enough to become the main preprocessing path. The final inference package 
keeps the ANTsPyX-related model family as part of the ensemble, while the core 
preprocessing is based on custom profile/similarity heuristics.

## Models

The final inference code references these trained artifacts:

| Artifact pattern | Role |
| --- | --- |
| `2model_fold_*.pth` | CNN family from the EDA6/similarity preprocessing path. |
| `ants_aug_model_fold_*.pth` | CNN family from the ANTsPyX preprocessing path. |
| `new*.pth` | Larger CNN family used with high final blend weight. |
| `profile_model.pth` | MLP trained on 1D image profiles. |
| `final_meta_classifier_no_native.joblib` | Final scikit-learn meta-classifier. |

These files are intentionally excluded from git because they are generated binary artifacts.

## Validation issue found after submission

After the competition work, I reviewed the stacking stage and found a data leakage risk in the meta-classifier training.

The issue: fold models were used to predict the full training set when building meta-classifier features. For a given training row, most fold models had already seen that row during training, and the remaining fold model may have seen it during validation/early stopping. This means the meta-classifier was trained on in-sample predictions instead of true out-of-fold predictions.

Impact:

- cross-validation log loss for the meta-classifier was probably too optimistic;
- meta-model feature weights were not fully reliable;
- the final high weight of the newer CNN family should be interpreted cautiously.

The intended fix is documented in [docs/oof_stacking_plan.md](docs/oof_stacking_plan.md). In short: create one shared fold split, generate predictions only for each fold's validation rows, train the meta-classifier on those OOF predictions, and compare single CNN / averaged ensemble / stacking on a clean hold-out set.

## How to run inference

The public repository does not include the competition data or trained weights, so `main.py` cannot be run end-to-end from the repository alone.

Expected local layout for inference:

```text
code_execution/
  data/
    submission_format.csv
    niftis/
      <uid>.nii.gz
  model/
    2model_fold_1.pth
    ...
    final_meta_classifier_no_native.joblib
  template_94xfri09.nii.gz
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python code_execution/main.py
```

The script writes:

```text
code_execution/submission.csv
```

## What I would improve next

- Rebuild stacking with true out-of-fold predictions.
- Save and version the shared fold assignment file.
- Keep a final hold-out split untouched until model selection is finished.
- Move notebook experiments into reproducible training scripts.
- Add MLflow tracking for preprocessing variants, model families, and validation results.
- Add small smoke tests for preprocessing functions using synthetic NIfTI volumes.
- Future interpretability work: generate 3D Grad-CAM maps for CNN models to inspect which brain regions contributed most to pathologic predictions.
- Add fold-safe data augmentation for CNN training: small rotations, translations, intensity scaling, and noise, applied only to training folds.

## Notes

This was an exploratory competition project, not a medical diagnostic system. The repository is shared as a portfolio/research project showing 3D medical image preprocessing, PyTorch inference, model ensembling, and a documented validation mistake with a concrete repair plan.

# A15 — Filter ugly exercises, rescore good ones & cut exercise segments

## Overview

This directory takes the ~180 2D exercise videos, filters out exercises with
poor form (predicted "ugly" by a CNN), rescales the remaining good
exercises onto a normalised 0–4 quality score, cuts each clip
to only the frames between the predicted start and stop of the exercise
movement using the A12 start/stop detection model, and finally augments
the cut sequences to expand the training dataset for downstream models.

---

## Pipeline

```
scores.csv (180 clips, raw quality scores)
    │
    ▼ filter_ugly_exercises.py
    │   • loads each clip from Datasets_all/kinect_good_preprocessed/
    │   • runs A_CNN.keras (sigmoid output: P(good))
    │   • threshold: ≥ 0.5 → GOOD, < 0.5 → UGLY
    │
    ├─ a15_predictions.csv   (179 clips processed, 1 missing Kinect file)
    ├─ a15_good_list.csv     (171 clips predicted GOOD)
    ├─ a15_ugly_list.csv     (  8 clips predicted UGLY)
    └─ a15_summary.txt       (text summary of the filter run)
                              │
                              ▼ rescore_good.py
                                  • min-max normalisation to 0–4
                                  • excludes A1_kinect (score=0.0) from
                                    the [min, max] range calculation
                                  • clips result to [0, 4]
                                  │
                                  └─ a15_good_rescaled.csv  (171 clips)
                                    │
                                    ▼ cut_with_a12.py
                                        • loads each clip from kinect_good_preprocessed/
                                        • engineers same 70 features as A12 training
                                          (distances, velocities, accelerations)
                                        • runs A_Kinect_Dense_relu_adam_bs64
                                          frame-by-frame → binary prediction
                                        • first 0→1 transition → START,
                                          last 1→0 transition → STOP
                                        • deletes sequences shorter than 30 frames
                                        │
                                        ├─ a15_cut/              (148 cut CSVs)
                                        └─ a15_cut_summary.csv   (per-clip metadata)
                                          │
                                          ▼ augment_cut_data.py
                                              • loads each cut CSV from a15_cut/
                                              • applies 4 spatial augmentations:
                                                — mirror on y-axis
                                                — rotate ±10° on y-axis
                                                — stretch/compress x,y,z
                                              • each variant inherits the original score
                                              │
                                              ├─ a15_cut_augmented/   (740 CSVs)
                                              └─ a15_augmented_data.csv   (740 rows)
```

---

## Files

| File | Rows | Description |
|---|---|---|
| `scores.csv` | 180 | Original raw scores from upstream (Var1=clip, Var2=score) |
| `a15_predictions.csv` | 179 | Per-clip original scores + model `good_probability` + `predicted_label` |
| `a15_good_list.csv` | 171 | Subset predicted GOOD |
| `a15_ugly_list.csv` | 8 | Subset predicted UGLY |
| `a15_good_rescaled.csv` | **171** | Good clips with min-max normalised score (0–4) |
| `a15_summary.txt` | — | Human-readable summary of the filter step |
| `a15_cut_summary.csv` | **171** | Per-clip cutting metadata (status, start/stop frames, lengths) |

### Directories

| Directory | Contents |
|---|---|
| `a15_cut/` | 148 cut CSV files (≥ 30 frames each), frames trimmed to predicted exercise segment |
| `a15_cut_augmented/` | 740 CSV files (148 original + 592 augmented), each with full frame-by-frame skeleton data |

### Utility scripts

| Script | Purpose |
|---|---|
| `filter_ugly_exercises.py` | Connects `scores.csv` with Kinect CSVs, runs A_CNN inference, splits good/ugly |
| `rescore_good.py` | Min-max normalises good-clip scores to [0, 4] |
| `rescore.py` | (Original reference) — simple `score / max * 4` rescale on all clips |
| `cut_with_a12.py` | Cuts each good clip to the exercise segment using the A12 start/stop classifier |
| `augment_cut_data.py` | Augments cut sequences via mirror, rotation, and stretch/compress transformations |

---

## Rescaling method (`rescore_good.py`)

The original `rescore.py` uses a simple linear rescale:

```
score_rescaled = (score / max_score) * 4
```

This clustered most good clips between 3 and 4, providing poor
discrimination.  The updated `rescore_good.py` uses **min-max
normalisation**, which spreads scores evenly across the full range:

```
score_rescaled = ((score - min_score) / (max_score - min_score)) * 4
               → clamped to [0, 4]
```

**Calibration set**: 170 good clips (all except A1_kinect, which has
score=0.0 — an artifact, not a meaningful observation).

| Statistic | Value |
|---|---|
| Min (maps to 0) | 0.8879 (A24_kinect) |
| Max (maps to 4) | 1.3354 (B2_kinect) |
| Range width | 0.4475 |

A1_kinect is **kept in the output** with its clipped score of 0.0, but
excluded from the min/max calculation so it does not compress the range for
all other clips.

---

## Cutting method (`cut_with_a12.py`)

After filtering and rescoring, each clip undergoes frame-level **start/stop
detection** using the best-performing model from A12:
`A_Kinect_Dense_relu_adam_bs64` (Dense architecture, 70 input features,
3-fold CV F1 = 0.949).

### Steps
1. **Feature engineering** — joint distances (hand-to-shoulder, head-to-hip,
   etc.), velocities, and accelerations are computed for every frame,
   matching the exact feature set used in A12 training.
2. **Frame classification** — the Dense model predicts exercise (1) vs
   non-exercise (0) for each frame independently.
3. **Segment detection** — the first 0→1 transition marks the **start**
   frame, and the last 1→0 transition marks the **stop** frame.
4. **Cut** — frames outside [start, stop] are discarded.
5. **Length filter** — sequences shorter than **30 frames** (≈ 1 s at
   30 fps) are considered too short and are removed.

### Results

| Metric | Count |
|---|---|
| Good clips processed | 171 |
| Cut OK (≥ 30 frames) | **148** |
| Too short, removed (< 30 frames) | 23 |
| Avg. cut length | ~129 frames (~4.3 s) |

### Output columns (`a15_cut_summary.csv`)

| Column | Description |
|---|---|
| `clip` | Exercise clip identifier |
| `status` | `ok`, `too_short_N_frames`, or `missing_kinect_csv` |
| `n_frames_input` | Number of frames in the original (uncut) CSV |
| `n_frames_cut` | Number of frames after cutting |
| `start_frame` | Predicted start frame (FrameNo value) |
| `stop_frame` | Predicted stop frame (FrameNo value) |
| `cut_path` | Path to the cut CSV file (empty if too short) |

---

## Augmentation method (`augment_cut_data.py`)

After cutting, each sequence is augmented in-place to produce 4 additional
variants per original clip, for a 5× expansion of the dataset (148 → 740).
Only spatial augmentations that preserve the exercise label and quality
score are applied.

### Augmentations

| Augmentation | Suffix | Description |
|---|---|---|
| Mirror on y-axis | `_mirror` | Negate all x-coordinates, mirroring left ↔ right |
| Rotate +10° on y-axis | `_rotate_pos` | 3D rotation matrix around y-axis (+10°) |
| Rotate −10° on y-axis | `_rotate_neg` | 3D rotation matrix around y-axis (−10°) |
| Stretch/compress | `_stretch` | Scale x by +5%, y by −5%, z by +2% |

Each augmented variant keeps its original `score_rescaled` and
`good_probability` — the augmentation changes only the skeleton
coordinates, not the underlying quality of the exercise.

### Input / output

| Item | Path |
|---|---|
| Input directory | `a15_cut/` — 148 frame-by-frame skeleton CSVs |
| Output directory | `a15_cut_augmented/` — 740 CSVs |
| Metadata | `a15_augmented_data.csv` — clip, score_rescaled, good_probability |

### Output columns (`a15_augmented_data.csv`)

| Column | Description |
|---|---|
| `clip` | Exercise clip identifier (e.g. `A100_kinect_mirror`) |
| `score_rescaled` | Min-max normalised quality score in [0, 4] (same as original) |
| `good_probability` | CNN sigmoid output — model confidence that the exercise is GOOD (same as original) |

---

## Final output columns (`a15_good_rescaled.csv`)

| Column | Description |
|---|---|
| `clip` | Exercise clip identifier (e.g. `A100_kinect`) |
| `score_rescaled` | Min-max normalised quality score in [0, 4] |
| `good_probability` | CNN sigmoid output — model confidence that the exercise is GOOD |

---

## Supplementary output (`a15_augmented_data.csv`)

See the [Augmentation method](#augmentation-method-augment_cut_datapy) section above.
This file combines original and augmented clips (740 rows) and is the
primary training input for downstream models that benefit from a larger,
spatially varied dataset.
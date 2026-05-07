# Dataset Augmentation for Processed Skeleton Data

This directory contains scripts and results for augmenting the processed skeleton-based classification data in the A13 project.

## Overview

The augmentation script applies geometric transformations to the processed skeleton data to increase dataset diversity and improve model generalization. The following transformations are applied:

1. **Mirror on y-axis**: Flips x-coordinates to simulate left-right mirroring
2. **Rotation on y-axis**: Rotates skeleton by ±10 degrees around the y-axis
3. **Stretch/Compress**: Scales coordinates by small percentages along x, y, z axes

## Files

- `augment_processed_data.py`: Main augmentation script
- `test_augmented_data.py`: Validation script for augmented data
- `README.md`: This documentation file

## Augmented Datasets

The following augmented datasets were created in the `Processed_Data/` directory:

- `processed_sequences_Good_vs_Bad_train_augmented.csv`: 455 samples (91 original + augmentations)
- `processed_sequences_Good_vs_Bad_test_augmented.csv`: 115 samples (23 original + augmentations) 
- `processed_sequences_kinect_good_vs_bad_not_preprocessed_train_augmented.csv`: 455 samples (91 original + augmentations)
- `processed_sequences_kinect_good_vs_bad_not_preprocessed_test_augmented.csv`: 115 samples (23 original + augmentations)

## Augmentation Details

Each original sample generates 4 additional samples:
- `_mirror`: Mirrored version
- `_rotate_pos`: Rotated +10 degrees
- `_rotate_neg`: Rotated -10 degrees  
- `_stretch`: Scaled by factors (x: 1.05, y: 0.95, z: 1.02)

## Important Notes

- Only original samples are augmented (not previously augmented ones)
- The augmentation preserves the temporal structure of the sequences
- Labels are preserved during augmentation (same as original sample)
- Each sequence consists of 10 frames with 102 features per frame (1020 total features)

## Testing

The augmented data has been validated with the test script `test_augmented_data.py` which confirms:
- Augmented data maintains label consistency 
- Original and augmented data can be used interchangeably in ML pipelines
- Data shapes remain compatible with existing processing workflows
- Augmentation increases dataset size by 5x (original + 4 augmentations)
# Classification Problems Definition

This directory contains prepared data for 2 classification problems with 2 approaches each, derived from the processed skeleton data.

## Problem Definitions

### Problem A: 3D Classification (Kinect-based)
- **Input**: 3D joint coordinates from Kinect sensor
- **Structure**: 13 joints × 3 dimensions (x, y, z) = 39 features per frame
- **Temporal**: 10 frames per sequence
- **Total features**: 39 × 10 = 390 features per sequence
- **Task**: Classify movement quality as Good (1) or Bad (0)

#### Approaches:
- **ADense**: Flattened features for dense neural networks
  - Shape: (samples, 390)
- **ACNN**: Structured features for convolutional neural networks  
  - Shape: (samples, 10, 13, 3) - [time_steps, joints, coordinates]

### Problem B: 2D Classification (PoseNet-based)
- **Input**: 2D joint coordinates from PoseNet/MediaPipe
- **Structure**: 13 joints × 2 dimensions (x, y) = 26 features per frame
- **Temporal**: 10 frames per sequence
- **Total features**: 26 × 10 = 260 features per sequence
- **Task**: Classify movement quality as Good (1) or Bad (0)

#### Approaches:
- **BDense**: Flattened features for dense neural networks
  - Shape: (samples, 260)
- **BCNN**: Structured features for convolutional neural networks
  - Shape: (samples, 10, 13, 2) - [time_steps, joints, coordinates]

## Data Organization

The prepared data is organized in the `prepared_data/` directory:

### ADense Files (3D, Flattened):
- `A_Dense_train_X.npy`: Training features (shape: samples×390)
- `A_Dense_train_y.npy`: Training labels
- `A_Dense_test_X.npy`: Test features (shape: samples×390)
- `A_Dense_test_y.npy`: Test labels
- `A_Dense_train_aug_X.npy`: Augmented training features
- `A_Dense_train_aug_y.npy`: Augmented training labels
- `A_Dense_test_aug_X.npy`: Augmented test features
- `A_Dense_test_aug_y.npy`: Augmented test labels

### ACNN Files (3D, Structured):
- `A_CNN_train_X.npy`: Training features (shape: samples×10×13×3)
- `A_CNN_train_y.npy`: Training labels
- `A_CNN_test_X.npy`: Test features (shape: samples×10×13×3)
- `A_CNN_test_y.npy`: Test labels
- `A_CNN_train_aug_X.npy`: Augmented training features
- `A_CNN_train_aug_y.npy`: Augmented training labels
- `A_CNN_test_aug_X.npy`: Augmented test features
- `A_CNN_test_aug_y.npy`: Augmented test labels

### BDense Files (2D, Flattened):
- `B_Dense_train_X.npy`: Training features (shape: samples×260)
- `B_Dense_train_y.npy`: Training labels
- `B_Dense_test_X.npy`: Test features (shape: samples×260)
- `B_Dense_test_y.npy`: Test labels
- `B_Dense_train_aug_X.npy`: Augmented training features
- `B_Dense_train_aug_y.npy`: Augmented training labels
- `B_Dense_test_aug_X.npy`: Augmented test features
- `B_Dense_test_aug_y.npy`: Augmented test labels

### BCNN Files (2D, Structured):
- `B_CNN_train_X.npy`: Training features (shape: samples×10×13×2)
- `B_CNN_train_y.npy`: Training labels
- `B_CNN_test_X.npy`: Test features (shape: samples×10×13×2)
- `B_CNN_test_y.npy`: Test labels
- `B_CNN_train_aug_X.npy`: Augmented training features
- `B_CNN_train_aug_y.npy`: Augmented training labels
- `B_CNN_test_aug_X.npy`: Augmented test features
- `B_CNN_test_aug_y.npy`: Augmented test labels

## Data Sources

The data was extracted from the original processed sequences by interpreting the first portion of each frame as joint coordinates:
- For 3D problem: First 39 values per frame interpreted as 13 joints × 3 coordinates
- For 2D problem: First 26 values per frame interpreted as 13 joints × 2 coordinates

## Usage Examples

```python
import numpy as np

# Load ADense data for training a dense network
X_train = np.load('prepared_data/A_Dense_train_X.npy')
y_train = np.load('prepared_data/A_Dense_train_y.npy')
X_test = np.load('prepared_data/A_Dense_test_X.npy')
y_test = np.load('prepared_data/A_Dense_test_y.npy')

# Load ACNN data for training a CNN
X_train_cnn = np.load('prepared_data/A_CNN_train_X.npy')
y_train_cnn = np.load('prepared_data/A_CNN_train_y.npy')
```

## Note on Augmented Data

All datasets include both original and augmented versions:
- Original data maintains the original 91 training + 23 test samples
- Augmented data includes 4 additional versions per original sample (mirror, rotate ±10°, stretch)
- Total augmented data: 455 training + 115 test samples
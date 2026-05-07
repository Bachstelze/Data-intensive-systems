# Processed Data Summary

## Overview
This document summarizes the preprocessing of the raw kinect data for the good vs bad classification task.

## Data Preprocessing Steps

### 1. Fixed Size Sequence Creation
- Each video sequence was converted to a fixed length of **c = 10 frames**
- Frames were selected **equidistantly** from the relevant portion of each video
- When a video had fewer than 10 frames in the relevant range, it was padded with zero frames

### 2. Frame Selection Method
- For videos with start/stop frame annotations (e.g., A1.start_stop_frames), only the frames between start and stop were considered
- For videos without start/stop annotations, the full sequence was used
- From the selected frame range, 10 equidistant frames were extracted

### 3. Coordinate Smoothing (Optional)
- Applied averaging of surrounding frames to smooth coordinate values
- Used a window size of 3 frames to compute local averages

### 4. Label Assignment
- **Good sequences**: Labeled as `1`
  - Files starting with `G` (e.g., G01, G02, ...)
  - File `A1`
- **Bad sequences**: Labeled as `0`
  - Files starting with `W` (e.g., W01, W02, ...)

## Final Dataset Statistics

| Metric | Value |
|--------|-------|
| Total number of sequences | 114 |
| Good sequences | 71 |
| Bad sequences | 43 |
| Frames per sequence | 10 |
| Features per frame | 102 |
| Total features per sequence | 1020 |

## File Structure

The processed data is stored in:
```
Processed_Data/
├── sequences.npy          # Numpy array of shape (114, 10, 102)
├── labels.npy             # Numpy array of shape (114,) with binary labels
└── processed_sequences_with_labels.csv  # CSV file with labels and filenames
```

## Data Format Details

- Each sequence contains 10 frames
- Each frame has 102 features (coordinates and metadata)
- The first 3 columns in each frame are typically metadata (frame number, timestamp, etc.)
- The remaining 99 columns represent kinect sensor coordinates
- Padded frames (when sequences were shorter than 10) have frame number -1

## Usage Example

```python
import numpy as np

# Load the data
sequences = np.load('Processed_Data/sequences.npy')  # Shape: (114, 10, 102)
labels = np.load('Processed_Data/labels.npy')        # Shape: (114,)

# Access first sequence
first_sequence = sequences[0]    # Shape: (10, 102)
first_label = labels[0]          # Either 0 (bad) or 1 (good)
```

## Notes
- The original data was sourced from two directories:
  - `Good vs Bad`: Contains 114 files with mixed good/bad labels
  - `kinect_good_vs_bad_not_preprocessed`: Skipped due to different format with headers
- Only the first directory was successfully processed
- The preprocessing ensures consistent input dimensions for machine learning models
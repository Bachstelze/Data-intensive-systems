# Classification Data Checker

This directory contains the script to check and correct outliers in the start/stop classification data.

## Overview

The `check_classification_data.py` script performs the following tasks:

1. **Loads classification data** from the `classification_data` directory
2. **Detects outliers** based on velocity and acceleration changes
3. **Analyzes label distribution** to identify potentially mislabeled frames
4. **Checks label transitions** for unusual patterns
5. **Compares cut and not-cut data** to detect frame count anomalies
6. **Provides interactive correction** for label issues
7. **Filters outlier files** to remove problematic files from the dataset

## Usage

Run the script from the command line:

```bash
cd /home/cyclonaut/Dokumente/Studium\ Växjö/data\ intensive\ project/end_epril_repo/Data-intensive-systems/A11/check_classification_data
python3 check_classification_data.py
```

## How It Works

### Outlier Detection

The script identifies potential outliers by checking for:
- Significant changes in joint speeds (> 2.0 units)
- High acceleration values (> 5.0 units)

### Label Analysis

Frames are flagged as potentially mislabeled if:
- A 'start' frame doesn't have a significant velocity change
- A 'stop' frame doesn't have a significant velocity change
- A 'neutral' frame has a significant velocity change

### Label Transition Checks

Unusual transitions are flagged:
- 'start' -> 'stop' or 'stop' -> 'start' within fewer than 5 frames

### Cut Frame Comparison

Compares start/stop frame counts between cut and not-cut data:
- Not-cut files should have more start/stop labels (they contain the full motion)
- Cut files should have fewer start/stop labels (boundaries are removed)
- Files where cut has >30% of not-cut start/stop frames are flagged as outliers

## Interactive Correction

When prompted, you can correct labels using these commands:

| Command | Action |
|---------|--------|
| `s` | Change label to 'start' |
| `t` | Change label to 'stop' |
| `n` | Change label to 'neutral' |
| `k` | Keep current label |
| `q` | Quit without saving |
| `a` | Accept all remaining changes |

## Output

The script saves outputs to:
```
check_classification_data/corrected_data/
├── corrected_classification_data.csv   # Manually corrected labels
├── filtered_classification_data.csv    # Filtered outlier files removed
```

## Files

- `check_classification_data.py` - Main script
- `README.md` - This file
- `corrected_data/` - Directory for corrected output files

## Thresholds

The following thresholds are used for detection (can be adjusted in the script):

| Threshold | Default Value | Description |
|-----------|---------------|-------------|
| VELOCITY_CHANGE_THRESHOLD | 2.0 | Minimum velocity change to flag as outlier |
| ACCELERATION_THRESHOLD | 5.0 | Minimum acceleration to flag as outlier |
| SPEED_THRESHOLD | 0.3 | Minimum speed for motion detection |
| CUT_START_STOP_RATIO_THRESHOLD | 0.3 | Max ratio of start/stop in cut vs not-cut |
| CUT_FRAME_COUNT_THRESHOLD | 0.5 | Min ratio of cut frames to not-cut frames |

## Prerequisites

- Python 3.x
- pandas
- numpy

The classification data should be generated first by running `prepare_classification_data.py`.
#!/usr/bin/env python3
"""
Rescore only the "good" exercises (from a15_good_list.csv) with min-max
normalisation to 0–4, excluding the A1_kinect outlier (score = 0.0) from
the calculation but keeping it in the output.

Previously used old/max*4 linear rescale, which bunched most values in 3–4.
Min-max normalisation spreads them fully across the 0–4 range.

  score_rescaled = ((score - min) / (max - min)) * 4
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

INPUT_CSV  = HERE / "a15_good_list.csv"
OUTPUT_CSV = HERE / "a15_good_rescaled.csv"

with open(INPUT_CSV, newline="") as f:
    reader = csv.DictReader(f)
    rows = [(r["clip"], float(r["score"]), float(r["good_probability"])) for r in reader]

print(f"Loaded {len(rows)} good clips from {INPUT_CSV.name}")

# Separate the outlier from the calculation pool
outlier_rows = [(c, s, p) for c, s, p in rows if c == "A1_kinect"]
calc_rows    = [(c, s, p) for c, s, p in rows if c != "A1_kinect"]

print(f"Excluded {len(outlier_rows)} outlier(s) (A1_kinect, score=0.0) from calculation")
print(f"Using {len(calc_rows)} clips for min-max range")

# Min-max normalisation on the filtered set
scores = [s for _, s, _ in calc_rows]
min_score = min(scores)
max_score = max(scores)
score_range = max_score - min_score

print(f"  Score range:  {min_score} – {max_score}")
print(f"  Range width:  {score_range}")

def rescale(val: float) -> float:
    """Min-max to 0–4, clipped to [0, 4]."""
    raw = ((val - min_score) / score_range) * 4
    return round(max(0.0, min(4.0, raw)), 6)

# Build output: calculation rows rescaled, then outlier appended at the end
rescaled = [(clip, rescale(val), proba) for clip, val, proba in calc_rows]
rescaled += [(clip, rescale(val), proba) for clip, val, proba in outlier_rows]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["clip", "score_rescaled", "good_probability"])
    writer.writerows(rescaled)

print(f"Written {len(rescaled)} rows to {OUTPUT_CSV.name}")

# Show a few examples
print("\nExamples (min-max normalised, 0–4):")
for clip, old, prob in calc_rows[:3] + calc_rows[len(calc_rows)//2:len(calc_rows)//2+1] + calc_rows[-3:] + outlier_rows:
    new = rescale(old)
    print(f"  {clip}: {old} → {new}  (P(good)={prob})")

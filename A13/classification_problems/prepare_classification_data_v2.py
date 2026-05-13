#!/usr/bin/env python3
"""Rebuild the prepared classification arrays from clean raw Kinect data.

Replaces the broken ``prepare_classification_problems.py`` whose
"first 39 features per frame" slice silently captured 3 metadata columns
(FrameNo, timestamp, padding-zero) from the 102-feature processed format,
producing 1.27e9-magnitude garbage in "joint 0" and shifting all later
joints by one axis. That made the BatchNorm-first-layer model learn on
fantasy features and always predict "good" when the app fed real
coordinate-scale inputs.

This v2 reads the original 40-column raw Kinect CSVs directly
(``A13/kinect_good_vs_bad_not_preprocessed/``) and builds clean
(10, 13, 3) sequences with real meter-scale joint coordinates.

Outputs to ``A13/classification_problems/prepared_data/`` the exact
file set expected by ``A13/dl_models/data_loader.py``:

    {A,B}_{Dense,CNN}_{train,train_aug,test}_{X,y}.npy
    {A,B}_{Dense,CNN}_{train_aug,test}_filenames.npy

Problem A = 3D (Kinect, 13 joints x 3 dims).
Problem B = 2D (x,y projection of the same Kinect data; the repo
does not contain PoseNet recordings for the Good-vs-Bad clips, so we
project rather than guess. The architecture and CV protocol are
unchanged; only the input channel count differs).

Augmentations applied to the training set only (test never augmented):
    _mirror      : negate x coordinates
    _rotate_pos  : +10 deg around vertical (Y) axis
    _rotate_neg  : -10 deg around vertical (Y) axis
    _stretch     : isotropic scale by 1.05

Run::

    python -m A13.classification_problems.prepare_classification_data_v2
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------- paths
THIS_DIR = Path(__file__).resolve().parent
RAW_DIR = THIS_DIR.parent / "kinect_good_vs_bad_not_preprocessed"
OUT_DIR = THIS_DIR / "prepared_data"

# --------------------------------------------------------------------- consts
FRAMES = 10
JOINTS = 13  # head + 6 upper-body + 6 lower-body, matches the CSV schema
DIMS = 3
RANDOM_STATE = 42
TEST_SIZE = 0.2
ROT_DEG = 10.0
STRETCH = 1.05


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ labeling
def label_from_filename(stem: str) -> int:
    """G* or A1 -> 1 (good); W* -> 0 (bad). Matches the original spec."""
    if stem == "A1" or stem.startswith("G"):
        return 1
    if stem.startswith("W"):
        return 0
    raise ValueError(f"Unknown label for {stem!r}")


# ----------------------------------------------------------------- load clip
def load_clip(csv_path: Path) -> np.ndarray:
    """Return (FRAMES, JOINTS, DIMS) float32 array of joint coords."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Drop FrameNo; the remaining 39 cols are 13 joints x (x, y, z).
    if "FrameNo" not in df.columns:
        raise ValueError(f"{csv_path.name}: expected a FrameNo column")
    coords = df.drop(columns=["FrameNo"]).values.astype("float32")
    n_rows, n_cols = coords.shape
    if n_cols != JOINTS * DIMS:
        raise ValueError(
            f"{csv_path.name}: expected {JOINTS * DIMS} coord cols, got {n_cols}"
        )

    # Equidistant subsample to FRAMES; if shorter, pad with last frame.
    if n_rows >= FRAMES:
        idx = np.linspace(0, n_rows - 1, FRAMES, dtype=int)
        seq = coords[idx]
    else:
        seq = np.zeros((FRAMES, n_cols), dtype="float32")
        seq[:n_rows] = coords
        if n_rows > 0:
            seq[n_rows:] = coords[-1]

    return seq.reshape(FRAMES, JOINTS, DIMS)


# ------------------------------------------------------------- augmentations
def aug_mirror(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    out[..., 0] = -out[..., 0]
    return out


def _rotate_y(seq: np.ndarray, deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    out = seq.copy()
    x = seq[..., 0]
    z = seq[..., 2]
    out[..., 0] = c * x + s * z
    out[..., 2] = -s * x + c * z
    return out


def aug_rotate_pos(seq: np.ndarray) -> np.ndarray:
    return _rotate_y(seq, +ROT_DEG)


def aug_rotate_neg(seq: np.ndarray) -> np.ndarray:
    return _rotate_y(seq, -ROT_DEG)


def aug_stretch(seq: np.ndarray) -> np.ndarray:
    return seq * STRETCH


AUGS = [
    ("_mirror", aug_mirror),
    ("_rotate_pos", aug_rotate_pos),
    ("_rotate_neg", aug_rotate_neg),
    ("_stretch", aug_stretch),
]


# ----------------------------------------------------------------- pipeline
def collect_clips() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(p for p in RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {RAW_DIR}")
    log(f"[1] reading {len(files)} clips from {RAW_DIR}")
    seqs, labels, names = [], [], []
    for i, p in enumerate(files):
        stem = p.stem
        try:
            y = label_from_filename(stem)
        except ValueError as e:
            log(f"    skip {stem}: {e}")
            continue
        seq = load_clip(p)
        seqs.append(seq)
        labels.append(y)
        names.append(stem)
        if (i + 1) % 25 == 0:
            log(f"    loaded {i + 1}/{len(files)}")
    X = np.stack(seqs).astype("float32")          # (N, 10, 13, 3)
    y = np.asarray(labels, dtype="int32")         # (N,)
    fn = np.asarray(names, dtype=object)          # (N,)
    log(f"    -> X {X.shape}  y {y.shape}  good={int(y.sum())}  bad={int((y == 0).sum())}")
    log(f"    coord scale: min={X.min():.3g}  max={X.max():.3g}  mean={X.mean():.3g}")
    return X, y, fn


def split(X, y, fn):
    log(f"[2] stratified split test_size={TEST_SIZE} random_state={RANDOM_STATE}")
    Xtr, Xte, ytr, yte, ftr, fte = train_test_split(
        X, y, fn, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log(f"    train: {Xtr.shape}  good={int(ytr.sum())}/{len(ytr)}")
    log(f"    test:  {Xte.shape}  good={int(yte.sum())}/{len(yte)}")
    return Xtr, ytr, ftr, Xte, yte, fte


def augment(Xtr, ytr, ftr):
    log(f"[3] augmenting train (originals + {len(AUGS)} variants each)")
    X_all = [Xtr]
    y_all = [ytr]
    f_all = [ftr]
    for suf, fn in AUGS:
        X_all.append(np.stack([fn(s) for s in Xtr]))
        y_all.append(ytr.copy())
        f_all.append(np.asarray([f"{n}{suf}" for n in ftr], dtype=object))
    X = np.concatenate(X_all, axis=0).astype("float32")
    y = np.concatenate(y_all, axis=0).astype("int32")
    f = np.concatenate(f_all, axis=0)
    log(f"    -> aug train: {X.shape}  good={int(y.sum())}/{len(y)}")
    return X, y, f


def save_problem(problem: str, dims_keep: int,
                 Xtr, ytr, ftr,
                 Xtr_aug, ytr_aug, ftr_aug,
                 Xte, yte, fte):
    """Slice last axis to ``dims_keep`` and write Dense+CNN variants."""
    def proj(X):
        return X[..., :dims_keep]

    Xtr_p = proj(Xtr)
    Xtr_aug_p = proj(Xtr_aug)
    Xte_p = proj(Xte)

    # Dense = flatten
    n_feat = FRAMES * JOINTS * dims_keep
    pairs_dense = {
        f"{problem}_Dense_train_X":          Xtr_p.reshape(len(Xtr_p), n_feat),
        f"{problem}_Dense_train_y":          ytr,
        f"{problem}_Dense_train_aug_X":      Xtr_aug_p.reshape(len(Xtr_aug_p), n_feat),
        f"{problem}_Dense_train_aug_y":      ytr_aug,
        f"{problem}_Dense_train_aug_filenames": ftr_aug,
        f"{problem}_Dense_test_X":           Xte_p.reshape(len(Xte_p), n_feat),
        f"{problem}_Dense_test_y":           yte,
        f"{problem}_Dense_test_filenames":   fte,
    }
    # CNN = keep (frames, joints, dims)
    pairs_cnn = {
        f"{problem}_CNN_train_X":          Xtr_p,
        f"{problem}_CNN_train_y":          ytr,
        f"{problem}_CNN_train_aug_X":      Xtr_aug_p,
        f"{problem}_CNN_train_aug_y":      ytr_aug,
        f"{problem}_CNN_train_aug_filenames": ftr_aug,
        f"{problem}_CNN_test_X":           Xte_p,
        f"{problem}_CNN_test_y":           yte,
        f"{problem}_CNN_test_filenames":   fte,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in {**pairs_dense, **pairs_cnn}.items():
        np.save(OUT_DIR / f"{name}.npy", arr)
    log(
        f"    wrote 16 files for problem {problem} "
        f"(Dense {n_feat}-feat, CNN {(FRAMES, JOINTS, dims_keep)})"
    )


def main():
    log("=" * 70)
    log("prepare_classification_data_v2: clean rebuild from raw Kinect CSVs")
    log("=" * 70)
    X, y, fn = collect_clips()
    Xtr, ytr, ftr, Xte, yte, fte = split(X, y, fn)
    Xtr_aug, ytr_aug, ftr_aug = augment(Xtr, ytr, ftr)

    log("[4] writing Problem A (3D Kinect, 13x3)")
    save_problem("A", 3, Xtr, ytr, ftr, Xtr_aug, ytr_aug, ftr_aug, Xte, yte, fte)
    log("[5] writing Problem B (2D x,y projection of Kinect, 13x2)")
    save_problem("B", 2, Xtr, ytr, ftr, Xtr_aug, ytr_aug, ftr_aug, Xte, yte, fte)
    log(f"[6] done. output dir: {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main() or 0)

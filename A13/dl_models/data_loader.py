"""Load the prepared classification data produced in Issue #9.

The directory ``A13/classification_problems/prepared_data`` contains:

* ``{P}_{M}_train_X.npy``       original train features
* ``{P}_{M}_train_y.npy``       original train labels
* ``{P}_{M}_train_aug_X.npy``   augmented train features (incl. originals)
* ``{P}_{M}_train_aug_y.npy``   augmented train labels
* ``{P}_{M}_test_X.npy``        held-out test features
* ``{P}_{M}_test_y.npy``        held-out test labels
* ``{P}_{M}_*_filenames.npy``   the source clip name (used to keep all
                                augmentations of one clip in the same CV fold)

with ``P in {A, B}`` and ``M in {Dense, CNN}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Resolve the prepared_data directory relative to this file so that the
# package works no matter from where the notebook / script is launched.
_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = (_THIS_DIR.parent / "classification_problems" / "prepared_data").resolve()


@dataclass
class Dataset:
    """Container holding all arrays for one (problem, model) combination."""

    problem: str          # "A" or "B"
    model_kind: str       # "Dense" or "CNN"
    X_train: np.ndarray   # original (un-augmented) train features
    y_train: np.ndarray
    X_train_aug: np.ndarray  # augmented train features (used for fitting)
    y_train_aug: np.ndarray
    train_groups: np.ndarray  # source-clip id per augmented train sample
    X_test: np.ndarray
    y_test: np.ndarray
    test_filenames: np.ndarray

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.X_train_aug.shape[1:]

    def summary(self) -> str:
        return (
            f"Problem {self.problem} / {self.model_kind}: "
            f"train_aug={self.X_train_aug.shape}, "
            f"test={self.X_test.shape}, "
            f"pos_train={int(self.y_train_aug.sum())}/{len(self.y_train_aug)}, "
            f"pos_test={int(self.y_test.sum())}/{len(self.y_test)}"
        )


def _load(name: str) -> np.ndarray:
    path = DATA_DIR / f"{name}.npy"
    return np.load(path, allow_pickle=True)


# Augmentation suffixes appended to source-clip filenames in the prepared data.
# The CV must group all augmented copies of one source clip together, so we
# strip these suffixes to recover the original clip id (e.g. ``A1_mirror`` -> ``A1``).
_AUG_SUFFIXES = ("_mirror", "_rotate_pos", "_rotate_neg", "_stretch")


def _source_clip_ids(filenames: np.ndarray) -> np.ndarray:
    out = np.empty(len(filenames), dtype=object)
    for i, name in enumerate(filenames):
        s = str(name)
        for suf in _AUG_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)]
                break
        out[i] = s
    return out


def load_dataset(problem: str, model_kind: str) -> Dataset:
    """Load arrays for problem ``A``/``B`` and ``Dense``/``CNN``."""

    if problem not in {"A", "B"}:
        raise ValueError(f"problem must be 'A' or 'B', got {problem!r}")
    if model_kind not in {"Dense", "CNN"}:
        raise ValueError(f"model_kind must be 'Dense' or 'CNN', got {model_kind!r}")

    prefix = f"{problem}_{model_kind}"
    return Dataset(
        problem=problem,
        model_kind=model_kind,
        X_train=_load(f"{prefix}_train_X").astype("float32"),
        y_train=_load(f"{prefix}_train_y").astype("int32"),
        X_train_aug=_load(f"{prefix}_train_aug_X").astype("float32"),
        y_train_aug=_load(f"{prefix}_train_aug_y").astype("int32"),
        train_groups=_source_clip_ids(_load(f"{prefix}_train_aug_filenames")),
        X_test=_load(f"{prefix}_test_X").astype("float32"),
        y_test=_load(f"{prefix}_test_y").astype("int32"),
        test_filenames=_load(f"{prefix}_test_filenames"),
    )


def load_all() -> dict[tuple[str, str], Dataset]:
    """Convenience helper returning the four datasets keyed by (problem, kind)."""

    return {(p, m): load_dataset(p, m) for p in ("A", "B") for m in ("Dense", "CNN")}

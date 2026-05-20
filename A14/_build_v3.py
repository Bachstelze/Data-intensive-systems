"""Build A14_Report_v3.ipynb from v2 + inlined source modules.

Standalone goal: v3 imports nothing from A13.* / A14.*.  All Python code from
A13/dl_models/models.py, A13/dl_models/data_loader.py, and A14/auto_rubric.py
is inlined.  Data + saved-model paths remain configurable; defaults keep the
notebook runnable from inside the repo with no edits.
"""
from __future__ import annotations
import json
import re
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V2 = HERE / "A14_Report_v2.ipynb"
V3 = HERE / "A14_Report_v3.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def file_text(p: Path) -> str:
    return p.read_text()


# ----------------------------------------------------------------------------
# Source of the inlined modules.  We strip the from-__future__ imports and the
# module docstring to avoid noisy top-of-cell prose, but otherwise leave the
# code byte-identical to the repo.
# ----------------------------------------------------------------------------
def strip_module_preamble(src: str) -> str:
    # Drop the leading module docstring and `from __future__` line.
    out = re.sub(r'^"""[\s\S]*?"""\n', "", src, count=1)
    out = re.sub(r"^from __future__ import [^\n]*\n", "", out, flags=re.MULTILINE)
    return out.lstrip("\n")


models_src = strip_module_preamble(file_text(REPO / "A13/dl_models/models.py"))
loader_src = strip_module_preamble(file_text(REPO / "A13/dl_models/data_loader.py"))
rubric_src = strip_module_preamble(file_text(REPO / "A14/auto_rubric.py"))

# Patch the data_loader so it reads from a configurable PREPARED_DIR set by the
# setup cell instead of a hard-coded path relative to __file__.
loader_src = loader_src.replace(
    "# Resolve the prepared_data directory relative to this file so that the\n"
    "# package works no matter from where the notebook / script is launched.\n"
    "_THIS_DIR = Path(__file__).resolve().parent\n"
    'DATA_DIR = (_THIS_DIR.parent / "classification_problems" / "prepared_data").resolve()',
    "# PREPARED_DIR is set by the §0 setup cell.  Use a private name to avoid\n"
    "# colliding with the auto_rubric DATA_DIR inlined in §0.3.",
)
# Re-bind the loader's internal reference from DATA_DIR -> PREPARED_DIR.
loader_src = loader_src.replace("path = DATA_DIR / f", "path = PREPARED_DIR / f")

# Patch auto_rubric so it (a) reads CSVs from KINECT_CSV_DIR set in §0, (b)
# writes artefacts to LABELS_DIR set in §0, and (c) exposes a callable
# `run_auto_rubric()` instead of relying on `if __name__ == "__main__"`.
rubric_src = rubric_src.replace(
    'DATA_DIR = Path(__file__).resolve().parent.parent / "A13" / "kinect_good_vs_bad_not_preprocessed"\n'
    'OUT_DIR  = Path(__file__).resolve().parent / "labels"\n'
    "OUT_DIR.mkdir(exist_ok=True)",
    "# DATA_DIR and OUT_DIR are set from §0 (KINECT_CSV_DIR, LABELS_DIR).\n"
    "DATA_DIR = KINECT_CSV_DIR\n"
    "OUT_DIR = LABELS_DIR\n"
    "OUT_DIR.mkdir(parents=True, exist_ok=True)",
)
rubric_src = rubric_src.replace(
    'if __name__ == "__main__":\n    main()',
    "def run_auto_rubric():\n    main()",
)

# ----------------------------------------------------------------------------
# Build v3 from v2: keep markdown verbatim, surgically rewrite the small set
# of cells that reach into A13.* / A14.* modules, and prepend the inlining.
# ----------------------------------------------------------------------------
nb = json.loads(V2.read_text())
cells = nb["cells"]

# --- 1. Replace the §0 setup cell -------------------------------------------
new_setup = '''from __future__ import annotations
import json, os, sys, time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration.  v3 is standalone: it does not import from A13.* or A14.*.
# Point these at any directory layout; defaults match the repo for
# convenience when running from inside <repo>/A14/.
# ---------------------------------------------------------------------------
NB_DIR = Path.cwd()
DEFAULT_REPO = NB_DIR if (NB_DIR / "A13").exists() else NB_DIR.parent

KINECT_CSV_DIR = Path(os.environ.get("A14_KINECT_CSV_DIR",
    DEFAULT_REPO / "A13" / "kinect_good_vs_bad_not_preprocessed"))
PREPARED_DIR   = Path(os.environ.get("A14_PREPARED_DIR",
    DEFAULT_REPO / "A13" / "classification_problems" / "prepared_data"))
SAVED_DIR      = Path(os.environ.get("A14_SAVED_DIR",
    DEFAULT_REPO / "A13" / "dl_models" / "saved"))
SWEEP_DIR      = Path(os.environ.get("A14_SWEEP_DIR",
    DEFAULT_REPO / "A13" / "dl_models" / "sweep_results"))
LABELS_DIR     = Path(os.environ.get("A14_LABELS_DIR",
    DEFAULT_REPO / "A14" / "labels"))
MEDIAPIPE_TASK = Path(os.environ.get("A14_MEDIAPIPE_TASK",
    DEFAULT_REPO / "A14" / "pose_landmarker_lite.task"))

# Optional sample video for the MediaPipe + ExercisePipeline demos.
SAMPLE_VIDEO_PATH: str | None = None

for label, p in [("kinect csvs", KINECT_CSV_DIR), ("prepared npy", PREPARED_DIR),
                 ("saved models", SAVED_DIR), ("sweep results", SWEEP_DIR),
                 ("rubric labels", LABELS_DIR)]:
    print(f"  {label:<14} {p}  {'OK' if p.exists() else 'MISSING'}")
'''
# locate the original setup cell (it sets REPO_ROOT) and replace it
for i, c in enumerate(cells):
    if c["cell_type"] == "code" and any("REPO_ROOT" in line for line in c["source"]):
        cells[i] = code(new_setup)
        break

# --- 2. Insert the three inlined-module cells right after §0 setup ----------
def make_inline_cells():
    return [
        md("### 0.1 Inlined: model factories (was `A13/dl_models/models.py`)"),
        code(models_src),
        md("### 0.2 Inlined: dataset loader (was `A13/dl_models/data_loader.py`)"),
        code(loader_src),
        md("### 0.3 Inlined: auto-rubric (was `A14/auto_rubric.py`)\n\n"
           "Call `run_auto_rubric()` to regenerate the four artefacts in `LABELS_DIR`."),
        code(rubric_src),
    ]

# insert after the setup code cell we just wrote
for i, c in enumerate(cells):
    if c["cell_type"] == "code" and any("KINECT_CSV_DIR" in line for line in c["source"]):
        cells[i+1:i+1] = make_inline_cells()
        break

# --- 3. Rewrite §2.4 "Rebuild champions" cell to use inlined names ---------
new_24 = '''import tensorflow as tf
# `build_dense`, `build_cnn`, `assert_param_budget`, `count_params` come from §0.1.
# `load_dataset` comes from §0.2.
champion_models: dict[str, dict[str, tf.keras.Model]] = {}
for P in ('A', 'B'):
    ds_d = load_dataset(P, 'Dense')
    ds_c = load_dataset(P, 'CNN')
    cd = sweep['champions'][P]['Dense']['config']
    cc = sweep['champions'][P]['CNN']['config']
    dense = build_dense(
        input_dim=ds_d.X_train_aug.shape[1],
        hidden_units=tuple(cd['hidden_units']),
        dropout=cd['dropout'], learning_rate=cd['lr'],
        name=f'{P}_dense_champion',
    )
    cnn = build_cnn(
        input_shape=tuple(ds_c.X_train_aug.shape[1:]),
        filters=tuple(cc['filters']),
        kernel_size=tuple(cc['kernel_size']),
        dense_units=cc['dense_units'],
        dropout=cc['dropout'], learning_rate=cc['lr'],
        name=f'{P}_cnn_champion',
    )
    assert_param_budget(dense, cnn, ratio=0.20)
    champion_models[P] = {'Dense': dense, 'CNN': cnn}
    print(f"Problem {P}: Dense={count_params(dense)}  CNN={count_params(cnn)}  "
          f"ratio={count_params(cnn)/count_params(dense)*100:.1f}%  (budget OK)")
'''
for i, c in enumerate(cells):
    if c["cell_type"] == "code" and any("from A13.dl_models import models as M" in line for line in c["source"]):
        cells[i] = code(new_24)
        break

# --- 3b. §3.2 FLOP estimate uses M.count_params indirectly? actually no, it
# just uses estimate_flops + champion_models.  But §3.1 uses M.count_params.
new_31 = '''# Lightweight latency micro-benchmark on champion CNN-A: single-clip forward pass.
import numpy as np
cnn_a = champion_models['A']['CNN']
x = np.zeros((1, *cnn_a.input_shape[1:]), dtype='float32')
# Warm up.
for _ in range(5):
    _ = cnn_a(x, training=False)
N = 50
t0 = time.perf_counter()
for _ in range(N):
    _ = cnn_a(x, training=False)
dt = (time.perf_counter() - t0) / N
print(f'Champion A CNN: {count_params(cnn_a)} params, '
      f'mean single-clip forward pass = {dt*1000:.2f} ms (untrained weights, CPU)')
'''
for i, c in enumerate(cells):
    if c["cell_type"] == "code" and any("Champion A CNN: {M.count_params" in line for line in c["source"]):
        cells[i] = code(new_31)
        break

# --- 4. Replace the JSON loaders (§2.1, §2.3) to use SWEEP_DIR / SAVED_DIR --
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if "REPO_ROOT / 'A13/dl_models/sweep_results/latest.json'" in s:
        c["source"] = s.replace(
            "REPO_ROOT / 'A13/dl_models/sweep_results/latest.json'",
            "SWEEP_DIR / 'latest.json'"
        ).splitlines(keepends=True)
    if "REPO_ROOT / 'A13/dl_models/saved/training_summary.json'" in "".join(c["source"]):
        c["source"] = "".join(c["source"]).replace(
            "REPO_ROOT / 'A13/dl_models/saved/training_summary.json'",
            "SAVED_DIR / 'training_summary.json'"
        ).splitlines(keepends=True)
    if "REPO_ROOT / 'A13/dl_models/saved/A_CNN.keras'" in "".join(c["source"]):
        c["source"] = "".join(c["source"]).replace(
            "REPO_ROOT / 'A13/dl_models/saved/A_CNN.keras'",
            "SAVED_DIR / 'A_CNN.keras'"
        ).splitlines(keepends=True)
    if "REPO_ROOT / 'A14/pose_landmarker_lite.task'" in "".join(c["source"]):
        c["source"] = "".join(c["source"]).replace(
            "REPO_ROOT / 'A14/pose_landmarker_lite.task'",
            "MEDIAPIPE_TASK"
        ).splitlines(keepends=True)

# --- 5. §5.3 smoke test currently does `ds_smoke = load_dataset(...)` and
# does NOT import from A13 (load_dataset is already in scope from §0.2).
# Just make sure there is no `from A13...` line lingering anywhere.
for c in cells:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    src = re.sub(r"^from A13\.[^\n]*\n", "", src, flags=re.MULTILINE)
    src = re.sub(r"^from A14\.mediapipe_pose_estimator[^\n]*\n",
                 "from A14.mediapipe_pose_estimator import MediaPipePoseEstimator, KEYPOINT_NAMES\n"
                 if False else "", src, flags=re.MULTILINE)
    c["source"] = src.splitlines(keepends=True)

# --- 6. Update the title cell and add a v3 preamble note --------------------
for c in cells:
    if c["cell_type"] == "markdown" and "".join(c["source"]).strip() == "# A14 — Report":
        c["source"] = ["# A14 — Report (v3, standalone)\n"]
        break

preamble = md(
    "*This is the standalone version of the v2 report.  All code from "
    "`A13/dl_models/models.py`, `A13/dl_models/data_loader.py`, and "
    "`A14/auto_rubric.py` is inlined below in §0.1–§0.3.  The notebook needs "
    "no `A13.*` / `A14.*` imports; only the data folders (CSV clips, prepared "
    "`.npy` arrays, saved `.keras` checkpoints, sweep / label JSON artefacts) "
    "have to exist at the paths configured in §0.*  All textual content, "
    "numbers, and code logic are identical to v2.*"
)
# place preamble right after the title
for i, c in enumerate(cells):
    if c["cell_type"] == "markdown" and "A14 — Report" in "".join(c["source"]):
        cells.insert(i+1, preamble)
        break

# --- 7. Add a §6.7 cell that actually executes the inlined rubric ----------
rubric_run = md(
    "### 6.7 Reproduce the rubric artefacts\n\n"
    "The cell below calls the inlined `run_auto_rubric()` (§0.3) to regenerate "
    "`rubric_features.csv`, `rubric_labels.csv`, `thresholds.json`, and "
    "`disagreements.csv` in `LABELS_DIR`.  Same code path as the standalone "
    "`A14/auto_rubric.py` script."
)
rubric_call = code("run_auto_rubric()")
cells.extend([rubric_run, rubric_call])

V3.write_text(json.dumps(nb, indent=1))
print(f"wrote {V3}  ({len(cells)} cells)")

"""Apply the §6.2 labelling rubric algorithmically to every clip and compare
against the filename labels (G* / A1 = GOOD, W* = BAD).

Two passes:

  Pass 1 (descriptive). Compute the five rubric features on the *active segment*
  of each clip (frames where the working wrist is materially elevated). Dump the
  per-clip feature table to A14/labels/rubric_features.csv. No labelling yet.

  Pass 2 (calibrated rubric). Set each rule's threshold from the empirical 90th
  percentile of the feature on filename-GOOD clips, so a well-calibrated rubric
  passes ~90 % of GOOD clips by construction. Apply the rubric to all clips.
  Compute Cohen's kappa vs filename labels. The BAD-clip pass rate is then the
  honest measure of how well mechanical rules recover the original judgement.

Outputs:
  A14/labels/rubric_features.csv     (per-clip features, no label)
  A14/labels/rubric_labels.csv       (per-clip features + calibrated label)
  A14/labels/disagreements.csv       (rubric vs filename disagreements)
  A14/labels/thresholds.json         (calibrated thresholds + provenance)

Run:
    .venv/bin/python A14/auto_rubric.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ---- configuration ---------------------------------------------------------
FPS              = 30
REST_FRAMES      = 10            # baseline window for rest pose
ACTIVE_FRACTION  = 0.30          # frames with hand_y > rest + 30 % of peak rise
CALIB_PERCENTILE = 90            # threshold = 90th percentile on GOOD clips

DATA_DIR = Path(__file__).resolve().parent.parent / "A13" / "kinect_good_vs_bad_not_preprocessed"
OUT_DIR  = Path(__file__).resolve().parent / "labels"
OUT_DIR.mkdir(exist_ok=True)


def filename_label(name: str) -> str:
    p = name.split('.')[0]
    if p.startswith('W'): return 'BAD'
    if p.startswith('G') or p == 'A1': return 'GOOD'
    return 'UNKNOWN'


def load_clip(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def features_one_clip(df: pd.DataFrame) -> dict | None:
    """Compute the five rubric features on the active segment."""
    lh_rise = df["left_hand_y"].max()  - df["left_hand_y"].iloc[:REST_FRAMES].mean()
    rh_rise = df["right_hand_y"].max() - df["right_hand_y"].iloc[:REST_FRAMES].mean()
    side    = "left" if lh_rise >= rh_rise else "right"
    other   = "right" if side == "left" else "left"

    def J(name: str) -> np.ndarray:
        return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy()

    sh, hd  = J(f"{side}_shoulder"), J(f"{side}_hand")
    contra_y = df[f"{other}_shoulder_y"].to_numpy()
    hip_c    = 0.5 * (J("left_hip") + J("right_hip"))

    rest_y = sh[:REST_FRAMES, 1].mean()
    rise   = float(hd[:, 1].max() - rest_y)
    if rise < 0.05:
        return None
    active = hd[:, 1] - rest_y > ACTIVE_FRACTION * rise
    if active.sum() < 5:
        return None

    peak_idx     = int(np.argmax(hd[:, 1]))
    peak_margin  = float(hd[peak_idx, 1] - sh[peak_idx, 1])
    lateral_max  = float(np.abs(hd[active, 0] - sh[active, 0]).max())
    contra_rise  = float(contra_y.max() - contra_y[:REST_FRAMES].mean())
    hip_x_disp   = float(hip_c[active, 0].max() - hip_c[active, 0].min())
    hip_z_disp   = float(hip_c[active, 2].max() - hip_c[active, 2].min())
    trunk_disp   = max(hip_x_disp, hip_z_disp)

    thresh = rest_y + 0.1 * rise
    above  = np.where(hd[:, 1] > thresh)[0]
    lift_s = max(1, peak_idx - int(above[0])) / FPS if len(above) else 0.0

    return {
        "working_side":  side,
        "peak_margin":   round(peak_margin, 3),
        "lateral_max":   round(lateral_max, 3),
        "contra_rise":   round(contra_rise, 3),
        "lift_seconds":  round(lift_s, 2),
        "trunk_disp":    round(trunk_disp, 3),
    }


def cohens_kappa(y1: list[str], y2: list[str]) -> float:
    n  = len(y1)
    po = sum(a == b for a, b in zip(y1, y2)) / n
    labels = sorted(set(y1) | set(y2))
    pe = sum((y1.count(l) / n) * (y2.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    rows = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        feats = features_one_clip(load_clip(f))
        if feats is None:
            continue
        feats["clip"]           = f.stem
        feats["filename_label"] = filename_label(f.name)
        rows.append(feats)

    feat = pd.DataFrame(rows)
    feat = feat[["clip", "filename_label", "working_side",
                 "peak_margin", "lateral_max", "contra_rise",
                 "lift_seconds", "trunk_disp"]]
    feat.to_csv(OUT_DIR / "rubric_features.csv", index=False)
    print(f"Features computed for {len(feat)} clips "
          f"({(feat.filename_label=='GOOD').sum()} GOOD / "
          f"{(feat.filename_label=='BAD').sum()} BAD).\n")

    # Calibrate thresholds from the GOOD distribution
    G  = feat[feat.filename_label == "GOOD"]
    p  = CALIB_PERCENTILE
    th = {
        "peak_margin_min":   float(np.percentile(G.peak_margin,   100 - p)),
        "lateral_max_max":   float(np.percentile(G.lateral_max,    p)),
        "contra_rise_max":   float(np.percentile(G.contra_rise,    p)),
        "lift_seconds_min":  float(np.percentile(G.lift_seconds, 100 - p)),
        "lift_seconds_max":  float(np.percentile(G.lift_seconds,   p)),
        "trunk_disp_max":    float(np.percentile(G.trunk_disp,     p)),
        "calibration":       f"{p}th percentile on filename-GOOD clips (n={len(G)})",
    }
    (OUT_DIR / "thresholds.json").write_text(json.dumps(th, indent=2))

    print(f"Calibrated thresholds ({p}th percentile of GOOD):")
    for k, v in th.items():
        if isinstance(v, float):
            print(f"  {k:<22s} {v:.3f}")
    print()

    def classify(row: pd.Series) -> tuple[str, dict[str, bool]]:
        rules = {
            "r1_plane":    row.lateral_max  <= th["lateral_max_max"],
            "r2_range":    row.peak_margin  >= th["peak_margin_min"],
            "r3_symmetry": row.contra_rise  <= th["contra_rise_max"],
            "r4_tempo":    th["lift_seconds_min"] <= row.lift_seconds <= th["lift_seconds_max"],
            "r5_trunk":    row.trunk_disp   <= th["trunk_disp_max"],
        }
        return ("GOOD" if all(rules.values()) else "BAD"), rules

    labels = []
    rule_cols: dict[str, list[bool]] = {k: [] for k in
        ("r1_plane","r2_range","r3_symmetry","r4_tempo","r5_trunk")}
    for _, row in feat.iterrows():
        lab, rules = classify(row)
        labels.append(lab)
        for k, v in rules.items():
            rule_cols[k].append(v)

    out = feat.copy()
    out["rubric_label"] = labels
    for k, vs in rule_cols.items():
        out[k] = vs
    out.to_csv(OUT_DIR / "rubric_labels.csv", index=False)

    known = out[out.filename_label != "UNKNOWN"]
    kappa = cohens_kappa(known.filename_label.tolist(), known.rubric_label.tolist())
    cm = pd.crosstab(known.filename_label, known.rubric_label, dropna=False) \
           .reindex(index=["GOOD","BAD"], columns=["GOOD","BAD"], fill_value=0)
    agree = int(cm.loc["GOOD","GOOD"] + cm.loc["BAD","BAD"])

    print("Confusion matrix (rows = filename, cols = rubric):")
    print(cm.to_string()); print()
    print(f"Agreement      : {agree}/{len(known)} = {agree/len(known):.1%}")
    print(f"Cohen's kappa  : {kappa:.3f}")

    print("\nPer-rule failure counts:")
    n_g = int((out.filename_label == "GOOD").sum())
    n_b = int((out.filename_label == "BAD").sum())
    for r in ("r1_plane","r2_range","r3_symmetry","r4_tempo","r5_trunk"):
        fails_g = int((~out[r] & (out.filename_label == "GOOD")).sum())
        fails_b = int((~out[r] & (out.filename_label == "BAD")).sum())
        print(f"  {r}: GOOD-fails={fails_g:>3d}/{n_g}   BAD-fails={fails_b:>3d}/{n_b}")

    dis = known[known.filename_label != known.rubric_label].copy()
    dis.to_csv(OUT_DIR / "disagreements.csv", index=False)
    print(f"\n{len(dis)} disagreements -> {OUT_DIR/'disagreements.csv'}")
    print(f"Per-clip table       -> {OUT_DIR/'rubric_labels.csv'}")
    print(f"Calibrated thresholds-> {OUT_DIR/'thresholds.json'}")


if __name__ == "__main__":
    main()

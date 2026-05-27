"""Batch evaluator for the A16 endpoint over locally-stored labeled clips.

Walks every video in ``A16/all_videos`` (gitignored), looks up the ground-truth
label from the A15 lists, runs the live ``run_pipeline_3d`` endpoint on each
clip, and prints a live per-clip report plus an aggregate confusion matrix at
the end. Designed to surface real misbehaviours fast without manual webcam
testing.

Usage::

    python -m A16.tests.batch_eval                 # all clips
    python -m A16.tests.batch_eval --limit 20      # first 20 clips
    python -m A16.tests.batch_eval --pattern A1    # only clips matching glob A1*
    python -m A16.tests.batch_eval --csv out.csv   # also dump per-clip results

Ground truth sources:
- ``A15_Data/a15_good_list.csv``  → GOOD clips (with reference score)
- ``A15_Data/a15_ugly_list.csv``  → UGLY clips
- ``A15_Data/scores.csv``         → fallback reference score for any clip

Clip filename ``A123.avi`` maps to ground-truth key ``A123_kinect``.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
VIDEOS_DIR = ROOT / "A16" / "all_videos"
A15_DIR = ROOT / "A15_Data"


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def _load_label_csv(path: Path, label: str) -> Dict[str, Dict[str, Any]]:
    """Read a15_good_list.csv / a15_ugly_list.csv into ``{clip_key: row}``."""
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("clip", "").strip()
            if not key:
                continue
            out[key] = {
                "label": label,
                "ref_score": float(row["score"]) if row.get("score") else None,
                "ref_good_prob": (
                    float(row["good_probability"])
                    if row.get("good_probability") else None
                ),
            }
    return out


def _load_scores_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)  # noqa: F841 — header skipped
        for row in reader:
            if len(row) < 2:
                continue
            try:
                out[row[0].strip()] = float(row[1])
            except ValueError:
                continue
    return out


def load_ground_truth() -> Dict[str, Dict[str, Any]]:
    gt = _load_label_csv(A15_DIR / "a15_good_list.csv", "GOOD")
    gt.update(_load_label_csv(A15_DIR / "a15_ugly_list.csv", "UGLY"))
    score_lookup = _load_scores_csv(A15_DIR / "scores.csv")
    for key, ref_score in score_lookup.items():
        gt.setdefault(key, {"label": None, "ref_score": ref_score,
                            "ref_good_prob": None})
        gt[key].setdefault("ref_score", ref_score)
    return gt


def _video_key(video_path: Path) -> str:
    """``A123.avi`` → ``A123_kinect`` (the key used by the A15 lists)."""
    return f"{video_path.stem}_kinect"


# ---------------------------------------------------------------------------
# Per-clip evaluation
# ---------------------------------------------------------------------------

def evaluate_clip(
    video_path: Path,
    threshold: float,
) -> Dict[str, Any]:
    """Run ``run_pipeline_3d`` on a single clip and return a flat result row."""
    # Local import so the script can be imported without TF/MediaPipe at
    # module-collection time (e.g. by pytest).
    from A16.service.endpoint import run_pipeline_3d

    t0 = time.monotonic()
    err: Optional[str] = None
    resp: Dict[str, Any] = {}
    try:
        resp = run_pipeline_3d(str(video_path), quality_threshold=threshold)
    except Exception as e:  # pragma: no cover — surfacing is the point
        err = f"{type(e).__name__}: {e}"
        resp = {"status": "EXCEPTION", "message": err}

    wall_ms = (time.monotonic() - t0) * 1000.0

    rec = resp.get("recording", {}) or {}
    cls = resp.get("classification", {}) or {}
    sc = resp.get("score", {}) or {}
    seg = resp.get("segment", {}) or {}

    return {
        "clip": video_path.stem,
        "status": resp.get("status"),
        "message": resp.get("message", ""),
        "rec_label": rec.get("quality_label"),
        "rec_conf": rec.get("quality_confidence"),
        "class_label": cls.get("label"),
        "class_conf": cls.get("confidence"),
        "score": sc.get("value"),
        "band": sc.get("band"),
        "start": seg.get("start_frame"),
        "stop": seg.get("stop_frame"),
        "wall_ms": round(wall_ms, 1),
        "error": err,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_cell(v: Any, n: int = 6) -> str:
    if v is None:
        return "-".rjust(n)
    if isinstance(v, float):
        return f"{v:.3f}".rjust(n)
    return str(v).rjust(n)


def print_row(idx: int, total: int, row: Dict[str, Any],
              gt: Dict[str, Any]) -> None:
    truth = gt.get("label") or "?"
    truth_score = gt.get("ref_score")
    ok_marker = " "
    if row["status"] == "EXCEPTION":
        ok_marker = "X"
    elif truth == "UGLY" and row["rec_label"] == "UGLY":
        ok_marker = "."
    elif truth == "UGLY" and row["rec_label"] != "UGLY":
        ok_marker = "!"  # false-positive-good (let an UGLY clip through)
    elif truth == "GOOD" and row["rec_label"] == "UGLY":
        ok_marker = "!"  # false-positive-ugly (rejected a GOOD clip)
    elif truth == "GOOD":
        ok_marker = "."

    truth_s = f"truth={truth}"
    if truth_score is not None:
        truth_s += f"(ref {truth_score:.2f})"

    print(
        f"[{idx:>3}/{total}] {ok_marker} {row['clip']:<6} "
        f"{truth_s:<22} status={row['status']:<22} "
        f"rec={row['rec_label']}/{_fmt_cell(row['rec_conf'])} "
        f"cls={row['class_label']}/{_fmt_cell(row['class_conf'])} "
        f"score={_fmt_cell(row['score'])} band={row['band']} "
        f"seg={row['start']}->{row['stop']} "
        f"({row['wall_ms']:.0f} ms)"
    )
    if row["error"]:
        print(f"        ERROR: {row['error']}")


def summarise(rows: List[Dict[str, Any]],
              gt: Dict[str, Dict[str, Any]]) -> None:
    n = len(rows)
    exceptions = [r for r in rows if r["status"] == "EXCEPTION"]
    rejected = [r for r in rows if r["rec_label"] == "UGLY"]
    ok = [r for r in rows if r["status"] == "OK"]

    # Confusion vs ground-truth (only clips we have a truth label for).
    tp = fp = tn = fn = 0
    unknown = 0
    for r in rows:
        truth = (gt.get(_video_key(VIDEOS_DIR / f"{r['clip']}.avi"), {}) or {}).get("label")
        if truth is None:
            unknown += 1
            continue
        pred_ugly = r["rec_label"] == "UGLY"
        if truth == "UGLY" and pred_ugly:
            tp += 1
        elif truth == "UGLY" and not pred_ugly:
            fn += 1
        elif truth == "GOOD" and pred_ugly:
            fp += 1
        elif truth == "GOOD" and not pred_ugly:
            tn += 1

    wall = [r["wall_ms"] for r in rows if r["wall_ms"] is not None]
    wall_total_s = sum(wall) / 1000.0 if wall else 0.0
    wall_avg_s = (sum(wall) / len(wall) / 1000.0) if wall else 0.0

    print("")
    print("=" * 70)
    print(f"Processed {n} clips in {wall_total_s:.1f}s "
          f"(avg {wall_avg_s:.2f}s/clip)")
    print(f"  OK              : {len(ok)}")
    print(f"  Rejected (UGLY) : {len(rejected)}")
    print(f"  Exceptions      : {len(exceptions)}")
    print(f"  No ground truth : {unknown}")
    print("")
    print("UGLY-gate confusion (truth UGLY = positive):")
    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    if tp + fn > 0:
        print(f"  Recall (catch-UGLY)   : {tp/(tp+fn):.2%}")
    if tn + fp > 0:
        print(f"  Specificity (pass-GOOD): {tn/(tn+fp):.2%}")

    if exceptions:
        print("")
        print(f"Exceptions ({len(exceptions)}):")
        for r in exceptions[:20]:
            print(f"  {r['clip']:<6}  {r['error']}")
        if len(exceptions) > 20:
            print(f"  ... and {len(exceptions) - 20} more")


def dump_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Per-clip CSV written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--videos-dir", default=str(VIDEOS_DIR),
                   help="Directory of .avi/.mp4 clips (default: A16/all_videos)")
    p.add_argument("--pattern", default="*",
                   help="Glob applied to filenames (e.g. 'A1*').")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N clips (0 = all).")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Recording-quality threshold (matches UI default).")
    p.add_argument("--csv", default="",
                   help="Optional path to dump per-clip results as CSV.")
    p.add_argument("--ext", default=".avi,.mp4,.mov,.webm",
                   help="Comma-separated extensions to include.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        print(f"ERROR: videos dir does not exist: {videos_dir}", file=sys.stderr)
        return 2

    allowed_exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}
    clips = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in allowed_exts
        and fnmatch.fnmatch(p.name, args.pattern)
        # Also accept patterns without extension, e.g. --pattern A1
        or (p.is_file() and p.suffix.lower() in allowed_exts
            and fnmatch.fnmatch(p.stem, args.pattern))
    )
    # De-dupe while preserving order
    seen = set()
    clips = [c for c in clips if not (c in seen or seen.add(c))]
    if args.limit > 0:
        clips = clips[: args.limit]

    if not clips:
        print(f"No clips matched in {videos_dir} (pattern={args.pattern!r}).")
        return 1

    gt_all = load_ground_truth()
    print(f"Loaded {len(gt_all)} ground-truth entries from A15_Data/")
    print(f"Evaluating {len(clips)} clip(s) from {videos_dir}\n")

    rows: List[Dict[str, Any]] = []
    for i, clip in enumerate(clips, 1):
        gt = gt_all.get(_video_key(clip), {"label": None, "ref_score": None})
        try:
            row = evaluate_clip(clip, threshold=args.threshold)
        except KeyboardInterrupt:
            print("\nInterrupted — summarising results so far...")
            break
        except Exception as e:  # pragma: no cover
            row = {
                "clip": clip.stem, "status": "EXCEPTION",
                "message": "outer-loop crash",
                "rec_label": None, "rec_conf": None,
                "class_label": None, "class_conf": None,
                "score": None, "band": None,
                "start": None, "stop": None,
                "wall_ms": 0.0,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            }
        rows.append(row)
        print_row(i, len(clips), row, gt)
        sys.stdout.flush()

    summarise(rows, gt_all)
    if args.csv:
        dump_csv(rows, Path(args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

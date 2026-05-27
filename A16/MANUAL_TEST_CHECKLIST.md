# A16 Final Endpoint — Manual Test Checklist

Use this list to verify the A16 tab end-to-end before the presentation.
Tick each box. If a step fails, capture what you saw and let the dev know
(do not hallucinate fixes).

**Scope:** UI behaviour, response shape, and integration. Model accuracy
is out of scope for this checklist (covered by A10/A11/A13/A15 reports).

---

## 0. Pre-flight

- [ ] `python -m pytest A4/ A16/ -v` passes locally. Expect **11 A16 tests + the A4 tests** all green.
- [ ] Model artefacts present in `models/`:
  - [ ] `week16_result.h5`
  - [ ] `week15_2d_to_3d.h5`
  - [ ] `week17_start_and_stop.h5`
  - [ ] `week17_start_and_stop.pkl`
  - [ ] `A_CNN.keras`
  - [ ] `scoring_model.keras`
  - [ ] `scoring_scaler.pkl`
  - [ ] `week16_scaler_X.pkl`, `week16_scaler_y.pkl`
- [ ] MediaPipe model present: `A14/pose_landmarker_lite.task`.
- [ ] `python app.py` launches without traceback. Console shows the existing
      `Initialising Exercise Pipeline` banner only **when** a tab triggers it
      (lazy load), **not** at startup.

---

## 1. Tab presence & layout

Open `http://127.0.0.1:7860` (local) or the HF Space URL (deployed).

- [ ] A new tab labelled **"A16 Final Endpoint"** is visible to the right
      of `Exercise Scoring (A15)`.
- [ ] The tab contains, top to bottom on each column:
  - **Left column:** intro markdown, a `Video` upload, a `Recording quality
    threshold` slider (default 0.6, range 0.1–0.9), a primary `Run A16
    endpoint` button.
  - **Right column:** `Status` textbox (read-only), a Markdown panel, a
    `3D skeleton animation` video output, a `Full response (A16 schema)`
    JSON viewer.

---

## 2. Happy path (good recording)

Use any good-quality exercise clip you have used for A14 / A15 demos.

- [ ] Upload the video, leave threshold at `0.6`, click **Run A16 endpoint**.
- [ ] During processing the console prints the existing
      `[1] Loading MediaPipe …` banner once.
- [ ] When it finishes:
  - [ ] **Status** textbox starts with `OK — score <number> (<BAND>...)`.
  - [ ] **Summary** Markdown shows non-`None` values for Recording, Segment,
        Classification, Score and Timing sections.
  - [ ] **3D skeleton animation** plays (it is the same `<stem>_skeleton.mp4`
        you'd get from A14).
  - [ ] **Full response JSON** has:
    - `endpoint == "A16"`, `variant == "3D"`, `schema_version == "1.0.0"`
    - `status == "OK"`
    - `score.value` is a number in `[0, 4]`
    - `score.band` matches the threshold: `<1` GREEN, `1–2` AMBER, `≥2` RED
    - `timing_ms.upstream_ms` > 0 and `timing_ms.total_ms` ≥ `upstream_ms`
    - `artefacts.cut_3d_csv` and `artefacts.full_3d_csv` are non-null paths
    - `warnings` is an empty list

> If the score is `null` but everything else looks fine, check the
> `status` field — `ERROR_SCORER` or `ERROR_TOO_SHORT_AFTER_CUT` is the
> expected failure mode and should already show a clear message.

---

## 3. Ugly recording path

Use a deliberately bad clip (occluded body, very dark, partial frame).
If you don't have one, raise the threshold to `0.9` on any normal clip.

- [ ] Click **Run A16 endpoint**.
- [ ] **Status** textbox starts with `REJECTED — ugly recording (conf <n>)`.
- [ ] **Full response JSON**:
  - `status == "REJECTED_UGLY_RECORDING"`
  - `recording.quality_label == "UGLY"`
  - `recording.quality_confidence` < `recording.threshold`
  - `score.value` is `null`
  - `classification.label` is `null`
  - `segment.start_frame` is `null`
  - `timing_ms.scorer_nn_ms == 0.0`

---

## 4. No-video path

- [ ] Click **Run A16 endpoint** without uploading anything.
- [ ] **Status** shows `ERROR_NO_VIDEO — No video provided.`
- [ ] **Full response JSON** `status == "ERROR_NO_VIDEO"`, all sections present but null.
- [ ] **No traceback** appears in the console.

---

## 5. Existing tabs still work (regression check)

Quickly verify nothing regressed in the other tabs:

- [ ] `📸 Image Processing` — still renders an annotated image.
- [ ] `🎥 Video Processing` — still renders an annotated video.
- [ ] `🧪 Video Pipeline` (A12) — still runs end-to-end.
- [ ] `Exercise Analysis (A14)` — still runs end-to-end.
- [ ] `Exercise Scoring (A15)` — still returns a score.

---

## 6. Deployment

After pushing to `main`:

- [ ] GitHub Actions run `Sync to Hugging Face hub` is green:
  - [ ] `Lint .py files` passes
  - [ ] `Lint notebooks` passes
  - [ ] `Run unit tests` passes (now runs `pytest A4/ A16/`)
  - [ ] `Push to hub` succeeds
- [ ] The HF Space rebuilds and the **A16 Final Endpoint** tab appears live.

---

## 7. What to capture for the presentation

- [ ] Screenshot of the A16 tab on a happy-path run (status + summary + JSON).
- [ ] Screenshot of the ugly rejection path (status + JSON).
- [ ] Mermaid architecture diagram from `A16_Report.ipynb` §1.
- [ ] One-line timing numbers from the happy-path JSON (`upstream_ms`,
      `scorer_nn_ms`, `total_ms`).

---

## 8. If something looks wrong

Do **not** guess. Note exactly:

1. Which step failed (section + checkbox text).
2. What the `status` and `message` fields said.
3. The first 5 lines of the console output around the failure.
4. Whether the same clip works in the A14 or A15 tab.

Then ping the dev with that info — most failures map directly to a
known model-loading issue documented in `A16_Report.ipynb` §8.

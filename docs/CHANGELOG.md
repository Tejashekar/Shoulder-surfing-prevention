# Changelog — v2.1 (2026-11-15)

## Bugs fixed

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | `detection/gaze_estimation.py` | `FaceMesh()` defaulted to `max_num_faces=1`, so the observer branch (`len(faces) >= 2`) was unreachable. `gaze_score` was **0.0 on all 10,722 logged frames**, including all 1,742 multi-face frames. The 0.32 gaze weight never contributed. | `max_num_faces=4`; raw measurement and mesh face count are now logged (`gaze_raw`, `mesh_face_count`). |
| 2 | `detection/proximity.py`, `gaze_estimation.py` | Observer selected by detector index (`faces[1]`, "skip index 0"). MediaPipe ordering is not stable; user and observer could swap. | User = largest face box; all other faces are observers (`split_user_and_observers`). |
| 3 | `main.py` | Logged `dwell_score = duration/5*20` uncapped (max seen 79.6) while the engine capped at 20. | Logs `risk_engine.last_dwell_score` (capped) plus `dwell_seconds`. |
| 4 | `pii/patterns.py` | `account` regex could not match `Account number: 123456789012` (colon **and** space). 0/4 recall in the pilot. | Flexible separator/whitespace. |
| 5 | `pii/patterns.py` | `email` failed 0/8 because Tesseract splits `demo.user@example.` and `invalid` into two tokens. | Allow one space after a domain dot. |
| 6 | `pii/patterns.py` | `phone` matched 7-digit fragments: 237 of 243 live-screen phone hits were timestamps/IDs/coordinates. `aadhaar` matched any 12 digits (duplicate of account/card). | Phone requires a 10-digit national number; Aadhaar requires 4-4-4 grouping; stronger categories suppress weaker ones on the same span. |
| 7 | `pii/ocr_engine.py` vs `experiments/evaluate_pii_manifest.py` | Live path binarised with a fixed threshold of 150; the evaluator fed raw images to Tesseract. Reported PII metrics did not describe the deployed pipeline. | Single `preprocess_for_ocr()` (adaptive threshold, dark-mode polarity inversion, 1.5x upscale for < 1600 px) used by both. Boxes are rescaled back to source pixels. |
| 8 | `overlay/overlay_engine.py` | Overlay latency column mixed real renders (~165 ms) with throttle early-returns (0.04 ms); no way to tell them apart. | `apply_selective_redaction` returns `RenderResult(rendered, reason, capture_ms, render_ms, total_ms)`; runtime log has `overlay_rendered`, `overlay_skip_reason`, `overlay_capture_ms`, `overlay_render_ms`. |
| 9 | `evaluation/analyze_results.py` | PII metrics pooled unlabeled live-screen rows with synthetic rows → "precision 0.05, 255 FP" artefact in `runtime_summary.csv`. | Live rows excluded; IoU-matched `pii_summary.csv` is the reportable source. Latency stats use rendered events only. |
| 10 | `evaluation/sensitivity_analysis.py` | Hard-coded crowd=18 and three weight sets; ignored `risk.json`. | Replays logged components through `RiskEngine.score_from_components` under 105 candidate configs; reads config file. |
| 11 | `experiments/evaluate_pii_manifest.py` | Kept only the first region per category per image. | One row per detected region; negatives per undetected category. |
| 12 | `pii/pii_detector.py` | Token-offset search by `text.find()` could mis-map repeated tokens. | Precomputed token spans. Keyword-anchored categories (password, account) box only the value (`(?P<value>...)`). |
| 13 | `pii/ocr_engine.py`, `overlay_engine.py` | `import pyautogui` at module import time → tests needed a display. | Lazy imports. |

## New

- `experiments/run_trials.py` — counterbalanced, seeded trial plans for the distance / gaze / robustness / ablation blocks; prompts per trial and launches `main.py` with the correct metadata and `--observer-present` label.
- `experiments/offline_ablation.py` — replays labelled runtime frames under full / no-gaze / no-proximity / no-dwell / no-pii / no-crowd / observer-only / content-only configurations; emits `results/ablation_offline.csv`.
- `experiments/generate_synthetic_screens.py --set extended` — 60-image seeded held-out set: 8 PII categories (incl. PAN, card), 4 themes (light/dark/low-contrast/paper), 3 font families, 14–34 px, noise, and 20 distractor strings (dates, IPs, versions, order IDs) as negatives. Ground-truth boxes cover the *value* span (what must be redacted).
- `main.py`: `--observer-present`, `--duration`, `--ocr-every`, `--no-crowd`, `--headless`; per-component contributions logged (`contrib_*`); user/observer boxes drawn in different colours.
- `experiment_logging.CsvLogger` migrates an older header in place (new columns appended; old rows padded).
- `evaluation/analyze_results.py --experiment/--trial`; per-experiment blocks; trial-level observer metrics from labelled runtime frames (`results/observer_trial_summary.csv`).

## Synthetic PII results after the fixes (container, Tesseract 5.3.4, IoU >= 0.50)

| Set | Images | Annotations | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| Starter (pilot paper set) — pilot build | 14 | 14 | 7 | 5 | 7 | 0.58 | 0.50 | 0.54 |
| Starter — v2.1 | 14 | 14 | 13 | 0 | 1 | 1.00 | 0.93 | 0.96 |
| Extended, seed 2026 (dev) — v2.1 | 60 | 99 | 90 | 3 | 9 | 0.97 | 0.91 | 0.94 |
| Extended, seed 2027 (held-out) — v2.1 | 60 | 91 | 76 | 3 | 15 | 0.96 | 0.84 | 0.89 |

The one starter miss is the deliberately truncated `partial-text` phone number. Re-run on the target Windows machine (Tesseract 5.5) before reporting; numbers above are from the development container.

## Still required (cannot be done without a camera and a second person)

Distance / gaze-dwell / robustness / ablation live trials. See the [experiment protocol](EXPERIMENT_PROTOCOL.md).

# Experiment protocol

Status: protocol and collection tools are implemented. `results/` is generated locally and is not version-controlled; never treat its contents as findings until they come from documented trials.

## Record the test environment

At the start of each session, record these values in a dated lab notebook or session metadata sheet: computer make/model; CPU model/core count; installed RAM; GPU/model/driver if used; Windows build; Python version; installed package versions (`python -m pip freeze`); webcam make/model, selected camera index, capture resolution/FPS; display make/model/resolution/scaling; Tesseract version/path; room and screen lighting description; camera position; and risk config file contents/hash. The runtime CSV logs actual observed camera frame dimensions, screen dimensions, per-frame processing time, process CPU%, memory, and measured OCR call duration.

Install the project requirements and Tesseract before any live run. Hardware values are deliberately not guessed here.

## Runtime logging

From `Version 2/`, use `python main.py --experiment-id EXP-YYYYMMDD-01 --trial-id T01 --condition baseline --lighting normal --distance 1.0 --gaze-condition toward --gaze-duration 2 --occlusion none --camera-position centered`. `--distance` must be measured physically; the system's face area is not a meter-valued distance sensor. Runtime records are appended to `results/runtime_logs.csv`; detected categories and boxes go to `results/pii_detection.csv`. The session runs until Escape. Webcam frames and screenshot pixels are not written by this path.

FPS is derived from start-to-start frame-loop intervals, so it includes the preceding loop's display wait and logging overhead. CPU is `psutil` process CPU percent and may exceed 100 on multi-core Windows systems; memory is process resident set size. `processing_time_ms` covers camera read, detection, risk calculation, due OCR, drawing, and overlay call, ending before the separate CPU/memory/screen-size sample and CSV write. OCR runs on the first frame and every 10 frames afterward even when no PII is found. Overlay latency is a client-side call interval from immediately before screen capture/redaction through OpenCV's `imshow`/`waitKey(1)` return; the paired `t_trigger`/`t_overlay_visible` fields are wall-clock ISO timestamps and the latency is computed from a monotonic timer. This is not a photodiode/display scanout measurement and needs independent validation before being described as physical pixel-visible latency.

## Observer distance experiment

Use measured marks at 0.5, 1.0, 1.5, and 2.0 m from the camera. Randomize or counterbalance distance order. At each distance run repeated observer-present trials and observer-absent control trials of equal duration (choose and report duration and trial count before collection; do not stop early based on outcomes). Keep camera, chair, lighting, user, and screen layout fixed. Record ground truth only from the scripted trial setup. Run live system with the matching `--distance` and `--trial-id`; manually inspect `observer_detected`, `threat_score`, and overlay time ranges in runtime logs. Then append one trial-level row using `python -m experiments.record_trial distance --distance 0.5 --observer-present true --predicted-observer-present true --threat-score 70 --latency-ms 123 --condition distance-0.5`. Replace example values only with observed values; commands shown here are syntax examples, not data. Include all positive and negative trials. Metrics are calculated only over rows with both labels.

The primary user is taken to be the largest face and every other face is an observer; this can fail if an observer is closer to the camera than the user. Proximity is face-box area. Treat the outcome as performance of this implementation under the specified camera/setup, not calibrated physical-distance accuracy.

## Gaze and dwell experiment

Use conditions `away` and `toward-screen`, with instructed gaze durations 0.5, 2, and 5 seconds. Measure durations with an external timer or scripted cue and record the actual condition; the command-line duration field is an annotation, not automatic gaze ground truth. Run repeated trials with matched observer distance and randomize condition order. Start `main.py` with explicit `--gaze-condition`, `--gaze-duration`, `--condition`, and unique trial ID. Compare the logged gaze proxy, dwell score, threat score, and `overlay_latency_ms` by condition. Do not report gaze classification accuracy: there is no independent eye-gaze label source or validated gaze classifier.

## Synthetic OCR and PII test

Create only generated test screens with fake values and no personal data. Run `python -m experiments.generate_synthetic_screens` to create starter samples across email, phone, password/passcode, ID-like string, account number, postal address, mixed categories, non-sensitive text, font size, contrast, font, partial text, and noise. Extend the set with additional layouts and PII examples before reporting robust results. The manifest includes `sample_id,image_path,ground_truth_pii_type,ground_truth_bbox,ground_truth_annotations,expected_synthetic_text,condition`. Type lists may be semicolon separated; boxes are JSON `[x,y,w,h]`. Expected text is synthetic; the evaluator uses it in memory only and writes a correctness flag, never text, to results.

Run `python -m experiments.evaluate_pii_manifest test_data\pii\manifest.csv`. Output rows compare category presence and retain box/confidence metadata, not recognized strings. `ocr_recognition_correct` is a normalized exact-substring/presence check, not character error rate; paper claims must describe that limitation. Then run `python -m experiments.evaluate_pii_annotations` for one-to-one, IoU-matched precision/recall/F1 (see `PII_EVALUATION.md`).

## Robustness

Use the full Cartesian condition grid: lighting bright/normal/dim; camera centered/left/right; face frontal/approximately 30°/approximately 45°; occlusion none/glasses/partial face; and distances 0.5/1.0/1.5/2.0 m. For every condition use positive and negative observer trials and a fixed, predeclared number/duration of repeats. Record actual setup in CLI flags and one row per trial with `python -m experiments.record_trial robustness ...`. Report missing or infeasible cells, do not infer them. The existing system has no lighting/orientation/occlusion classifier; these are test conditions, not measured component outputs.

## Ablation and weight sensitivity

Run matched trials with the default configuration, then configurations disabling exactly one of gaze, proximity, dwell, or PII. Use `--no-gaze`, `--no-proximity`, `--no-dwell`, or `--no-pii` for live runs; use `--risk-config` when testing a copied JSON config. Retain all other weights, thresholds, and trial conditions. `RiskEngine(config_path=..., features=...)` supports these switches. Record feature values and labels in the ablation CSV. Do not compare non-matched conditions.

Current heuristic values are gaze 0.32, proximity 0.28, PII 0.30, crowd +18, dwell up to +20 saturated at 5 seconds. No-action and selective-blur activation both start at 35; risk-level medium/high boundaries are 35/70; fake overlay starts at 75. Weights are not normalized and the score has a separate crowd term; these are implementation heuristics. Until enough labeled independent sessions exist, perform sensitivity analyses over documented plausible settings and report the chosen values as heuristic, not learned or statistically calibrated. After labeled runs, `python -m evaluation.sensitivity_analysis` evaluates candidate component weights and thresholds only where runtime and ground-truth trial IDs match. This is a sensitivity comparison, not training or automatic calibration. Preserve train/test separation by participant/session if later calibration is attempted.

## Runtime, latency, and usability

Run the camera loop continuously for a predeclared duration (recommend at least 10 minutes after warm-up) for each environment; include warm-up handling in the protocol and retain all logged samples. `evaluation/analyze_results.py` computes count, mean, median, population standard deviation, min, max, and nearest-rank 95th percentile for runtime fields and grouped classifier results where labels exist. `evaluation/generate_plots.py` plots only available measured CSV rows. Per-frame trigger/visible timestamps and GUI-call latency are logged for overlay activations; do not substitute a guessed value.

Usability tasks require a consented, approved human-subject protocol where applicable. Use participant codes only. Predefine tasks, success criteria, rating scale anchors, and overlay conditions; record each response using `python -m experiments.record_trial usability ...`. Do not put names or free-text identifiers in comments. Obtain institutional ethics approval/IRB determination before recruitment where required by the institution or jurisdiction.

## Reproduction commands

From this folder: `python -m pip install -r requirements.txt`; install Tesseract separately and optionally set `TESSERACT_CMD`; `python -m unittest discover -s tests -v`; `python -m experiments.generate_synthetic_screens`; collect live/manual trials; then `python -m evaluation.analyze_results`, `python -m evaluation.sensitivity_analysis`, and `python -m evaluation.generate_plots`. Record exact command, config, package versions, equipment, trial exclusions, and deviations. Do not fill output CSVs with demonstration rows.

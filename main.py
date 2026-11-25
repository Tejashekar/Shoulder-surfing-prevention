"""Live shoulder-surfing monitor with privacy-safe experiment logging.

v2.1 changes versus the pilot build:
- Observer = every face except the largest (primary user); no index assumption.
- FaceMesh now tracks up to 4 faces, so the gaze proxy actually fires.
- `dwell_score` logged is the capped value used in the score.
- Overlay telemetry carries `overlay_rendered`, `overlay_capture_ms` and
  `overlay_render_ms`; latency statistics must filter on `overlay_rendered`.
- Raw (pre-Kalman) gaze measurement and per-component contributions are logged
  so offline ablation can replay the score without a camera.
"""
import cv2
import os
import time
import uuid
import psutil
import argparse
import atexit

from detection.face_detection import FaceDetector
from detection.gaze_estimation import GazeEstimator
from detection.proximity import estimate_proximity_score, split_user_and_observers
from risk.risk_engine import RiskEngine

from pii.pii_detector import detect_pii
from overlay.overlay_engine import apply_blur
from experiment_logging import RuntimeLogger, CsvLogger, PII_FIELDS, utc_now

HERE = os.path.dirname(os.path.abspath(__file__))
cli = argparse.ArgumentParser(description="Local shoulder-surfing monitor with privacy-safe experiment logs")
cli.add_argument("--experiment-id", default=None)
cli.add_argument("--trial-id", default="")
cli.add_argument("--condition", default="")
cli.add_argument("--lighting", default="")
cli.add_argument("--distance", type=float, choices=(0.5, 1.0, 1.5, 2.0), default=None, help="Manually measured observer distance in metres")
cli.add_argument("--gaze-condition", default="")
cli.add_argument("--gaze-duration", type=float, default=None, help="Ground-truth gaze duration in seconds")
cli.add_argument("--occlusion", default="")
cli.add_argument("--camera-position", default="")
cli.add_argument("--observer-present", choices=("true", "false"), default="", help="Ground-truth label for this trial (researcher-entered)")
cli.add_argument("--duration", type=float, default=None, help="Stop automatically after N seconds")
cli.add_argument("--risk-config", default=os.path.join(HERE, "configs", "risk.json"))
cli.add_argument("--ocr-every", type=int, default=10, help="Run OCR every N frames")
cli.add_argument("--no-gaze", action="store_true")
cli.add_argument("--no-proximity", action="store_true")
cli.add_argument("--no-dwell", action="store_true")
cli.add_argument("--no-pii", action="store_true")
cli.add_argument("--no-crowd", action="store_true")
cli.add_argument("--headless", action="store_true", help="Do not show the webcam monitor window")
args = cli.parse_args()
SESSION_ID = uuid.uuid4().hex
EXPERIMENT_ID = args.experiment_id or SESSION_ID
runtime_log = RuntimeLogger(os.path.join(HERE, "results"), SESSION_ID)
pii_log = CsvLogger(os.path.join(HERE, "results", "pii_detection.csv"), PII_FIELDS)
process = psutil.Process(os.getpid())

detector = FaceDetector()
gaze = GazeEstimator()
risk_engine = RiskEngine(config_path=args.risk_config, features={
    "gaze": not args.no_gaze, "proximity": not args.no_proximity,
    "dwell": not args.no_dwell, "pii": not args.no_pii, "crowd": not args.no_crowd,
})
atexit.register(detector.release)
atexit.register(gaze.close)
atexit.register(cv2.destroyAllWindows)

frame_count = 0
last_pii_regions = []
last_ocr_time_ms = ""
last_ocr_executed = False
prev_frame_start = None
frame_id = 0
session_start = time.perf_counter()
screen_size = ""
try:
    import pyautogui
    sw, sh = pyautogui.size()
    screen_size = f"{sw}x{sh}"
except Exception:
    pass

while True:
    frame_started = time.perf_counter()
    if args.duration is not None and frame_started - session_start >= args.duration:
        break
    fps = (1.0 / (frame_started - prev_frame_start)) if prev_frame_start and frame_started > prev_frame_start else ""
    prev_frame_start = frame_started
    frame = detector.get_frame()
    if frame is None:
        break
    camera_resolution = f"{frame.shape[1]}x{frame.shape[0]}"

    frame = cv2.resize(frame, (640, 480))

    faces = detector.detect_faces(frame)
    face_count = len(faces)
    user_box, observer_boxes = split_user_and_observers(faces)
    frame_id += 1
    frame_count += 1

    gaze_score = gaze.estimate_intruder_score(frame)
    proximity_score = estimate_proximity_score(faces, frame.shape)

    if frame_count == 1 or frame_count % max(1, args.ocr_every) == 0:
        ocr_started = time.perf_counter()
        last_ocr_executed = True
        last_pii_regions = detect_pii()
        last_ocr_time_ms = round((time.perf_counter() - ocr_started) * 1000, 3)
        for region in last_pii_regions:
            pii_log.append({"timestamp": utc_now(), "experiment_id": SESSION_ID,
                            "trial_id": args.trial_id, "sample_id": "live-screen",
                            "condition": "live", "detected_pii_type": region.get("label", ""),
                            "detected_bbox": region.get("bbox", ()), "confidence": region.get("confidence", ""),
                            "ocr_detected": True, "pii_detected": True})
    else:
        last_ocr_executed = False

    pii_score = max((region.get("sensitivity", 0) for region in last_pii_regions), default=0)

    score = risk_engine.compute_risk(face_count, gaze_score, proximity_score, pii_score)
    level = risk_engine.get_risk_level(score)

    if not args.headless:
        if user_box:
            x, y, w, h = user_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            cv2.putText(frame, "user", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        for (x, y, w, h) in observer_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "observer", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        color = (0, 255, 0) if level == "LOW" else (0, 255, 255) if level == "MEDIUM" else (0, 0, 255)
        cv2.putText(frame, f"RISK: {level} ({score:.0f}/100)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"G {gaze_score*100:.0f}  P {proximity_score:.0f}  D {risk_engine.last_dwell_score:.0f}  S {pii_score:.0f}",
                    (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Webcam Monitor", frame)

    redaction_mode = risk_engine.overlay_mode(score)
    overlay_triggered = bool(last_pii_regions) and redaction_mode != "none"
    render = None
    overlay_trigger_timestamp = ""
    if overlay_triggered:
        overlay_trigger_timestamp = utc_now()
        render = apply_blur(last_pii_regions, threat_score=score, mode=redaction_mode)
    overlay_rendered = bool(render and render.rendered)
    overlay_visible_ts = utc_now() if overlay_rendered else ""

    completed = time.perf_counter()
    processing_ms = (completed - frame_started) * 1000
    try:
        cpu = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        cpu, memory_mb = "", ""
    comps = risk_engine.last_components
    runtime_log.log_frame(
        frame_id=frame_id, face_count=face_count, face_detected=face_count > 0,
        observer_detected=face_count > 1, gaze_score=round(gaze_score, 4),
        gaze_raw=round(gaze.last_raw_measurement, 4), mesh_face_count=gaze.last_face_count,
        proximity_score=round(proximity_score, 3), dwell_score=round(risk_engine.last_dwell_score, 3),
        dwell_seconds=round(risk_engine.duration_counter, 3),
        pii_score=pii_score, pii_count=len(last_pii_regions),
        pii_types=sorted({r.get("label", "") for r in last_pii_regions}),
        contrib_crowd=round(comps.get("crowd", 0), 3), contrib_gaze=round(comps.get("gaze", 0), 3),
        contrib_proximity=round(comps.get("proximity", 0), 3), contrib_pii=round(comps.get("pii", 0), 3),
        contrib_dwell=round(comps.get("dwell", 0), 3),
        threat_score=score, risk_level=level, overlay_mode=redaction_mode,
        overlay_triggered=overlay_triggered, overlay_rendered=overlay_rendered,
        overlay_skip_reason=(render.reason if render and not render.rendered else ""),
        overlay_activation_timestamp=overlay_visible_ts,
        t_trigger=overlay_trigger_timestamp,
        t_overlay_visible=overlay_visible_ts,
        overlay_latency_ms=round(render.total_ms, 3) if overlay_rendered else "",
        overlay_capture_ms=round(render.capture_ms, 3) if overlay_rendered else "",
        overlay_render_ms=round(render.render_ms, 3) if overlay_rendered else "",
        processing_time_ms=round(processing_ms, 3), fps=round(fps, 3) if fps != "" else "",
        cpu_percent=cpu, memory_mb=round(memory_mb, 3) if memory_mb != "" else "",
        ocr_execution=last_ocr_executed, ocr_time_ms=last_ocr_time_ms if last_ocr_executed else "",
        camera_resolution=camera_resolution, screen_resolution=screen_size,
        experiment_id=EXPERIMENT_ID, trial_id=args.trial_id, condition=args.condition,
        ground_truth_observer_present=args.observer_present,
        lighting_condition=args.lighting, observer_distance=args.distance,
        gaze_condition=args.gaze_condition, occlusion_condition=args.occlusion,
        gaze_duration_s=args.gaze_duration, camera_position=args.camera_position,
        use_gaze=risk_engine.config["features"]["gaze"],
        use_proximity=risk_engine.config["features"]["proximity"],
        use_dwell=risk_engine.config["features"]["dwell"],
        use_pii=risk_engine.config["features"]["pii"],
        use_crowd=risk_engine.config["features"].get("crowd", True),
        risk_weight_gaze=risk_engine.config["weights"]["gaze"],
        risk_weight_proximity=risk_engine.config["weights"]["proximity"],
        risk_weight_pii=risk_engine.config["weights"]["pii"],
        risk_crowd_points=risk_engine.config["crowd_points"],
        risk_dwell_max_points=risk_engine.config["dwell_max_points"],
        risk_dwell_saturation_s=risk_engine.config["dwell_saturation_seconds"],
        threshold_no_action=risk_engine.config["thresholds"]["no_action"],
        threshold_selective_blur=risk_engine.config["thresholds"]["selective_blur"],
        threshold_fake_overlay=risk_engine.config["thresholds"]["fake_overlay"],
        threshold_risk_medium=risk_engine.config["thresholds"]["risk_medium"],
        threshold_risk_high=risk_engine.config["thresholds"]["risk_high"],
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

print(f"Session {SESSION_ID} ({EXPERIMENT_ID}/{args.trial_id}): {frame_id} frames logged.")

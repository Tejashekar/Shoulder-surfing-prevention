"""Append-only CSV logging; intentionally excludes OCR text and raw frames."""
import csv
import json
import os
import uuid
from datetime import datetime, timezone

RUNTIME_FIELDS = ["timestamp", "session_id", "frame_id", "face_count", "face_detected", "observer_detected", "gaze_score", "proximity_score", "dwell_score", "pii_score", "pii_count", "pii_types", "threat_score", "overlay_mode", "overlay_triggered", "t_trigger", "t_overlay_visible", "overlay_activation_timestamp", "overlay_latency_ms", "processing_time_ms", "fps", "cpu_percent", "memory_mb", "ocr_execution", "ocr_time_ms", "camera_resolution", "screen_resolution", "lighting_condition", "observer_distance", "gaze_condition", "gaze_duration_s", "occlusion_condition", "camera_position", "use_gaze", "use_proximity", "use_dwell", "use_pii", "risk_weight_gaze", "risk_weight_proximity", "risk_weight_pii", "risk_crowd_points", "risk_dwell_max_points", "risk_dwell_saturation_s", "threshold_no_action", "threshold_selective_blur", "threshold_fake_overlay", "threshold_risk_medium", "threshold_risk_high", "experiment_id", "trial_id", "condition", "ground_truth_observer_present", "risk_level", "gaze_raw", "mesh_face_count", "dwell_seconds", "contrib_crowd", "contrib_gaze", "contrib_proximity", "contrib_pii", "contrib_dwell", "overlay_rendered", "overlay_skip_reason", "overlay_capture_ms", "overlay_render_ms", "use_crowd"]
PII_FIELDS = ["timestamp", "experiment_id", "trial_id", "sample_id", "condition", "ground_truth_pii_type", "ground_truth_bbox", "detected_pii_type", "detected_bbox", "confidence", "ocr_detected", "ocr_recognition_correct", "pii_detected"]

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

def latency_ms(t_trigger, t_visible):
    """Convert same-clock trigger/visible timestamps into a nonnegative duration."""
    value = (float(t_visible) - float(t_trigger)) * 1000.0
    if value < 0:
        raise ValueError("visible timestamp precedes trigger timestamp")
    return value

class CsvLogger:
    def __init__(self, path, fields):
        self.path, self.fields = os.fspath(path), list(fields)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()
        else:
            self._migrate_header()

    def _migrate_header(self):
        """If the file was written with an older (shorter) schema whose columns are a
        prefix of the current one, rewrite the header so new columns append cleanly.
        Existing rows keep their values; new columns read back as empty strings."""
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing = next(reader, [])
            if existing == self.fields or not existing:
                return
            if self.fields[:len(existing)] != existing:
                raise ValueError(f"{self.path}: schema differs and is not a prefix of the current schema; migrate manually")
            rows = list(reader)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(self.fields)
            pad = [""] * (len(self.fields) - len(existing))
            for r in rows:
                w.writerow(r + pad)

    def append(self, row):
        safe = {key: row.get(key, "") for key in self.fields}
        for key in ("pii_types", "ground_truth_bbox", "detected_bbox"):
            if isinstance(safe.get(key), (list, tuple, dict)):
                safe[key] = json.dumps(safe[key], separators=(",", ":"))
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(safe)

class RuntimeLogger:
    def __init__(self, results_dir="results", session_id=None):
        self.session_id = session_id or uuid.uuid4().hex
        self.csv = CsvLogger(os.path.join(results_dir, "runtime_logs.csv"), RUNTIME_FIELDS)

    def log_frame(self, **values):
        values["session_id"] = self.session_id
        values.setdefault("timestamp", utc_now())
        self.csv.append(values)

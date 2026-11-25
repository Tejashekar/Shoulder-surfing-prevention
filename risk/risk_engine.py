import json
import os
import time


class RiskEngine:
    def __init__(self, config_path=None, features=None):
        self.duration_counter = 0
        self.last_score = 0.0
        self.last_dwell_score = 0.0
        self.last_components = {}
        self.last_frame_time = None
        self.config = {
            "weights": {"gaze": 0.32, "proximity": 0.28, "pii": 0.30},
            "crowd_points": 18.0, "dwell_max_points": 20.0,
            "dwell_saturation_seconds": 5.0,
            "thresholds": {"no_action": 35.0, "selective_blur": 35.0, "fake_overlay": 75.0, "risk_medium": 35.0, "risk_high": 70.0},
            "features": {"gaze": True, "proximity": True, "dwell": True, "pii": True},
        }
        if config_path:
            with open(config_path, encoding="utf-8") as f:
                supplied = json.load(f)
            for section in ("weights", "thresholds", "features"):
                self.config[section].update(supplied.get(section, {}))
            for key in ("crowd_points", "dwell_max_points", "dwell_saturation_seconds"):
                if key in supplied:
                    self.config[key] = float(supplied[key])
        if features:
            self.config["features"].update(features)

    def _coerce_score(self, value, true_value=100.0, false_value=0.0):
        if isinstance(value, bool):
            return true_value if value else false_value

        if isinstance(value, str):
            lowered = value.lower()
            if lowered == "near":
                return 100.0
            if lowered == "far":
                return 0.0

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return false_value

        if numeric <= 1.0:
            return max(0.0, min(100.0, numeric * 100.0))
        return max(0.0, min(100.0, numeric))

    def compute_risk(self, face_count, gaze_signal, proximity, pii_signal=0.0, now=None):
        gaze_score = self._coerce_score(gaze_signal)
        proximity_score = self._coerce_score(proximity)
        pii_score = self._coerce_score(pii_signal)

        now = time.monotonic() if now is None else float(now)
        elapsed = 0.0 if self.last_frame_time is None else max(0.0, min(now - self.last_frame_time, 1.0))
        self.last_frame_time = now
        if face_count > 1:
            self.duration_counter += elapsed
        else:
            self.duration_counter = max(0.0, self.duration_counter - elapsed * 2.0)

        dwell_score = min(self.duration_counter / max(self.config["dwell_saturation_seconds"], 1e-9), 1.0) * self.config["dwell_max_points"]
        crowd_score = self.config["crowd_points"] if face_count > 1 else 0.0
        features = self.config["features"]

        self.last_components = {
            "crowd": crowd_score if features.get("crowd", True) else 0.0,
            "gaze": gaze_score * self.config["weights"]["gaze"] if features["gaze"] else 0.0,
            "proximity": proximity_score * self.config["weights"]["proximity"] if features["proximity"] else 0.0,
            "pii": pii_score * self.config["weights"]["pii"] if features["pii"] else 0.0,
            "dwell": dwell_score if features["dwell"] else 0.0,
        }
        self.last_dwell_score = dwell_score
        score = sum(self.last_components.values())

        self.last_score = max(0.0, min(100.0, score))
        return round(self.last_score, 2)

    @staticmethod
    def score_from_components(face_count, gaze_score, proximity_score, pii_score, dwell_score, config):
        """Pure recomputation of the fused score from logged 0-100 component values.

        Used by offline ablation/sensitivity replay so that logged runtime frames
        can be re-scored under alternative configurations without a camera.
        """
        f, w = config["features"], config["weights"]
        crowd = config["crowd_points"] if face_count > 1 and f.get("crowd", True) else 0.0
        dwell = min(float(dwell_score), config["dwell_max_points"]) if f["dwell"] else 0.0
        total = (crowd
                 + (float(gaze_score) * w["gaze"] if f["gaze"] else 0.0)
                 + (float(proximity_score) * w["proximity"] if f["proximity"] else 0.0)
                 + (float(pii_score) * w["pii"] if f["pii"] else 0.0)
                 + dwell)
        return round(max(0.0, min(100.0, total)), 2)

    def get_risk_level(self, score):
        thresholds = self.config["thresholds"]
        if score < thresholds["risk_medium"]:
            return "LOW"
        elif score < thresholds["risk_high"]:
            return "MEDIUM"
        else:
            return "HIGH"

    def overlay_mode(self, score):
        thresholds = self.config["thresholds"]
        if score >= thresholds["fake_overlay"]:
            return "fake"
        if score >= thresholds["selective_blur"]:
            return "blur"
        return "none"

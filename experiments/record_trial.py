"""Record manually observed experimental ground truth; never invent labels.

Examples:
  python -m experiments.record_trial distance --observer-present true --distance 1.0
  python -m experiments.record_trial robustness --observer-present false --lighting dim
  python -m experiments.record_trial usability --participant-id P001 --task-id T1
"""
import argparse
import csv
import os
import uuid
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
SCHEMAS = {
    "distance": ("observer_detection.csv", ["timestamp", "experiment_id", "trial_id", "condition", "distance_m", "ground_truth_observer_present", "predicted_observer_present", "threat_score", "overlay_mode", "detection_latency_ms", "notes"]),
    "gaze": ("runtime_logs.csv", None),
    "gaze_trial": ("gaze_results.csv", ["timestamp", "experiment_id", "trial_id", "condition", "gaze_condition", "target_dwell_s", "measured_dwell_s", "mean_gaze_score", "mean_dwell_score", "mean_threat_score", "time_to_overlay_ms", "overlay_mode", "notes"]),
    "robustness": ("robustness_results.csv", ["timestamp", "experiment_id", "trial_id", "condition", "lighting_condition", "camera_position", "face_orientation", "occlusion_condition", "distance_m", "ground_truth_observer_present", "predicted_observer_present", "threat_score", "detection_latency_ms", "notes"]),
    "ablation": ("ablation_results.csv", ["timestamp", "experiment_id", "trial_id", "condition", "use_gaze", "use_proximity", "use_dwell", "use_pii", "ground_truth_observer_present", "predicted_observer_present", "threat_score", "overlay_latency_ms", "fps", "cpu_percent", "memory_mb"]),
    "usability": ("usability_results.csv", ["timestamp", "experiment_id", "trial_id", "participant_id", "condition", "task_id", "task_completion_time_s", "task_success", "error_count", "overlay_interruptions", "user_rating", "annoyance_rating", "comments"]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=SCHEMAS)
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--trial-id", default=None)
    p.add_argument("--condition", default="")
    p.add_argument("--observer-present", choices=("true", "false"))
    p.add_argument("--distance", type=float, choices=(0.5, 1.0, 1.5, 2.0))
    p.add_argument("--predicted-observer-present", choices=("true", "false"))
    p.add_argument("--threat-score", type=float)
    p.add_argument("--overlay-mode", default="")
    p.add_argument("--latency-ms", type=float)
    p.add_argument("--lighting", default="")
    p.add_argument("--camera-position", default="")
    p.add_argument("--face-orientation", default="")
    p.add_argument("--occlusion", default="")
    p.add_argument("--participant-id", default="")
    p.add_argument("--task-id", default="")
    p.add_argument("--completion-time", type=float)
    p.add_argument("--success", choices=("true", "false"))
    p.add_argument("--errors", type=int)
    p.add_argument("--interruptions", type=int)
    p.add_argument("--rating", type=float)
    p.add_argument("--annoyance", type=float)
    p.add_argument("--notes", default="")
    p.add_argument("--gaze-condition", default="")
    p.add_argument("--target-dwell", type=float)
    p.add_argument("--measured-dwell", type=float)
    p.add_argument("--mean-gaze-score", type=float)
    p.add_argument("--mean-dwell-score", type=float)
    p.add_argument("--mean-threat-score", type=float)
    p.add_argument("--overlay-latency", type=float)
    p.add_argument("--use-gaze", choices=("true", "false"))
    p.add_argument("--use-proximity", choices=("true", "false"))
    p.add_argument("--use-dwell", choices=("true", "false"))
    p.add_argument("--use-pii", choices=("true", "false"))
    p.add_argument("--fps", type=float)
    p.add_argument("--cpu-percent", type=float)
    p.add_argument("--memory-mb", type=float)
    a = p.parse_args()
    filename, fields = SCHEMAS[a.kind]
    if fields is None:
        p.error("use gaze_trial to record trial-level gaze/dwell summaries; frame measurements are in runtime_logs.csv")
    exp = a.experiment_id or uuid.uuid4().hex
    trial = a.trial_id or uuid.uuid4().hex[:12]
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
           "experiment_id": exp, "trial_id": trial, "condition": a.condition,
           "distance_m": a.distance, "ground_truth_observer_present": a.observer_present,
           "predicted_observer_present": a.predicted_observer_present, "threat_score": a.threat_score,
           "overlay_mode": a.overlay_mode, "detection_latency_ms": a.latency_ms,
           "overlay_latency_ms": a.overlay_latency,
           "lighting_condition": a.lighting, "camera_position": a.camera_position,
           "face_orientation": a.face_orientation, "occlusion_condition": a.occlusion,
           "participant_id": a.participant_id, "task_id": a.task_id,
           "task_completion_time_s": a.completion_time, "task_success": a.success,
           "error_count": a.errors, "overlay_interruptions": a.interruptions,
           "user_rating": a.rating, "annoyance_rating": a.annoyance, "comments": a.notes, "notes": a.notes}
    row.update({"gaze_condition": a.gaze_condition, "target_dwell_s": a.target_dwell,
                "measured_dwell_s": a.measured_dwell, "mean_gaze_score": a.mean_gaze_score,
                "mean_dwell_score": a.mean_dwell_score, "mean_threat_score": a.mean_threat_score,
                "time_to_overlay_ms": a.overlay_latency, "use_gaze": a.use_gaze,
                "use_proximity": a.use_proximity, "use_dwell": a.use_dwell, "use_pii": a.use_pii,
                "fps": a.fps, "cpu_percent": a.cpu_percent, "memory_mb": a.memory_mb})
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, filename)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(f"Recorded manual trial {trial} in {path}; verify this row against the observed trial.")


if __name__ == "__main__":
    main()

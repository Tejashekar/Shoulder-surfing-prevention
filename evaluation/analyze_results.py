"""Summarize collected CSV data. Empty/missing files are never populated.

v2.1:
- Runtime stats may be filtered to one experiment/trial (--experiment, --trial)
  and are also emitted per experiment.
- Overlay latency uses only rows with overlay_rendered=True (older rows without
  that column fall back to latency > 1 ms to exclude throttle early-returns).
- PII metrics are computed only on synthetic-manifest rows (sample_id != live-screen),
  which carry ground truth; live rows have no labels and were previously inflating FP.
- Observer-detection metrics are computed from (a) observer_detection.csv trial rows
  and (b) runtime frames that carry ground_truth_observer_present, aggregated per trial
  (a trial is predicted positive if the system flagged an observer on >= 50% of frames
  or the max threat score reached the blur threshold).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

try:
    from .metrics import binary_metrics, descriptive
except ImportError:
    from metrics import binary_metrics, descriptive

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
LIVE_SAMPLE = "live-screen"


def read_csv(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _flag(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def rendered_latencies(rows):
    out = []
    for r in rows:
        v = r.get("overlay_latency_ms", "")
        if not v:
            continue
        if "overlay_rendered" in r and r.get("overlay_rendered", "") != "":
            if _flag(r["overlay_rendered"]):
                out.append(v)
        elif float(v) > 1.0:  # legacy rows: drop throttle early-returns
            out.append(v)
    return out


def trial_level_observer_predictions(runtime, frame_fraction=0.5):
    """Aggregate labelled runtime frames into one prediction per (experiment, trial)."""
    groups = defaultdict(list)
    for r in runtime:
        if r.get("ground_truth_observer_present") in ("true", "false") and r.get("trial_id"):
            groups[(r.get("experiment_id", ""), r["trial_id"])].append(r)
    trials = []
    for key, rows in groups.items():
        truth = rows[0]["ground_truth_observer_present"] == "true"
        flagged = sum(_flag(r.get("observer_detected")) for r in rows) / len(rows)
        threat = [float(r["threat_score"]) for r in rows if r.get("threat_score")]
        blur_t = float(rows[0].get("threshold_selective_blur") or 35)
        predicted = flagged >= frame_fraction or (max(threat, default=0) >= blur_t)
        trials.append({"experiment_id": key[0], "trial_id": key[1], "truth": truth, "predicted": predicted,
                       "flagged_fraction": flagged, "max_threat": max(threat, default=0),
                       "condition": rows[0].get("condition", ""), "distance": rows[0].get("observer_distance", ""),
                       "frames": len(rows)})
    return trials


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default=None, help="Only include runtime rows with this experiment_id")
    p.add_argument("--trial", default=None)
    args = p.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    summary = []

    runtime_all = read_csv("runtime_logs.csv")
    runtime = [r for r in runtime_all
               if (args.experiment is None or r.get("experiment_id") == args.experiment)
               and (args.trial is None or r.get("trial_id") == args.trial)]

    def runtime_block(rows, tag):
        for field in ("fps", "cpu_percent", "memory_mb", "processing_time_ms", "ocr_time_ms", "gaze_score", "proximity_score", "dwell_score", "pii_score", "threat_score"):
            d = descriptive([r.get(field) for r in rows if r.get(field)])
            if d["count"]:
                summary.append({"metric": field, "condition": tag, **d})
        lat = rendered_latencies(rows)
        if lat:
            summary.append({"metric": "overlay_latency_ms_rendered", "condition": tag, **descriptive(lat)})
        for field in ("overlay_capture_ms", "overlay_render_ms"):
            d = descriptive([r.get(field) for r in rows if r.get(field)])
            if d["count"]:
                summary.append({"metric": field, "condition": tag, **d})
        n_rendered = sum(_flag(r.get("overlay_rendered")) for r in rows)
        n_trig = sum(_flag(r.get("overlay_triggered")) for r in rows)
        if n_trig:
            summary.append({"metric": "overlay_triggered_frames", "condition": tag, "count": n_trig})
            summary.append({"metric": "overlay_rendered_frames", "condition": tag, "count": n_rendered})
        if rows:
            summary.append({"metric": "frames", "condition": tag, "count": len(rows)})
            summary.append({"metric": "multi_face_frames", "condition": tag, "count": sum(int(r.get("face_count") or 0) > 1 for r in rows)})

    runtime_block(runtime, "all")
    for exp in sorted({r.get("experiment_id", "") for r in runtime}):
        runtime_block([r for r in runtime if r.get("experiment_id") == exp], f"exp:{exp}")

    # Observer detection — trial rows
    observer = read_csv("observer_detection.csv")
    labeled = [r for r in observer if r.get("ground_truth_observer_present") in ("true", "false") and r.get("predicted_observer_present") in ("true", "false")]
    if labeled:
        m = binary_metrics([r["ground_truth_observer_present"] == "true" for r in labeled], [r["predicted_observer_present"] == "true" for r in labeled])
        for name, value in m.items():
            summary.append({"metric": "observer_" + name, "count": len(labeled), "mean": value})
        for dist in sorted({r.get("distance_m", "") for r in labeled}):
            g = [r for r in labeled if r.get("distance_m", "") == dist]
            m = binary_metrics([r["ground_truth_observer_present"] == "true" for r in g], [r["predicted_observer_present"] == "true" for r in g])
            for name in ("accuracy", "precision", "recall", "F1", "FPR", "FNR"):
                summary.append({"metric": "observer_" + name, "condition": f"distance:{dist}", "count": len(g), "mean": m[name]})

    # Observer detection — labelled runtime frames aggregated per trial
    trials = trial_level_observer_predictions(runtime_all)
    if trials:
        m = binary_metrics([t["truth"] for t in trials], [t["predicted"] for t in trials])
        for name, value in m.items():
            summary.append({"metric": "observer_trial_" + name, "count": len(trials), "mean": value})
        for dist in sorted({t["distance"] for t in trials}):
            g = [t for t in trials if t["distance"] == dist]
            m = binary_metrics([t["truth"] for t in g], [t["predicted"] for t in g])
            for name in ("accuracy", "precision", "recall", "F1"):
                summary.append({"metric": "observer_trial_" + name, "condition": f"distance:{dist}", "count": len(g), "mean": m[name]})
        with open(os.path.join(RESULTS, "observer_trial_summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(trials[0].keys())); w.writeheader(); w.writerows(trials)

    # PII — synthetic rows only (live rows have no ground truth)
    pii_rows = [r for r in read_csv("pii_detection.csv") if r.get("sample_id") and r.get("sample_id") != LIVE_SAMPLE]
    if pii_rows:
        experiments = sorted({r.get("experiment_id", "") for r in pii_rows})
        latest = experiments[-1] if experiments else ""
        rows = [r for r in pii_rows if r.get("experiment_id") == latest]
        m = binary_metrics([bool(r.get("ground_truth_pii_type")) for r in rows], [bool(r.get("detected_pii_type")) for r in rows])
        for name, value in m.items():
            summary.append({"metric": "pii_category_" + name, "category": "all", "condition": f"exp:{latest}", "count": len(rows), "mean": value})
        ocr = {(r["sample_id"]): r.get("ocr_recognition_correct") for r in rows if r.get("ocr_recognition_correct") not in ("", None)}
        if ocr:
            summary.append({"metric": "ocr_exact_presence_accuracy", "condition": f"exp:{latest}", "count": len(ocr), "mean": sum(_flag(v) for v in ocr.values()) / len(ocr)})
        summary.append({"metric": "pii_note", "condition": "see pii_summary.csv (IoU-matched) for reportable P/R/F1"})

    robustness = read_csv("robustness_results.csv")
    robust_labeled = [r for r in robustness if r.get("ground_truth_observer_present") in ("true", "false") and r.get("predicted_observer_present") in ("true", "false")]
    for condition in sorted({r.get("condition", "") for r in robust_labeled}):
        group = [r for r in robust_labeled if r.get("condition", "") == condition]
        m = binary_metrics([r["ground_truth_observer_present"] == "true" for r in group], [r["predicted_observer_present"] == "true" for r in group])
        for name, value in m.items():
            summary.append({"metric": "robustness_" + name, "condition": condition, "count": len(group), "mean": value})

    ablation = read_csv("ablation_results.csv")
    ablation_labeled = [r for r in ablation if r.get("ground_truth_observer_present") in ("true", "false") and r.get("predicted_observer_present") in ("true", "false")]
    for condition in sorted({r.get("condition", "") for r in ablation_labeled}):
        group = [r for r in ablation_labeled if r.get("condition", "") == condition]
        m = binary_metrics([r["ground_truth_observer_present"] == "true" for r in group], [r["predicted_observer_present"] == "true" for r in group])
        for name, value in m.items():
            summary.append({"metric": "ablation_" + name, "condition": condition, "count": len(group), "mean": value})

    usability = read_csv("usability_results.csv")
    for field in ("task_completion_time_s", "error_count", "overlay_interruptions", "user_rating", "annoyance_rating"):
        d = descriptive([r.get(field) for r in usability])
        if d["count"]:
            summary.append({"metric": "usability_" + field, **d})

    path = os.path.join(RESULTS, "runtime_summary.csv")
    fields = ["metric", "condition", "category", "count", "mean", "median", "std", "min", "max", "p95"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)
    print(f"Wrote {path} from {len(runtime)} runtime frames, {len(observer)} observer rows, {len(trials)} labelled trials.")
    if not runtime and not observer and not pii_rows and not robustness and not ablation and not usability:
        print("No measured data found; summary contains headers only.")


if __name__ == "__main__":
    main()

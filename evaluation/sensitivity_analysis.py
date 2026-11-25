"""Weight/threshold sensitivity on labelled trials by replaying logged components.

Ground truth comes from either observer_detection.csv (trial rows) or the
`ground_truth_observer_present` column of runtime_logs.csv (v2.1 live runs with
--observer-present). The logged 0-100 component values (gaze_score x100,
proximity_score, pii_score, dwell_score, face_count) are re-fused with
`RiskEngine.score_from_components` under each candidate configuration.

A trial is predicted "observer present" if its maximum replayed score reaches
the candidate threshold. This is a sensitivity comparison on the collected
trials, not training; report held-out results before calling any value
"calibrated".
"""
import argparse
import csv
import itertools
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from risk.risk_engine import RiskEngine
try:
    from .metrics import binary_metrics
except ImportError:
    from metrics import binary_metrics

BASE = os.path.join(ROOT, "results")


def read(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labelled_trials():
    """Return {(experiment_id, trial_id): (truth_bool, [frame rows])}."""
    truth = {}
    for r in read("observer_detection.csv"):
        if r.get("trial_id") and r.get("ground_truth_observer_present") in ("true", "false"):
            truth[(r.get("experiment_id", ""), r["trial_id"])] = r["ground_truth_observer_present"] == "true"
    frames = {}
    for r in read("runtime_logs.csv"):
        key = (r.get("experiment_id", ""), r.get("trial_id", ""))
        if r.get("ground_truth_observer_present") in ("true", "false"):
            truth.setdefault(key, r["ground_truth_observer_present"] == "true")
        if key in truth:
            try:
                row = {k: float(r.get(k) or 0) for k in ("gaze_score", "proximity_score", "pii_score", "dwell_score", "face_count")}
            except ValueError:
                continue
            frames.setdefault(key, []).append(row)
    return {k: (truth[k], frames[k]) for k in frames}


def replay(trials, config, threshold):
    ys, ps = [], []
    for truth, rows in trials.values():
        scores = [RiskEngine.score_from_components(
            r["face_count"], r["gaze_score"] * 100.0 if r["gaze_score"] <= 1.0 else r["gaze_score"],
            r["proximity_score"], r["pii_score"], r["dwell_score"], config) for r in rows]
        ys.append(truth)
        ps.append(max(scores, default=0) >= threshold)
    return ys, ps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "risk.json"))
    args = p.parse_args(argv)
    base = RiskEngine(config_path=args.config).config
    trials = load_labelled_trials()
    weight_sets = [(0.2, 0.2, 0.2), (0.32, 0.28, 0.30), (0.4, 0.3, 0.2), (0.3, 0.4, 0.2), (0.4, 0.4, 0.1)]
    crowd_sets = (0.0, 10.0, 18.0)
    thresholds = (25, 35, 45, 50, 60, 70, 75)
    candidates = []
    for (wg, wp, ws), crowd, threshold in itertools.product(weight_sets, crowd_sets, thresholds):
        cfg = json.loads(json.dumps(base))
        cfg["weights"] = {"gaze": wg, "proximity": wp, "pii": ws}
        cfg["crowd_points"] = crowd
        ys, ps = replay(trials, cfg, threshold)
        if ys:
            candidates.append({"w_gaze": wg, "w_proximity": wp, "w_pii": ws, "crowd_points": crowd,
                               "threshold": threshold, "n_trials": len(ys), **binary_metrics(ys, ps)})
    out = os.path.join(BASE, "sensitivity_analysis.csv")
    fields = ["w_gaze", "w_proximity", "w_pii", "crowd_points", "threshold", "n_trials", "TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "F1", "FPR", "FNR"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(candidates)
    print(f"Wrote {len(candidates)} candidate evaluations to {out} using {len(trials)} labelled trials.")
    if not candidates:
        print("No labelled trials found; run main.py with --observer-present true/false and --trial-id, or record observer_detection.csv rows.")


if __name__ == "__main__":
    main()

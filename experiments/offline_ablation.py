"""Offline component ablation by replaying logged runtime frames.

Every runtime frame logs its 0-100 component inputs (gaze, proximity, PII,
dwell, face count). Those inputs do not depend on the fusion configuration, so
the fused score can be recomputed under each ablation (gaze off, proximity off,
dwell off, PII off, crowd off, observer-only, content-only) without re-running
the camera. For labelled trials (ground_truth_observer_present present either in
runtime_logs.csv or observer_detection.csv) the script reports trial-level
detection metrics and the number of overlay-activating frames per configuration.

This answers the reviewer request for an ablation without a second data
collection, provided the labelled trials were collected with the v2.1 runtime
(the pilot build logged gaze_score = 0 on every frame, so a gaze ablation on
pilot data is uninformative and is flagged).
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from risk.risk_engine import RiskEngine
from evaluation.metrics import binary_metrics, descriptive
from evaluation.sensitivity_analysis import load_labelled_trials

ABLATIONS = {
    "full": {},
    "no-gaze": {"gaze": False},
    "no-proximity": {"proximity": False},
    "no-dwell": {"dwell": False},
    "no-pii": {"pii": False},
    "no-crowd": {"crowd": False},
    "observer-only": {"pii": False},
    "content-only": {"gaze": False, "proximity": False, "dwell": False, "crowd": False},
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "risk.json"))
    p.add_argument("--output", default=os.path.join(ROOT, "results", "ablation_offline.csv"))
    args = p.parse_args(argv)
    base = RiskEngine(config_path=args.config).config
    blur_t = base["thresholds"]["selective_blur"]
    trials = load_labelled_trials()
    if not trials:
        print("No labelled trials; nothing to ablate. Collect trials with --observer-present and --trial-id.")
        return
    all_frames = [r for _, rows in trials.values() for r in rows]
    gaze_live = any(r["gaze_score"] > 0 for r in all_frames)

    rows_out = []
    for name, switches in ABLATIONS.items():
        cfg = json.loads(json.dumps(base))
        cfg["features"].update(switches)
        ys, ps, activations, maxes = [], [], 0, []
        for truth, rows in trials.values():
            scores = [RiskEngine.score_from_components(
                r["face_count"], r["gaze_score"] * 100.0 if r["gaze_score"] <= 1.0 else r["gaze_score"],
                r["proximity_score"], r["pii_score"], r["dwell_score"], cfg) for r in rows]
            ys.append(truth)
            ps.append(max(scores, default=0) >= blur_t)
            activations += sum(s >= blur_t for s in scores)
            maxes.append(max(scores, default=0))
        m = binary_metrics(ys, ps)
        d = descriptive(maxes)
        rows_out.append({"condition": name, "n_trials": len(ys), "activating_frames": activations,
                         "frames": len(all_frames), "mean_max_threat": round(d["mean"], 2),
                         **{k: m[k] for k in ("TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "F1", "FPR", "FNR")},
                         "note": "gaze_score was 0 on all frames; no-gaze equals full" if (name == "no-gaze" and not gaze_live) else ""})
    fields = list(rows_out[0].keys())
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows_out)
    print(f"Wrote {args.output}: {len(trials)} labelled trials, {len(all_frames)} frames.")
    for r in rows_out:
        print(f"  {r['condition']:<14} acc={r['accuracy']:.2f} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['F1']:.2f} FPR={r['FPR']:.2f} activations={r['activating_frames']} {r['note']}")


if __name__ == "__main__":
    main()

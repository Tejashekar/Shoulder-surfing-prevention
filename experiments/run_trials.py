"""Guided collection of labelled observer trials.

Builds a counterbalanced trial list (distance x observer-present/absent x
repeats, plus optional gaze and robustness blocks), prompts the researcher to
stage each condition, then launches `main.py` for a fixed duration with the
matching metadata and ground-truth label. Trial order is shuffled with a seed
and written to results/trial_plan_<experiment>.csv before anything runs.

Examples (from Version 2/):
  python -m experiments.run_trials --experiment-id OBS-01 --block distance --repeats 5 --duration 15
  python -m experiments.run_trials --experiment-id GAZE-01 --block gaze --repeats 3 --duration 10
  python -m experiments.run_trials --experiment-id ROB-01 --block robustness --repeats 2 --duration 10
  python -m experiments.run_trials --experiment-id ABL-01 --block ablation --repeats 3 --duration 15

Ground truth is what the researcher staged, never what the system detected.
Keep the same user seated, the same screen content (a synthetic PII screen from
test_data/), and the same lighting within a block.
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")


def build_plan(block, repeats, seed):
    rnd = random.Random(seed)
    plan = []
    if block == "distance":
        for d in (0.5, 1.0, 1.5, 2.0):
            for present in (True, False):
                for r in range(repeats):
                    plan.append({"condition": f"distance-{d}m-{'obs' if present else 'noobs'}", "distance": d,
                                 "observer_present": present, "gaze_condition": "toward" if present else "",
                                 "lighting": "normal", "camera_position": "centered", "occlusion": "none", "extra": []})
    elif block == "gaze":
        for gaze in ("toward", "away"):
            for dwell in (0.5, 2.0, 5.0):
                for r in range(repeats):
                    plan.append({"condition": f"gaze-{gaze}-{dwell}s", "distance": 1.0, "observer_present": True,
                                 "gaze_condition": gaze, "gaze_duration": dwell, "lighting": "normal",
                                 "camera_position": "centered", "occlusion": "none", "extra": []})
        for r in range(repeats):
            plan.append({"condition": "gaze-control-noobs", "distance": 1.0, "observer_present": False,
                         "gaze_condition": "", "lighting": "normal", "camera_position": "centered", "occlusion": "none", "extra": []})
    elif block == "robustness":
        cells = [("dim", "centered", "none"), ("bright", "centered", "none"), ("normal", "left", "none"),
                 ("normal", "right", "none"), ("normal", "centered", "glasses"), ("normal", "centered", "partial-face"),
                 ("normal", "centered", "angled-30"), ("normal", "centered", "angled-45")]
        for lighting, cam, occ in cells:
            for present in (True, False):
                for r in range(repeats):
                    plan.append({"condition": f"rob-{lighting}-{cam}-{occ}-{'obs' if present else 'noobs'}", "distance": 1.0,
                                 "observer_present": present, "gaze_condition": "toward" if present else "",
                                 "lighting": lighting, "camera_position": cam, "occlusion": occ, "extra": []})
    elif block == "ablation":
        # Live ablation: same staged condition, one feature disabled per run.
        for flag, name in ((None, "full"), ("--no-gaze", "no-gaze"), ("--no-proximity", "no-proximity"),
                           ("--no-dwell", "no-dwell"), ("--no-pii", "no-pii")):
            for present in (True, False):
                for r in range(repeats):
                    plan.append({"condition": f"abl-{name}-{'obs' if present else 'noobs'}", "distance": 1.0,
                                 "observer_present": present, "gaze_condition": "toward" if present else "",
                                 "lighting": "normal", "camera_position": "centered", "occlusion": "none",
                                 "extra": [flag] if flag else []})
    rnd.shuffle(plan)
    for i, t in enumerate(plan, 1):
        t["trial_id"] = f"T{i:03}"
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--block", choices=("distance", "gaze", "robustness", "ablation"), required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--duration", type=float, default=15.0, help="seconds per trial")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start-at", type=int, default=1, help="resume from this trial number")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    plan = build_plan(a.block, a.repeats, a.seed)
    os.makedirs(RESULTS, exist_ok=True)
    plan_path = os.path.join(RESULTS, f"trial_plan_{a.experiment_id}.csv")
    with open(plan_path, "w", newline="", encoding="utf-8") as f:
        fields = ["trial_id", "condition", "distance", "observer_present", "gaze_condition", "gaze_duration", "lighting", "camera_position", "occlusion", "extra"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for t in plan:
            w.writerow({k: (" ".join(t[k]) if k == "extra" else t.get(k, "")) for k in fields})
    print(f"{len(plan)} trials planned ({a.block}, {a.repeats} repeats, {a.duration}s each, ~{len(plan) * (a.duration + 12) / 60:.0f} min). Plan: {plan_path}")
    for i, t in enumerate(plan, 1):
        if i < a.start_at:
            continue
        print("\n" + "=" * 70)
        print(f"Trial {i}/{len(plan)}  id={t['trial_id']}  condition={t['condition']}")
        print(f"  OBSERVER: {'PRESENT at %.1f m, looking %s' % (t['distance'], t['gaze_condition'].upper()) if t['observer_present'] else 'ABSENT (user alone)'}")
        if t.get("gaze_duration"):
            print(f"  Observer looks at the screen for {t['gaze_duration']} s, then away (use a timer).")
        print(f"  lighting={t['lighting']}  camera={t['camera_position']}  occlusion={t['occlusion']}  extra={' '.join(t['extra']) or '-'}")
        if a.dry_run:
            continue
        input("  Stage the condition, then press Enter to start recording...")
        cmd = [a.python, os.path.join(ROOT, "main.py"), "--experiment-id", a.experiment_id, "--trial-id", t["trial_id"],
               "--condition", t["condition"], "--lighting", t["lighting"], "--distance", str(t["distance"]),
               "--gaze-condition", t["gaze_condition"], "--occlusion", t["occlusion"], "--camera-position", t["camera_position"],
               "--observer-present", "true" if t["observer_present"] else "false", "--duration", str(a.duration)] + t["extra"]
        if t.get("gaze_duration"):
            cmd += ["--gaze-duration", str(t["gaze_duration"])]
        print("  running:", " ".join(cmd[1:]))
        for s in (3, 2, 1):
            print(f"  starting in {s}..."); time.sleep(1)
        subprocess.run(cmd, cwd=ROOT)
    print("\nBlock complete. Next: python -m evaluation.analyze_results && python -m experiments.offline_ablation && python -m evaluation.sensitivity_analysis")


if __name__ == "__main__":
    main()

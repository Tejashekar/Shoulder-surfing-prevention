"""Create plots only from collected CSV rows. No example values are generated."""
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(RESULTS, "plots")

def rows(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path): return []
    with open(path, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def series_plot(data, field, title, ylabel, filename, plt):
    valid = [(i, float(r[field])) for i, r in enumerate(data) if r.get(field, "")]
    if not valid: return
    plt.figure(figsize=(8, 4.5)); plt.plot([p[0] for p in valid], [p[1] for p in valid], linewidth=1)
    plt.xlabel("Recorded sample"); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, filename), dpi=300); plt.close()

def main():
    os.makedirs(PLOTS, exist_ok=True)
    runtime = rows("runtime_logs.csv")
    pii = rows("pii_detection.csv")
    observer_rows = rows("observer_detection.csv")
    has_data = bool(runtime or pii or observer_rows or rows("robustness_results.csv") or rows("ablation_results.csv"))
    if not has_data:
        print(f"No measured CSV rows found; no plot generated. Output directory: {PLOTS}")
        return
    import matplotlib.pyplot as plt
    for field, title, ylabel, filename in [
        ("fps", "Measured FPS over time", "Frames per second", "fps_over_time.png"),
        ("cpu_percent", "Measured process CPU over time", "CPU (%)", "cpu_over_time.png"),
        ("memory_mb", "Measured process memory over time", "Resident memory (MB)", "memory_over_time.png"),
        ("processing_time_ms", "Measured frame processing latency", "Milliseconds", "processing_latency.png"),
    ]: series_plot(runtime, field, title, ylabel, filename, plt)
    latency = [float(r["overlay_latency_ms"]) for r in runtime if r.get("overlay_latency_ms")]
    if not latency:
        latency = [float(r["detection_latency_ms"]) for r in observer_rows if r.get("detection_latency_ms")]
    if latency:
        plt.figure(figsize=(7, 4.5)); plt.hist(latency, bins="auto"); plt.xlabel("Detection latency (ms)"); plt.ylabel("Trials")
        plt.title("Measured observer detection latency"); plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "overlay_latency_distribution.png"), dpi=300); plt.close()
    # Plot threat by manually entered dwell/gaze trial records only when annotated rows exist.
    if runtime:
        grouped = {}
        for r in runtime:
            try: grouped.setdefault((r.get("gaze_condition", ""), r.get("condition", "")), []).append(float(r["threat_score"]))
            except (ValueError, KeyError): pass
        groups = [(k, v) for k, v in grouped.items() if any(k)]
        if groups:
            plt.figure(figsize=(8, 4.5))
            labels = [" / ".join(x for x in k if x) for k, _ in groups]
            plt.boxplot([v for _, v in groups], labels=labels); plt.xticks(rotation=25, ha="right")
            plt.ylabel("Threat score"); plt.title("Measured threat scores by logged condition")
            plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "threat_by_condition.png"), dpi=300); plt.close()
    categories = sorted({r.get("ground_truth_pii_type") or r.get("detected_pii_type") for r in pii} - {""})
    if pii and categories:
        recalls = []
        for category in categories:
            group = [r for r in pii if r.get("ground_truth_pii_type") == category or r.get("detected_pii_type") == category]
            positives = [r for r in group if r.get("ground_truth_pii_type")]
            recalls.append(sum(bool(r.get("detected_pii_type")) for r in positives) / len(positives) if positives else 0)
        plt.figure(figsize=(8, 4.5)); plt.bar(categories, recalls); plt.ylim(0, 1)
        plt.ylabel("Recall"); plt.xlabel("PII category"); plt.title("Measured synthetic PII category recall")
        plt.xticks(rotation=25, ha="right"); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, "pii_category_recall.png"), dpi=300); plt.close()
    print(f"Plots derived from recorded CSV data are in {PLOTS}; no plot is emitted for missing measurements.")

if __name__ == "__main__": main()

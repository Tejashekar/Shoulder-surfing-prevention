"""Binary classification metrics without third-party dependencies."""
import math


def binary_metrics(truth, prediction):
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction must have equal length")
    tp = sum(bool(t) and bool(p) for t, p in zip(truth, prediction))
    tn = sum(not bool(t) and not bool(p) for t, p in zip(truth, prediction))
    fp = sum(not bool(t) and bool(p) for t, p in zip(truth, prediction))
    fn = sum(bool(t) and not bool(p) for t, p in zip(truth, prediction))
    div = lambda a, b: a / b if b else 0.0
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": div(tp + tn, tp + tn + fp + fn),
            "precision": precision, "recall": recall,
            "F1": div(2 * precision * recall, precision + recall),
            "FPR": div(fp, fp + tn), "FNR": div(fn, fn + tp)}


def descriptive(values):
    xs = sorted(float(x) for x in values if x not in (None, ""))
    if not xs:
        return {k: "" for k in ("count", "mean", "median", "std", "min", "max", "p95")}
    n = len(xs)
    p95 = xs[max(0, math.ceil(.95 * n) - 1)]
    mean = sum(xs) / n
    return {"count": n, "mean": mean, "median": (xs[(n-1)//2] + xs[n//2]) / 2,
            "std": math.sqrt(sum((x-mean)**2 for x in xs) / n),
            "min": xs[0], "max": xs[-1], "p95": p95}

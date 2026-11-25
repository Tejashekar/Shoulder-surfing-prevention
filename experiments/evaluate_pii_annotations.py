"""Evaluate PII detections at annotation level without rerunning OCR.

The manifest supplies ground-truth annotations. The raw detection CSV supplies
detected regions. A true positive requires the same normalized category and
IoU >= 0.50. Matching is one-to-one and deterministic: eligible pairs are
sorted by descending IoU, then ground-truth index, then detection index.

Outputs are overwritten deterministically:
  results/pii_annotation_results.csv
  results/pii_false_positive_results.csv
  results/pii_summary.csv

The input raw CSV is never modified.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MANIFEST = os.path.join(ROOT, "test_data", "pii", "manifest.csv")
DEFAULT_RAW = os.path.join(ROOT, "results", "pii_detection.csv")
DEFAULT_RESULTS = os.path.join(ROOT, "results")
DEFAULT_IOU_THRESHOLD = 0.50
REQUIRED_CATEGORIES = ("email", "phone", "password", "aadhaar", "pan", "card", "account", "address")

ANNOTATION_FIELDS = [
    "sample_id", "experiment_id", "condition", "ground_truth_pii_type",
    "detected_pii_type", "ground_truth_bbox", "detected_bbox", "iou",
    "confidence", "ocr_detected", "ocr_recognition_correct",
    "category_correct", "detection_correct", "iou_threshold",
]
FALSE_POSITIVE_FIELDS = [
    "sample_id", "experiment_id", "condition", "detected_pii_type",
    "detected_bbox", "iou", "confidence", "ocr_detected",
    "ocr_recognition_correct", "reason", "source_row_count", "iou_threshold",
]
SUMMARY_FIELDS = [
    "scope", "experiment_id", "category", "condition",
    "total_ground_truth_annotations", "TP", "FP", "FN", "precision",
    "recall", "F1", "OCR_exact_presence_rate", "iou_threshold",
]


@dataclass(frozen=True)
class GroundTruth:
    sample_id: str
    condition: str
    pii_type: str
    bbox: tuple[float, float, float, float]
    expected_text: str = ""


@dataclass(frozen=True)
class Detection:
    sample_id: str
    experiment_id: str
    condition: str
    pii_type: str
    bbox: tuple[float, float, float, float]
    confidence: str = ""
    ocr_detected: str = ""
    ocr_recognition_correct: str = ""
    source_row_count: int = 1


def normalize_type(value) -> str:
    return str(value or "").strip().casefold()


def parse_bool(value):
    value = str(value or "").strip().casefold()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return ""


def parse_bbox(value) -> tuple[float, float, float, float] | None:
    if value in (None, "", []):
        return None
    try:
        parsed = value if isinstance(value, (list, tuple)) else json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        return None
    try:
        x, y, w, h = (float(item) for item in parsed)
    except (TypeError, ValueError):
        return None
    if w < 0 or h < 0:
        return None
    return x, y, w, h


def bbox_iou(first, second) -> float:
    """Return intersection-over-union for ``(x, y, width, height)`` boxes."""
    a, b = parse_bbox(first), parse_bbox(second)
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _annotation_rows(row: dict) -> list[GroundTruth]:
    annotations = []
    raw_annotations = row.get("ground_truth_annotations", "")
    if raw_annotations:
        try:
            parsed = json.loads(raw_annotations)
            if isinstance(parsed, list):
                annotations = parsed
        except json.JSONDecodeError:
            annotations = []
    if not annotations:
        fallback_type = normalize_type(row.get("ground_truth_pii_type"))
        fallback_box = parse_bbox(row.get("ground_truth_bbox"))
        if fallback_type and fallback_box is not None:
            annotations = [{"type": fallback_type, "bbox": fallback_box}]
    result = []
    for item in annotations:
        pii_type = normalize_type(item.get("type"))
        box = parse_bbox(item.get("bbox"))
        if pii_type and box is not None:
            result.append(GroundTruth(
                sample_id=str(row.get("sample_id", "")),
                condition=str(row.get("condition", "")),
                pii_type=pii_type,
                bbox=box,
                expected_text=str(row.get("expected_synthetic_text", "")),
            ))
    return result


def load_manifest(path: str) -> tuple[dict[str, list[GroundTruth]], dict[str, dict]]:
    annotations: dict[str, list[GroundTruth]] = {}
    metadata: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id", ""))
            metadata[sample_id] = row
            annotations[sample_id] = _annotation_rows(row)
    return annotations, metadata


def _raw_experiment_rows(path: str, sample_ids: set[str]) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle)
                if row.get("sample_id", "") in sample_ids and row.get("experiment_id", "")]


def _deduplicate_detections(rows: Iterable[dict], metadata: dict[str, dict]) -> tuple[list[Detection], dict[tuple, dict]]:
    grouped: dict[tuple, dict] = {}
    for row in rows:
        detected_type = normalize_type(row.get("detected_pii_type"))
        box = parse_bbox(row.get("detected_bbox"))
        if not detected_type or box is None or not parse_bool(row.get("pii_detected")):
            continue
        sample_id = str(row.get("sample_id", ""))
        experiment_id = str(row.get("experiment_id", ""))
        key = (experiment_id, sample_id, detected_type, tuple(round(v, 6) for v in box))
        if key not in grouped:
            grouped[key] = {
                "row": row, "count": 0,
            }
        grouped[key]["count"] += 1
    detections = []
    for key in sorted(grouped):
        item = grouped[key]
        row = item["row"]
        sample_id = str(row.get("sample_id", ""))
        detections.append(Detection(
            sample_id=sample_id,
            experiment_id=str(row.get("experiment_id", "")),
            condition=str(metadata.get(sample_id, {}).get("condition", row.get("condition", ""))),
            pii_type=normalize_type(row.get("detected_pii_type")),
            bbox=parse_bbox(row.get("detected_bbox")),
            confidence=str(row.get("confidence", "")),
            ocr_detected=str(row.get("ocr_detected", "")),
            ocr_recognition_correct=str(row.get("ocr_recognition_correct", "")),
            source_row_count=item["count"],
        ))
    return detections, grouped


def _sample_ocr_info(rows: Iterable[dict]) -> dict[tuple[str, str], tuple[object, object]]:
    values: dict[tuple[str, str], dict[str, list]] = {}
    for row in rows:
        key = (str(row.get("experiment_id", "")), str(row.get("sample_id", "")))
        entry = values.setdefault(key, {"ocr": [], "exact": []})
        for target, field in (("ocr", "ocr_detected"), ("exact", "ocr_recognition_correct")):
            parsed = parse_bool(row.get(field))
            if parsed != "":
                entry[target].append(parsed)
    result = {}
    for key, entry in values.items():
        result[key] = (
            True if True in entry["ocr"] else False if entry["ocr"] else "",
            True if True in entry["exact"] else False if entry["exact"] else "",
        )
    return result


def _json_box(box) -> str:
    return json.dumps([round(float(v), 6) for v in box], separators=(",", ":")) if box is not None else ""


def _candidate_key(iou: float, gt_index: int, detection_index: int):
    return (-iou, gt_index, detection_index)


def evaluate_group(ground_truth: list[GroundTruth], detections: list[Detection],
                   ocr_info: tuple[object, object], iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                   experiment_id: str = ""):
    """Evaluate one sample/run group and return annotation rows and FP rows."""
    eligible = []
    for gt_index, gt in enumerate(ground_truth):
        for detection_index, detection in enumerate(detections):
            if gt.pii_type == detection.pii_type:
                overlap = bbox_iou(gt.bbox, detection.bbox)
                if overlap >= iou_threshold:
                    eligible.append((overlap, gt_index, detection_index))
    assigned_gt, assigned_detection = set(), set()
    matches = {}
    for overlap, gt_index, detection_index in sorted(eligible, key=lambda p: _candidate_key(*p)):
        if gt_index not in assigned_gt and detection_index not in assigned_detection:
            assigned_gt.add(gt_index)
            assigned_detection.add(detection_index)
            matches[gt_index] = (detection_index, overlap)

    annotation_rows = []
    for gt_index, gt in enumerate(ground_truth):
        match = matches.get(gt_index)
        diagnostic = None
        if match is None:
            available = [(bbox_iou(gt.bbox, detection.bbox), index, detection)
                         for index, detection in enumerate(detections) if index not in assigned_detection]
            if available:
                diagnostic = max(available, key=lambda p: (p[0], -p[1]))
        if match is not None:
            detection_index, overlap = match
            detection = detections[detection_index]
            category_correct, detection_correct = True, True
        elif diagnostic is not None:
            overlap, detection_index, detection = diagnostic
            category_correct, detection_correct = (gt.pii_type == detection.pii_type), False
        else:
            overlap, detection = 0.0, None
            category_correct, detection_correct = False, False
        annotation_rows.append({
            "sample_id": gt.sample_id,
            "experiment_id": experiment_id or (detections[0].experiment_id if detections else ""),
            "condition": gt.condition,
            "ground_truth_pii_type": gt.pii_type,
            "detected_pii_type": detection.pii_type if detection else "",
            "ground_truth_bbox": _json_box(gt.bbox),
            "detected_bbox": _json_box(detection.bbox) if detection else "",
            "iou": round(overlap, 6),
            "confidence": detection.confidence if detection else "",
            "ocr_detected": ocr_info[0],
            "ocr_recognition_correct": ocr_info[1],
            "category_correct": category_correct,
            "detection_correct": detection_correct,
            "iou_threshold": iou_threshold,
        })

    false_positive_rows = []
    for index, detection in enumerate(detections):
        if index in assigned_detection:
            continue
        false_positive_rows.append({
            "sample_id": detection.sample_id,
            "experiment_id": detection.experiment_id,
            "condition": detection.condition,
            "detected_pii_type": detection.pii_type,
            "detected_bbox": _json_box(detection.bbox),
            "iou": round(max((bbox_iou(gt.bbox, detection.bbox) for gt in ground_truth), default=0.0), 6),
            "confidence": detection.confidence,
            "ocr_detected": detection.ocr_detected,
            "ocr_recognition_correct": detection.ocr_recognition_correct,
            "reason": "unmatched_detection",
            "source_row_count": detection.source_row_count,
            "iou_threshold": iou_threshold,
        })
    return annotation_rows, false_positive_rows


def _metric_row(scope, experiment_id, category, condition, gt_count, tp, fp, fn, ocr_rate, threshold):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"scope": scope, "experiment_id": experiment_id, "category": category,
            "condition": condition, "total_ground_truth_annotations": gt_count,
            "TP": tp, "FP": fp, "FN": fn, "precision": round(precision, 6),
            "recall": round(recall, 6), "F1": round(f1, 6),
            "OCR_exact_presence_rate": round(ocr_rate, 6), "iou_threshold": threshold}


def _ocr_rate(ocr_info, sample_keys, sample_conditions, sample_types,
              experiment_id="", category="", condition=""):
    selected = []
    for key in sorted(sample_keys):
        exp, sample_id = key
        if experiment_id and exp != experiment_id:
            continue
        if condition and sample_conditions.get(sample_id, "") != condition:
            continue
        if category and category not in sample_types.get(sample_id, set()):
            continue
        selected.append(key)
    values = [ocr_info[key][1] for key in selected if key in ocr_info and ocr_info[key][1] != ""]
    return sum(bool(v) for v in values) / len(values) if values else 0.0


def evaluate(manifest_path=DEFAULT_MANIFEST, raw_path=DEFAULT_RAW,
             results_dir=DEFAULT_RESULTS, iou_threshold=DEFAULT_IOU_THRESHOLD):
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be between 0 and 1")
    annotations, metadata = load_manifest(manifest_path)
    sample_ids = set(annotations)
    raw_rows = _raw_experiment_rows(raw_path, sample_ids)
    if not raw_rows:
        raise ValueError("No evaluator rows matching manifest sample IDs were found in the raw CSV")
    experiment_ids = sorted({row["experiment_id"] for row in raw_rows})
    ocr_info = _sample_ocr_info(raw_rows)
    all_annotations, all_false_positives = [], []
    sample_keys = [(experiment_id, sample_id) for experiment_id in experiment_ids
                   for sample_id in sorted(sample_ids)]
    sample_conditions = {sample_id: str(row.get("condition", "")) for sample_id, row in metadata.items()}
    sample_types = {sample_id: {gt.pii_type for gt in values} for sample_id, values in annotations.items()}
    for experiment_id in experiment_ids:
        experiment_rows = [row for row in raw_rows if row["experiment_id"] == experiment_id]
        detections, _ = _deduplicate_detections(experiment_rows, metadata)
        for sample_id in sorted(sample_ids):
            sample_detections = [d for d in detections if d.sample_id == sample_id]
            condition = str(metadata[sample_id].get("condition", ""))
            group_rows, fp_rows = evaluate_group(
                annotations.get(sample_id, []), sample_detections,
                ocr_info.get((experiment_id, sample_id), ("", "")), iou_threshold,
                experiment_id=experiment_id,
            )
            all_annotations.extend(group_rows)
            all_false_positives.extend(fp_rows)

    # Add experiment IDs to negative/empty-annotation groups through FPs only;
    # S07 correctly creates no annotation row and no FP when no PII was detected.
    summary = []
    def add_summary(scope, exp_id="", category="", condition="", ann_rows=None, fp_rows=None, sample_filter=None):
        ann_rows = ann_rows or []
        fp_rows = fp_rows or []
        if sample_filter:
            ann_rows = [r for r in ann_rows if sample_filter(r)]
            fp_rows = [r for r in fp_rows if sample_filter(r)]
        tp = sum(bool(r["detection_correct"]) for r in ann_rows)
        fn = len(ann_rows) - tp
        fp = len(fp_rows)
        keys = {(r["experiment_id"], r["sample_id"]) for r in ann_rows}
        keys.update((r["experiment_id"], r["sample_id"]) for r in fp_rows)
        if not keys:
            keys = set(sample_keys)
        summary.append(_metric_row(
            scope, exp_id, category, condition, len(ann_rows), tp, fp, fn,
            _ocr_rate(ocr_info, sample_keys, sample_conditions, sample_types,
                      experiment_id=exp_id, category=category, condition=condition),
            iou_threshold,
        ))

    add_summary("overall", ann_rows=all_annotations, fp_rows=all_false_positives)
    for experiment_id in experiment_ids:
        add_summary("experiment", experiment_id, ann_rows=all_annotations, fp_rows=all_false_positives,
                    sample_filter=lambda r, e=experiment_id: r["experiment_id"] == e)
    for category in REQUIRED_CATEGORIES:
        add_summary("category", category=category, ann_rows=all_annotations, fp_rows=all_false_positives,
                    sample_filter=lambda r, c=category: r.get("ground_truth_pii_type", r.get("detected_pii_type", "")) == c)
    conditions = sorted({str(row.get("condition", "")) for row in metadata.values()})
    for condition in conditions:
        add_summary("condition", condition=condition, ann_rows=all_annotations, fp_rows=all_false_positives,
                    sample_filter=lambda r, c=condition: r["condition"] == c)

    os.makedirs(results_dir, exist_ok=True)
    annotation_path = os.path.join(results_dir, "pii_annotation_results.csv")
    fp_path = os.path.join(results_dir, "pii_false_positive_results.csv")
    summary_path = os.path.join(results_dir, "pii_summary.csv")
    with open(annotation_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS); writer.writeheader(); writer.writerows(all_annotations)
    with open(fp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FALSE_POSITIVE_FIELDS); writer.writeheader(); writer.writerows(all_false_positives)
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS); writer.writeheader(); writer.writerows(summary)
    return annotation_path, fp_path, summary_path, len(all_annotations), len(all_false_positives), experiment_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    args = parser.parse_args()
    outputs = evaluate(args.manifest, args.raw, args.results_dir, args.iou_threshold)
    print(f"Wrote annotation-level PII evaluation: {outputs[0]}")
    print(f"Wrote false-positive evaluation: {outputs[1]}")
    print(f"Wrote summary metrics: {outputs[2]}")
    print(f"Processed {outputs[3]} ground-truth annotation records and {outputs[4]} unmatched detections across experiment IDs: {', '.join(outputs[5])}")


if __name__ == "__main__":
    main()

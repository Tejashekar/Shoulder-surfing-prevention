"""Evaluate synthetic screenshots listed in an annotation CSV.

Manifest columns: sample_id,image_path,ground_truth_pii_type,ground_truth_bbox,
ground_truth_annotations,expected_synthetic_text,condition. Types are
semicolon-separated; bbox is JSON `[x,y,w,h]`. Expected text must be synthetic.
Only categories, boxes, and correctness flags are written to experiment output.

v2.1: images go through `pii.pii_detector.detect_pii`, i.e. the same
preprocessing + Tesseract configuration as the live screen path, so the
reported metrics describe the deployed pipeline. Pass --raw to reproduce the
earlier behaviour (unprocessed image straight into Tesseract).
"""
import argparse
import csv
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from pii.pii_detector import regions_from_ocr_data
from pii.ocr_engine import extract_text_data, preprocess_for_ocr
from pii.patterns import PATTERNS
from experiment_logging import CsvLogger, PII_FIELDS


def normalize(text):
    return re.sub(r"\s+", " ", (text or "")).strip().casefold()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", help="CSV annotations for synthetic images only")
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--raw", action="store_true", help="Skip live-path preprocessing (legacy behaviour)")
    p.add_argument("--output", default=os.path.join(ROOT, "results", "pii_detection.csv"))
    args = p.parse_args()
    experiment_id = args.experiment_id or ("PII-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    out = CsvLogger(args.output, PII_FIELDS)
    totals = {"samples": 0, "ocr_correct": 0, "regions": 0}
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row["image_path"]
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(os.path.abspath(args.manifest)), path)
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Cannot read synthetic image: {path}")
            if args.raw:
                data = extract_text_data(image)
            else:
                data = extract_text_data(preprocess_for_ocr(image), original_shape=image.shape)
            detected = regions_from_ocr_data(data)
            totals["regions"] += len(detected)
            annotations = []
            if row.get("ground_truth_annotations", ""):
                annotations = json.loads(row["ground_truth_annotations"])
            truth_types = {str(x.get("type", "")).lower() for x in annotations if x.get("type")}
            if not truth_types:
                truth_types = {x.strip().lower() for x in row.get("ground_truth_pii_type", "").split(";") if x.strip()}
            boxes = {str(x.get("type", "")).lower(): x.get("bbox", "") for x in annotations}
            recognized = ""
            expected = row.get("expected_synthetic_text", "")
            ocr_text = normalize(" ".join(data.get("text", [])))
            if expected:
                recognized = normalize(expected) in ocr_text
                totals["samples"] += 1
                totals["ocr_correct"] += bool(recognized)
            ocr_detected = bool([t for t in data.get("text", []) if str(t).strip()])
            stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            # One row per detected region (so multiple regions per category are
            # kept for IoU matching) plus one negative row per undetected category.
            detected_categories = set()
            for region in detected:
                category = region["label"]
                detected_categories.add(category)
                out.append({"timestamp": stamp, "experiment_id": experiment_id,
                            "trial_id": row.get("sample_id", ""), "sample_id": row.get("sample_id", ""),
                            "condition": row.get("condition", ""),
                            "ground_truth_pii_type": category if category in truth_types else "",
                            "ground_truth_bbox": boxes.get(category, "") if category in truth_types else "",
                            "detected_pii_type": category, "detected_bbox": region.get("bbox", ""),
                            "confidence": region.get("confidence", ""), "ocr_detected": ocr_detected,
                            "ocr_recognition_correct": recognized, "pii_detected": True})
            for category in sorted(PATTERNS):
                if category in detected_categories:
                    continue
                out.append({"timestamp": stamp, "experiment_id": experiment_id,
                            "trial_id": row.get("sample_id", ""), "sample_id": row.get("sample_id", ""),
                            "condition": row.get("condition", ""),
                            "ground_truth_pii_type": category if category in truth_types else "",
                            "ground_truth_bbox": boxes.get(category, "") if category in truth_types else "",
                            "detected_pii_type": "", "detected_bbox": "", "confidence": "",
                            "ocr_detected": ocr_detected, "ocr_recognition_correct": recognized,
                            "pii_detected": False})
    print(f"Experiment {experiment_id}: {totals['regions']} regions over the manifest; "
          f"OCR exact-presence {totals['ocr_correct']}/{totals['samples']} (not a character-level OCR metric). "
          f"Run experiments.evaluate_pii_annotations for IoU-based P/R/F1.")


if __name__ == "__main__":
    main()

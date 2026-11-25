# Annotation-level PII/OCR evaluation

This evaluation is a post-hoc analysis of `results/pii_detection.csv`. It does not rerun or modify the OCR engine or the PII detector, and it never changes the raw CSV.

## Current raw evaluator limitation

The original manifest evaluator emitted one row for every configured regex category per image. It retained only the first detected region for each category, did not compare boxes, repeated image-level OCR flags on every category row, and appended multiple experiment runs to the same CSV. Therefore those rows are evidence of OCR/PII output, not annotation-level TP/FP/FN records.

## Matching rule

The clean evaluator reads `ground_truth_annotations` from the manifest. If that column is unavailable, it falls back to the single type and box columns. Each detected region is read from raw rows where `pii_detected=True` and a detected type and box are present. Exact duplicate rows within one experiment/sample/type/box are collapsed and their count is retained as `source_row_count`.

The default IoU threshold is **0.50**. For boxes represented as `(x, y, width, height)`:

`IoU = area(intersection) / area(union)`.

A true positive requires both normalized PII category equality and `IoU >= 0.50`. Matching is one-to-one. Eligible pairs are sorted deterministically by descending IoU, then ground-truth index, then detection index. A ground-truth annotation can therefore produce at most one TP.

For diagnostic visibility, an unmatched annotation records the highest-overlap remaining detection even when its category is wrong or its IoU is below threshold. Such a detection is not a TP. The annotation is an FN and the unmatched detection is an FP. Every unmatched detected region, including detections in S07, is written to the false-positive file. S08 is evaluated as two independent annotations: one email and one phone. S07 has zero ground-truth annotations; with no detected region it contributes neither TP, FP, nor FN.

Definitions:

- `TP`: a ground-truth annotation with one matched detection satisfying type and IoU rules.
- `FN`: a ground-truth annotation without such a match.
- `FP`: a detected region not consumed by a valid one-to-one match.
- `precision = TP / (TP + FP)`; zero when the denominator is zero.
- `recall = TP / (TP + FN)`; zero when the denominator is zero.
- `F1 = 2 * precision * recall / (precision + recall)`; zero when both are zero.
- `OCR_exact_presence_rate`: image-level rate from the raw evaluator's `ocr_recognition_correct` flag. It is counted once per experiment/sample image, never once per OCR token or annotation. It is an exact normalized-presence check, not character error rate.

## Outputs

- `results/pii_annotation_results.csv`: one row per ground-truth annotation per evaluator experiment.
- `results/pii_false_positive_results.csv`: one row per unmatched detected region.
- `results/pii_summary.csv`: overall, per-experiment, category, and condition metrics. Category rows include email, phone, password, aadhaar, account, and address. Condition rows are generated from the manifest, including normal, mixed-layout, small-font, low-contrast, large-font, partial-text, alternate-font, and noisy-background when present.

## Reproducible command

From `Version 2/`:

```powershell
python -m experiments.evaluate_pii_annotations `
  --manifest test_data\pii\manifest.csv `
  --raw results\pii_detection.csv `
  --iou-threshold 0.50
```

The output files are regenerated deterministically. The raw `pii_detection.csv` is never overwritten.

import unittest

from experiments.evaluate_pii_annotations import (
    Detection,
    GroundTruth,
    _deduplicate_detections,
    bbox_iou,
    evaluate_group,
)


def gt(pii_type="email", bbox=(0, 0, 10, 10), sample_id="S01", condition="normal"):
    return GroundTruth(sample_id, condition, pii_type, bbox)


def det(pii_type="email", bbox=(0, 0, 10, 10), sample_id="S01", experiment_id="E1"):
    return Detection(sample_id, experiment_id, "normal", pii_type, bbox, "90", "True", "True")


class AnnotationEvaluationTests(unittest.TestCase):
    def test_exact_match_is_true_positive(self):
        annotations, false_positives = evaluate_group([gt()], [det()], (True, True), experiment_id="E1")
        self.assertEqual(annotations[0]["detection_correct"], True)
        self.assertEqual(annotations[0]["category_correct"], True)
        self.assertAlmostEqual(annotations[0]["iou"], 1.0)
        self.assertEqual(false_positives, [])

    def test_correct_category_but_poor_iou_is_fn_and_fp(self):
        annotations, false_positives = evaluate_group([gt()], [det(bbox=(20, 20, 5, 5))], (True, True), experiment_id="E1")
        self.assertTrue(annotations[0]["category_correct"])
        self.assertFalse(annotations[0]["detection_correct"])
        self.assertEqual(len(false_positives), 1)

    def test_wrong_category_is_fn_and_fp(self):
        annotations, false_positives = evaluate_group([gt("email")], [det("phone")], (True, True), experiment_id="E1")
        self.assertEqual(annotations[0]["detected_pii_type"], "phone")
        self.assertFalse(annotations[0]["category_correct"])
        self.assertFalse(annotations[0]["detection_correct"])
        self.assertEqual(len(false_positives), 1)

    def test_false_positive_without_ground_truth(self):
        annotations, false_positives = evaluate_group([], [det("phone")], (True, True), experiment_id="E1")
        self.assertEqual(annotations, [])
        self.assertEqual(len(false_positives), 1)

    def test_false_negative(self):
        annotations, false_positives = evaluate_group([gt()], [], (True, False), experiment_id="E1")
        self.assertEqual(annotations[0]["experiment_id"], "E1")
        self.assertEqual(annotations[0]["detected_pii_type"], "")
        self.assertFalse(annotations[0]["detection_correct"])
        self.assertEqual(false_positives, [])

    def test_multiple_annotations_in_one_image(self):
        annotations, false_positives = evaluate_group(
            [gt("email", (0, 0, 10, 10), "S08", "mixed-layout"), gt("phone", (0, 20, 10, 10), "S08", "mixed-layout")],
            [det("email", (0, 0, 10, 10), "S08"), det("phone", (0, 20, 10, 10), "S08")],
            (True, True), experiment_id="E1")
        self.assertEqual(len(annotations), 2)
        self.assertEqual(sum(row["detection_correct"] for row in annotations), 2)
        self.assertEqual(false_positives, [])

    def test_negative_sample_with_no_detection(self):
        annotations, false_positives = evaluate_group([], [], (True, True), experiment_id="E1")
        self.assertEqual(annotations, [])
        self.assertEqual(false_positives, [])

    def test_duplicate_detection_rows_are_collapsed(self):
        rows = [
            {"sample_id": "S01", "experiment_id": "E1", "condition": "normal", "detected_pii_type": "email", "detected_bbox": "[0,0,10,10]", "confidence": "90", "ocr_detected": "True", "ocr_recognition_correct": "True", "pii_detected": "True"},
            {"sample_id": "S01", "experiment_id": "E1", "condition": "normal", "detected_pii_type": "email", "detected_bbox": "[0,0,10,10]", "confidence": "90", "ocr_detected": "True", "ocr_recognition_correct": "True", "pii_detected": "True"},
        ]
        detections, _ = _deduplicate_detections(rows, {"S01": {"condition": "normal"}})
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].source_row_count, 2)

    def test_iou_is_deterministic(self):
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (5, 5, 10, 10)), 25 / 175)


if __name__ == "__main__":
    unittest.main()

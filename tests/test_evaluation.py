import csv
import os
import re
import tempfile
import unittest
from evaluation.metrics import binary_metrics, descriptive
from experiment_logging import CsvLogger, RUNTIME_FIELDS, latency_ms
from risk.risk_engine import RiskEngine
from pii.patterns import PATTERNS
from pii.pii_detector import regions_from_ocr_data
from pii.tesseract_config import DEFAULT_TESSERACT_CMD, resolve_tesseract_cmd
from detection.proximity import estimate_proximity_score, split_user_and_observers
from detection.gaze_estimation import Kalman1D, alignment_score


def _ocr(words):
    n = len(words)
    return {"text": words, "left": [10 * i for i in range(n)], "top": [0] * n,
            "width": [8] * n, "height": [10] * n, "conf": [90] * n}


class EvaluationTests(unittest.TestCase):
    def test_tesseract_path_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = os.path.join(td, "ocr.json")
            with open(config_path, "w", encoding="utf-8") as config_file:
                config_file.write('{"tesseract_cmd": "D:/tools/tesseract.exe"}')
            self.assertEqual(resolve_tesseract_cmd(config_path, {}), "D:/tools/tesseract.exe")
            self.assertEqual(resolve_tesseract_cmd(config_path, {"TESSERACT_CMD": "E:/custom/tesseract.exe"}), "E:/custom/tesseract.exe")
            with open(config_path, "w", encoding="utf-8") as config_file:
                config_file.write("{}")
            self.assertEqual(resolve_tesseract_cmd(config_path, {}), DEFAULT_TESSERACT_CMD)

    def test_binary_metrics(self):
        m = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((m["TP"], m["TN"], m["FP"], m["FN"]), (1, 1, 1, 1))
        self.assertEqual(m["F1"], .5)

    def test_descriptive(self):
        d = descriptive([1, 2, 3, 4])
        self.assertEqual(d["median"], 2.5); self.assertEqual(d["p95"], 4)

    def test_risk_threshold_and_ablation(self):
        e = RiskEngine(features={"gaze": False, "proximity": False, "dwell": False, "pii": False})
        self.assertEqual(e.compute_risk(1, 100, 100, 100, now=1), 0)
        self.assertEqual(e.overlay_mode(34), "none")
        self.assertEqual(e.overlay_mode(40), "blur")
        self.assertEqual(e.overlay_mode(75), "fake")

    def test_default_risk_weights_and_elapsed_dwell(self):
        e = RiskEngine(features={"dwell": False})
        self.assertEqual(e.compute_risk(1, 100, 100, 100, now=10), 90.0)
        self.assertEqual(e.last_components["gaze"], 32.0)
        d = RiskEngine(features={"gaze": False, "proximity": False, "pii": False})
        d.compute_risk(2, 0, 0, 0, now=10)
        d.compute_risk(2, 0, 0, 0, now=10.25)
        self.assertAlmostEqual(d.duration_counter, 0.25)
        self.assertAlmostEqual(d.last_dwell_score, 1.0)

    def test_dwell_is_capped(self):
        d = RiskEngine(features={"gaze": False, "proximity": False, "pii": False})
        for t in range(0, 20):
            d.compute_risk(2, 0, 0, 0, now=10 + t)
        self.assertEqual(d.last_dwell_score, 20.0)

    def test_score_from_components_matches_live(self):
        e = RiskEngine(features={"dwell": False})
        live = e.compute_risk(2, 0.5, 40, 65, now=5)
        replay = RiskEngine.score_from_components(2, 50, 40, 65, 0.0, e.config)
        self.assertEqual(live, replay)
        content_only = dict(e.config, features={**e.config["features"], "gaze": False, "proximity": False, "crowd": False})
        self.assertEqual(RiskEngine.score_from_components(2, 50, 40, 65, 0.0, content_only), 19.5)

    def test_pii_patterns(self):
        self.assertRegex("test@example.org", re.compile(PATTERNS["email"], re.I))
        self.assertRegex("password: demo-only", re.compile(PATTERNS["password"], re.I))
        self.assertRegex("Account number: 123456789012", re.compile(PATTERNS["account"], re.I))
        self.assertRegex("Account number : 123456789012", re.compile(PATTERNS["account"], re.I))
        self.assertRegex("1111 2222 3333", PATTERNS["aadhaar"])
        self.assertIsNone(re.search(PATTERNS["aadhaar"], "123456789012"))
        self.assertRegex("+1 202 555 0101", PATTERNS["phone"])
        self.assertRegex("202-555-0101", PATTERNS["phone"])
        self.assertRegex("98765 43210", PATTERNS["phone"])
        # Live-screen noise that the old pattern matched:
        self.assertIsNone(re.search(PATTERNS["phone"], "2026-10-02T21:16:07"))
        self.assertIsNone(re.search(PATTERNS["phone"], "build 4.11.0.86"))
        self.assertIsNone(re.search(PATTERNS["phone"], "1920x1200 at 1786,1150"))

    def test_suppression_card_not_reported_as_phone_or_aadhaar(self):
        r = regions_from_ocr_data(_ocr(["Card:", "4111", "1111", "1111", "1111"]))
        self.assertEqual({x["label"] for x in r}, {"card"})
        r = regions_from_ocr_data(_ocr(["Account", "number:", "123456789012"]))
        self.assertEqual({x["label"] for x in r}, {"account"})

    def test_multitoken_bbox_and_no_text(self):
        data = {"text": ["demo", "user@example.com"], "left": [1, 10], "top": [2, 4],
                "width": [5, 40], "height": [8, 10]}
        r = regions_from_ocr_data(data)
        email = next(x for x in r if x["label"] == "email")
        self.assertEqual(email["bbox"], (10, 4, 40, 10)); self.assertNotIn("text", email)

    def test_repeated_token_offsets(self):
        # Same token twice: the second "a@b.io" must map to the second box.
        data = _ocr(["a@b.io", "x", "a@b.io"])
        boxes = sorted(x["bbox"][0] for x in regions_from_ocr_data(data))
        self.assertEqual(boxes, [0, 20])

    def test_user_observer_split_and_proximity(self):
        faces = [(0, 0, 50, 50), (100, 100, 200, 200), (300, 300, 20, 20)]
        user, observers = split_user_and_observers(faces)
        self.assertEqual(user, (100, 100, 200, 200))
        self.assertEqual(observers[0], (0, 0, 50, 50))
        self.assertEqual(estimate_proximity_score([(0, 0, 200, 200)], (480, 640)), 0.0)
        self.assertGreater(estimate_proximity_score(faces, (480, 640)), 0.0)

    def test_alignment_and_kalman(self):
        self.assertEqual(alignment_score(0.5, 0.4, 0.6), 1.0)
        self.assertEqual(alignment_score(0.7, 0.4, 0.6), 0.0)
        k = Kalman1D()
        self.assertEqual(k.update(1.0), 1.0)
        self.assertLess(k.update(0.0), 1.0)

    def test_csv_allowlist_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "runtime.csv")
            logger = CsvLogger(path, RUNTIME_FIELDS)
            logger.append({"frame_id": 2, "pii_types": ["email"], "ocr_text": "should not log"})
            with open(path, newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["frame_id"], "2")
            with open(path, encoding="utf-8") as f:
                self.assertNotIn("should not log", f.read())

    def test_csv_header_migration(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "runtime.csv")
            old = RUNTIME_FIELDS[:5]
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(old); w.writerow(["t", "s", "1", "2", "True"])
            CsvLogger(path, RUNTIME_FIELDS).append({"frame_id": 3})
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["frame_id"], "1"); self.assertEqual(rows[0]["overlay_rendered"], "")
            self.assertEqual(rows[1]["frame_id"], "3")

    def test_latency_measurement(self):
        self.assertAlmostEqual(latency_ms(10.0, 10.125), 125.0)
        with self.assertRaises(ValueError):
            latency_ms(2.0, 1.0)


if __name__ == "__main__":
    unittest.main()

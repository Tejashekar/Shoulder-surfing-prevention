import unittest
from unittest.mock import patch

import numpy as np

from overlay import overlay_engine


class OverlayRateLimitTests(unittest.TestCase):
    def setUp(self):
        overlay_engine._window_created = False
        overlay_engine._last_render_time = 0.0

    @patch.object(overlay_engine.cv2, "waitKey")
    @patch.object(overlay_engine, "_set_protected_window_visible")
    @patch.object(overlay_engine.cv2, "imshow")
    @patch.object(overlay_engine.cv2, "namedWindow")
    @patch.object(overlay_engine, "_screenshot")
    @patch.object(overlay_engine.time, "monotonic", side_effect=[1.0, 1.01, 1.05, 1.30, 1.31])
    def test_reuses_window_and_reports_throttled_calls(
        self, _clock, screenshot, named_window, imshow, set_visible, _wait_key
    ):
        screenshot.return_value = np.zeros((40, 40, 3), dtype=np.uint8)
        regions = [{"bbox": (5, 5, 20, 20), "label": "test", "sensitivity": 50}]

        first = overlay_engine.apply_blur(regions)
        second = overlay_engine.apply_blur(regions)
        third = overlay_engine.apply_blur(regions)

        self.assertTrue(first.rendered); self.assertEqual(first.reason, "rendered")
        self.assertFalse(second.rendered); self.assertEqual(second.reason, "throttled")
        self.assertTrue(third.rendered)
        self.assertGreaterEqual(first.total_ms, 0.0)
        self.assertEqual(screenshot.call_count, 2)
        self.assertEqual(named_window.call_count, 1)
        self.assertEqual(imshow.call_count, 2)
        self.assertEqual(set_visible.call_args_list[0].args, (False,))
        self.assertEqual(set_visible.call_args_list[1].args, (True,))

    def test_no_regions_is_not_a_render(self):
        self.assertEqual(overlay_engine.apply_blur([]).reason, "no-regions")

    def test_redaction_modifies_only_region(self):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        out = overlay_engine.redact_image(img.copy(), [{"bbox": (10, 10, 20, 20), "label": "email", "sensitivity": 60}], 80, "fake")
        self.assertTrue((out[0:10, :, :] == 128).all())
        self.assertFalse((out[10:30, 10:30, :] == 128).all())

    @patch.object(overlay_engine.os, "name", "nt")
    @patch.object(overlay_engine.ctypes, "WinDLL", create=True)
    def test_hides_native_window_without_activating_it(self, win_dll):
        user32 = win_dll.return_value
        user32.FindWindowW.return_value = 123
        overlay_engine._set_protected_window_visible(False)
        user32.ShowWindow.assert_called_once_with(123, 0)


if __name__ == "__main__":
    unittest.main()

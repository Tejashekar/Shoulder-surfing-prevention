"""Selective screen redaction preview.

v2.1 changes:
- `apply_selective_redaction` returns a `RenderResult` so the caller can tell a
  real capture/render apart from a throttled early return. Overlay-latency
  statistics must only use rendered events.
- PyAutoGUI is imported lazily (headless tests, no display).
"""
import cv2
import numpy as np
import time
import ctypes
import os
from dataclasses import dataclass


_WINDOW_NAME = "Protected Screen"
_MIN_RENDER_INTERVAL_SECONDS = 0.2
_window_created = False
_last_render_time = 0.0


@dataclass(frozen=True)
class RenderResult:
    rendered: bool
    reason: str = ""          # "rendered" | "throttled" | "no-regions"
    capture_ms: float = 0.0   # screenshot time
    render_ms: float = 0.0    # redaction + imshow + waitKey time
    total_ms: float = 0.0


def _screenshot():
    import pyautogui
    return pyautogui.screenshot()


def _set_protected_window_visible(visible):
    """Use Win32 visibility directly so desktop capture excludes this window."""
    if os.name == "nt":
        try:
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
            user32.FindWindowW.restype = wintypes.HWND
            user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
            user32.ShowWindow.restype = wintypes.BOOL
            hwnd = user32.FindWindowW(None, _WINDOW_NAME)
            if hwnd:
                # SW_HIDE=0; SW_SHOWNOACTIVATE=4 restores without stealing focus.
                user32.ShowWindow(hwnd, 4 if visible else 0)
                return
        except (AttributeError, OSError):
            pass

    cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE, 1 if visible else 0)


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _fake_value(label, original_text=""):
    return "REDACTED"


def _normalize_region(region):
    if isinstance(region, dict):
        bbox = region.get("bbox", (0, 0, 0, 0))
        label = region.get("label", "")
        text = region.get("text", "")
        sensitivity = float(region.get("sensitivity", 50))
        return bbox, label, text, sensitivity

    if len(region) >= 5:
        x, y, w, h, label = region[:5]
        text = region[5] if len(region) > 5 else ""
        return (x, y, w, h), label, text, 50.0

    return (0, 0, 0, 0), "", "", 50.0


def redact_image(img, regions, threat_score=0.0, mode="blur"):
    """Apply blur/fake redaction to `regions` of `img` in place and return it."""
    for region in regions:
        (x, y, w, h), label, original_text, sensitivity = _normalize_region(region)
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(w), int(h)
        roi = img[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        alpha = _clamp(0.25 + (float(threat_score) / 100.0) * 0.6 + sensitivity / 300.0, 0.25, 0.95)

        if mode == "fake":
            overlay = roi.copy()
            overlay[:] = (245, 245, 245)
            fake_text = _fake_value(label, original_text)
            font_scale = max(0.35, min(1.0, w / max(len(fake_text), 1) / 12.0))
            cv2.putText(overlay, fake_text, (5, max(15, int(h * 0.65))),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 2, cv2.LINE_AA)
            img[y:y + h, x:x + w] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
        else:
            kernel = int(15 + alpha * 40)
            kernel = kernel if kernel % 2 == 1 else kernel + 1
            blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
            img[y:y + h, x:x + w] = cv2.addWeighted(blurred, alpha, roi, 1 - alpha, 0)
    return img


def apply_selective_redaction(regions, threat_score=0.0, mode="blur"):
    global _window_created, _last_render_time

    if not regions:
        return RenderResult(False, "no-regions")

    now = time.monotonic()
    if _window_created and now - _last_render_time < _MIN_RENDER_INTERVAL_SECONDS:
        return RenderResult(False, "throttled")

    t0 = time.perf_counter()
    if _window_created:
        _set_protected_window_visible(False)
        cv2.waitKey(1)
    try:
        screenshot = _screenshot()
    finally:
        if _window_created:
            _set_protected_window_visible(True)
            cv2.waitKey(1)
    t1 = time.perf_counter()

    img = np.array(screenshot)
    redact_image(img, regions, threat_score=threat_score, mode=mode)

    if not _window_created:
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
        _window_created = True
    cv2.imshow(_WINDOW_NAME, img)
    cv2.waitKey(1)
    t2 = time.perf_counter()
    _last_render_time = time.monotonic()
    return RenderResult(True, "rendered",
                        capture_ms=(t1 - t0) * 1000.0,
                        render_ms=(t2 - t1) * 1000.0,
                        total_ms=(t2 - t0) * 1000.0)


def apply_blur(regions, threat_score=0.0, mode="blur"):
    return apply_selective_redaction(regions, threat_score=threat_score, mode=mode)

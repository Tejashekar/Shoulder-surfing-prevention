"""Screen capture and Tesseract OCR.

`preprocess_for_ocr` is the single preprocessing path shared by the live loop
and the offline synthetic evaluation, so reported PII metrics describe the same
pipeline that runs at runtime. Adaptive thresholding replaces the earlier fixed
threshold (150), which erased low-contrast grey text, and dark-mode
screens are inverted before binarisation.
"""
import cv2
import numpy as np
import pytesseract

from pii.tesseract_config import resolve_tesseract_cmd

pytesseract.pytesseract.tesseract_cmd = resolve_tesseract_cmd()

# Tesseract page-segmentation mode 6 = assume a uniform block of text; this is
# more stable than the default (3) for UI screenshots with scattered widgets.
TESSERACT_CONFIG = "--psm 6"


def preprocess_for_ocr(image):
    """Convert a BGR/RGB/greyscale image into a binarised greyscale image."""
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey = image
    # Tesseract expects dark text on a light background. Dark-mode UIs are the
    # opposite; detect polarity from the median intensity and invert if needed.
    if np.median(grey) < 110:
        grey = cv2.bitwise_not(grey)
    # Upscale small text slightly; Tesseract prefers ~30 px x-height.
    h, w = grey.shape[:2]
    if w < 1600:
        grey = cv2.resize(grey, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    binary = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)
    return cv2.medianBlur(binary, 3)


def _scale_factor(original_shape, processed_shape):
    return original_shape[1] / processed_shape[1], original_shape[0] / processed_shape[0]


def capture_screen(persist=False):
    """Grab the desktop via PyAutoGUI and return it as a raw BGR image.

    Preprocessing is applied by `detect_pii` so that live and offline paths are
    identical. PyAutoGUI is imported lazily so headless test environments can
    import this module without a display.
    """
    import pyautogui
    screenshot = pyautogui.screenshot()
    img = np.array(screenshot)  # RGB
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def extract_text_data(image, original_shape=None):
    """Run Tesseract and return word-level data in the *original* pixel space.

    `preprocess_for_ocr` may upscale the image; bounding boxes are rescaled back
    so that overlay regions and evaluation boxes line up with the source.
    """
    data = pytesseract.image_to_data(image, config=TESSERACT_CONFIG,
                                     output_type=pytesseract.Output.DICT)
    if original_shape is not None:
        sx, sy = _scale_factor(original_shape, image.shape)
        data["left"] = [int(v * sx) for v in data["left"]]
        data["width"] = [int(v * sx) for v in data["width"]]
        data["top"] = [int(v * sy) for v in data["top"]]
        data["height"] = [int(v * sy) for v in data["height"]]
    return data

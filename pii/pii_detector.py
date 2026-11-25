import re
from pii.patterns import PATTERNS, SUPPRESSES

SENSITIVITY = {
    "aadhaar": 90,
    "pan": 80,
    "card": 95,
    "email": 60,
    "phone": 65,
    "password": 95,
    "account": 90,
    "address": 60,
}


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def regions_from_ocr_data(data, include_text=False):
    """Map regex matches in OCR words to union bounding boxes (testable offline).

    Matching runs on the space-joined token string. Character offsets of every
    token are computed once so each regex match maps to the tokens it overlaps.
    Stronger categories suppress weaker ones on the same span (e.g. a card
    number is not also reported as a phone number).
    """
    words = [str(word or "") for word in data.get("text", [])]
    text = " ".join(words)

    token_spans, cursor = [], 0
    for word in words:
        token_spans.append((cursor, cursor + len(word)))
        cursor += len(word) + 1

    raw_matches = []  # (category, start, end, matched_text)
    for key, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            # Box only the sensitive value when the pattern exposes one.
            if "value" in match.groupdict() and match.group("value") is not None:
                start, end, matched = match.start("value"), match.end("value"), match.group("value")
            else:
                start, end, matched = match.start(), match.end(), match.group(0)
            raw_matches.append((key, start, end, matched))

    kept = []
    for key, start, end, matched in raw_matches:
        suppressed = any(
            other_key != key
            and key in SUPPRESSES.get(other_key, ())
            and _overlaps((start, end), (other_start, other_end))
            for other_key, other_start, other_end, _ in raw_matches
        )
        if not suppressed:
            kept.append((key, start, end, matched))

    regions = []
    for key, start, end, matched in kept:
        spans = [i for i, (s, e) in enumerate(token_spans) if words[i] and s < end and e > start]
        if not spans:
            continue
        x0 = min(int(data["left"][i]) for i in spans)
        y0 = min(int(data["top"][i]) for i in spans)
        x1 = max(int(data["left"][i]) + int(data["width"][i]) for i in spans)
        y1 = max(int(data["top"][i]) + int(data["height"][i]) for i in spans)
        region = {"bbox": (x0, y0, x1 - x0, y1 - y0), "label": key,
                  "sensitivity": SENSITIVITY.get(key, 50)}
        try:
            confidences = [float(data["conf"][i]) for i in spans if float(data["conf"][i]) >= 0]
            region["confidence"] = sum(confidences) / len(confidences) if confidences else ""
        except (KeyError, TypeError, ValueError):
            region["confidence"] = ""
        if include_text:
            region["text"] = matched
        regions.append(region)
    return regions


def detect_pii(image=None, include_text=False, preprocess=True):
    """Detect categories and union boxes. OCR text is discarded by default.

    `image` permits synthetic image tests without capturing the desktop and is
    passed through the same preprocessing as the live screen capture so that
    offline evaluation reflects the live pipeline. Set include_text=True only
    for controlled debugging; never use it in study logs.
    """
    from pii.ocr_engine import capture_screen, extract_text_data, preprocess_for_ocr
    if image is None:
        image = capture_screen()
    original_shape = image.shape
    if preprocess:
        image = preprocess_for_ocr(image)
    data = extract_text_data(image, original_shape=original_shape)
    return regions_from_ocr_data(data, include_text=include_text)

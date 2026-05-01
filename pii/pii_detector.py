import re
from pii.patterns import PATTERNS
from pii.ocr_engine import capture_screen, extract_text_data

def detect_pii():
    image = capture_screen()
    data = extract_text_data(image)

    regions = []

    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue

        for key, pattern in PATTERNS.items():
            if re.match(pattern, text):
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]

                regions.append((x, y, w, h, key))

    return regions
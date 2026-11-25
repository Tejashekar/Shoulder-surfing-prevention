"""Generate synthetic OCR screenshots and annotations (not experimental results).

Two sets are produced:

* `test_data/pii/`          — the original 14-image starter set (kept for
                               comparability with the pilot paper).
* `test_data/pii_extended/` — a larger held-out set (default 60 images) built
                               from a seeded random combination of layouts,
                               fonts, sizes, themes, PII categories, and
                               *distractor* strings (dates, IPs, version numbers,
                               timestamps, order IDs, coordinates) that are
                               negatives for every category. Use --seed to make
                               a different draw; report the seed with results.

All values are fake. No real personal data is used.
"""
import argparse
import csv
import json
import os
import random
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BASE = os.path.join(ROOT, "test_data", "pii")
BASE_EXT = os.path.join(ROOT, "test_data", "pii_extended")

FONT_CANDIDATES = {
    "sans": ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    "mono": ["C:/Windows/Fonts/cour.ttf", "C:/Windows/Fonts/consola.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
    "serif": ["C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/georgia.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"],
}


def load_font(family, size):
    for path in FONT_CANDIDATES.get(family, []):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------- starter set
def _after_label(text):
    """Split point: after the first ':' / '=' / '#' / '.' label separator, if any."""
    for sep in (": ", "= ", "# ", ":", "=", "#"):
        if sep in text:
            return sep
    return ""


CASES = [
    ("email", "Email: demo.user@example.invalid", "normal"),
    ("phone", "Phone: +1 202 555 0101", "normal"),
    ("password", "Password: sample-only-42", "normal"),
    ("aadhaar", "ID-like: 1111 2222 3333", "normal"),
    ("account", "Account number: 123456789012", "normal"),
    ("address", "Address: 12 Example Street", "normal"),
    ("", "Meeting notes: bring the blue notebook", "normal"),
    ("email;phone", "Email: demo.user@example.invalid\nPhone: 202-555-0101", "mixed-layout"),
    ("password", "Password: sample-only-42", "small-font"),
    ("email", "Email: demo.user@example.invalid", "low-contrast"),
    ("account", "Account number: 123456789012", "large-font"),
    ("phone", "Phone: 202-555-01", "partial-text"),
    ("address", "Address: 12 Example Street", "alternate-font"),
    ("email", "Email: demo.user@example.invalid", "noisy-background"),
]


def generate_starter():
    os.makedirs(BASE, exist_ok=True)
    manifest = os.path.join(BASE, "manifest.csv")
    font_normal, font_small = load_font("sans", 30), load_font("sans", 15)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "image_path", "ground_truth_pii_type", "ground_truth_bbox", "ground_truth_annotations", "expected_synthetic_text", "condition"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for idx, (category, text, condition) in enumerate(CASES, 1):
            image = Image.new("RGB", (1000, 240), "white")
            draw = ImageDraw.Draw(image)
            font = font_small if condition == "small-font" else font_normal
            fill = (175, 175, 175) if condition == "low-contrast" else (20, 20, 20)
            if condition == "large-font":
                font = load_font("sans", 46)
            if condition == "alternate-font":
                font = load_font("mono", 30)
            if condition == "partial-text":
                text = "Phone: 202-555-01"
            line_boxes, y = [], 45
            for line in text.split("\n"):
                draw.text((24, y), line, font=font, fill=fill)
                # Value-only ground truth (the span to redact), consistent with the extended set.
                sep = _after_label(line)
                label = line.split(sep, 1)[0] + sep if sep else ""
                prefix_w = draw.textlength(label, font=font) if label else 0
                box = draw.textbbox((24 + prefix_w, y), line[len(label):], font=font)
                line_boxes.append([int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])])
                y += 70
            if condition == "noisy-background":
                for px in range(0, 1000, 19):
                    for py in range(0, 240, 17):
                        draw.point((px, py), fill=(235, 235, 235))
            path = f"sample_{idx:02}.png"; image.save(os.path.join(BASE, path))
            categories = category.split(";") if category else []
            annotations = [{"type": cat, "bbox": box} for cat, box in zip(categories, line_boxes)]
            writer.writerow({"sample_id": f"S{idx:02}", "image_path": path, "ground_truth_pii_type": category,
                             "ground_truth_bbox": json.dumps(annotations[0]["bbox"]) if annotations else "",
                             "ground_truth_annotations": json.dumps(annotations),
                             "expected_synthetic_text": text, "condition": condition})
    print(f"Generated {len(CASES)} starter fixture images in {BASE}; these are test inputs, not measurements.")


# --------------------------------------------------------------- extended set
PII_LINES = {
    "email": ["Email: priya.k@example.invalid", "Contact: support.desk@mail-example.org", "Reply to: r.mehta042@mail.invalid",
              "User: dev_team+alerts@example.co.in", "mail: arjun.rao@campus.example.edu"],
    "phone": ["Phone: +91 98765 43210", "Mobile: 98450 12345", "Call: +1 202 555 0147", "Tel: 080-2345-6789", "Phone: (415) 555-0199"],
    "password": ["Password: Tr0ub4dor&3", "PIN: 4821", "passcode = winter-2026", "Password: hunter2!", "PIN : 775531"],
    "aadhaar": ["Aadhaar: 2345 6789 0123", "UID 9876 5432 1098", "ID: 4567-8901-2345", "Aadhaar No. 1122 3344 5566"],
    "pan": ["PAN: ABCDE1234F", "PAN No: PQRSX9876L", "Permanent Account Number: KLMNO4567P"],
    "card": ["Card: 4111 1111 1111 1111", "Visa 5500-0000-0000-0004", "Card number 3400 0000 0000 0009"],
    "account": ["Account number: 123456789012", "A/c No: 00112233445566", "Acct #: 9988776655", "Account : 3456789012"],
    "address": ["Address: 12 Example Street", "Addr: 221 Baker Road", "Office: 45 MG Road", "Home: 7 Lake View Lane", "Ship to: 1600 Elm Avenue"],
}
DISTRACTORS = [
    "Build 4.11.0.86 (release)", "Updated 2026-10-02T21:16:07", "Order ID: ORD-20261002-7781", "IP 192.168.1.254",
    "Window 1920x1200 at 1786,1150", "Version 2.3.1", "Invoice #INV-000123", "Temperature 36.6 C", "Meeting at 14:30 on 02/10/2026",
    "Ticket 55831 closed", "Commit 7f3a9c2", "Latency 783.54 ms (p95)", "Quarter revenue 1,234,567", "Room 404, Block C",
    "Score 98/100", "Seat 12A, Gate 7", "Pages 5-8 of 22", "Ref: 2026/CS/118", "Batch 2027 - CSE", "CGPA 9.1",
]
VALUE_SEPARATOR = {cat: _after_label for cat in ("email", "phone", "password", "aadhaar", "pan", "card", "account", "address")}

THEMES = {"light": ((255, 255, 255), (20, 20, 20)), "dark": ((32, 34, 40), (230, 230, 230)),
          "low-contrast": ((255, 255, 255), (170, 170, 170)), "paper": ((248, 244, 232), (40, 40, 40))}


def generate_extended(count, seed):
    rnd = random.Random(seed)
    os.makedirs(BASE_EXT, exist_ok=True)
    manifest = os.path.join(BASE_EXT, "manifest.csv")
    fields = ["sample_id", "image_path", "ground_truth_pii_type", "ground_truth_bbox", "ground_truth_annotations", "expected_synthetic_text", "condition"]
    categories = list(PII_LINES)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for idx in range(1, count + 1):
            width, height = rnd.choice([(1000, 320), (1280, 400), (800, 480), (1600, 360)])
            theme = rnd.choice(list(THEMES))
            family = rnd.choice(["sans", "sans", "mono", "serif"])
            size = rnd.choice([14, 18, 22, 28, 34])
            bg, fg = THEMES[theme]
            n_pii = rnd.choice([0, 1, 1, 2, 2, 3])
            n_dis = rnd.choice([1, 2, 3])
            chosen = rnd.sample(categories, n_pii)
            lines = [(cat, rnd.choice(PII_LINES[cat])) for cat in chosen] + [("", d) for d in rnd.sample(DISTRACTORS, n_dis)]
            rnd.shuffle(lines)
            image = Image.new("RGB", (width, height), bg)
            draw = ImageDraw.Draw(image)
            font = load_font(family, size)
            x0 = rnd.choice([20, 40, 80])
            y = rnd.choice([20, 40])
            annotations, texts = [], []
            for cat, text in lines:
                draw.text((x0, y), text, font=font, fill=fg)
                if cat:
                    # Ground truth is the sensitive *value*, i.e. the span that
                    # should be redacted, not the descriptive label before it.
                    sep = VALUE_SEPARATOR[cat](text)
                    label = text.split(sep, 1)[0] + sep if sep else ""
                    value_text = text[len(label):]
                    prefix_w = draw.textlength(label, font=font) if label else 0
                    box = draw.textbbox((x0 + prefix_w, y), value_text, font=font)
                    annotations.append({"type": cat, "bbox": [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]})
                texts.append(text)
                y += int(size * 2.2)
            if rnd.random() < 0.2:
                speck = tuple(c - 20 if c > 100 else c + 20 for c in bg)
                for px in range(0, width, 23):
                    for py in range(0, height, 19):
                        draw.point((px, py), fill=speck)
                theme += "+noise"
            path = f"ext_{idx:03}.png"; image.save(os.path.join(BASE_EXT, path))
            writer.writerow({"sample_id": f"E{idx:03}", "image_path": path,
                             "ground_truth_pii_type": ";".join(a["type"] for a in annotations),
                             "ground_truth_bbox": json.dumps(annotations[0]["bbox"]) if annotations else "",
                             "ground_truth_annotations": json.dumps(annotations),
                             "expected_synthetic_text": "\n".join(texts),
                             "condition": f"{theme}|{family}|{size}px"})
    print(f"Generated {count} extended fixture images (seed {seed}) in {BASE_EXT}; these are test inputs, not measurements.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", choices=("starter", "extended", "both"), default="both")
    p.add_argument("--count", type=int, default=60)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    if a.set in ("starter", "both"):
        generate_starter()
    if a.set in ("extended", "both"):
        generate_extended(a.count, a.seed)


if __name__ == "__main__":
    main()

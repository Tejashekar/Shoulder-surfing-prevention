"""OCR smoke check for a supplied synthetic image; never prints OCR text."""
import argparse
import os
import cv2
import pytesseract
from pytesseract import Output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a synthetic test image")
    args = parser.parse_args()
    path = os.path.abspath(args.image)
    if "test_data" not in path.lower():
        parser.error("only use a synthetic image stored under test_data")
    image = cv2.imread(path)
    if image is None:
        parser.error(f"could not read image: {path}")
    words = [w for w in pytesseract.image_to_data(image, output_type=Output.DICT)["text"] if str(w).strip()]
    print(f"OCR smoke check complete: {len(words)} non-empty tokens detected; text was not displayed or saved.")

if __name__ == "__main__": main()

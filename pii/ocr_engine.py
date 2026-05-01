import pytesseract
import pyautogui
import cv2
import numpy as np

# SET YOUR PATH HERE
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def capture_screen():
    # Take screenshot
    screenshot = pyautogui.screenshot()

    # Convert to numpy array
    img = np.array(screenshot)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply threshold (improves OCR)
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]

    # Reduce noise
    img = cv2.medianBlur(img, 3)

    return img


def extract_text_data(image):
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT
    )
    return data
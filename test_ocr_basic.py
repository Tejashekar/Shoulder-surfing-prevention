import pytesseract
import pyautogui
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = pyautogui.screenshot()
img = np.array(img)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

text = pytesseract.image_to_string(img)

print("RAW OCR OUTPUT:")
print(repr(text))
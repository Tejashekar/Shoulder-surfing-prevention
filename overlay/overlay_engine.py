import cv2
import pyautogui
import numpy as np

def apply_blur(regions):
    # Take screenshot
    screenshot = pyautogui.screenshot()
    img = np.array(screenshot)

    # Apply blur to detected regions
    for (x, y, w, h, label) in regions:
        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        blurred = cv2.GaussianBlur(roi, (35, 35), 0)
        img[y:y+h, x:x+w] = blurred

    # Show result
    cv2.imshow("Protected Screen", img)
    cv2.waitKey(1)
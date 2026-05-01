from pii.pii_detector import detect_pii

regions = detect_pii()

for r in regions:
    print("Detected:", r)


from pii.pii_detector import detect_pii
from overlay.overlay_engine import apply_blur

while True:
    regions = detect_pii()

    if regions:
        print("Detected:", regions)

    apply_blur(regions)
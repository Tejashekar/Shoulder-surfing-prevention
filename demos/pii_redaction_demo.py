import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pii.pii_detector import detect_pii
from overlay.overlay_engine import apply_blur

regions = detect_pii()

for region in regions:
    print("Detected:", region)

if regions:
    apply_blur(regions, threat_score=80, mode="fake")
import cv2

from detection.face_detection import FaceDetector
from detection.gaze_estimation import GazeEstimator
from detection.proximity import estimate_proximity
from risk.risk_engine import RiskEngine

from pii.pii_detector import detect_pii
from overlay.overlay_engine import apply_blur

# Initialize modules
detector = FaceDetector()
gaze = GazeEstimator()
risk_engine = RiskEngine()

frame_count = 0
gaze_flag = False

while True:
    frame = detector.get_frame()
    if frame is None:
        break

    # Resize for performance
    frame = cv2.resize(frame, (640, 480))

    faces = detector.detect_faces(frame)
    face_count = len(faces)

    # Run gaze less frequently (performance optimization)
    frame_count += 1
    if face_count > 1 and frame_count % 5 == 0:
        gaze_flag = gaze.detect_intruder(frame)
    elif face_count <= 1:
        gaze_flag = False

    # Proximity
    proximity = estimate_proximity(faces)

    # Risk calculation
    score = risk_engine.compute_risk(face_count, gaze_flag, proximity)
    level = risk_engine.get_risk_level(score)

    # Draw faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show risk level
    color = (0, 255, 0) if level == "LOW" else (0, 255, 255) if level == "MEDIUM" else (0, 0, 255)
    cv2.putText(frame, f"RISK: {level}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Webcam Monitor", frame)

    # 🔥 ONLY run OCR + blur if risk is HIGH
    if level == "HIGH":
        regions = detect_pii()

        if regions:
            print("PII Detected:", regions)
            apply_blur(regions)

    # Exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

detector.release()
cv2.destroyAllWindows()
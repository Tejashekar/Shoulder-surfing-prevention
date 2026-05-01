from detection.face_detection import FaceDetector
from detection.gaze_estimation import GazeEstimator
from detection.proximity import estimate_proximity
from risk.risk_engine import RiskEngine
import cv2

detector = FaceDetector()
gaze = GazeEstimator()
risk_engine = RiskEngine()

while True:
    frame = detector.get_frame()
    if frame is None:
        break

    faces = detector.detect_faces(frame)
    face_count = len(faces)

    gaze_flag = gaze.detect_intruder(frame)
    proximity = estimate_proximity(faces)

    score = risk_engine.compute_risk(face_count, gaze_flag, proximity)
    level = risk_engine.get_risk_level(score)

    # Draw faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display risk
    text = f"RISK: {level}"
    color = (0, 255, 0) if level == "LOW" else (0, 255, 255) if level == "MEDIUM" else (0, 0, 255)

    cv2.putText(frame, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    print(f"Faces: {face_count} | Gaze: {gaze_flag} | Proximity: {proximity} | Score: {score} | Level: {level}")

    cv2.imshow("Risk Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

detector.release()
cv2.destroyAllWindows()
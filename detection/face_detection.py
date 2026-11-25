import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Ensure correct format
        frame = frame.astype('uint8')
        return frame

    def detect_faces(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)

        faces = []

        if result.detections:
            h, w, _ = frame.shape

            for det in result.detections:
                bbox = det.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                width = max(0, min(width, w - x))
                height = max(0, min(height, h - y))

                faces.append((x, y, width, height))

        return faces

    def release(self):
        self.cap.release()
        self.detector.close()

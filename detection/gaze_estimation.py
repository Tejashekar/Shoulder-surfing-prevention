import cv2
import mediapipe as mp

class GazeEstimator:
    def __init__(self):
        self.mp_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_mesh.FaceMesh(refine_landmarks=True)

    def detect_intruder(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            return False

        faces = result.multi_face_landmarks

        if len(faces) < 2:
            return False

        # Skip main user (face 0)
        for i, face in enumerate(faces):
            if i == 0:
                continue

            # Get key landmarks
            nose = face.landmark[1]
            left_face = face.landmark[234]
            right_face = face.landmark[454]

            # Check if face is centered (looking forward)
            center = (left_face.x + right_face.x) / 2

            if abs(nose.x - center) < 0.05:
                return True  # looking forward

        return False
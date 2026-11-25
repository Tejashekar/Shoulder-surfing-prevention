import cv2
import mediapipe as mp


class Kalman1D:
    def __init__(self, process_variance=0.02, measurement_variance=0.08):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = None
        self.error_covariance = 1.0

    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            return measurement

        self.error_covariance += self.process_variance
        gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimate = self.estimate + gain * (measurement - self.estimate)
        self.error_covariance = (1 - gain) * self.error_covariance
        return self.estimate


def alignment_score(nose_x, left_x, right_x, tolerance=0.12):
    """Face-pose alignment proxy in [0, 1].

    0 = nose far from the face centre (head turned away); 1 = nose centred
    (head facing the camera/screen). This is a pose proxy, not eye gaze.
    """
    center_x = (left_x + right_x) / 2.0
    offset = abs(nose_x - center_x)
    return max(0.0, 1.0 - min(offset / tolerance, 1.0))


def _landmark_area(face):
    xs = [lm.x for lm in face.landmark]
    ys = [lm.y for lm in face.landmark]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


class GazeEstimator:
    """Observer face-pose alignment proxy.

    The primary user is assumed to be the largest detected face (closest to
    the camera); every other face is treated as a potential observer. This
    replaces the earlier assumption that MediaPipe returns the user at index 0,
    which is not guaranteed.
    """

    MAX_FACES = 4

    def __init__(self, max_faces=None, process_variance=0.02, measurement_variance=0.08):
        self.mp_mesh = mp.solutions.face_mesh
        # FaceMesh defaults to max_num_faces=1, which made the observer branch
        # unreachable (gaze_score was 0.0 on every logged frame). Allow several.
        self.mesh = self.mp_mesh.FaceMesh(
            max_num_faces=max_faces or self.MAX_FACES,
            refine_landmarks=True,
        )
        self.filter = Kalman1D(process_variance, measurement_variance)
        self.last_raw_measurement = 0.0
        self.last_face_count = 0

    def estimate_intruder_score(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        faces = list(result.multi_face_landmarks or [])
        self.last_face_count = len(faces)
        if len(faces) < 2:
            self.last_raw_measurement = 0.0
            return float(self.filter.update(0.0))

        # Largest face = primary user; remaining faces = candidate observers.
        faces.sort(key=_landmark_area, reverse=True)
        measurement = 0.0
        for face in faces[1:]:
            nose = face.landmark[1]
            left_face = face.landmark[234]
            right_face = face.landmark[454]
            measurement = max(measurement, alignment_score(nose.x, left_face.x, right_face.x))

        self.last_raw_measurement = measurement
        return float(self.filter.update(measurement))

    def detect_intruder(self, frame):
        return self.estimate_intruder_score(frame) >= 0.6

    def close(self):
        self.mesh.close()

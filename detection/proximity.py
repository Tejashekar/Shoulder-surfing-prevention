"""Relative-proximity proxy for observer faces.

The primary user is taken to be the largest face box (closest to the camera).
Every other face is a candidate observer; the score is driven by the largest
observer box. Face-box area is a camera-specific proxy, not calibrated distance.
"""

# Fraction of the frame area at which an observer face saturates the score.
_SATURATION_FRACTION = 0.18


def split_user_and_observers(faces):
    """Return (user_box, [observer boxes]) sorted by box area, largest first."""
    if not faces:
        return None, []
    ordered = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    return ordered[0], ordered[1:]


def estimate_proximity_score(faces, frame_shape=None):
    _, observers = split_user_and_observers(faces)
    if not observers:
        return 0.0

    _, _, w, h = observers[0]
    area = max(0, w * h)

    if frame_shape is not None:
        frame_height, frame_width = frame_shape[:2]
        frame_area = max(1, frame_height * frame_width)
        normalized = area / frame_area
        return max(0.0, min(100.0, (normalized / _SATURATION_FRACTION) * 100.0))

    if area > 50000:
        return 100.0
    if area > 25000:
        return 70.0
    if area > 12000:
        return 40.0
    return 15.0


def estimate_proximity(faces):
    return "near" if estimate_proximity_score(faces) >= 50.0 else "far"

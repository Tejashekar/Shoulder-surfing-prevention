def estimate_proximity(faces):
    if len(faces) < 2:
        return "far"

    # second face (intruder)
    _, _, w, h = faces[1]
    area = w * h

    if area > 50000:
        return "near"
    else:
        return "far"
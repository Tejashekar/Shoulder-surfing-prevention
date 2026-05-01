class RiskEngine:
    def __init__(self):
        # Weights (tune later for paper)
        self.w_face = 2
        self.w_gaze = 3
        self.w_proximity = 2
        self.w_duration = 1

        self.duration_counter = 0

    def compute_risk(self, face_count, gaze_flag, proximity):
        score = 0

        # Face factor
        if face_count > 1:
            score += self.w_face

        # Gaze factor
        if gaze_flag:
            score += self.w_gaze

        # Proximity factor
        if proximity == "near":
            score += self.w_proximity

        # Duration factor
        self.duration_counter += 1 if face_count > 1 else -1
        self.duration_counter = max(0, self.duration_counter)

        score += min(self.duration_counter, 3)

        return score

    def get_risk_level(self, score):
        if score <= 2:
            return "LOW"
        elif score <= 5:
            return "MEDIUM"
        else:
            return "HIGH"
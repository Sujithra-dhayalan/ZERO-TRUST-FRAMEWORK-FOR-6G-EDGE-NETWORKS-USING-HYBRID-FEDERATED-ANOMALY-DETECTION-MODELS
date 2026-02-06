import numpy as np

class ZeroTrustEnforcer:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def calculate_trust_score(self, reconstruction_error):
        # Higher MSE = Lower Trust
        trust_score = np.exp(-reconstruction_error * 10)
        return np.clip(trust_score, 0, 1)

    def access_control(self, trust_score):
        if trust_score > 0.8: return "GRANT"
        elif trust_score > 0.5: return "RESTRICT (MFA Required)"
        else: return "DENY (Immediate Block)"
"""
calibration.py
--------------
Phase 7: calibrate W/D/L probabilities so they're not just accurate on
average but reliable (a 30%-confidence prediction should be right ~30%
of the time). Uses isotonic regression per outcome class, fit on a
VALIDATION fold only.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class WDLCalibrator:
    def __init__(self):
        self.calibrators = {}  # one IsotonicRegression per class

    def fit(self, raw_probs: dict, actual_outcomes: list):
        """
        raw_probs: dict of {"home_win": np.array, "draw": np.array, "away_win": np.array}
                   uncalibrated probabilities from the score model, on a VALIDATION fold.
        actual_outcomes: list of "home_win"/"draw"/"away_win" strings, same order/length.
        """
        for cls in ["home_win", "draw", "away_win"]:
            binary_target = np.array([1 if o == cls else 0 for o in actual_outcomes])
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_probs[cls], binary_target)
            self.calibrators[cls] = iso

    def transform(self, raw_probs: dict) -> dict:
        calibrated = {cls: self.calibrators[cls].transform(raw_probs[cls]) for cls in raw_probs}
        # renormalise so the three classes sum to 1 per row
        total = sum(calibrated.values())
        return {cls: calibrated[cls] / total for cls in calibrated}

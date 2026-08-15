"""
pipeline.py
-----------
Phase 12: the single entry point your Streamlit app calls. Ties every
module together exactly as specified in the roadmap's core prediction
function.

NOTE ON WHAT'S "REAL" HERE: the individual modules (cleaning, rolling
features, ratings, lineup aggregation, Elo, Dixon-Coles, evaluation)
are all fully functional and tested against your actual uploaded data.
The GoalModelSet (Poisson/RF/GBM ensemble) needs to be FIT on your
full historical match-level training table before predict_match() can
return real numbers - right now it will raise a clear error if you
call it before fitting, rather than silently returning fake output.
"""

import pandas as pd
import numpy as np

from cleaning import build_clean_dataset
from rolling_features import build_player_feature_table
from player_ratings import add_role_ratings
from lineup_features import build_match_row, build_lineup_vector
from elo import extract_match_results, compute_chronological_elo
from goal_models import GoalModelSet
from score_models import dixon_coles_matrix, topn_from_matrix, outcome_probs_from_matrix
from calibration import WDLCalibrator


class PredictionPipeline:
    def __init__(self):
        self.feature_table = None      # player-match level, post ratings
        self.elo_table = None
        self.home_goal_model = GoalModelSet()
        self.away_goal_model = GoalModelSet()
        self.dc_rho = -0.13            # default; overwrite via fit_dixon_coles_rho
        self.calibrator = WDLCalibrator()
        self.is_fitted = False

    def build_features(self, player_csv_paths: list):
        """Phase 1 + 2A + 2B + 2C in one call."""
        clean = build_clean_dataset(player_csv_paths)
        feats = build_player_feature_table(clean)
        rated = add_role_ratings(feats)
        self.feature_table = rated

        results = extract_match_results(clean)
        self.elo_table = compute_chronological_elo(results)
        return self.feature_table

    def get_player_state_before(self, player: str, date) -> pd.Series:
        """Most recent feature row for a player strictly before `date` - leakage-safe lookup."""
        if self.feature_table is None:
            raise RuntimeError("Call build_features() first.")
        hist = self.feature_table[
            (self.feature_table["Player"] == player) & (self.feature_table["Date"] < date)
        ]
        if len(hist) == 0:
            raise ValueError(f"No history for {player} before {date} - cannot build a leakage-safe rating.")
        return hist.iloc[-1]

    def build_match_training_row(self, home_xi_names: list, away_xi_names: list, date) -> dict:
        home_rows = pd.DataFrame([self.get_player_state_before(p, date) for p in home_xi_names])
        away_rows = pd.DataFrame([self.get_player_state_before(p, date) for p in away_xi_names])
        row = build_match_row(home_rows, away_rows)
        return row

    def fit_goal_models(self, training_table: pd.DataFrame, feature_cols: list, train_frac=0.7, val_frac=0.15):
        """
        training_table: one row per historical match, with feature_cols
        (from build_match_training_row, accumulated over your season)
        plus 'HomeGoals' and 'AwayGoals' columns and a date column.
        """
        from goal_models import chronological_split

        train, val, test = chronological_split(training_table, "date", train_frac, val_frac)
        self.home_goal_model.fit_and_validate(
            train[feature_cols], train["HomeGoals"], val[feature_cols], val["HomeGoals"]
        )
        self.away_goal_model.fit_and_validate(
            train[feature_cols], train["AwayGoals"], val[feature_cols], val["AwayGoals"]
        )
        self.is_fitted = True
        return test  # hand back the held-out fold for further evaluation

    def predict_match(self, home_xi_names: list, away_xi_names: list, date, max_goals: int = 7) -> dict:
        if not self.is_fitted:
            raise RuntimeError(
                "Goal models are not fitted yet. Call fit_goal_models() on your full "
                "historical training table first - predict_match() will not fabricate "
                "a result from an untrained model."
            )
        row = self.build_match_training_row(home_xi_names, away_xi_names, date)
        X = pd.DataFrame([row])
        lam_home = float(self.home_goal_model.predict(X)[0])
        lam_away = float(self.away_goal_model.predict(X)[0])

        matrix = dixon_coles_matrix(lam_home, lam_away, self.dc_rho, max_goals)
        top5 = topn_from_matrix(matrix, 5)
        outcome = outcome_probs_from_matrix(matrix)

        return {
            "lambda_home": lam_home,
            "lambda_away": lam_away,
            "top5_scorelines": top5,
            "outcome_probs": outcome,
            "matrix": matrix,
            "lineup_row": row,
        }

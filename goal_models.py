"""
goal_models.py
--------------
Phase 5 of the roadmap: replace the fixed 0.60/0.40 LR+RF blend with
candidate models compared honestly on held-out data, then a LEARNED
(not fixed) ensemble weight.

Design note: these are thin wrappers so you can swap models in/out of
the same interface (fit -> predict lambda_home, lambda_away) without
rewriting pipeline.py. Train TWO separate models (home goals, away
goals) as the roadmap recommends, since that's simpler to debug than
a joint model and works fine with Poisson/NegBinom/GBM.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


class GoalModelSet:
    """
    One instance predicts ONE target (e.g. HomeGoals). Fit separate
    instances for home and away.
    """

    def __init__(self):
        self.models = {
            "poisson": PoissonRegressor(alpha=0.5, max_iter=500),
            "random_forest": RandomForestRegressor(
                n_estimators=200, max_depth=6, random_state=42
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
            ),
        }
        self.fold_mae = {}
        self.ensemble_weights = None

    def fit_and_validate(self, X_train, y_train, X_val, y_val):
        """
        Fits every candidate on X_train/y_train, scores each on
        X_val/y_val (a chronologically LATER fold - never a random
        split), and derives inverse-MAE ensemble weights from the
        validation performance (learned, not fixed at 0.6/0.4).
        """
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = np.clip(model.predict(X_val), 0, None)
            mae = mean_absolute_error(y_val, preds)
            self.fold_mae[name] = mae

        eps = 1e-6
        inv = {name: 1.0 / (mae + eps) for name, mae in self.fold_mae.items()}
        total = sum(inv.values())
        self.ensemble_weights = {name: w / total for name, w in inv.items()}
        return self.fold_mae, self.ensemble_weights

    def predict(self, X):
        """Weighted ensemble prediction using the learned weights."""
        if self.ensemble_weights is None:
            raise RuntimeError("Call fit_and_validate before predict.")
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            preds += self.ensemble_weights[name] * np.clip(model.predict(X), 0, None)
        return np.clip(preds, 0.05, None)  # lambda must stay positive for Poisson


def chronological_split(df: pd.DataFrame, date_col: str, train_frac: float = 0.7, val_frac: float = 0.15):
    """
    Three-way chronological split: train / validation / test.
    test here is your DEVELOPMENT test fold, not the locked 38-match
    benchmark - keep that one completely separate and untouched.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


if __name__ == "__main__":
    # Synthetic smoke test only (proves the code runs correctly) -
    # this is NOT a claim about real predictive performance. Real
    # numbers require your actual match-level training table from
    # lineup_features.py + elo.py.
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 5))
    true_lambda = np.clip(1.4 + 0.5 * X[:, 0] - 0.3 * X[:, 1], 0.1, None)
    y = rng.poisson(true_lambda)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["y"] = y
    df["date_idx"] = np.arange(n)

    train, val, test = chronological_split(df, "date_idx")
    gm = GoalModelSet()
    mae, weights = gm.fit_and_validate(
        train[[f"f{i}" for i in range(5)]], train["y"],
        val[[f"f{i}" for i in range(5)]], val["y"],
    )
    print("Validation MAE per model:", mae)
    print("Learned ensemble weights:", weights)
    test_preds = gm.predict(test[[f"f{i}" for i in range(5)]])
    print("Test MAE (ensemble):", mean_absolute_error(test["y"], test_preds))

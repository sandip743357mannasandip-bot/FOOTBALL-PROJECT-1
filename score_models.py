"""
score_models.py
----------------
Phase 6 of the roadmap - the module most directly responsible for
Top-5/Top-4/Top-3/Top-2 hit rate.

Implements:
  - independent_poisson_matrix: your CURRENT method (baseline, E0)
  - dixon_coles_matrix: independent Poisson + the Dixon & Coles (1997)
    tau(x,y) correction for the four low-score cells (0-0, 1-0, 0-1, 1-1)
  - fit_dixon_coles_rho: fits the single correlation parameter rho by
    maximum likelihood on your own historical (HomeGoals, AwayGoals,
    lambda_home, lambda_away) rows
  - rank_scorelines / topn_from_matrix: turns a probability matrix into
    a ranked Top-N list, used both for prediction and for evaluation
"""

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar


def independent_poisson_matrix(lam_home: float, lam_away: float, max_goals: int = 7) -> np.ndarray:
    """P(Home=i, Away=j) assuming independence. Shape (max_goals+1, max_goals+1)."""
    home_probs = poisson.pmf(np.arange(max_goals + 1), lam_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lam_away)
    matrix = np.outer(home_probs, away_probs)
    return matrix / matrix.sum()  # renormalise for grid truncation


def _dc_tau(x: int, y: int, lam_home: float, lam_away: float, rho: float) -> float:
    """Dixon-Coles correction factor for low scorelines only."""
    if x == 0 and y == 0:
        return 1 - lam_home * lam_away * rho
    elif x == 0 and y == 1:
        return 1 + lam_home * rho
    elif x == 1 and y == 0:
        return 1 + lam_away * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def dixon_coles_matrix(lam_home: float, lam_away: float, rho: float, max_goals: int = 7) -> np.ndarray:
    """Independent Poisson matrix with the DC tau correction applied to (0,0),(1,0),(0,1),(1,1)."""
    matrix = independent_poisson_matrix(lam_home, lam_away, max_goals)
    for x in range(2):
        for y in range(2):
            matrix[x, y] *= _dc_tau(x, y, lam_home, lam_away, rho)
    matrix = np.clip(matrix, 0, None)  # tau can occasionally push a cell slightly negative
    return matrix / matrix.sum()


def dc_neg_log_likelihood(rho: float, rows: list) -> float:
    """
    rows: list of (home_goals, away_goals, lam_home, lam_away) from your
    OWN historical training data (walk-forward - not the locked 38-match set).
    """
    ll = 0.0
    for hg, ag, lh, la in rows:
        base = poisson.pmf(hg, lh) * poisson.pmf(ag, la)
        tau = _dc_tau(min(hg, 1), min(ag, 1), lh, la, rho) if hg <= 1 and ag <= 1 else 1.0
        p = max(base * tau, 1e-10)
        ll += np.log(p)
    return -ll


def fit_dixon_coles_rho(rows: list, bounds=(-0.5, 0.5)) -> float:
    """
    Fit rho by MLE. rows: list of (home_goals, away_goals, lam_home, lam_away)
    from TRAINING/VALIDATION folds only - never from the locked test set.
    """
    result = minimize_scalar(dc_neg_log_likelihood, bounds=bounds, args=(rows,), method="bounded")
    return float(result.x)


def rank_scorelines(matrix: np.ndarray) -> list:
    """Returns [(home_goals, away_goals, prob), ...] sorted probability descending."""
    entries = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            entries.append((i, j, float(matrix[i, j])))
    entries.sort(key=lambda t: t[2], reverse=True)
    return entries


def topn_from_matrix(matrix: np.ndarray, n: int) -> list:
    return rank_scorelines(matrix)[:n]


def outcome_probs_from_matrix(matrix: np.ndarray) -> dict:
    """P(Home Win), P(Draw), P(Away Win) from the full scoreline matrix."""
    home_win = float(np.sum(np.triu(matrix, k=1)))   # i > j means row>col -> need lower triangle actually
    # careful: matrix[i, j] = P(home=i, away=j). Home win means i > j.
    n = matrix.shape[0]
    home_win = sum(matrix[i, j] for i in range(n) for j in range(n) if i > j)
    draw = sum(matrix[i, j] for i in range(n) for j in range(n) if i == j)
    away_win = sum(matrix[i, j] for i in range(n) for j in range(n) if i < j)
    return {"home_win": home_win, "draw": draw, "away_win": away_win}


def actual_scoreline_rank(matrix: np.ndarray, actual_home: int, actual_away: int) -> int:
    """1-indexed rank of the true scoreline within the full ranked list. Useful diagnostic
    the roadmap specifically recommends: if actual is frequently rank 6-10, the ranking/
    calibration layer -- not the goal model -- is where to focus next."""
    ranked = rank_scorelines(matrix)
    for idx, (i, j, _) in enumerate(ranked, start=1):
        if i == actual_home and j == actual_away:
            return idx
    return -1  # actual score fell outside the max_goals grid


if __name__ == "__main__":
    # Sanity check with plausible Real Madrid home-favourite lambdas
    lam_h, lam_a = 2.1, 0.9
    indep = independent_poisson_matrix(lam_h, lam_a)
    dc = dixon_coles_matrix(lam_h, lam_a, rho=-0.13)

    print("Independent Poisson Top-5:")
    for i, j, p in topn_from_matrix(indep, 5):
        print(f"  {i}-{j}: {p:.4f}")
    print("Dixon-Coles (rho=-0.13) Top-5:")
    for i, j, p in topn_from_matrix(dc, 5):
        print(f"  {i}-{j}: {p:.4f}")
    print("Independent outcome probs:", outcome_probs_from_matrix(indep))
    print("Dixon-Coles outcome probs:", outcome_probs_from_matrix(dc))

"""
evaluation.py
-------------
Phase 8 + 9: the evaluation harness every experiment (E0-E9) gets
scored with, and the walk-forward split that keeps the 38-match
benchmark locked and untouched until the final test.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, f1_score, confusion_matrix

from score_models import rank_scorelines, actual_scoreline_rank


def topn_hit_rate(matches: list, n: int) -> float:
    """
    matches: list of dicts, each with keys 'matrix' (the scoreline
    probability matrix), 'actual_home', 'actual_away'.
    """
    hits = 0
    for m in matches:
        top = rank_scorelines(m["matrix"])[:n]
        if any(i == m["actual_home"] and j == m["actual_away"] for i, j, _ in top):
            hits += 1
    return hits / len(matches)


def exact_score_accuracy(matches: list) -> float:
    return topn_hit_rate(matches, 1)


def rank_diagnostics(matches: list) -> pd.DataFrame:
    """
    Roadmap's specific diagnostic: record the rank of the true scoreline
    for every match, even when it falls outside Top-5. If ranks cluster
    at 6-10, focus on calibration/ranking; if they're scattered past 15,
    focus on the goal model (lambda estimates themselves are off).
    """
    rows = []
    for m in matches:
        r = actual_scoreline_rank(m["matrix"], m["actual_home"], m["actual_away"])
        rows.append({"date": m.get("date"), "actual_rank": r})
    return pd.DataFrame(rows)


def full_scoreline_report(matches: list) -> dict:
    return {
        "top1_exact": topn_hit_rate(matches, 1),
        "top2": topn_hit_rate(matches, 2),
        "top3": topn_hit_rate(matches, 3),
        "top4": topn_hit_rate(matches, 4),
        "top5": topn_hit_rate(matches, 5),
        "n_matches": len(matches),
    }


def outcome_report(y_true: list, y_pred: list, probs: dict) -> dict:
    """
    y_true, y_pred: lists of "home_win"/"draw"/"away_win" strings.
    probs: dict of arrays, {"home_win": [...], "draw": [...], "away_win": [...]}
           aligned with y_true, used for log loss / Brier.
    """
    classes = ["home_win", "draw", "away_win"]
    y_true_idx = [classes.index(c) for c in y_true]
    prob_matrix = np.column_stack([probs[c] for c in classes])
    prob_matrix = np.clip(prob_matrix, 1e-6, 1 - 1e-6)
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

    accuracy = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    ll = log_loss(y_true_idx, prob_matrix, labels=[0, 1, 2])
    # multi-class Brier: mean squared error between one-hot and prob vector
    onehot = np.zeros_like(prob_matrix)
    for row, idx in enumerate(y_true_idx):
        onehot[row, idx] = 1
    brier = float(np.mean(np.sum((prob_matrix - onehot) ** 2, axis=1)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "log_loss": ll,
        "brier_score": brier,
        "confusion_matrix": pd.DataFrame(cm, index=[f"actual_{c}" for c in classes],
                                          columns=[f"pred_{c}" for c in classes]),
    }


def walk_forward_folds(df: pd.DataFrame, date_col: str, n_folds: int = 4, min_train_frac: float = 0.4):
    """
    Yields (train_df, val_df) chronological folds, each fold's train set
    growing to include all prior folds - never a random shuffle.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    start = int(n * min_train_frac)
    step = (n - start) // n_folds
    for k in range(n_folds):
        train_end = start + k * step
        val_end = start + (k + 1) * step if k < n_folds - 1 else n
        if train_end >= val_end:
            continue
        yield df.iloc[:train_end], df.iloc[train_end:val_end]


class ExperimentLog:
    """Minimal experiment tracker matching the roadmap's experiment_log.csv."""

    def __init__(self, path="experiments/experiment_log.csv"):
        self.path = path
        self.rows = []

    def log(self, experiment_id: str, notes: str, **metrics):
        row = {"experiment_id": experiment_id, "notes": notes}
        row.update(metrics)
        self.rows.append(row)

    def save(self):
        import os
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)

"""
lineup_features.py
-------------------
Phase 3 of the roadmap: convert a selected Playing XI (11 player rows,
each already carrying attack_rating / creation_rating / defense_rating
from player_ratings.py, as of the match date) into one structured
lineup-strength vector per team per match.
"""

import pandas as pd
import numpy as np


def build_lineup_vector(xi_rows: pd.DataFrame) -> dict:
    """
    xi_rows: dataframe of exactly the 11 selected players' feature rows
    for ONE match (already filtered to Date < match_date via your
    leakage-safe history load - these ratings must be each player's
    most recent rating BEFORE the match, not from the match itself).

    Returns a flat dict - one row of the eventual training table.
    """
    if len(xi_rows) == 0:
        raise ValueError("build_lineup_vector received an empty Playing XI")

    fwd_mid = xi_rows[xi_rows["PosGroup"].isin(["FWD", "MID"])]
    defenders = xi_rows[xi_rows["PosGroup"].isin(["DEF", "GK"])]

    def safe_mean(s: pd.Series) -> float:
        return float(s.mean()) if len(s) else 0.0

    vector = {
        "attack_strength": safe_mean(fwd_mid["attack_rating"]),
        "creation_strength": safe_mean(xi_rows["creation_rating"]),
        "defense_strength": safe_mean(defenders["defense_rating"]),
        "overall_attack_rating": safe_mean(xi_rows["attack_rating"]),
        "overall_defense_rating": safe_mean(xi_rows["defense_rating"]),
        # minutes_stability: are these players used to playing full matches
        # recently, or are we fielding a rotated/rusty XI? Lower = less stable.
        "minutes_stability": safe_mean(xi_rows["Minutes_roll5"]) if "Minutes_roll5" in xi_rows else np.nan,
        "n_players_used": len(xi_rows),
    }
    return vector


def build_matchup_features(home_vector: dict, away_vector: dict) -> dict:
    """Phase 3 MATCHUP vector - relative strength differentials."""
    return {
        "home_attack_minus_away_defense": home_vector["attack_strength"] - away_vector["defense_strength"],
        "away_attack_minus_home_defense": away_vector["attack_strength"] - home_vector["defense_strength"],
        "attack_gap_home_minus_away": home_vector["attack_strength"] - away_vector["attack_strength"],
        "defense_gap_home_minus_away": home_vector["defense_strength"] - away_vector["defense_strength"],
    }


def build_match_row(home_xi: pd.DataFrame, away_xi: pd.DataFrame) -> dict:
    """Combine both teams' lineup vectors + matchup features into one training row."""
    home_vec = build_lineup_vector(home_xi)
    away_vec = build_lineup_vector(away_xi)
    matchup = build_matchup_features(home_vec, away_vec)

    row = {}
    row.update({f"home_{k}": v for k, v in home_vec.items()})
    row.update({f"away_{k}": v for k, v in away_vec.items()})
    row.update(matchup)
    return row

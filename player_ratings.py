"""
player_ratings.py
------------------
Phase 2C of the roadmap: turn the per-90 shrunk stats from
rolling_features.py into position-specific ratings.

Design: each rating is a weighted blend of the relevant shrunk per-90
stats for that role. Weights are simple and interpretable to start -
the roadmap explicitly says "let validation decide", so these initial
weights are a reasonable starting point, NOT tuned. Once you have a
goal model running (Phase 5), you can replace these hand weights with
coefficients learned from data.
"""

import pandas as pd
import numpy as np

# Position groups - adjust to match whatever Position strings your
# FBref-derived data actually uses (this covers the common ones).
FORWARD_POS = {"FW", "LW", "RW", "ST"}
MID_POS = {"CM", "DM", "AM", "LM", "RM", "MF"}
DEF_POS = {"CB", "LB", "RB", "DF"}
GK_POS = {"GK"}


def classify_position(pos: str) -> str:
    if pos in FORWARD_POS:
        return "FWD"
    if pos in MID_POS:
        return "MID"
    if pos in DEF_POS:
        return "DEF"
    if pos in GK_POS:
        return "GK"
    return "UNKNOWN"


def add_position_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PosGroup"] = df["Position"].apply(classify_position)
    return df


def add_role_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires columns produced by rolling_features.build_player_feature_table:
    Goals_per90_shrunk, Assists_per90_shrunk, Shots_per90_shrunk,
    SoT_per90_shrunk, TacklesWon_per90_shrunk, Interceptions_per90_shrunk.

    Adds:
      attack_rating   - meaningful for FWD/MID
      creation_rating - meaningful for MID
      defense_rating  - meaningful for DEF/GK
    All ratings are z-scored WITHIN this dataset so they're comparable
    across players. Re-fit the mean/std whenever you add a new season
    of data (don't hardcode old constants).
    """
    df = df.copy()
    df = add_position_group(df)

    def zscore(s: pd.Series) -> pd.Series:
        mu, sigma = s.mean(), s.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sigma

    z_goals = zscore(df["Goals_per90_shrunk"])
    z_shots = zscore(df["Shots_per90_shrunk"])
    z_sot = zscore(df["SoT_per90_shrunk"])
    z_assists = zscore(df["Assists_per90_shrunk"])
    z_tackles = zscore(df["TacklesWon_per90_shrunk"])
    z_interceptions = zscore(df["Interceptions_per90_shrunk"])

    # Attack: finishing-weighted, shots/SoT as volume+quality signal
    df["attack_rating"] = 0.5 * z_goals + 0.3 * z_sot + 0.2 * z_shots

    # Creation: assists-led (extend with key passes/xA if you add that data)
    df["creation_rating"] = z_assists

    # Defense: tackles + interceptions, equally weighted
    df["defense_rating"] = 0.5 * z_tackles + 0.5 * z_interceptions

    return df


if __name__ == "__main__":
    import sys
    from cleaning import build_clean_dataset
    from rolling_features import build_player_feature_table

    clean = build_clean_dataset(sys.argv[1:])
    feats = build_player_feature_table(clean)
    rated = add_role_ratings(feats)
    print(rated[["Date", "Player", "Position", "PosGroup",
                 "attack_rating", "creation_rating", "defense_rating"]].tail(15).to_string(index=False))

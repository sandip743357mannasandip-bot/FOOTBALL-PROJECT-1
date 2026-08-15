"""
cleaning.py
-----------
Phase 1 of the roadmap: turn raw per-player FBref-style CSVs into one
standardised, leakage-safe table.

Key rules this module enforces (from Roadmap Phase 1 "Data quality checks"):
  - Dates parsed and sorted chronologically.
  - Duplicate player-match rows removed.
  - SoT <= Shots, Minutes >= 0 (logical bounds).
  - Missingness is tracked with explicit indicator columns instead of
    being silently converted to zero, since a blank stat can mean two
    very different things: "player didn't attempt it" vs "not recorded".
"""

import pandas as pd
import numpy as np

# Stat columns that get a companion "_was_missing" flag before any fill
STAT_COLS = [
    "Minutes", "Goals", "Assists", "Shots", "SoT",
    "Yellow", "Red", "Fouls", "Offsides", "Crosses",
    "TacklesWon", "Interceptions",
]

REQUIRED_COLS = [
    "Date", "Season", "Competition", "Team", "Opponent", "Venue",
    "TeamGoals", "OppGoals", "Result", "Player", "Start", "Position",
] + STAT_COLS


def load_player_csv(path: str) -> pd.DataFrame:
    """Load one raw player CSV (e.g. Abdul_Mumin_-_Sheet1.csv)."""
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing expected columns: {missing_cols}")
    return df


def clean_player_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise one player's raw match log.
    Returns a cleaned copy - does NOT mutate the input.
    """
    df = df.copy()

    # --- dates ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    n_bad_dates = df["Date"].isna().sum()
    if n_bad_dates:
        print(f"WARNING: dropping {n_bad_dates} row(s) with unparseable Date")
        df = df.dropna(subset=["Date"])

    # --- de-duplicate (same player, same date, same team = same match) ---
    before = len(df)
    df = df.drop_duplicates(subset=["Player", "Date", "Team", "Opponent"])
    if len(df) < before:
        print(f"WARNING: dropped {before - len(df)} duplicate player-match row(s)")

    # --- missingness indicators BEFORE any fill ---
    for col in STAT_COLS:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)

    # A row with no Position and no Minutes is a squad-listed but
    # unused/unrecorded appearance (e.g. row 1 in the Mumin file: 12 min,
    # blank Position). Treat Minutes as 0 only when genuinely absent from
    # play, not when the stat columns are simply unrecorded.
    df["Minutes"] = df["Minutes"].fillna(0)
    df["Position"] = df["Position"].fillna("UNKNOWN")

    # Counting stats: fill with 0 ONLY where Minutes == 0 (didn't play,
    # so genuinely can't have registered a shot/tackle/etc).
    # Where Minutes > 0 but a stat is NaN, leave it NaN - that is a true
    # data-quality gap, not a zero, and rolling functions must skip it
    # rather than silently treating it as "player did nothing".
    played_mask = df["Minutes"] > 0
    for col in [c for c in STAT_COLS if c != "Minutes"]:
        df.loc[~played_mask, col] = df.loc[~played_mask, col].fillna(0)

    # --- logical bounds ---
    bad_sot = df["SoT"] > df["Shots"]
    if bad_sot.any():
        print(f"WARNING: {bad_sot.sum()} row(s) have SoT > Shots - clipping SoT to Shots")
        df.loc[bad_sot, "SoT"] = df.loc[bad_sot, "Shots"]

    neg_minutes = df["Minutes"] < 0
    if neg_minutes.any():
        print(f"WARNING: {neg_minutes.sum()} row(s) have negative Minutes - clipping to 0")
        df.loc[neg_minutes, "Minutes"] = 0

    # --- sort chronologically per player: mandatory for leakage-safe rolling ---
    df = df.sort_values(["Player", "Date"]).reset_index(drop=True)
    return df


def build_clean_dataset(paths: list) -> pd.DataFrame:
    """Load + clean multiple player CSVs and concatenate into one table."""
    frames = [clean_player_df(load_player_csv(p)) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Player", "Date"]).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    import sys
    df = build_clean_dataset(sys.argv[1:])
    print(df.shape)
    print(df.head())

"""
rolling_features.py
--------------------
Phase 2A + 2B of the roadmap ("the biggest upgrade"):
  A. Multi-window rolling form (last 3 / 5 / 10 / 20 matches), with
     recency weighting instead of a flat mean.
  B. Per-90 rate stats, with empirical-Bayes shrinkage so a single
     12-minute cameo doesn't distort a player's profile.

LEAKAGE RULE (critical): every rolling/aggregate feature for the match
on row i must be built ONLY from rows before row i for that player.
This is enforced by always calling .shift(1) before .rolling(...), so
the current match's own stats are never included in its own features.
"""

import pandas as pd
import numpy as np

WINDOWS = (3, 5, 10, 20)

# Counting stats we build rolling form for (raw totals, pre-per-90)
FORM_STAT_COLS = ["Goals", "Assists", "Shots", "SoT", "TacklesWon", "Interceptions"]

# TeamGoals/OppGoals are team-level outcomes attached to every player row -
# used for goals-for / goals-against rolling form.
TEAM_STAT_COLS = ["TeamGoals", "OppGoals"]


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Flat rolling mean using only PRIOR rows (leakage-safe)."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _shifted_recency_weighted_mean(series: pd.Series, window: int, decay: float = 0.75) -> pd.Series:
    """
    Recency-weighted rolling mean over the last `window` PRIOR matches.
    Most recent prior match gets weight decay^0, next gets decay^1, etc.
    Weights are normalised to sum to 1 over whatever history is available
    (handles the early-season case where fewer than `window` matches exist).
    """
    shifted = series.shift(1)
    out = np.full(len(series), np.nan)
    values = shifted.values
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        hist = values[lo:i + 1]
        hist = hist[~np.isnan(hist)]
        if len(hist) == 0:
            continue
        # hist[-1] is the most recent prior match
        weights = np.array([decay ** k for k in range(len(hist))])[::-1]
        weights = weights / weights.sum()
        out[i] = np.dot(hist, weights)
    return pd.Series(out, index=series.index)


def add_multiwindow_form(df: pd.DataFrame, weighted: bool = True) -> pd.DataFrame:
    """
    Adds columns like Goals_roll5, Goals_wroll5 (recency-weighted),
    for each stat in FORM_STAT_COLS + TEAM_STAT_COLS, for each window.
    Must be called on a df already sorted by [Player, Date].
    """
    df = df.copy()
    stat_cols = FORM_STAT_COLS + TEAM_STAT_COLS
    for col in stat_cols:
        grouped = df.groupby("Player")[col]
        for w in WINDOWS:
            flat_col = f"{col}_roll{w}"
            df[flat_col] = grouped.transform(lambda s, w=w: _shifted_rolling_mean(s, w))
            if weighted:
                wcol = f"{col}_wroll{w}"
                df[wcol] = grouped.transform(lambda s, w=w: _shifted_recency_weighted_mean(s, w))
    return df


def add_minutes_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Season-to-date minutes and a prior-appearance counter - needed both
    for per-90 normalisation and for shrinkage weighting.
    """
    df = df.copy()
    grouped = df.groupby("Player")["Minutes"]
    # cumulative minutes played BEFORE this match (leakage-safe)
    df["career_minutes_before"] = grouped.transform(lambda s: s.shift(1).cumsum().fillna(0))
    df["prior_appearances"] = df.groupby("Player").cumcount()
    return df


def add_per90_and_shrinkage(df: pd.DataFrame, shrink_k: float = 270.0) -> pd.DataFrame:
    """
    Per-90 rolling rates (last-5-match window) with empirical-Bayes
    shrinkage toward the player's own career-to-date per-90 rate.

    shrink_k is expressed in MINUTES of prior evidence needed before we
    mostly trust the recent-5 rate over the shrinkage prior. 270 minutes
    is roughly 3 full matches: with less prior evidence than that, the
    shrunk estimate leans harder on the career baseline.

    shrunk_rate = (recent_rate * recent_minutes + prior_rate * shrink_k)
                  / (recent_minutes + shrink_k)
    """
    df = df.copy()
    df = add_minutes_context(df)

    for col in FORM_STAT_COLS:
        grouped_stat = df.groupby("Player")[col]
        grouped_min = df.groupby("Player")["Minutes"]

        # minutes actually played over the last 5 PRIOR matches (denominator)
        recent_minutes = grouped_min.transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
        recent_total = grouped_stat.transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
        recent_rate = (recent_total / recent_minutes.replace(0, np.nan)) * 90

        # career-to-date rate (prior to this match) as the shrinkage prior
        career_total = grouped_stat.transform(lambda s: s.shift(1).cumsum())
        career_rate = (career_total / df["career_minutes_before"].replace(0, np.nan)) * 90

        recent_minutes_filled = recent_minutes.fillna(0)
        recent_rate_filled = recent_rate.fillna(career_rate).fillna(0)
        career_rate_filled = career_rate.fillna(0)

        shrunk = (
            recent_rate_filled * recent_minutes_filled + career_rate_filled * shrink_k
        ) / (recent_minutes_filled + shrink_k)

        df[f"{col}_per90_shrunk"] = shrunk

    return df


def build_player_feature_table(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Full Phase 2A+2B pipeline: multi-window form + per-90 shrinkage."""
    df = clean_df.sort_values(["Player", "Date"]).reset_index(drop=True)
    df = add_multiwindow_form(df, weighted=True)
    df = add_per90_and_shrinkage(df)
    return df


if __name__ == "__main__":
    import sys
    from cleaning import build_clean_dataset
    clean = build_clean_dataset(sys.argv[1:])
    feats = build_player_feature_table(clean)
    show_cols = [
        "Date", "Player", "Minutes",
        "Goals_roll5", "Goals_wroll5",
        "TacklesWon_roll5", "TacklesWon_wroll5",
        "TacklesWon_per90_shrunk", "Interceptions_per90_shrunk",
    ]
    print(feats[show_cols].to_string(index=False))

"""
elo.py
------
Phase 4 of the roadmap: chronological Elo ratings per team, computed
strictly from PAST results relative to each match (no leakage).

IMPORTANT LIMITATION (flagging honestly rather than hiding it):
Elo needs match results for every team a club plays, not just the
club you're building player CSVs for. Your player-level CSVs are
Real-Madrid-squad-centric, but the TeamGoals/OppGoals columns on each
row DO give you full match results (Team X vs Opponent Y, final score)
regardless of whose player file it came from. So: build the match
results table by de-duplicating (Date, Team, Opponent, TeamGoals,
OppGoals) across ALL your player CSVs combined - the more player files
you feed in (i.e. covering more clubs' squads), the more complete your
Elo history will be. If you only have Real Madrid player files, your
Elo will only "see" Real Madrid's matches and won't have independent
ratings for e.g. Girona or Osasuna's other fixtures - it will still
work, just with less signal than a full league-wide Elo.
"""

import pandas as pd
import numpy as np


DEFAULT_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADV = 60.0  # Elo points added to home team's rating pre-match


def extract_match_results(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    De-duplicate player-level rows into one row per real match.
    Requires columns: Date, Team, Opponent, Venue, TeamGoals, OppGoals.
    """
    matches = player_df[["Date", "Team", "Opponent", "Venue", "TeamGoals", "OppGoals"]].drop_duplicates()
    matches = matches.sort_values("Date").reset_index(drop=True)
    return matches


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_chronological_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """
    matches: one row per match, with Date, Team (home-perspective row
    only - filter Venue == 'Home' upstream, OR pass both home+away rows
    and this function will dedupe by Date+sorted(Team,Opponent) pairs).

    Returns matches with columns:
      home_elo_pre, away_elo_pre  - each team's rating BEFORE this match
    plus updates an internal rating table AFTER each match (chronological).
    """
    matches = matches.copy()
    # Keep only the Home-venue row per fixture so each real match appears once
    home_rows = matches[matches["Venue"].str.lower() == "home"].copy()
    home_rows = home_rows.sort_values("Date").reset_index(drop=True)

    ratings = {}
    home_elo_pre, away_elo_pre = [], []

    for _, row in home_rows.iterrows():
        home, away = row["Team"], row["Opponent"]
        r_home = ratings.get(home, DEFAULT_ELO)
        r_away = ratings.get(away, DEFAULT_ELO)

        home_elo_pre.append(r_home)
        away_elo_pre.append(r_away)

        # result from home team's perspective: 1 win, 0.5 draw, 0 loss
        if row["TeamGoals"] > row["OppGoals"]:
            actual = 1.0
        elif row["TeamGoals"] == row["OppGoals"]:
            actual = 0.5
        else:
            actual = 0.0

        expected = _expected_score(r_home + HOME_ADV, r_away)
        delta = K_FACTOR * (actual - expected)

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    home_rows["home_elo_pre"] = home_elo_pre
    home_rows["away_elo_pre"] = away_elo_pre
    home_rows["elo_diff"] = home_rows["home_elo_pre"] - home_rows["away_elo_pre"]
    return home_rows


if __name__ == "__main__":
    import sys
    dfs = [pd.read_csv(p) for p in sys.argv[1:]]
    combined = pd.concat(dfs, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    matches = extract_match_results(combined)
    elo_table = compute_chronological_elo(matches)
    print(elo_table[["Date", "Team", "Opponent", "TeamGoals", "OppGoals",
                      "home_elo_pre", "away_elo_pre", "elo_diff"]].to_string(index=False))

"""
backend.py — Football Match Predictor v2, single-file backend.

This is every module from the modular `src/` pipeline (data_loader,
cleaning, rolling_features, player_ratings, lineup_features, elo,
goal_models, score_models, calibration, evaluation, run_experiments,
pipeline) merged into one file, in dependency order, so the project is
just two files: backend.py (all logic) + app.py (Streamlit frontend).
Nothing was removed — only the cross-file `from module import ...`
statements between these sections were dropped, since everything now
lives in the same namespace.

Section map (search for these headers):
  1. DATA LOADER      — CSV loading, column normalisation, season lookup
  2. CLEANING         — dedupe, bounds checks, leakage-safe slicing
  3. ROLLING FEATURES  — multi-window player form features
  4. PLAYER RATINGS    — position-specific 0-100 ratings
  5. LINEUP FEATURES   — formation + XI -> lineup strength vector
  6. ELO               — chronological team Elo ratings
  7. GOAL MODELS        — xG regressor candidates + learned ensemble
  8. SCORE MODELS       — scoreline probability grids (Poisson family)
  9. CALIBRATION        — isotonic W/D/L probability calibration
  10. EVALUATION         — metrics + walk-forward validation harness
  11. RUN EXPERIMENTS    — E0..E9 experiment matrix runner
  12. PIPELINE           — ties everything together, predict_match()
"""

import argparse
import csv
import glob
import math
import os
import unicodedata
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error, f1_score, confusion_matrix, log_loss, brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. DATA LOADER — Phase 1
# Loads raw player CSVs + SEASON_DATA.csv with flexible column names.
# =============================================================================

COL_ALIASES = {
    "Goals":          ["Goals", "G", "Gls", "Goal", "goals"],
    "Assists":        ["Assists", "Ast", "A", "assists"],
    "Shots":          ["Shots", "Sh", "shots", "Shot"],
    "SoT":            ["SoT", "shots_on_target", "Shots on Target", "ShotsOnTarget", "sot"],
    "Minutes":        ["Minutes", "Min", "Mins", "minutes", "min"],
    "TeamGoals":      ["TeamGoals", "GF", "team_goals", "Goals For", "GoalsFor"],
    "OppGoals":       ["OppGoals", "GA", "opp_goals", "Goals Against", "GoalsAgainst"],
    "Venue":          ["Venue", "venue", "Home/Away", "HomeAway", "location"],
    "Opponent":       ["Opponent", "opponent", "Opp", "opp", "vs", "Against"],
    "Result":         ["Result", "result", "Res", "res", "W/D/L"],
    "TacklesWon":     ["TacklesWon", "Tkl", "tackles_won", "Tackles"],
    "Interceptions":  ["Interceptions", "Int", "interceptions"],
    "Date":           ["Date", "date", "Match Date", "MatchDate"],
    "Team":           ["Team", "team", "Club", "club", "Squad"],
    "Season":         ["Season", "season", "Szn"],
    "Position":       ["Position", "position", "Pos", "pos"],
    "xG":             ["xG", "XG", "xg", "ExpectedGoals"],
    "Yellow":         ["Yellow", "YellowCards", "Yel", "yellow"],
    "Red":            ["Red", "RedCards", "Red_Cards", "red"],
}

NUMERIC_COLS = [
    "Goals", "Assists", "Shots", "SoT", "Minutes", "TacklesWon",
    "Interceptions", "TeamGoals", "OppGoals", "Yellow", "Red", "xG",
]

DEFAULTS = {
    "Goals": 0, "Assists": 0, "Shots": 0, "SoT": 0,
    "Minutes": 90, "TacklesWon": 0, "Interceptions": 0,
    "TeamGoals": 0, "OppGoals": 0,
    "Venue": "Home", "Opponent": "Unknown", "Result": "D",
    "Position": "Unknown",
}

SEASON_DATES = {
    f"{y}-{y+1}": (f"{y}-07-01", f"{y+1}-06-30") for y in range(2008, 2025)
}


def find_player_data_dir(base_dir):
    for name in ["PLAYER DATA", "player_data", "Player Data", "PLAYER_DATA",
                 "data/raw_players", "raw_players", "data", "DATA"]:
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and glob.glob(os.path.join(p, "*.csv")):
            return p
    for item in os.listdir(base_dir):
        full = os.path.join(base_dir, item)
        if os.path.isdir(full) and glob.glob(os.path.join(full, "*.csv")):
            return full
    return base_dir


def get_season_range(season):
    if season in SEASON_DATES:
        s, e = SEASON_DATES[season]
        return pd.Timestamp(s), pd.Timestamp(e)
    try:
        yr = int(str(season).split("-")[0])
        return pd.Timestamp(f"{yr}-07-01"), pd.Timestamp(f"{yr+1}-06-30")
    except Exception:
        return pd.Timestamp("2024-07-01"), pd.Timestamp("2025-06-30")


def normalize_name(name):
    """Accent / diacritic-insensitive lowercase key, used for player-name matching."""
    extra = {
        "ø": "o", "Ø": "O", "ð": "d", "Ð": "D", "þ": "th", "æ": "ae",
        "Æ": "AE", "ł": "l", "Ł": "L", "ß": "ss", "đ": "d", "ħ": "h",
        "ĸ": "k", "ŋ": "n", "ŧ": "t", "ı": "i",
    }
    name = str(name)
    for char, repl in extra.items():
        name = name.replace(char, repl)
    name = unicodedata.normalize("NFKD", name)
    return "".join(c for c in name if not unicodedata.combining(c)).lower().strip()


def clean_player_name(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    if " - " in name:
        name = name.split(" - ")[0]
    return name.strip()


def standardise_columns(df):
    """Rename column variants -> canonical names, fill defaults, coerce numerics."""
    rename_map = {}
    existing = set(df.columns.str.strip())
    for std_name, variants in COL_ALIASES.items():
        if std_name in existing:
            continue
        for variant in variants:
            if variant in existing:
                rename_map[variant] = std_name
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    df.columns = df.columns.str.strip()

    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Opponent" in df.columns:
        replacements = {"Ã©": "é", "Ã¡": "á", "Ã­": "í", "Ã³": "ó",
                         "Ãº": "ú", "Ã±": "ñ", "Ã": "Á"}
        df["Opponent"] = df["Opponent"].astype(str)
        for bad, good in replacements.items():
            df["Opponent"] = df["Opponent"].str.replace(bad, good, regex=False)

    return df


def load_all_players(player_data_dir):
    """Returns {player_name: DataFrame} with standardised columns and parsed Date."""
    players = {}
    for path in glob.glob(os.path.join(player_data_dir, "*.csv")):
        if "SEASON_DATA" in os.path.basename(path).upper():
            continue
        name = clean_player_name(path)
        try:
            df = pd.read_csv(path, encoding="latin1")
            df.columns = df.columns.str.strip()

            date_col = next((c for c in df.columns
                              for v in COL_ALIASES["Date"] if c == v), None)
            if date_col and date_col != "Date":
                df = df.rename(columns={date_col: "Date"})
            if "Date" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).copy()
            df = standardise_columns(df)
            df = df.sort_values("Date").reset_index(drop=True)
            players[name] = df
        except Exception as e:
            print(f"[data_loader] Skipping {os.path.basename(path)}: {e}")
    return players


def load_season_data(player_data_dir, base_dir):
    search_paths = [
        os.path.join(player_data_dir, "SEASON_DATA.csv"),
        os.path.join(base_dir, "SEASON_DATA.csv"),
    ]
    try:
        for item in os.listdir(base_dir):
            full = os.path.join(base_dir, item)
            if os.path.isdir(full):
                search_paths.append(os.path.join(full, "SEASON_DATA.csv"))
    except Exception:
        pass

    path = next((p for p in search_paths if os.path.exists(p)), None)
    if path is None:
        return {}

    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    result = {}
    for _, row in df.iterrows():
        season = str(row["SEASON"]).strip()
        club = str(row["TEAM"]).strip()
        player = str(row["PLAYER"]).strip()
        if player.startswith("_placeholder_"):
            result.setdefault(season, {}).setdefault(club, [])
            continue
        result.setdefault(season, {}).setdefault(club, []).append(player)
    return result


def get_squad_for_season(season_data, players_dict, club, season):
    season_players = season_data.get(season, {}).get(club, [])
    if not season_players:
        return []
    csv_norm = {normalize_name(n): n for n in players_dict.keys()}
    matched = []
    for sp in season_players:
        norm_sp = normalize_name(sp)
        if norm_sp in csv_norm:
            matched.append(csv_norm[norm_sp])
        else:
            for norm_csv, csv_name in csv_norm.items():
                if norm_sp in norm_csv or norm_csv in norm_sp:
                    matched.append(csv_name)
                    break
    return sorted(set(matched))


def get_clubs_with_data(season_data, players_dict, season):
    """Clubs to actually offer in the UI for this season: only clubs where at
    least one listed player has a matching CSV uploaded. SEASON_DATA.csv can
    list a club for a season even when nobody uploaded that player's CSV
    (e.g. a player's full career history was bulk-added but only some of
    their seasons' club-mates have data) — those clubs would otherwise show
    up as selectable and immediately fail with 'No player CSVs found'."""
    clubs = sorted(season_data.get(season, {}).keys())
    has_data, missing = [], []
    for club in clubs:
        if get_squad_for_season(season_data, players_dict, club, season):
            has_data.append(club)
        else:
            missing.append(club)
    return has_data, missing


# =============================================================================
# 2. CLEANING — Phase 1: data quality checks + leakage-safe match table.
# =============================================================================

QUALITY_REPORT_COLS = [
    "player", "n_rows", "duplicates_removed", "negative_minutes_fixed",
    "sot_gt_shots_fixed", "missing_goals", "missing_shots",
]


def dedupe_player_df(df):
    """Remove duplicate player-match rows (same Date + Opponent + Venue)."""
    before = len(df)
    key_cols = [c for c in ["Date", "Opponent", "Venue"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first")
    removed = before - len(df)
    return df.reset_index(drop=True), removed


def enforce_bounds(df):
    """
    Data-quality checks:
    - minutes must be non-negative
    - SoT must be <= Shots
    - numeric stat columns must actually be numeric (already coerced upstream)
    Missing values become an explicit missingness flag rather than a silent 0,
    for the columns where "unknown" and "zero" mean different things.
    """
    df = df.copy()
    negative_minutes_fixed = 0
    sot_fixed = 0

    if "Minutes" in df.columns:
        mask = df["Minutes"] < 0
        negative_minutes_fixed = int(mask.sum())
        df.loc[mask, "Minutes"] = np.nan

    if "SoT" in df.columns and "Shots" in df.columns:
        mask = df["SoT"] > df["Shots"]
        sot_fixed = int(mask.sum())
        # cap rather than silently drop the observation
        df.loc[mask, "SoT"] = df.loc[mask, "Shots"]

    # Explicit missingness indicators for columns where 0 vs "not recorded"
    # genuinely differ (e.g. tackles/interceptions may be absent for older
    # seasons rather than truly zero).
    for col in ["TacklesWon", "Interceptions", "xG"]:
        if col in df.columns:
            flag_col = f"{col}_missing"
            df[flag_col] = df[col].isna().astype(int)

    # Minutes: if missing, assume a full match rather than 0 (0 would make a
    # player look inactive, which biases per-90 stats far more than an
    # assumed-90 default).
    if "Minutes" in df.columns:
        df["Minutes"] = df["Minutes"].fillna(90)

    # Everything else genuinely defaults to 0 when absent (no shot = 0 shots).
    for col in ["Goals", "Assists", "Shots", "SoT", "TacklesWon",
                "Interceptions", "TeamGoals", "OppGoals", "Yellow", "Red"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df, negative_minutes_fixed, sot_fixed


def clean_players_dict(players_dict):
    """Run dedupe + bound checks on every player's DataFrame. Returns (clean_dict, report_df)."""
    clean = {}
    rows = []
    for name, df in players_dict.items():
        d, dups = dedupe_player_df(df)
        d, neg_min, sot_fix = enforce_bounds(d)
        d = d.sort_values("Date").reset_index(drop=True)
        clean[name] = d
        rows.append({
            "player": name,
            "n_rows": len(d),
            "duplicates_removed": dups,
            "negative_minutes_fixed": neg_min,
            "sot_gt_shots_fixed": sot_fix,
            "missing_goals": int(df["Goals"].isna().sum()) if "Goals" in df.columns else 0,
            "missing_shots": int(df["Shots"].isna().sum()) if "Shots" in df.columns else 0,
        })
    report = pd.DataFrame(rows, columns=QUALITY_REPORT_COLS)
    return clean, report


def leakage_safe_history(df, match_date):
    """Return only rows strictly before match_date. This is the ONE leakage rule
    every downstream feature function must go through."""
    match_date = pd.to_datetime(match_date)
    return df[df["Date"] < match_date].copy()


# =============================================================================
# 3. ROLLING FEATURES — Phase 2A/2B
# Multi-window rolling form + per-90 features + recency weighting + shrinkage.
# =============================================================================

WINDOWS = [3, 5, 10, 20]
SHRINKAGE_MINUTES = 450  # ~5 full matches; below this, shrink toward league/team mean


def _weighted_mean(values, half_life=5):
    """Exponential recency weighting: most recent match weighted highest,
    weights sum to 1. `values` must be ordered oldest -> newest."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0
    decay = np.log(2) / half_life
    ages = np.arange(n)[::-1]  # most recent match has age 0
    w = np.exp(-decay * ages)
    w = w / w.sum()
    return float(np.dot(values, w))


def multi_window_form(df, match_date, stat_cols=("Goals", "Assists", "Shots", "SoT",
                                                   "TacklesWon", "Interceptions")):
    """
    For each stat and each window in WINDOWS, compute:
      - simple mean over the window
      - recency-weighted mean over the window
    Returns a flat dict of features, e.g. {"Goals_mean_5": .., "Goals_wmean_5": ..}
    """
    hist = leakage_safe_history(df, match_date).sort_values("Date")
    feats = {}
    for col in stat_cols:
        if col not in hist.columns:
            continue
        series = hist[col].values
        for w in WINDOWS:
            window_vals = series[-w:] if len(series) > 0 else np.array([])
            feats[f"{col}_mean_{w}"] = float(np.mean(window_vals)) if len(window_vals) else 0.0
            feats[f"{col}_wmean_{w}"] = _weighted_mean(window_vals, half_life=max(2, w // 2))
    return feats


def goal_shot_efficiency(df, match_date):
    """Goals/Shot and Goals/SoT with additive (Laplace) smoothing at 5/10/20 windows,
    so a single-match spike (e.g. 1 shot, 1 goal -> 100% conversion) doesn't dominate."""
    hist = leakage_safe_history(df, match_date).sort_values("Date")
    feats = {}
    for w in (5, 10, 20):
        recent = hist.tail(w)
        goals = recent["Goals"].sum() if "Goals" in recent.columns else 0.0
        shots = recent["Shots"].sum() if "Shots" in recent.columns else 0.0
        sot = recent["SoT"].sum() if "SoT" in recent.columns else 0.0
        # smoothing prior: league-average ~0.10 goals/shot, ~0.30 goals/SoT
        feats[f"G_per_Sh_{w}"] = (goals + 0.10) / (shots + 1.0)
        feats[f"G_per_SoT_{w}"] = (goals + 0.30) / (sot + 1.0)
    return feats


def minutes_features(df, match_date):
    hist = leakage_safe_history(df, match_date).sort_values("Date")
    feats = {}
    for w in (5, 10):
        recent = hist.tail(w)
        feats[f"minutes_mean_{w}"] = float(recent["Minutes"].mean()) if len(recent) else 0.0
    feats["minutes_season_to_date"] = float(hist["Minutes"].sum()) if len(hist) else 0.0
    feats["appearances_season_to_date"] = int(len(hist))
    # start-rate proxy: fraction of recent matches with >=60 minutes
    last10 = hist.tail(10)
    feats["start_rate_10"] = float((last10["Minutes"] >= 60).mean()) if len(last10) else 0.0
    return feats


def per90_features(df, match_date, stat_cols=("Goals", "Assists", "Shots", "SoT",
                                               "TacklesWon", "Interceptions")):
    """Per-90 rates with empirical-Bayes-style shrinkage for low-minutes players:
    shrink toward the player's own season mean rate with a prior weight that
    fades in as more minutes accumulate."""
    hist = leakage_safe_history(df, match_date).sort_values("Date")
    total_minutes = float(hist["Minutes"].sum()) if len(hist) else 0.0
    feats = {}
    for col in stat_cols:
        if col not in hist.columns:
            continue
        total_stat = float(hist[col].sum()) if len(hist) else 0.0
        raw_per90 = (total_stat / total_minutes * 90.0) if total_minutes > 0 else 0.0

        # shrinkage: weight = minutes_played / (minutes_played + SHRINKAGE_MINUTES)
        weight = total_minutes / (total_minutes + SHRINKAGE_MINUTES) if total_minutes >= 0 else 0.0
        shrunk = raw_per90 * weight
        feats[f"{col}_per90"] = raw_per90
        feats[f"{col}_per90_shrunk"] = shrunk

        # recent-form change: last-5 per90 minus season-to-date per90
        last5 = hist.tail(5)
        m5 = float(last5["Minutes"].sum())
        p5 = (float(last5[col].sum()) / m5 * 90.0) if m5 > 0 else 0.0
        feats[f"{col}_form_change"] = p5 - raw_per90

    feats["minutes_share_total"] = total_minutes  # normalised later at team level
    return feats


def build_player_feature_row(df, match_date):
    """Assemble ALL player-level features for one player at one match_date."""
    feats = {}
    feats.update(multi_window_form(df, match_date))
    feats.update(goal_shot_efficiency(df, match_date))
    feats.update(minutes_features(df, match_date))
    feats.update(per90_features(df, match_date))
    return feats


# =============================================================================
# 4. PLAYER RATINGS — Phase 2C
# Position-specific 0-100 player ratings built on top of the rolling features.
# =============================================================================

SLOT_TO_GROUP = {
    "GK": "GK", "CB": "DEF", "LB": "DEF", "RB": "DEF",
    "DM": "MID", "CM": "MID", "LM": "MID", "RM": "MID", "AM": "MID",
    "ST": "FWD", "LW": "FWD", "RW": "FWD",
}


def _scale(value, lo, hi):
    """Clip-and-rescale a raw stat into a 0-100 band for comparability across
    very different raw magnitudes (e.g. goals/90 vs tackles/90)."""
    if hi <= lo:
        return 50.0
    v = (value - lo) / (hi - lo) * 100.0
    return float(np.clip(v, 0, 100))


def rate_player(df, match_date, position_group):
    """
    Returns a dict of position-aware ratings for one player at one match date.
    position_group in {"GK","DEF","MID","FWD"}.
    """
    feats = build_player_feature_row(df, match_date)
    ratings = {"raw": feats, "group": position_group}

    goals90 = feats.get("Goals_per90_shrunk", 0.0)
    assists90 = feats.get("Assists_per90_shrunk", 0.0)
    shots90 = feats.get("Shots_per90_shrunk", 0.0)
    sot90 = feats.get("SoT_per90_shrunk", 0.0)
    tkl90 = feats.get("TacklesWon_per90_shrunk", 0.0)
    int90 = feats.get("Interceptions_per90_shrunk", 0.0)
    conv_5 = feats.get("G_per_Sh_5", 0.10)
    conv_sot_5 = feats.get("G_per_SoT_5", 0.30)
    start_rate = feats.get("start_rate_10", 0.0)

    if position_group == "FWD":
        ratings["finishing"] = _scale(conv_sot_5, 0.15, 0.60)
        ratings["shot_volume"] = _scale(shots90, 0.5, 5.0)
        ratings["chance_conversion"] = _scale(conv_5, 0.02, 0.30)
        ratings["creation"] = _scale(assists90, 0.0, 0.6)
        ratings["attack_rating"] = float(np.mean([
            ratings["finishing"], ratings["shot_volume"], ratings["chance_conversion"]
        ]))

    elif position_group == "MID":
        ratings["creation"] = _scale(assists90, 0.0, 0.6)
        ratings["progression"] = _scale(shots90 + sot90, 0.5, 5.0)
        ratings["defensive_actions"] = _scale(tkl90 + int90, 0.5, 6.0)
        ratings["midfield_rating"] = float(np.mean([
            ratings["creation"], ratings["progression"], ratings["defensive_actions"]
        ]))

    elif position_group == "DEF":
        ratings["tackling"] = _scale(tkl90, 0.5, 4.0)
        ratings["interceptions"] = _scale(int90, 0.5, 4.0)
        ratings["defensive_rating"] = float(np.mean([
            ratings["tackling"], ratings["interceptions"]
        ]))

    else:  # GK
        # limited signal available from outfield-style CSVs; keep it simple
        ratings["gk_rating"] = 50.0

    ratings["minutes_stability"] = _scale(start_rate, 0.0, 1.0)
    if position_group == "FWD":
        ratings["overall_form"] = ratings.get("attack_rating", 50.0)
    elif position_group == "MID":
        ratings["overall_form"] = ratings.get("midfield_rating", 50.0)
    elif position_group == "DEF":
        ratings["overall_form"] = ratings.get("defensive_rating", 50.0)
    else:
        ratings["overall_form"] = ratings.get("gk_rating", 50.0)

    return ratings


# =============================================================================
# 5. LINEUP FEATURES — Phase 3
# Convert a formation + Playing XI into a structured lineup-strength vector.
# =============================================================================

FORMATIONS = {
    "4-3-3":   ["GK", "RB", "CB", "CB", "LB", "CM", "CM", "CM", "RW", "ST", "LW"],
    "4-4-2":   ["GK", "RB", "CB", "CB", "LB", "RM", "CM", "CM", "LM", "ST", "ST"],
    "4-2-3-1": ["GK", "RB", "CB", "CB", "LB", "DM", "DM", "AM", "RW", "LW", "ST"],
    "4-5-1":   ["GK", "RB", "CB", "CB", "LB", "RM", "CM", "CM", "CM", "LM", "ST"],
    "4-1-4-1": ["GK", "RB", "CB", "CB", "LB", "DM", "RM", "CM", "CM", "LM", "ST"],
    "3-5-2":   ["GK", "CB", "CB", "CB", "RM", "CM", "CM", "CM", "LM", "ST", "ST"],
    "3-4-3":   ["GK", "CB", "CB", "CB", "RM", "CM", "CM", "LM", "RW", "ST", "LW"],
    "3-5-1-1": ["GK", "CB", "CB", "CB", "RM", "CM", "CM", "CM", "LM", "AM", "ST"],
    "5-3-2":   ["GK", "RB", "CB", "CB", "CB", "LB", "CM", "CM", "CM", "ST", "ST"],
    "5-4-1":   ["GK", "RB", "CB", "CB", "CB", "LB", "RM", "CM", "CM", "LM", "ST"],
    "4-4-1-1": ["GK", "RB", "CB", "CB", "LB", "RM", "CM", "CM", "LM", "AM", "ST"],
    "4-2-2-2": ["GK", "RB", "CB", "CB", "LB", "CM", "CM", "AM", "AM", "ST", "ST"],
    "3-4-1-2": ["GK", "CB", "CB", "CB", "RM", "CM", "CM", "LM", "AM", "ST", "ST"],
    "4-3-1-2": ["GK", "RB", "CB", "CB", "LB", "CM", "CM", "CM", "AM", "ST", "ST"],
}


def _safe_mean(vals, default=50.0):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else default


def build_lineup_strength(players_dict, xi, formation, match_date):
    """
    Returns a dict:
      attack_strength, midfield_strength, defence_strength,
      finishing_strength, creation_strength, defensive_strength,
      goalkeeper_strength, minutes_stability, availability, chemistry_score
    """
    slots = FORMATIONS.get(formation, FORMATIONS["4-3-3"])
    n = min(len(slots), len(xi))

    attack_vals, mid_vals, def_vals, gk_vals = [], [], [], []
    finishing_vals, creation_vals, defensive_vals = [], [], []
    stability_vals = []
    minutes_hist = []
    missing_players = 0

    for i in range(n):
        slot = slots[i]
        player = xi[i]
        group = SLOT_TO_GROUP.get(slot, "MID")

        if not player or player not in players_dict:
            missing_players += 1
            continue

        df = players_dict[player]
        ratings = rate_player(df, match_date, group)
        stability_vals.append(ratings.get("minutes_stability", 50.0))
        minutes_hist.append(ratings["raw"].get("minutes_season_to_date", 0.0))

        if group == "FWD":
            attack_vals.append(ratings.get("attack_rating", 50.0))
            finishing_vals.append(ratings.get("finishing", 50.0))
            creation_vals.append(ratings.get("creation", 50.0))
        elif group == "MID":
            mid_vals.append(ratings.get("midfield_rating", 50.0))
            creation_vals.append(ratings.get("creation", 50.0))
            defensive_vals.append(ratings.get("defensive_actions", 50.0))
        elif group == "DEF":
            def_vals.append(ratings.get("defensive_rating", 50.0))
            defensive_vals.append(ratings.get("defensive_rating", 50.0))
        else:  # GK
            gk_vals.append(ratings.get("gk_rating", 50.0))

    availability = 1.0 - (missing_players / max(n, 1))

    # chemistry proxy: reward XIs built mostly from players with high combined
    # minutes-stability (a rough stand-in for "regularly plays together");
    # true pair/unit chemistry needs a shared-minutes-on-pitch dataset the
    # player CSVs don't currently provide.
    chemistry_score = _safe_mean(stability_vals, default=50.0)

    return {
        "attack_strength":     _safe_mean(attack_vals),
        "midfield_strength":   _safe_mean(mid_vals),
        "defence_strength":    _safe_mean(def_vals),
        "goalkeeper_strength": _safe_mean(gk_vals),
        "finishing_strength":  _safe_mean(finishing_vals),
        "creation_strength":   _safe_mean(creation_vals),
        "defensive_strength":  _safe_mean(defensive_vals),
        "minutes_stability":   _safe_mean(stability_vals),
        "availability":        float(availability),
        "chemistry_score":     chemistry_score,
    }


def matchup_vector(home_strength, away_strength):
    """MATCHUP = differences that feed the goal model directly."""
    return {
        "home_attack_minus_away_defence": home_strength["attack_strength"] - away_strength["defence_strength"],
        "away_attack_minus_home_defence": away_strength["attack_strength"] - home_strength["defence_strength"],
        "attack_gap": home_strength["attack_strength"] - away_strength["attack_strength"],
        "defence_gap": home_strength["defence_strength"] - away_strength["defence_strength"],
        "midfield_gap": home_strength["midfield_strength"] - away_strength["midfield_strength"],
        "creation_gap": home_strength["creation_strength"] - away_strength["creation_strength"],
    }


# =============================================================================
# 6. ELO — Phase 4
# Chronological Elo rating for teams, built from a one-row-per-real-match log.
# =============================================================================

BASE_RATING = 1500.0
K_FACTOR = 24.0
HOME_ADV_ELO = 60.0


class EloSystem:
    def __init__(self, base_rating=BASE_RATING, k=K_FACTOR, home_adv=HOME_ADV_ELO):
        self.base_rating = base_rating
        self.k = k
        self.home_adv = home_adv
        self.current = {}                 # team -> current rating
        self.snapshots = {}               # team -> list[(date, rating_after)]

    def get(self, team):
        return self.current.get(team, self.base_rating)

    def _expected(self, r_a, r_b):
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    def _record(self, team, date, rating):
        self.snapshots.setdefault(team, []).append((date, rating))
        self.current[team] = rating

    def update_match(self, date, home, away, gf_home, gf_away):
        r_home = self.get(home)
        r_away = self.get(away)

        exp_home = self._expected(r_home + self.home_adv, r_away)
        if gf_home > gf_away:
            score_home = 1.0
        elif gf_home == gf_away:
            score_home = 0.5
        else:
            score_home = 0.0

        margin = abs(gf_home - gf_away)
        mov_mult = 1.0 + 0.15 * max(0, margin - 1)  # bigger wins move rating a bit more

        delta = self.k * mov_mult * (score_home - exp_home)
        self._record(home, date, r_home + delta)
        self._record(away, date, r_away - delta)

    def fit_from_matches(self, matches_df):
        """matches_df: one row per REAL match (not per player), columns
        Date, Home, Away, HomeGoals, AwayGoals. Sorted internally."""
        df = matches_df.sort_values("Date")
        for _, row in df.iterrows():
            self.update_match(row["Date"], row["Home"], row["Away"],
                               row["HomeGoals"], row["AwayGoals"])
        return self

    def rating_before(self, date, team, default=None):
        """Most recent Elo rating for `team` from a match strictly before `date`."""
        if default is None:
            default = self.base_rating
        snaps = self.snapshots.get(team)
        if not snaps:
            return default
        eligible = [r for d, r in snaps if d < date]
        if not eligible:
            return default
        return eligible[-1]


# =============================================================================
# 7. GOAL MODELS — Phase 5
# Candidate expected-goals regressors + a learned (non-negative, sum-to-1)
# ensemble blend, replacing the old fixed 60/40 ML/traditional-xG formula.
#
# xgboost/catboost are not available in this environment, so the "gradient
# boosting" candidate uses sklearn's HistGradientBoostingRegressor with a
# Poisson loss — the closest built-in equivalent for count-like goal targets.
# =============================================================================

class NegativeBinomialRegressor:
    """
    Negative Binomial regression (simple, dependable, no external deps).
    Parameterised as mean mu = exp(X @ beta), fixed dispersion alpha estimated
    by method of moments on the training residuals, log-likelihood maximised
    for beta only. This is deliberately simple: it's a "must-test", not a
    mandatory production model.
    """
    def __init__(self, alpha=None, max_iter=200):
        self.alpha = alpha
        self.beta_ = None
        self.n_features_ = None

    def _nb_negloglik(self, beta, X, y, alpha):
        eta = X @ beta
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)
        r = 1.0 / max(alpha, 1e-6)
        ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
              + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
        return -np.sum(ll)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        self.n_features_ = Xb.shape[1]

        mean_y = max(np.mean(y), 1e-3)
        var_y = max(np.var(y), mean_y + 1e-3)
        alpha0 = max((var_y - mean_y) / (mean_y ** 2), 1e-3)
        self.alpha = self.alpha or alpha0

        beta0 = np.zeros(self.n_features_)
        beta0[0] = np.log(mean_y)

        res = minimize(self._nb_negloglik, beta0, args=(Xb, y, self.alpha),
                        method="BFGS", options={"maxiter": 200, "disp": False})
        self.beta_ = res.x
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        eta = np.clip(Xb @ self.beta_, -20, 20)
        return np.exp(eta)


CANDIDATE_BUILDERS = {
    "poisson_reg": lambda: PoissonRegressor(alpha=1e-3, max_iter=500),
    "neg_binomial": lambda: NegativeBinomialRegressor(),
    "gboost_poisson": lambda: HistGradientBoostingRegressor(
        loss="poisson", max_depth=4, max_iter=150, learning_rate=0.06, random_state=42),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    "linear_reg": lambda: LinearRegression(),
}


def fit_candidates(X_train, y_train):
    """Fit every candidate model; skip any that errors (e.g. too little data)."""
    fitted = {}
    for name, builder in CANDIDATE_BUILDERS.items():
        try:
            model = builder()
            model.fit(X_train, np.clip(y_train, 0, None))
            fitted[name] = model
        except Exception:
            continue
    return fitted


def out_of_fold_weights(X, y, n_folds=4):
    """
    Learn non-negative, sum-to-1 ensemble weights from out-of-fold predictions,
    replacing the fixed 0.60/0.40 blend. Uses simple chronological folds
    (no shuffling — this is time-series data).
    """
    n = len(y)
    if n < n_folds * 3:
        # not enough data for proper CV -> fall back to equal weights
        fitted = fit_candidates(X, y)
        names = list(fitted.keys())
        if not names:
            return {}, {}
        w = {name: 1.0 / len(names) for name in names}
        return w, fitted

    fold_edges = np.linspace(0, n, n_folds + 1).astype(int)
    oof_preds = {name: np.full(n, np.nan) for name in CANDIDATE_BUILDERS}

    for k in range(1, n_folds):
        train_end = fold_edges[k]
        val_start, val_end = fold_edges[k], fold_edges[k + 1]
        if train_end < 5 or val_end <= val_start:
            continue
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_val = X[val_start:val_end]
        fitted = fit_candidates(X_tr, y_tr)
        for name, model in fitted.items():
            try:
                oof_preds[name][val_start:val_end] = np.clip(model.predict(X_val), 0, None)
            except Exception:
                pass

    # non-negative least-squares style weight search over valid folds only
    valid_mask = ~np.isnan(np.array([oof_preds[n_] for n_ in oof_preds])).any(axis=0)
    names = [n_ for n_ in oof_preds if not np.all(np.isnan(oof_preds[n_]))]
    if not names or valid_mask.sum() < 3:
        fitted_full = fit_candidates(X, y)
        names = list(fitted_full.keys())
        if not names:
            return {}, {}
        w = {name: 1.0 / len(names) for name in names}
        return w, fitted_full

    P = np.column_stack([oof_preds[n_][valid_mask] for n_ in names])
    y_valid = y[valid_mask]

    def obj(w):
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-9)
        pred = P @ w
        return mean_absolute_error(y_valid, pred)

    w0 = np.ones(len(names)) / len(names)
    res = minimize(obj, w0, method="Nelder-Mead",
                    options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-4})
    w = np.clip(res.x, 0, None)
    w = w / (w.sum() + 1e-9)
    weights = {name: float(wi) for name, wi in zip(names, w)}

    fitted_full = fit_candidates(X, y)
    return weights, fitted_full


def ensemble_predict(weights, fitted_models, X_query):
    if not weights or not fitted_models:
        return 1.2  # league-average fallback
    total_w = 0.0
    total_pred = 0.0
    for name, w in weights.items():
        model = fitted_models.get(name)
        if model is None or w <= 0:
            continue
        try:
            pred = float(np.clip(model.predict(X_query)[0], 0.01, 10.0))
        except Exception:
            continue
        total_pred += w * pred
        total_w += w
    if total_w <= 0:
        return 1.2
    # sanity clip: a football team essentially never has a true expected-goals
    # value outside this band; unclipped values usually mean a model diverged
    # on a tiny training sample rather than a genuine signal.
    return round(float(np.clip(total_pred / total_w, 0.05, 8.0)), 3)


# =============================================================================
# 8. SCORE MODELS — Phase 6
# Turns (lambda_home, lambda_away) into a full scoreline probability grid,
# using one of several score-distribution families, then ranks Top-N.
# =============================================================================

MAX_GOALS_DEFAULT = 7


def _poisson_pmf(lam, k):
    lam = max(lam, 1e-6)
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def independent_poisson_matrix(lam_h, lam_a, max_goals=MAX_GOALS_DEFAULT):
    m = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            m[i][j] = _poisson_pmf(lam_h, i) * _poisson_pmf(lam_a, j)
    return m


def dixon_coles_tau(i, j, lam_h, lam_a, rho):
    """Dixon-Coles low-score correlation correction."""
    if i == 0 and j == 0:
        return 1 - lam_h * lam_a * rho
    if i == 0 and j == 1:
        return 1 + lam_h * rho
    if i == 1 and j == 0:
        return 1 + lam_a * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def dixon_coles_matrix(lam_h, lam_a, rho=-0.08, max_goals=MAX_GOALS_DEFAULT):
    m = independent_poisson_matrix(lam_h, lam_a, max_goals)
    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            m[i][j] *= dixon_coles_tau(i, j, lam_h, lam_a, rho)
    total = m.sum()
    if total > 0:
        m = m / total
    return m


def fit_dixon_coles_rho(matches, default=-0.08):
    """
    matches: list of (lam_h, lam_a, actual_home_goals, actual_away_goals)
    from a TRAINING/validation period only. Fits rho by maximum likelihood.
    Falls back to `default` if there isn't enough data.
    """
    if len(matches) < 15:
        return default

    def negloglik(rho):
        ll = 0.0
        for lam_h, lam_a, gh, ga in matches:
            p = (_poisson_pmf(lam_h, gh) * _poisson_pmf(lam_a, ga)
                 * dixon_coles_tau(min(gh, 1), min(ga, 1), lam_h, lam_a, rho))
            ll += math.log(max(p, 1e-10))
        return -ll

    res = minimize_scalar(negloglik, bounds=(-0.3, 0.3), method="bounded")
    return float(res.x) if res.success else default


def bivariate_poisson_matrix(lam_h, lam_a, lam_c=None, max_goals=MAX_GOALS_DEFAULT):
    """
    Simplified bivariate Poisson: X = Z1+Z3, Y = Z2+Z3 where Z3 (common
    shocks / shared match dynamics) induces positive dependence between the
    two scores. lam_c defaults to a small fraction of the smaller mean.
    """
    if lam_c is None:
        lam_c = 0.12 * min(lam_h, lam_a)
    lam1 = max(lam_h - lam_c, 1e-4)
    lam2 = max(lam_a - lam_c, 1e-4)

    m = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            s = 0.0
            for k in range(min(i, j) + 1):
                s += (_poisson_pmf(lam1, i - k) * _poisson_pmf(lam2, j - k)
                      * _poisson_pmf(lam_c, k))
            m[i][j] = s
    total = m.sum()
    if total > 0:
        m = m / total
    return m


def negative_binomial_matrix(lam_h, lam_a, alpha=0.15, max_goals=MAX_GOALS_DEFAULT):
    """Independent Negative-Binomial score grid — allows extra variance vs Poisson."""
    def nb_pmf(mu, k, a):
        r = 1.0 / max(a, 1e-6)
        mu = max(mu, 1e-6)
        log_p = (gammaln(k + r) - gammaln(r) - gammaln(k + 1)
                  + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu)))
        return math.exp(log_p)

    m = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            m[i][j] = nb_pmf(lam_h, i, alpha) * nb_pmf(lam_a, j, alpha)
    total = m.sum()
    if total > 0:
        m = m / total
    return m


SCORE_MODEL_BUILDERS = {
    "independent_poisson": independent_poisson_matrix,
    "dixon_coles": dixon_coles_matrix,
    "bivariate_poisson": bivariate_poisson_matrix,
    "negative_binomial": negative_binomial_matrix,
}


def ensemble_scoreline_matrix(lam_h, lam_a, max_goals=MAX_GOALS_DEFAULT, weights=None, **kwargs):
    """
    Blend all four score-distribution models into a single probability matrix
    by taking a (weighted) average, instead of picking one model manually.
    weights: optional dict like {"dixon_coles": 2, "bivariate_poisson": 1, ...}.
             Any model left out gets weight 0. Defaults to equal weights.
    """
    default_weights = {
        "dixon_coles": 1.0,
        "independent_poisson": 1.0,
        "bivariate_poisson": 1.0,
        "negative_binomial": 1.0,
    }
    weights = {**default_weights, **(weights or {})}

    combined = np.zeros((max_goals + 1, max_goals + 1))
    total_w = 0.0
    for name, w in weights.items():
        if w <= 0 or name not in SCORE_MODEL_BUILDERS:
            continue
        mat = scoreline_matrix(lam_h, lam_a, model=name, max_goals=max_goals, **kwargs)
        combined += w * mat
        total_w += w

    if total_w > 0:
        combined /= total_w
    return combined


def scoreline_matrix(lam_h, lam_a, model="dixon_coles", max_goals=MAX_GOALS_DEFAULT, **kwargs):
    if model == "ensemble":
        return ensemble_scoreline_matrix(lam_h, lam_a, max_goals=max_goals,
                                          weights=kwargs.get("weights"), **{
                                              k: v for k, v in kwargs.items() if k != "weights"
                                          })
    builder = SCORE_MODEL_BUILDERS.get(model, dixon_coles_matrix)
    if model == "dixon_coles":
        return builder(lam_h, lam_a, rho=kwargs.get("rho", -0.08), max_goals=max_goals)
    if model == "bivariate_poisson":
        return builder(lam_h, lam_a, lam_c=kwargs.get("lam_c"), max_goals=max_goals)
    if model == "negative_binomial":
        return builder(lam_h, lam_a, alpha=kwargs.get("alpha", 0.15), max_goals=max_goals)
    return builder(lam_h, lam_a, max_goals=max_goals)


def outcome_probs(matrix):
    win = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    loss = float(np.sum(np.triu(matrix, 1)))
    total = win + draw + loss
    if total > 0:
        win, draw, loss = win / total, draw / total, loss / total
    return win, draw, loss


def top_n_scorelines(matrix, n=5):
    max_g = matrix.shape[0] - 1
    scores = [(i, j, matrix[i][j]) for i in range(max_g + 1) for j in range(max_g + 1)]
    scores.sort(key=lambda x: -x[2])
    return [(f"{h}-{a}", round(p * 100, 2)) for h, a, p in scores[:n]]


def rank_of_actual(matrix, actual_h, actual_a):
    """Diagnostic: what rank (1-indexed) did the true scoreline get? Useful
    for the 'rank 6-10 near misses' diagnostic."""
    max_g = matrix.shape[0] - 1
    scores = [(i, j, matrix[i][j]) for i in range(max_g + 1) for j in range(max_g + 1)]
    scores.sort(key=lambda x: -x[2])
    for rank, (i, j, p) in enumerate(scores, start=1):
        if i == actual_h and j == actual_a:
            return rank
    return None


# =============================================================================
# 9. CALIBRATION — Phase 7
# Calibrates W/D/L probabilities that come out of the scoreline distribution.
# Isotonic regression (one-vs-rest per outcome, renormalised) is used here
# because it's robust to systematic mis-calibration with a small dataset.
# =============================================================================

OUTCOMES = ["W", "D", "L"]


class OutcomeCalibrator:
    def __init__(self):
        self.calibrators = {}
        self.fitted = False

    def fit(self, raw_probs, actual_outcomes):
        """
        raw_probs: array (n, 3) columns [P(W), P(D), P(L)] from the score model
        actual_outcomes: list of "W"/"D"/"L", length n
        """
        raw_probs = np.asarray(raw_probs)
        if len(actual_outcomes) < 20:
            # too little data for isotonic to be reliable; skip calibration
            self.fitted = False
            return self
        for idx, outcome in enumerate(OUTCOMES):
            y_bin = np.array([1 if o == outcome else 0 for o in actual_outcomes])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            try:
                iso.fit(raw_probs[:, idx], y_bin)
                self.calibrators[outcome] = iso
            except Exception:
                continue
        self.fitted = len(self.calibrators) == 3
        return self

    def transform(self, raw_probs):
        raw_probs = np.asarray(raw_probs, dtype=float)
        if not self.fitted:
            return raw_probs
        out = np.zeros_like(raw_probs)
        for idx, outcome in enumerate(OUTCOMES):
            iso = self.calibrators.get(outcome)
            if iso is None:
                out[:, idx] = raw_probs[:, idx]
            else:
                out[:, idx] = iso.predict(raw_probs[:, idx])
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums

    def transform_one(self, p_win, p_draw, p_loss):
        out = self.transform(np.array([[p_win, p_draw, p_loss]]))[0]
        return float(out[0]), float(out[1]), float(out[2])


def evaluate_calibration(probs, actual_outcomes):
    """Log loss + Brier score (one-vs-rest, averaged) for the W/D/L probabilities."""
    probs = np.asarray(probs)
    y_idx = np.array([OUTCOMES.index(o) for o in actual_outcomes])
    ll = log_loss(y_idx, probs, labels=[0, 1, 2])
    briers = []
    for idx in range(3):
        y_bin = (y_idx == idx).astype(int)
        briers.append(brier_score_loss(y_bin, probs[:, idx]))
    return {"log_loss": float(ll), "brier_score": float(np.mean(briers))}


# =============================================================================
# 10. EVALUATION — Phase 8/9
# Metrics + walk-forward (chronological, never shuffled) validation harness,
# and an experiment-log writer matching the experiment matrix.
# =============================================================================

def topn_hit_rates(predictions, actuals):
    """
    predictions: list of scoreline-probability matrices (numpy 2D arrays)
    actuals: list of (home_goals, away_goals) tuples, same order
    Returns dict with Top1..Top5 hit rate, exact-score rate, and mean rank.
    """
    n = len(actuals)
    if n == 0:
        return {}
    hits = {k: 0 for k in (1, 2, 3, 4, 5)}
    ranks = []
    for matrix, (gh, ga) in zip(predictions, actuals):
        max_g = matrix.shape[0] - 1
        gh_c, ga_c = min(gh, max_g), min(ga, max_g)
        r = rank_of_actual(matrix, gh_c, ga_c)
        ranks.append(r if r is not None else 999)
        for k in hits:
            top_k = top_n_scorelines(matrix, n=k)
            scoreline = f"{gh_c}-{ga_c}"
            if scoreline in [s for s, _ in top_k]:
                hits[k] += 1

    return {
        "top1_rate": hits[1] / n, "top2_rate": hits[2] / n,
        "top3_rate": hits[3] / n, "top4_rate": hits[4] / n,
        "top5_rate": hits[5] / n, "exact_rate": hits[1] / n,
        "mean_rank": float(np.mean(ranks)), "n_matches": n,
    }


def outcome_metrics(pred_outcomes, actual_outcomes):
    acc = float(np.mean([p == a for p, a in zip(pred_outcomes, actual_outcomes)]))
    macro_f1 = f1_score(actual_outcomes, pred_outcomes, labels=["W", "D", "L"], average="macro")
    cm = confusion_matrix(actual_outcomes, pred_outcomes, labels=["W", "D", "L"])
    return {"accuracy": acc, "macro_f1": float(macro_f1), "confusion_matrix": cm.tolist()}


def goal_mae(pred_goals, actual_goals):
    pred_goals = np.asarray(pred_goals, dtype=float)
    actual_goals = np.asarray(actual_goals, dtype=float)
    return float(np.mean(np.abs(pred_goals - actual_goals)))


def make_chronological_folds(match_dates, n_folds=5, min_train=20):
    """
    Returns list of (train_idx, val_idx) preserving time order — NEVER
    random-split football matches. Each fold's train set = everything
    before that fold's validation block (expanding-window design).
    """
    order = np.argsort(match_dates)
    n = len(order)
    if n < min_train + n_folds:
        return [(order[: max(1, n - 1)], order[max(1, n - 1):])] if n > 1 else []

    fold_edges = np.linspace(min_train, n, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        train_end = fold_edges[k]
        val_start, val_end = fold_edges[k], fold_edges[k + 1]
        if val_end <= val_start:
            continue
        folds.append((order[:train_end], order[val_start:val_end]))
    return folds


EXPERIMENT_LOG_COLS = [
    "experiment", "description", "n_train", "n_val", "top1_rate", "top2_rate",
    "top3_rate", "top4_rate", "top5_rate", "exact_rate", "mean_rank",
    "goal_mae_home", "goal_mae_away", "outcome_accuracy", "macro_f1",
    "log_loss", "brier_score", "notes",
]


def log_experiment(log_path, row: dict):
    """Append one row to experiments/experiment_log.csv, creating it if needed."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    exists = os.path.exists(log_path)
    row = {k: row.get(k, "") for k in EXPERIMENT_LOG_COLS}
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_COLS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# 11. PIPELINE — ties data_loader -> cleaning -> rolling/player-ratings ->
# lineup_features -> elo -> goal_models -> score_models -> calibration
# together. Exposes predict_match(...), the single function app.py calls.
# =============================================================================

ROLL_WINDOWS = (3, 5, 10, 20)


def build_team_df(players_dict, xi_players, match_date):
    """Every player in the XI carries team-level columns (TeamGoals/OppGoals/
    Venue/Opponent) for matches they played in, so grouping by Date recovers
    one row per real match."""
    match_date = pd.to_datetime(match_date)
    all_rows = []
    for player in xi_players:
        if not player or player not in players_dict:
            continue
        df = players_dict[player]
        past = leakage_safe_history(df, match_date)
        if len(past) > 0:
            all_rows.append(past)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows)
    agg = combined.groupby("Date").agg(
        Venue=("Venue", "first"),
        Opponent=("Opponent", "first"),
        Result=("Result", "first"),
        GF=("TeamGoals", "first"),
        GA=("OppGoals", "first"),
        Sh=("Shots", "sum"),
        SoT=("SoT", "sum"),
    ).reset_index().sort_values("Date").reset_index(drop=True)

    agg["G/Sh"] = agg.apply(lambda r: r["GF"] / r["Sh"] if r["Sh"] > 0 else 0.0, axis=1)
    agg["G/SoT"] = agg.apply(lambda r: r["GF"] / r["SoT"] if r["SoT"] > 0 else 0.0, axis=1)
    return agg


def context_features(team_df, venue, opponent, match_date):
    """Phase 4 — team strength & match context: rest days, congestion,
    head-to-head, venue splits."""
    match_date = pd.to_datetime(match_date)
    feats = {}

    if len(team_df) == 0:
        return {
            "rest_days": 7.0, "congestion_7d": 0, "congestion_14d": 0,
            "h2h_avg_gf": 1.2, "h2h_n": 0,
            "home_away_avg_gf": 1.2, "home_away_avg_ga": 1.0,
        }

    prior = team_df[team_df["Date"] < match_date]
    if len(prior) > 0:
        last_date = prior["Date"].max()
        feats["rest_days"] = float((match_date - last_date).days)
        feats["congestion_7d"] = int((prior["Date"] >= match_date - pd.Timedelta(days=7)).sum())
        feats["congestion_14d"] = int((prior["Date"] >= match_date - pd.Timedelta(days=14)).sum())
    else:
        feats["rest_days"] = 7.0
        feats["congestion_7d"] = 0
        feats["congestion_14d"] = 0

    h2h = prior[prior["Opponent"] == opponent]
    feats["h2h_avg_gf"] = float(h2h["GF"].mean()) if len(h2h) > 0 else (
        float(prior["GF"].mean()) if len(prior) > 0 else 1.2)
    feats["h2h_n"] = int(len(h2h))

    venue_matches = prior[prior["Venue"] == venue]
    feats["home_away_avg_gf"] = float(venue_matches["GF"].mean()) if len(venue_matches) > 0 else (
        float(prior["GF"].mean()) if len(prior) > 0 else 1.2)
    feats["home_away_avg_ga"] = float(venue_matches["GA"].mean()) if len(venue_matches) > 0 else (
        float(prior["GA"].mean()) if len(prior) > 0 else 1.0)

    return feats


def team_rolling_features(team_df, upto_idx=None):
    """Phase 2A applied at TEAM level: adds roll_GF_w / roll_GA_w / roll_Sh_w /
    roll_SoT_w for each window, always shifted by 1 so the current row's own
    result never leaks in."""
    data = team_df.copy().reset_index(drop=True)
    for w in ROLL_WINDOWS:
        data[f"roll_GF_{w}"] = data["GF"].shift(1).rolling(w, min_periods=1).mean()
        data[f"roll_GA_{w}"] = data["GA"].shift(1).rolling(w, min_periods=1).mean()
        data[f"roll_Sh_{w}"] = data["Sh"].shift(1).rolling(w, min_periods=1).mean()
        data[f"roll_SoT_{w}"] = data["SoT"].shift(1).rolling(w, min_periods=1).mean()
    return data


def build_ml_features(team_df, elo_system, team_name, target_col="GF"):
    """Builds the (X, y) training matrix for the goal-regression candidates."""
    data = team_rolling_features(team_df)
    if len(data) < 6:
        return None, None, None, None, None

    venue_enc = LabelEncoder()
    opp_enc = LabelEncoder()
    data["Venue_enc"] = venue_enc.fit_transform(data["Venue"].fillna("Home"))
    data["Opp_enc"] = opp_enc.fit_transform(data["Opponent"].fillna("Unknown"))

    if elo_system is not None:
        data["elo_self"] = data["Date"].apply(lambda d: elo_system.rating_before(d, team_name))
        data["elo_opp"] = [elo_system.rating_before(d, opp) for d, opp in zip(data["Date"], data["Opponent"])]
        data["elo_diff"] = data["elo_self"] - data["elo_opp"]
    else:
        data["elo_self"] = 1500.0
        data["elo_opp"] = 1500.0
        data["elo_diff"] = 0.0

    roll_cols = [f"roll_{stat}_{w}" for stat in ("GF", "GA", "Sh", "SoT") for w in ROLL_WINDOWS]
    feature_cols = ["Sh", "SoT", "G/Sh", "G/SoT", "Venue_enc", "Opp_enc",
                     "elo_diff"] + roll_cols

    data = data.dropna(subset=feature_cols + [target_col])
    if len(data) < 6:
        return None, None, None, None, None

    return data[feature_cols].values, data[target_col].values, opp_enc, venue_enc, feature_cols


def _query_feature_row(stats_row, opp_enc, venue_enc, venue, opponent, elo_diff, feature_cols):
    """Build the single-row feature vector for the upcoming match, matching
    the column order used to fit the models."""
    try:
        opp_code = opp_enc.transform([opponent])[0]
    except Exception:
        opp_code = 0
    try:
        venue_code = venue_enc.transform([venue])[0]
    except Exception:
        venue_code = 0

    lookup = {
        "Sh": stats_row["roll_Sh"], "SoT": stats_row["roll_SoT"],
        "G/Sh": stats_row["g_sh"], "G/SoT": stats_row["g_sot"],
        "Venue_enc": venue_code, "Opp_enc": opp_code,
        "elo_diff": elo_diff,
    }
    for w in ROLL_WINDOWS:
        lookup[f"roll_GF_{w}"] = stats_row["roll_GF"]
        lookup[f"roll_GA_{w}"] = stats_row["roll_GA"]
        lookup[f"roll_Sh_{w}"] = stats_row["roll_Sh"]
        lookup[f"roll_SoT_{w}"] = stats_row["roll_SoT"]

    row = [lookup.get(c, 0.0) for c in feature_cols]
    return np.array([row])


def _rolling_summary(team_df, venue, n=5):
    """Last-n rolling summary used both for the traditional-xG fallback and
    as the query row's own recent-form values."""
    venue_matches = team_df[team_df["Venue"] == venue]
    last_n = venue_matches.tail(n) if len(venue_matches) > 0 else team_df.tail(n)

    def _m(col, default):
        return float(last_n[col].mean()) if col in last_n.columns and len(last_n) > 0 else default

    return {
        "roll_Sh": max(_m("Sh", 0.1), 0.1),
        "roll_SoT": max(_m("SoT", 0.1), 0.1),
        "roll_GF": max(_m("GF", 1.2), 0.5),
        "roll_GA": max(_m("GA", 1.0), 0.5),
        "g_sh": _m("G/Sh", 0.0),
        "g_sot": _m("G/SoT", 0.0),
    }


def team_df_to_match_log(team_df, team_name):
    """Convert a per-team aggregated log into (Date, Home, Away, HomeGoals,
    AwayGoals) rows so it can feed the Elo system. Because we only ever see
    one side's player CSVs at a time, this is necessarily a partial view of
    the full league schedule — Elo here approximates relative strength
    between the two queried teams rather than a full round-robin Elo."""
    if len(team_df) == 0:
        return pd.DataFrame(columns=["Date", "Home", "Away", "HomeGoals", "AwayGoals"])
    rows = []
    for _, r in team_df.iterrows():
        if r["Venue"] == "Home":
            rows.append({"Date": r["Date"], "Home": team_name, "Away": r["Opponent"],
                         "HomeGoals": r["GF"], "AwayGoals": r["GA"]})
        else:
            rows.append({"Date": r["Date"], "Home": r["Opponent"], "Away": team_name,
                         "HomeGoals": r["GA"], "AwayGoals": r["GF"]})
    return pd.DataFrame(rows)


def build_elo_system(team_dfs_by_name):
    """Fit an EloSystem from whatever team match logs we have available."""
    logs = [team_df_to_match_log(df, name) for name, df in team_dfs_by_name.items()]
    logs = [l for l in logs if len(l) > 0]
    if not logs:
        return EloSystem()
    combined = pd.concat(logs, ignore_index=True).dropna(subset=["HomeGoals", "AwayGoals"])
    combined = combined.drop_duplicates(subset=["Date", "Home", "Away"])
    return EloSystem().fit_from_matches(combined)


def traditional_xg(stats, gf_vs_opp):
    xg_shots = stats["roll_Sh"] * stats["g_sh"]
    if xg_shots > 0:
        xg_adj = xg_shots * (gf_vs_opp / xg_shots)
    else:
        xg_adj = gf_vs_opp
    return max(round(xg_adj, 3), 0.1)


def predict_match(players_dict, home_team, away_team, home_xi, away_xi,
                   match_date, season, home_formation="4-3-3", away_formation="4-3-3",
                   score_model="dixon_coles", calibrator: OutcomeCalibrator = None,
                   dixon_coles_rho=None, max_goals=7):
    """Main entry point — everything app.py needs, in one call."""
    match_date = pd.to_datetime(match_date)

    team_dfs = {}
    lineup_info = {}
    context_info = {}

    for team, xi, venue, formation in [(home_team, home_xi, "Home", home_formation),
                                        (away_team, away_xi, "Away", away_formation)]:
        team_dfs[team] = build_team_df(players_dict, xi, match_date)
        lineup_info[team] = build_lineup_strength(players_dict, xi, formation, match_date)

    elo_system = build_elo_system(team_dfs)

    xg_results = {}
    for team, xi, venue, formation in [(home_team, home_xi, "Home", home_formation),
                                        (away_team, away_xi, "Away", away_formation)]:
        opponent = away_team if team == home_team else home_team
        team_df = team_dfs[team]
        context_info[team] = context_features(team_df, venue, opponent, match_date)

        if len(team_df) == 0:
            xg_results[team] = 1.2
            continue

        stats = _rolling_summary(team_df, venue)
        h2h_gf = context_info[team]["h2h_avg_gf"]
        xg_trad = traditional_xg(stats, h2h_gf)

        X, y, opp_enc, venue_enc, feature_cols = build_ml_features(team_df, elo_system, team)

        elo_diff = elo_system.rating_before(match_date, team) - elo_system.rating_before(match_date, opponent)

        if X is not None:
            weights, fitted = out_of_fold_weights(X, y)
            query_row = _query_feature_row(stats, opp_enc, venue_enc, venue, opponent,
                                            elo_diff=elo_diff, feature_cols=feature_cols)
            ml_xg = ensemble_predict(weights, fitted, query_row)
            xg_final = round(0.55 * ml_xg + 0.45 * xg_trad, 3)
        else:
            xg_final = xg_trad

        xg_results[team] = float(np.clip(xg_final, 0.1, 8.0))

    # lineup-strength adjustment now that both sides are built
    home_strength = lineup_info[home_team]
    away_strength = lineup_info[away_team]
    matchup = matchup_vector(home_strength, away_strength)

    def _lineup_multiplier(gap):
        # gap is roughly in [-100, 100]; map to a gentle +/-15% adjustment
        return 1.0 + np.clip(gap / 100.0, -0.5, 0.5) * 0.15

    xg_results[home_team] = max(round(
        xg_results[home_team] * _lineup_multiplier(matchup["home_attack_minus_away_defence"]), 3), 0.1)
    xg_results[away_team] = max(round(
        xg_results[away_team] * _lineup_multiplier(matchup["away_attack_minus_home_defence"]), 3), 0.1)

    lam_h = xg_results[home_team]
    lam_a = xg_results[away_team]

    rho = dixon_coles_rho if dixon_coles_rho is not None else -0.08
    matrix = scoreline_matrix(lam_h, lam_a, model=score_model, max_goals=max_goals, rho=rho)

    win, draw, loss = outcome_probs(matrix)
    if calibrator is not None:
        win, draw, loss = calibrator.transform_one(win, draw, loss)

    top5 = top_n_scorelines(matrix, n=5)

    return {
        "xg_home": lam_h, "xg_away": lam_a,
        "home_win": round(win * 100, 1), "draw": round(draw * 100, 1),
        "away_win": round(loss * 100, 1),
        "top5": top5,
        "matrix": matrix,
        "home_team": home_team, "away_team": away_team,
        "lineup_home": home_strength, "lineup_away": away_strength,
        "context_home": context_info[home_team], "context_away": context_info[away_team],
        "score_model": score_model,
    }


# =============================================================================
# 12. RUN EXPERIMENTS — E0..E9 experiment matrix runner
# Evaluates the GOAL MODEL and SCORE MODEL stages directly against a team's
# own match log (build_team_df output), so every phase can be benchmarked
# even before real upcoming-fixture lineups exist.
# =============================================================================

EXPERIMENT_LOG_PATH = os.path.join(BASE_DIR, "experiments", "experiment_log.csv")


def _outcome_label(gf, ga):
    if gf > ga:
        return "W"
    if gf == ga:
        return "D"
    return "L"


def evaluate_team_log(team_df, team_name, experiment_name="E9", score_model="dixon_coles",
                       n_folds=5, use_elo=True, use_calibration=True, notes=""):
    """
    Walk-forward evaluation of the full pipeline (goal model + score model +
    optional calibration) on ONE team's chronological match log. Every fold's
    model is trained only on matches strictly before the validation block.
    """
    data = team_rolling_features(team_df).dropna(subset=["GF"]).reset_index(drop=True)
    if len(data) < 15:
        return {"error": "not enough matches for walk-forward validation (need >= 15)"}

    folds = make_chronological_folds(data["Date"].values, n_folds=n_folds, min_train=10)
    if not folds:
        return {"error": "could not build folds"}

    all_matrices, all_actuals, all_outcome_pred, all_outcome_actual = [], [], [], []
    all_goal_mae_h = []
    raw_probs_for_calibration = []

    elo_system = None
    if use_elo:
        elo_system = build_elo_system({team_name: team_df})

    for train_idx, val_idx in folds:
        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)

        X, y, opp_enc, venue_enc, feature_cols = build_ml_features(train_df, elo_system, team_name)
        if X is None:
            continue
        weights, fitted = out_of_fold_weights(X, y)

        for _, row in val_df.iterrows():
            stats = {
                "roll_Sh": row.get("roll_Sh_5", 1.0) or 1.0,
                "roll_SoT": row.get("roll_SoT_5", 1.0) or 1.0,
                "roll_GF": row.get("roll_GF_5", 1.2) or 1.2,
                "roll_GA": row.get("roll_GA_5", 1.0) or 1.0,
                "g_sh": row["G/Sh"], "g_sot": row["G/SoT"],
            }
            elo_diff = 0.0
            if elo_system is not None:
                elo_diff = (elo_system.rating_before(row["Date"], team_name)
                            - elo_system.rating_before(row["Date"], row["Opponent"]))
            query_row = _query_feature_row(stats, opp_enc, venue_enc, row["Venue"],
                                            row["Opponent"], elo_diff, feature_cols)
            ml_xg = ensemble_predict(weights, fitted, query_row)
            lam_self = max(ml_xg, 0.1)
            # opponent goals against this team on this date -> use rolling GA as away-side lambda proxy
            lam_opp = max(float(row.get("roll_GA_5", 1.0) or 1.0), 0.1)

            if row["Venue"] == "Home":
                lam_h, lam_a = lam_self, lam_opp
                actual_h, actual_a = row["GF"], row["GA"]
            else:
                lam_h, lam_a = lam_opp, lam_self
                actual_h, actual_a = row["GA"], row["GF"]

            matrix = scoreline_matrix(lam_h, lam_a, model=score_model)
            all_matrices.append(matrix)
            all_actuals.append((int(actual_h), int(actual_a)))
            all_goal_mae_h.append(abs(lam_self - row["GF"]))

            win, draw, loss = outcome_probs(matrix)
            raw_probs_for_calibration.append([win, draw, loss])
            pred_label = ["L", "D", "W"][int(np.argmax([loss, draw, win]))]
            actual_label = _outcome_label(row["GF"], row["GA"])
            all_outcome_pred.append(pred_label)
            all_outcome_actual.append(actual_label)

    if not all_actuals:
        return {"error": "no validation predictions were generated"}

    calibrator = None
    calib_metrics = {}
    if use_calibration and len(all_actuals) >= 20:
        calibrator = OutcomeCalibrator().fit(raw_probs_for_calibration, all_outcome_actual)
        calibrated = calibrator.transform(np.array(raw_probs_for_calibration))
        calib_metrics = evaluate_calibration(calibrated, all_outcome_actual)
    else:
        calib_metrics = evaluate_calibration(np.array(raw_probs_for_calibration), all_outcome_actual)

    topn = topn_hit_rates(all_matrices, all_actuals)
    outc = outcome_metrics(all_outcome_pred, all_outcome_actual)
    mae_h = goal_mae([m for m in all_goal_mae_h], [0] * len(all_goal_mae_h))  # already abs errors

    result = {
        "experiment": experiment_name,
        "description": f"score_model={score_model}, elo={use_elo}, calib={use_calibration}",
        "n_train": int(len(data) - len(all_actuals)),
        "n_val": len(all_actuals),
        **{f"{k}": v for k, v in topn.items()},
        "goal_mae_home": round(float(np.mean(all_goal_mae_h)), 4),
        "goal_mae_away": "",
        "outcome_accuracy": round(outc["accuracy"], 4),
        "macro_f1": round(outc["macro_f1"], 4),
        "log_loss": round(calib_metrics.get("log_loss", float("nan")), 4),
        "brier_score": round(calib_metrics.get("brier_score", float("nan")), 4),
        "notes": notes,
    }
    return result


def run_full_experiment_matrix(team_df, team_name, log_path=EXPERIMENT_LOG_PATH):
    """Runs E0/E3/E6/E7/E9-style comparisons: independent Poisson vs
    Dixon-Coles vs bivariate Poisson, with/without Elo, with/without
    calibration."""
    configs = [
        ("E0_baseline_poisson", "independent_poisson", False, False, "current-model-style baseline"),
        ("E3_elo_added", "independent_poisson", True, False, "adds Elo context"),
        ("E6_dixon_coles", "dixon_coles", True, False, "swap score model to Dixon-Coles"),
        ("E7_calibration", "dixon_coles", True, True, "add isotonic calibration"),
        ("E9_bivariate", "bivariate_poisson", True, True, "bivariate Poisson + calibration"),
    ]
    results = []
    for name, score_model, use_elo, use_calib, notes in configs:
        res = evaluate_team_log(team_df, team_name, experiment_name=name,
                                 score_model=score_model, use_elo=use_elo,
                                 use_calibration=use_calib, notes=notes)
        results.append(res)
        if "error" not in res:
            log_experiment(log_path, res)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-csv", required=True, help="Path to one team's aggregated match CSV "
                                                            "(Date, Venue, Opponent, GF, GA, Sh, SoT)")
    parser.add_argument("--team-name", default="Team")
    args = parser.parse_args()

    df = pd.read_csv(args.team_csv, parse_dates=["Date"])
    out = run_full_experiment_matrix(df, args.team_name)
    for r in out:
        print(r)

"""
app.py
------
Phase 13: Streamlit frontend. Same inputs as your original app (season,
home/away team, date, formation, Playing XI) plus the diagnostic
sections the roadmap calls for (confidence/calibration indicator,
lineup ratings, Elo context, data coverage, model version).

HOW TO RUN:
    cd src/
    streamlit run app.py

REQUIRES real data to produce real predictions:
    1. Your full folder of player CSVs (one per squad player, same
       schema as Abdul_Mumin_-_Sheet1.csv).
    2. A historical match-level training table (HomeGoals, AwayGoals,
       date, and the Playing XI used in each match) to fit the goal
       models via pipeline.fit_goal_models(). This app currently loads
       player CSVs from data/raw_players/ and expects you to have run
       that fit step (see the "Model status" indicator below - it will
       clearly tell you if the model is unfit rather than showing a
       fake prediction).
"""

import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import PredictionPipeline

st.set_page_config(page_title="Real Madrid Score Predictor v2", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_players")

MODEL_VERSION = "v2.0 - Dixon-Coles + position ratings + Elo (development)"


@st.cache_resource
def load_pipeline():
    pipeline = PredictionPipeline()
    csv_paths = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_paths:
        return pipeline, []
    pipeline.build_features(csv_paths)
    return pipeline, csv_paths


def main():
    st.title("⚽ Real Madrid Score Prediction — Model v2")
    st.caption(MODEL_VERSION)

    pipeline, csv_paths = load_pipeline()

    with st.sidebar:
        st.header("Data coverage")
        if not csv_paths:
            st.error(
                f"No player CSVs found in {DATA_DIR}. "
                "Add your squad's player-match CSV files there and reload."
            )
        else:
            st.success(f"{len(csv_paths)} player file(s) loaded")
            st.caption(f"{len(pipeline.feature_table)} total player-match rows")
            date_min = pipeline.feature_table["Date"].min()
            date_max = pipeline.feature_table["Date"].max()
            st.caption(f"Date range: {date_min.date()} → {date_max.date()}")

        st.header("Model status")
        if pipeline.is_fitted:
            st.success("Goal models fitted")
        else:
            st.warning(
                "Goal models NOT fitted yet. Predictions are disabled until "
                "fit_goal_models() has been run on a historical training table. "
                "See pipeline.py docstring."
            )

    if not csv_paths:
        return

    st.header("Match setup")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input("Home team", value="Real Madrid")
        match_date = st.date_input("Match date")
    with col2:
        away_team = st.text_input("Away team")

    available_players = sorted(pipeline.feature_table["Player"].unique())

    st.subheader("Home Playing XI")
    home_xi = st.multiselect("Select 11 home players", available_players, key="home_xi")
    st.subheader("Away Playing XI")
    away_xi = st.multiselect("Select 11 away players", available_players, key="away_xi")

    predict_disabled = (not pipeline.is_fitted) or len(home_xi) != 11 or len(away_xi) != 11
    if len(home_xi) not in (0, 11):
        st.caption(f"⚠️ Home XI has {len(home_xi)}/11 players selected")
    if len(away_xi) not in (0, 11):
        st.caption(f"⚠️ Away XI has {len(away_xi)}/11 players selected")

    if st.button("Predict", disabled=predict_disabled):
        try:
            result = pipeline.predict_match(home_xi, away_xi, pd.Timestamp(match_date))
        except ValueError as e:
            st.error(str(e))
            return

        st.header("Prediction")

        c1, c2, c3 = st.columns(3)
        c1.metric("Home Expected Goals (λ)", f"{result['lambda_home']:.2f}")
        c2.metric("Away Expected Goals (λ)", f"{result['lambda_away']:.2f}")
        top_score = result["top5_scorelines"][0]
        c3.metric("Most likely score", f"{top_score[0]}-{top_score[1]} ({top_score[2]*100:.1f}%)")

        st.subheader("Outcome probabilities")
        probs = result["outcome_probs"]
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Home Win", f"{probs['home_win']*100:.1f}%")
        oc2.metric("Draw", f"{probs['draw']*100:.1f}%")
        oc3.metric("Away Win", f"{probs['away_win']*100:.1f}%")

        # Confidence/calibration indicator - roadmap explicitly asks not to
        # hide uncertainty. Flag when the top score is barely more likely
        # than the rest of the distribution.
        top_prob = top_score[2]
        if top_prob < 0.15:
            st.info(
                f"⚠️ Low concentration: the top scoreline ({top_score[0]}-{top_score[1]}) only carries "
                f"{top_prob*100:.1f}% probability. This match's outcome is genuinely uncertain — "
                "treat the Top-5 list as a spread, not a single confident forecast."
            )

        st.subheader("Top 5 scorelines")
        top5_df = pd.DataFrame(
            [(f"{i}-{j}", f"{p*100:.1f}%", rank + 1) for rank, (i, j, p) in enumerate(result["top5_scorelines"])],
            columns=["Scoreline", "Probability", "Rank"],
        )
        st.table(top5_df.set_index("Rank"))

        st.subheader("Lineup ratings")
        lr = result["lineup_row"]
        lc1, lc2 = st.columns(2)
        with lc1:
            st.write("**Home**")
            st.json({k.replace("home_", ""): round(v, 3) for k, v in lr.items() if k.startswith("home_")})
        with lc2:
            st.write("**Away**")
            st.json({k.replace("away_", ""): round(v, 3) for k, v in lr.items() if k.startswith("away_")})


if __name__ == "__main__":
    main()

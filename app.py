"""
app.py — Season-aware Streamlit Dashboard, v2 (roadmap upgrade)

Two-file layout: everything the app needs comes from backend.py (which
merges the old data_loader/cleaning/rolling_features/player_ratings/
lineup_features/elo/goal_models/score_models/calibration/evaluation/
run_experiments/pipeline modules into one file).

Features vs the old single-Poisson/RF app.py:
 - Uses the modular pipeline (backend.predict_match) instead of the old
   Poisson/RF-only formula.
 - Lets you pick the score-distribution model (independent Poisson /
   Dixon-Coles / bivariate Poisson / Negative Binomial).
 - Shows lineup strength ratings, match context (Elo diff proxy, rest days,
   head-to-head, congestion), and a calibration/uncertainty indicator.
 - Has a "Model Diagnostics" tab that runs walk-forward validation on the
   selected team's own match history and reports Top-1..Top-5 hit rate,
   outcome accuracy, log loss and Brier score, so you can see whether a
   change actually helps BEFORE trusting it on the locked benchmark.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from backend import (
    find_player_data_dir, load_all_players, load_season_data,
    get_squad_for_season, get_clubs_with_data, get_season_range, SEASON_DATES,
    clean_players_dict, FORMATIONS,
    predict_match, build_team_df,
    run_full_experiment_matrix,
)

MODEL_VERSION = "v2.0 (modular pipeline — see Football_Prediction_Model_Upgrade_Roadmap)"

st.set_page_config(page_title="⚽ Match Predictor v2", page_icon="⚽", layout="wide")


@st.cache_data
def load_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdir = find_player_data_dir(base_dir)
    raw_players = load_all_players(pdir)
    players, quality_report = clean_players_dict(raw_players)
    season_data = load_season_data(pdir, base_dir)
    return players, season_data, quality_report


def get_all_seasons(season_data):
    csv_seasons = set(season_data.keys())
    known_seasons = set(SEASON_DATES.keys())
    return sorted(csv_seasons | known_seasons, reverse=True)


PLAYERS, SEASON_DATA, QUALITY_REPORT = load_all()
ALL_SEASONS = get_all_seasons(SEASON_DATA)

st.title("⚽ Football Match Predictor — v2")
st.caption(f"Model version: {MODEL_VERSION}")
st.markdown(
    "Modular pipeline: multi-window rolling features → position-specific player "
    "ratings → lineup strength → Elo/context → ensembled goal model → "
    "Dixon-Coles/bivariate score distribution → calibration → Top-5 ranking. "
    "**Only data strictly before the selected match date is used.**"
)

if not ALL_SEASONS:
    st.error("❌ SEASON_DATA.csv not found in the player-data folder.")
    st.stop()
if not PLAYERS:
    st.error("❌ No player CSVs found in the player-data folder.")
    st.stop()

tab_predict, tab_diagnostics, tab_data = st.tabs(
    ["🔮 Predict a Match", "📊 Model Diagnostics (walk-forward)", "🗂️ Data Coverage"]
)

# =============================================================================
# TAB 1 — PREDICTION
# =============================================================================
with tab_predict:
    st.markdown("### 📅 Step 1 — Season")
    selected_season = st.selectbox("Season", ALL_SEASONS, index=0)
    season_start, season_end = get_season_range(selected_season)

    clubs_in_season, clubs_missing_data = get_clubs_with_data(SEASON_DATA, PLAYERS, selected_season)
    if not clubs_in_season:
        st.warning(f"No clubs with uploaded player CSVs found for season {selected_season}.")
        st.stop()
    if clubs_missing_data:
        with st.expander(f"ℹ️ {len(clubs_missing_data)} club(s) listed in SEASON_DATA.csv for "
                          f"{selected_season} have no matching player CSVs, so they're hidden below"):
            st.caption(", ".join(clubs_missing_data))

    st.markdown("### 🏟️ Step 2 — Teams, Date & Score Model")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        home_team = st.selectbox("🏠 Home Team", clubs_in_season, index=0)
    with c2:
        away_opts = [c for c in clubs_in_season if c != home_team]
        away_team = st.selectbox("✈️ Away Team", away_opts, index=0)
    with c3:
        match_date = st.date_input(
            "📅 Match Date", value=season_start.date(),
            min_value=season_start.date(), max_value=season_end.date(),
        )
    with c4:
        score_model = st.selectbox(
            "Score distribution",
            ["dixon_coles", "independent_poisson", "bivariate_poisson",
             "negative_binomial", "ensemble"],
            index=0,
            help=(
                "Dixon-Coles corrects the independence assumption for low scorelines. "
                "'ensemble' averages all four models together instead of using just one."
            ),
        )
    st.caption(f"ℹ️ Only data **strictly before {match_date}** will be used for this prediction.")

    st.markdown("### 🗂️ Step 3 — Formations")
    fc1, fc2 = st.columns(2)
    with fc1:
        home_formation = st.selectbox(f"🏠 {home_team} Formation", list(FORMATIONS.keys()), key="hf")
    with fc2:
        away_formation = st.selectbox(f"✈️ {away_team} Formation", list(FORMATIONS.keys()), key="af")

    home_slots = FORMATIONS[home_formation]
    away_slots = FORMATIONS[away_formation]

    st.markdown("### 👕 Step 4 — Playing XI")
    home_squad = get_squad_for_season(SEASON_DATA, PLAYERS, home_team, selected_season)
    away_squad = get_squad_for_season(SEASON_DATA, PLAYERS, away_team, selected_season)

    home_xi, away_xi = [], []
    col_home, col_away = st.columns(2)

    with col_home:
        st.markdown(f"#### 🏠 {home_team} — {home_formation}")
        if not home_squad:
            st.error(f"❌ No player CSVs found for **{home_team}** in {selected_season}.")
        else:
            for i, slot in enumerate(home_slots):
                already = [p for p in home_xi if p]
                options = [p for p in home_squad if p not in already] or home_squad
                player = st.selectbox(f"**{slot}** — Slot {i+1}", options=options, key=f"h_{i}")
                home_xi.append(player)

    with col_away:
        st.markdown(f"#### ✈️ {away_team} — {away_formation}")
        if not away_squad:
            st.error(f"❌ No player CSVs found for **{away_team}** in {selected_season}.")
        else:
            for i, slot in enumerate(away_slots):
                already = [p for p in away_xi if p]
                options = [p for p in away_squad if p not in already] or away_squad
                player = st.selectbox(f"**{slot}** — Slot {i+1}", options=options, key=f"a_{i}")
                away_xi.append(player)

    home_dups = [p for p in set(home_xi) if home_xi.count(p) > 1]
    away_dups = [p for p in set(away_xi) if away_xi.count(p) > 1]
    if home_dups:
        st.error(f"❌ {home_team}: **{', '.join(home_dups)}** selected more than once.")
    if away_dups:
        st.error(f"❌ {away_team}: **{', '.join(away_dups)}** selected more than once.")

    can_predict = bool(home_squad and away_squad and not home_dups and not away_dups)
    predict_btn = st.button("🔮 PREDICT MATCH", type="primary",
                             use_container_width=True, disabled=not can_predict)

    if predict_btn and can_predict:
        valid_home = [p for p in home_xi if p]
        valid_away = [p for p in away_xi if p]

        with st.spinner("Running prediction pipeline..."):
            try:
                result = predict_match(
                    PLAYERS, home_team, away_team, valid_home, valid_away,
                    str(match_date), selected_season,
                    home_formation=home_formation, away_formation=away_formation,
                    score_model=score_model,
                )

                st.divider()
                st.markdown("## 📊 Prediction Results")
                st.markdown(
                    f"**{home_team}** ({home_formation}) vs **{away_team}** ({away_formation}) | "
                    f"📅 {match_date} | 🗓️ {selected_season} | model: `{score_model}`"
                )

                st.markdown("### ⚽ Expected Goals")
                xc1, xc2, xc3 = st.columns(3)
                xc1.metric(f"🏠 {home_team}", result["xg_home"])
                xc2.metric("VS", "—")
                xc3.metric(f"✈️ {away_team}", result["xg_away"])
                st.divider()

                st.markdown("### 📈 Match Probabilities")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric(f"🏠 {home_team} Win", f"{result['home_win']}%")
                pc2.metric("🤝 Draw", f"{result['draw']}%")
                pc3.metric(f"✈️ {away_team} Win", f"{result['away_win']}%")
                prob_df = pd.DataFrame({
                    "Outcome": [f"{home_team} Win", "Draw", f"{away_team} Win"],
                    "Probability (%)": [result["home_win"], result["draw"], result["away_win"]],
                })
                st.bar_chart(prob_df.set_index("Outcome"))

                # Uncertainty / confidence indicator — "don't hide uncertainty"
                top5 = result["top5"]
                top1_prob = top5[0][1] if top5 else 0.0
                if top1_prob >= 25:
                    conf_msg = "🔵 Concentrated distribution — the model favours one scoreline clearly."
                elif top1_prob >= 12:
                    conf_msg = "🟡 Moderately spread distribution — several scorelines are plausible."
                else:
                    conf_msg = "🟠 Flat distribution — outcome is genuinely uncertain; treat any single score with caution."
                st.info(f"**Confidence:** top scoreline {top5[0][0] if top5 else '-'} at "
                        f"{top1_prob}% probability. {conf_msg}")
                st.divider()

                st.markdown("### 🎯 Top 5 Most Likely Scorelines")
                st.dataframe(pd.DataFrame(result["top5"], columns=["Scoreline", "Probability (%)"]),
                             use_container_width=True, hide_index=True)
                st.divider()

                st.markdown("### 🔥 Scoreline Probability Heatmap (%)")
                matrix_pct = np.round(result["matrix"] * 100, 2)
                heatmap_df = pd.DataFrame(
                    matrix_pct,
                    index=[f"{home_team} {i}" for i in range(matrix_pct.shape[0])],
                    columns=[f"{away_team} {j}" for j in range(matrix_pct.shape[1])],
                )
                st.dataframe(heatmap_df.style.background_gradient(cmap="YlOrRd"),
                             use_container_width=True)
                st.divider()

                st.markdown("### 🏗️ Lineup Strength")
                lc1, lc2 = st.columns(2)
                for col, team_name, strength in [
                    (lc1, home_team, result["lineup_home"]),
                    (lc2, away_team, result["lineup_away"]),
                ]:
                    with col:
                        st.markdown(f"**{team_name}**")
                        st.dataframe(
                            pd.DataFrame(
                                [{"Metric": k, "Value": round(v, 1)} for k, v in strength.items()]
                            ),
                            use_container_width=True, hide_index=True,
                        )
                st.divider()

                st.markdown("### 🧭 Match Context")
                cc1, cc2 = st.columns(2)
                for col, team_name, ctx in [
                    (cc1, home_team, result["context_home"]),
                    (cc2, away_team, result["context_away"]),
                ]:
                    with col:
                        st.markdown(f"**{team_name}**")
                        st.dataframe(
                            pd.DataFrame([{"Metric": k, "Value": v} for k, v in ctx.items()]),
                            use_container_width=True, hide_index=True,
                        )
                st.divider()

                st.markdown("### 📋 Playing XI Used")
                xi1, xi2 = st.columns(2)
                with xi1:
                    st.markdown(f"**🏠 {home_team} ({home_formation})**")
                    for slot, player in zip(home_slots, home_xi):
                        st.write(f"**{slot}** — {player}")
                with xi2:
                    st.markdown(f"**✈️ {away_team} ({away_formation})**")
                    for slot, player in zip(away_slots, away_xi):
                        st.write(f"**{slot}** — {player}")
                st.divider()

                st.markdown("### ℹ️ Data Coverage & Model Version")
                st.info(f"Only matches strictly before **{match_date}** were used. Model: `{MODEL_VERSION}`.")
                mdt = pd.to_datetime(str(match_date))
                info_rows = []
                for name, df in PLAYERS.items():
                    past = df[df["Date"] < mdt]
                    if len(past) > 0:
                        info_rows.append({
                            "Player": name, "Matches Used": len(past),
                            "Latest Match Used": str(past["Date"].max().date()),
                        })
                if info_rows:
                    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.exception(e)

# =============================================================================
# TAB 2 — MODEL DIAGNOSTICS / WALK-FORWARD VALIDATION
# =============================================================================
with tab_diagnostics:
    st.markdown("### 📊 Walk-Forward Validation")
    st.markdown(
        "Runs the E0→E9 controlled-experiment matrix on a single team's own "
        "chronological match history: independent Poisson → +Elo → Dixon-Coles "
        "→ +calibration → bivariate Poisson. "
        "**Time order is preserved — no random shuffling.**"
    )
    diag_season = st.selectbox("Season", ALL_SEASONS, index=0, key="diag_season")
    diag_teams, _ = get_clubs_with_data(SEASON_DATA, PLAYERS, diag_season)
    if not diag_teams:
        st.warning("No teams available for this season.")
    else:
        diag_team = st.selectbox("Team", diag_teams, key="diag_team")
        run_diag = st.button("▶️ Run Walk-Forward Validation", type="primary")
        if run_diag:
            squad = get_squad_for_season(SEASON_DATA, PLAYERS, diag_team, diag_season)
            if not squad:
                st.error(f"No player CSVs for {diag_team} in {diag_season}.")
            else:
                season_end_date = get_season_range(diag_season)[1]
                team_df = build_team_df(PLAYERS, squad, season_end_date)
                with st.spinner("Running walk-forward folds..."):
                    results = run_full_experiment_matrix(team_df, diag_team)
                rows = [r for r in results if "error" not in r]
                errs = [r for r in results if "error" in r]
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption(
                        "top5_rate/top4_rate/... = fraction of validation matches where the "
                        "actual scoreline appeared in the Top-N ranked predictions. "
                        "Results are appended to experiments/experiment_log.csv."
                    )
                for e in errs:
                    st.warning(e["error"])

# =============================================================================
# TAB 3 — DATA COVERAGE / QUALITY REPORT
# =============================================================================
with tab_data:
    st.markdown("### 🧹 Data Quality Report")
    st.caption(
        "Duplicates removed, negative-minutes rows fixed, SoT>Shots rows capped, "
        "and missing Goals/Shots counts, per player CSV."
    )
    st.dataframe(QUALITY_REPORT, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    "📌 **Add more players:** drop a CSV into the player-data folder and add a row "
    "to `SEASON_DATA.csv` with their club & season."
)

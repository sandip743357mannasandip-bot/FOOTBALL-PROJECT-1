# La Liga Score Prediction — Model v2

Implements the Upgrade Roadmap: leakage-safe multi-window features,
position-specific player ratings, lineup strength aggregation, Elo,
a learned goal-model ensemble, and Dixon-Coles scoreline generation.

## Status

| Module                 | Status                                              |
|-------------------------|-----------------------------------------------------|
| cleaning.py              | Tested on real data (Abdul_Mumin CSV)              |
| rolling_features.py      | Tested on real data                                 |
| player_ratings.py        | Tested on real data                                 |
| lineup_features.py       | Written, needs a real 11-player XI to test          |
| elo.py                   | Tested on real data                                 |
| goal_models.py           | Tested on synthetic data — needs your full training table |
| score_models.py (Dixon-Coles) | Tested — verified it shifts probability toward 1-1 / draws as expected |
| calibration.py           | Written, needs validation-fold predictions to fit    |
| evaluation.py            | Tested on synthetic data                            |
| pipeline.py               | Orchestration layer, ready once goal models are fit |
| app.py (Streamlit)        | Ready to run once data/raw_players/ is populated and the model is fit |

## To actually get real predictions, I need from you:

1. **All squad player CSVs** (same schema as Abdul_Mumin_-_Sheet1.csv) —
   drop them into `data/raw_players/`. The more players/seasons, the
   better the rolling form and Elo history.
2. **A match-level results/lineup file**: for every historical Real
   Madrid match, the date, opponent, venue, final score, and the 11
   players who started. Without this, `fit_goal_models()` has nothing
   to train on — the goal model, Dixon-Coles rho, and calibration all
   depend on it.
3. Confirmation of your **locked 38-match benchmark file** (the one
   this whole roadmap is trying to beat) so I don't accidentally use
   it for tuning.

## Run

```bash
cd src/
streamlit run app.py
```

The sidebar will tell you honestly whether the model is fitted —
it will not fabricate predictions from an unfitted model.

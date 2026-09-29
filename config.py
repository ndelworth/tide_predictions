from pathlib import Path

import pandas as pd

STATION = {
    "name":    "Pointe-du-Chêne",
    "lat":     46.24061389,
    "lon":     -64.53000556,
    "iwls_id": "64b6e5ec8027cb190816a0c0",
}

START     = pd.Timestamp("2024-03-14 00:00", tz="UTC")   # the first IFS run Open-Meteo archives
TRAIN_END = pd.Timestamp("2026-07-14 01:00", tz="UTC")

PAST_HOURS = 48
HORIZON    = 48

RUN_AVAILABLE_AFTER = pd.Timedelta(hours=8)   # ECMWF publishes a run ~6-7 h after it starts
MAX_RUN_AGE_HOURS   = 36
RUN_MAX_LEAD        = MAX_RUN_AGE_HOURS + HORIZON

WX_VARS = ["pressure_msl", "wind_u_10m", "wind_v_10m", "precipitation", "wind_gusts_10m",
           "wind_stress_u", "wind_stress_v"]

ROOT       = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"
MODEL_PATH = ROOT / "models" / "surge_model.pt"

WLO_CSV       = DATA_DIR / "raw" / "wlo.csv"
WX_ACTUAL_CSV = DATA_DIR / "raw" / "weather_actual.csv"
WX_RUNS_CSV   = DATA_DIR / "raw" / "weather_runs.csv"

ACTUAL_WITH_SURGE_CSV = DATA_DIR / "actual_weather_with_surge.csv"
FORECAST_RUNS_NPZ     = DATA_DIR / "forecast_runs.npz"

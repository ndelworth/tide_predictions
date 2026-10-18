import argparse
import json
import sys

import numpy as np
import pandas as pd
import requests
import torch

from config import (HORIZON, MAX_RUN_AGE_HOURS, MODEL_PATH, PAST_HOURS, RUN_AVAILABLE_AFTER, RUN_MAX_LEAD,
                    STATION, WLO_CSV, WX_ACTUAL_CSV, WX_RUNS_CSV, WX_VARS)
from dataset import tide_at, with_wind_stress
from download import ACTUAL_WEATHER_URL, FORECAST_RUN_URL, WATER_LEVEL_URL, get_weather
from features import (IS_FUTURE, MIN_FRACTION_FORECAST_WEATHER_KNOWN, MIN_FRACTION_PAST_SURGE_OBSERVED,
                      WINDOW_OFFSETS, time_features)
from model import SurgeTransformer

RECENT_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
RAW_WEATHER_VARS   = ["pressure_msl", "wind_u_10m", "wind_v_10m", "precipitation", "wind_gusts_10m"]


class NotEnoughData(Exception):
    pass


def live_water_level(start, end):
    response = requests.get(WATER_LEVEL_URL, timeout=60, params={
        "time-series-code": "wlo", "resolution": "SIXTY_MINUTES", "station-id": STATION["iwls_id"],
        "from": f"{start:%Y-%m-%dT%H:%M:%SZ}", "to": f"{end:%Y-%m-%dT%H:%M:%SZ}"})
    response.raise_for_status()
    rows = response.json()
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    return pd.Series(df["value"].to_numpy(float), index=pd.to_datetime(df["eventDate"], utc=True)).groupby(level=0).last()


def weather_or_none(url, **params):
    try:
        return get_weather(url, **params)
    except Exception as e:
        print(f"  weather request to {url} failed: {e}")
        return None


def live_actual_weather(start, end):
    frames = []
    archive = weather_or_none(ACTUAL_WEATHER_URL, start_date=f"{start:%Y-%m-%d}", end_date=f"{end:%Y-%m-%d}")
    if archive is not None:
        frames.append(archive.set_index("time").dropna(how="all"))
    recent = weather_or_none(RECENT_WEATHER_URL, past_days=3, forecast_days=1)
    if recent is not None:
        frames.append(recent.set_index("time"))
    if not frames:
        return pd.DataFrame(columns=RAW_WEATHER_VARS)
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.combine_first(frame)
    return combined


def live_forecast_run(issued):
    run = (issued - RUN_AVAILABLE_AFTER).floor("6h")
    while (issued - run) <= pd.Timedelta(hours=MAX_RUN_AGE_HOURS):
        df = weather_or_none(FORECAST_RUN_URL, run=f"{run:%Y-%m-%dT%H:%M}", forecast_hours=RUN_MAX_LEAD + 1)
        if df is not None and df[RAW_WEATHER_VARS].notna().any().any():
            return run, df.set_index("time")
        print(f"  IFS run {run} not available yet, trying the one before")
        run -= pd.Timedelta(hours=6)
    raise NotEnoughData(f"no IFS run within {MAX_RUN_AGE_HOURS} h of {issued}")


def replay_water_level(start, end):
    wlo = pd.read_csv(WLO_CSV, index_col="time", parse_dates=["time"])["wlo"]
    return wlo.loc[start:end]


def replay_actual_weather(start, end):
    return pd.read_csv(WX_ACTUAL_CSV, index_col="time", parse_dates=["time"]).loc[start:end]


def replay_forecast_run(issued):
    runs = pd.read_csv(WX_RUNS_CSV, parse_dates=["run", "time"])
    available = runs["run"].drop_duplicates()
    available = available[(available + RUN_AVAILABLE_AFTER <= issued)
                          & (issued - available <= pd.Timedelta(hours=MAX_RUN_AGE_HOURS))]
    if available.empty:
        raise NotEnoughData(f"no IFS run in {WX_RUNS_CSV} within {MAX_RUN_AGE_HOURS} h of {issued}")
    run = available.max()
    return run, runs[runs["run"] == run].drop(columns=["run", "lead"]).set_index("time")


def build_window(issued, water_level, actual_weather, run, run_weather):
    times = issued + pd.to_timedelta(WINDOW_OFFSETS, unit="h")
    tide = tide_at(times)

    lookback = pd.date_range(issued - pd.Timedelta(hours=PAST_HOURS), issued, freq="h")
    surge_lookback = water_level.reindex(lookback).to_numpy(float) - tide_at(lookback)
    past_surge = surge_lookback[-PAST_HOURS:]
    last_surge = pd.Series(surge_lookback).ffill(limit=PAST_HOURS).iloc[-1]
    surge = np.where(IS_FUTURE, last_surge, np.concatenate([past_surge, np.full(HORIZON, np.nan)]))

    leads = np.asarray((times - run) / pd.Timedelta(hours=1)).round().astype(int)
    run_hours = pd.DatetimeIndex(run + pd.to_timedelta(np.arange(RUN_MAX_LEAD + 1), unit="h"))
    run_weather = with_wind_stress(run_weather[RAW_WEATHER_VARS].reindex(run_hours)).to_numpy(np.float32)
    forecast = np.full((len(times), len(WX_VARS)), np.nan, np.float32)
    in_run = (leads >= 1) & (leads <= RUN_MAX_LEAD)
    forecast[in_run] = run_weather[leads[in_run]]
    actual = with_wind_stress(actual_weather[RAW_WEATHER_VARS].reindex(times)).to_numpy(np.float32)
    use_forecast = IS_FUTURE[:, None] | np.isfinite(forecast)
    weather = np.where(use_forecast, forecast, actual)

    past_observed = np.isfinite(past_surge).mean()
    forecast_known = np.isfinite(forecast[IS_FUTURE]).mean()
    if past_observed < MIN_FRACTION_PAST_SURGE_OBSERVED:
        raise NotEnoughData(f"only {past_observed:.0%} of the last {PAST_HOURS} h of water level observed")
    if forecast_known < MIN_FRACTION_FORECAST_WEATHER_KNOWN:
        raise NotEnoughData(f"IFS run {run} covers only {forecast_known:.0%} of the next {HORIZON} h")
    missing_past_weather = ~np.isfinite(weather[~IS_FUTURE]).all(axis=1)
    if missing_past_weather.any():
        print(f"  note: {missing_past_weather.sum()} past hours have no weather (the model treats them as average)")

    return (times, tide,
            torch.from_numpy(weather[None]), torch.from_numpy(surge.astype(np.float32)[None]),
            torch.from_numpy(time_features(times)[None]))


def load_model():
    model = SurgeTransformer()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return model.eval()


def forecast(issued, replay=False):
    water_level_source, weather_source, run_source = (
        (replay_water_level, replay_actual_weather, replay_forecast_run) if replay else
        (live_water_level, live_actual_weather, live_forecast_run))
    start = issued - pd.Timedelta(hours=PAST_HOURS + 1)
    run, run_weather = run_source(issued)
    print(f"Forecast issued {issued}, IFS run {run} ({(issued - run) / pd.Timedelta(hours=1):.0f} h old)")
    times, tide, weather, surge, time = build_window(
        issued, water_level_source(start, issued), weather_source(start, issued), run, run_weather)
    with torch.no_grad():
        predicted_surge = load_model()(weather, surge, time)[0].numpy().astype(float)
    future = IS_FUTURE
    return {
        "issued": f"{issued:%Y-%m-%dT%H:%M:%SZ}",
        "weather_run": f"{run:%Y-%m-%dT%H:%M:%SZ}",
        "station": STATION["name"],
        "forecast": [
            {"valid_time": f"{t:%Y-%m-%dT%H:%M:%SZ}", "lead_hours": int(lead),
             "water_level": round(td + s, 4), "tide": round(td, 4), "surge": round(s, 4)}
            for t, lead, td, s in zip(times[future], WINDOW_OFFSETS[future], tide[future], predicted_surge)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="48 h water-level forecast for Pointe-du-Chêne, issued this hour")
    parser.add_argument("--out", default="forecast.json")
    parser.add_argument("--replay", metavar="ISSUED_UTC", help="rebuild a past forecast from data/raw instead")
    args = parser.parse_args()

    issued = (pd.Timestamp(args.replay) if args.replay else pd.Timestamp.now(tz="UTC")).floor("h")
    if issued.tzinfo is None:
        issued = issued.tz_localize("UTC")
    try:
        result = forecast(issued, replay=bool(args.replay))
    except NotEnoughData as e:
        print(f"Not forecasting: {e}")
        sys.exit(1)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    levels = [row["water_level"] for row in result["forecast"]]
    print(f"Wrote {args.out}: {len(levels)} hours, water level {min(levels):.2f} to {max(levels):.2f} m")


if __name__ == "__main__":
    main()

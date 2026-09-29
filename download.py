import time

import numpy as np
import pandas as pd
import requests

from config import RUN_AVAILABLE_AFTER, RUN_MAX_LEAD, START, STATION, WLO_CSV, WX_ACTUAL_CSV, WX_RUNS_CSV

WATER_LEVEL_URL    = f"https://api-iwls.dfo-mpo.gc.ca/api/v1/stations/{STATION['iwls_id']}/data"
ACTUAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_RUN_URL   = "https://single-runs-api.open-meteo.com/v1/forecast"
WATER_LEVEL_MAX_DAYS_PER_REQUEST = 30


def date_chunks(end, days):
    starts = pd.date_range(START, end, freq=f"{days}D", inclusive="left")
    return zip(starts, [*starts[1:], end])


def get_weather(url, **params):
    response = requests.get(url, timeout=60, params={
        "latitude": STATION["lat"], "longitude": STATION["lon"], "models": "ecmwf_ifs",
        "hourly": "pressure_msl,precipitation,wind_gusts_10m,wind_speed_10m,wind_direction_10m",
        "timezone": "UTC", "wind_speed_unit": "ms", **params})
    if response.status_code == 400 and "not available" in response.text:
        return None
    response.raise_for_status()
    hourly = pd.DataFrame(response.json()["hourly"])
    speed = hourly["wind_speed_10m"].astype(float)
    blowing_from = np.deg2rad(hourly["wind_direction_10m"].astype(float))
    return pd.DataFrame({
        "time":           pd.to_datetime(hourly["time"], utc=True),
        "pressure_msl":   hourly["pressure_msl"].astype(float),
        "wind_u_10m":     -speed * np.sin(blowing_from),
        "wind_v_10m":     -speed * np.cos(blowing_from),
        "precipitation":  hourly["precipitation"].astype(float),
        "wind_gusts_10m": hourly["wind_gusts_10m"].astype(float),
    })


def download_water_level(end):
    rows = []
    for start, stop in date_chunks(end, WATER_LEVEL_MAX_DAYS_PER_REQUEST):
        response = requests.get(WATER_LEVEL_URL, timeout=60, params={
            "time-series-code": "wlo", "resolution": "SIXTY_MINUTES", "station-id": STATION["iwls_id"],
            "from": f"{start:%Y-%m-%dT%H:%M:%SZ}", "to": f"{stop:%Y-%m-%dT%H:%M:%SZ}"})
        response.raise_for_status()
        rows += response.json()
        print(f"  water level {start.date()}: {len(rows):,} rows so far")
        time.sleep(2.5)
    df = pd.DataFrame(rows)
    df = pd.DataFrame({"time": pd.to_datetime(df["eventDate"], utc=True), "wlo": df["value"]})
    df.drop_duplicates("time").to_csv(WLO_CSV, index=False)


def download_actual_weather(end):
    frames = []
    for start, stop in date_chunks(end, 365):
        frames.append(get_weather(ACTUAL_WEATHER_URL, start_date=f"{start:%Y-%m-%d}", end_date=f"{stop:%Y-%m-%d}"))
        print(f"  actual weather {start.date()}: {len(frames[-1]):,} rows")
        time.sleep(2.5)
    pd.concat(frames).drop_duplicates("time").to_csv(WX_ACTUAL_CSV, index=False)


def download_forecast_runs(end):
    first = START
    if WX_RUNS_CSV.exists():
        first = pd.read_csv(WX_RUNS_CSV, usecols=["run"], parse_dates=["run"])["run"].max() + pd.Timedelta(hours=6)
    runs = pd.date_range(first.ceil("6h"), (end - RUN_AVAILABLE_AFTER).floor("6h"), freq="6h")
    print(f"  {len(runs):,} forecast runs to fetch, {runs.min()} -> {runs.max()}")
    for i, run in enumerate(runs, 1):
        df = get_weather(FORECAST_RUN_URL, run=f"{run:%Y-%m-%dT%H:%M}", forecast_hours=RUN_MAX_LEAD + 1)
        if df is not None:
            lead = ((df["time"] - run) / pd.Timedelta(hours=1)).round().astype(int)
            df.insert(0, "run", run)
            df.insert(2, "lead", lead)
            df[lead.between(0, RUN_MAX_LEAD)].to_csv(WX_RUNS_CSV, mode="a", header=not WX_RUNS_CSV.exists(),
                                                     index=False, float_format="%.6g")
        if i % 100 == 0:
            print(f"  {i:,}/{len(runs):,} runs")
        time.sleep(0.3)


if __name__ == "__main__":
    end = pd.Timestamp.now(tz="UTC").floor("h")
    WLO_CSV.parent.mkdir(parents=True, exist_ok=True)
    download_water_level(end)
    download_actual_weather(end)
    download_forecast_runs(end)

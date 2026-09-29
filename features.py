import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from config import (ACTUAL_WITH_SURGE_CSV, FORECAST_RUNS_NPZ, HORIZON, MAX_RUN_AGE_HOURS, PAST_HOURS,
                    RUN_AVAILABLE_AFTER, WX_VARS)

WINDOW         = PAST_HOURS + HORIZON
WINDOW_OFFSETS = np.arange(-PAST_HOURS + 1, HORIZON + 1)
LEADS          = np.arange(1, HORIZON + 1)
IS_FUTURE      = WINDOW_OFFSETS > 0
TIME_FEATURES  = ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]

MIN_FRACTION_PAST_SURGE_OBSERVED    = 0.5
MIN_FRACTION_FORECAST_WEATHER_KNOWN = 0.9


def time_features(times):
    hour = 2 * np.pi * times.hour.to_numpy() / 24
    day = 2 * np.pi * times.dayofyear.to_numpy() / 365.25
    return np.stack([np.sin(hour), np.cos(hour), np.sin(day), np.cos(day)], axis=1).astype(np.float32)


def forecast_run_weather(times):
    with np.load(FORECAST_RUNS_NPZ) as saved:
        run_times = pd.to_datetime(saved["run_times"], utc=True)
        forecasts = saved["forecasts"]
    newest_run = (run_times + RUN_AVAILABLE_AFTER).searchsorted(times, side="right") - 1
    run_age = np.asarray((times - run_times[newest_run]) / pd.Timedelta(hours=1)).astype(int)
    has_run = (newest_run >= 0) & (run_age <= MAX_RUN_AGE_HOURS)
    lead = run_age[:, None] + WINDOW_OFFSETS
    after_run_start = has_run[:, None] & (lead >= 1)
    weather = np.full(lead.shape + (len(WX_VARS),), np.nan, np.float32)
    run = np.broadcast_to(newest_run[:, None], lead.shape)
    weather[after_run_start] = forecasts[run[after_run_start], lead[after_run_start]]
    return weather


class ForecastDataset:
    def __init__(self, through=None):
        self.df = pd.read_csv(ACTUAL_WITH_SURGE_CSV, index_col="time", parse_dates=["time"]).loc[:through]
        self.times = self.df.index
        self.surge = self.df["surge"].to_numpy(np.float32)
        self.last_surge = pd.Series(self.surge).ffill(limit=PAST_HOURS).to_numpy()
        self.actual_weather = self.df[WX_VARS].to_numpy(np.float32)
        self.forecast_weather = forecast_run_weather(self.times)
        self.time_features = time_features(self.times)

        past_surge_observed = pd.Series(np.isfinite(self.surge)).rolling(PAST_HOURS).mean().to_numpy()
        forecast_weather_known = np.isfinite(self.forecast_weather[:, IS_FUTURE]).mean(axis=(1, 2))
        self.usable = ((past_surge_observed >= MIN_FRACTION_PAST_SURGE_OBSERVED)
                       & (forecast_weather_known >= MIN_FRACTION_FORECAST_WEATHER_KNOWN))
        self.usable[:PAST_HOURS - 1] = self.usable[-HORIZON:] = False

    def windows(self, issue_rows):
        issue_rows = np.asarray(issue_rows)
        rows = issue_rows[:, None] + WINDOW_OFFSETS
        forecast_weather = self.forecast_weather[issue_rows]
        # Once the forecast run has started, it replaces the actual weather: live, nothing newer exists.
        use_forecast = IS_FUTURE[:, None] | np.isfinite(forecast_weather)
        weather = np.where(use_forecast, forecast_weather, self.actual_weather[rows])
        surge = np.where(IS_FUTURE, self.last_surge[issue_rows, None], self.surge[rows])
        target = self.surge[rows[:, IS_FUTURE]]
        return TensorDataset(*map(torch.from_numpy, (weather, surge, self.time_features[rows], target)))

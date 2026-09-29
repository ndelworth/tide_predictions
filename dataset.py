import numpy as np
import pandas as pd

from config import (ACTUAL_WITH_SURGE_CSV, FORECAST_RUNS_NPZ, RUN_MAX_LEAD, START, TRAIN_END,
                    WLO_CSV, WX_ACTUAL_CSV, WX_RUNS_CSV, WX_VARS)

TIDAL_PERIODS_HOURS = {
    "O1": 25.8194, "K1": 23.9345, "S1": 24.0000, "P1": 24.0659, "Q1": 26.8684,
    "2Q1": 28.0063, "rho1": 26.7230, "J1": 23.0985, "OO1": 22.3061, "M1": 24.8412,
    "M2": 12.4206, "S2": 12.0000, "N2": 12.6583, "K2": 11.9672, "L2": 12.1916, "lambda2": 12.2218,
    "2N2": 12.9055, "mu2": 12.8717, "nu2": 12.6260, "T2": 12.0164, "R2": 11.9840, "2SM2": 11.6070,
    "M3": 8.2803, "MK3": 8.1774, "2MK3": 8.3864, "M4": 6.2103, "MS4": 6.1033, "MN4": 6.2695,
    "S4": 6.0000, "M6": 4.1402, "S6": 4.0000, "M8": 3.1051,
    "Sa": 8766.0, "Ssa": 4383.0, "MSf": 354.37, "Mm": 661.3, "Mf": 327.9,
}


def harmonic_basis(n_hours):
    phase = np.outer(np.arange(n_hours), 2 * np.pi / np.array(list(TIDAL_PERIODS_HOURS.values())))
    return np.column_stack([np.ones(n_hours), np.cos(phase), np.sin(phase)])


def with_wind_stress(weather):
    speed = np.hypot(weather["wind_u_10m"], weather["wind_v_10m"])
    return weather.assign(wind_stress_u=weather["wind_u_10m"] * speed,
                          wind_stress_v=weather["wind_v_10m"] * speed)[WX_VARS]


def build_actual_weather_with_surge():
    water_level = pd.read_csv(WLO_CSV, index_col="time", parse_dates=["time"])["wlo"]
    hours = pd.date_range(START, water_level.last_valid_index(), freq="h")
    water_level = water_level.reindex(hours)

    X = harmonic_basis(len(hours))
    fit_hours = (hours <= TRAIN_END) & water_level.notna().to_numpy()
    tide = X @ np.linalg.lstsq(X[fit_hours], water_level[fit_hours].to_numpy(), rcond=None)[0]

    weather = pd.read_csv(WX_ACTUAL_CSV, index_col="time", parse_dates=["time"]).reindex(hours)
    df = pd.DataFrame({"wlo": water_level, "tide": tide, "surge": water_level - tide})
    df = df.join(with_wind_stress(weather)).rename_axis("time")
    df.to_csv(ACTUAL_WITH_SURGE_CSV, float_format="%.6g")
    print(f"Saved {ACTUAL_WITH_SURGE_CSV}: {hours[0]} -> {hours[-1]}, water level observed "
          f"{water_level.notna().mean():.1%}, tide fit on {fit_hours.sum():,} hours up to TRAIN_END")


def build_forecast_runs():
    runs = pd.read_csv(WX_RUNS_CSV, parse_dates=["run"]).set_index(["run", "lead"])
    run_times = runs.index.unique("run").sort_values()
    every_lead = pd.MultiIndex.from_product([run_times, range(RUN_MAX_LEAD + 1)])
    forecasts = with_wind_stress(runs.reindex(every_lead)).to_numpy(np.float32)
    forecasts = forecasts.reshape(len(run_times), RUN_MAX_LEAD + 1, len(WX_VARS))
    np.savez_compressed(FORECAST_RUNS_NPZ, run_times=run_times.as_unit("ns").asi8, forecasts=forecasts)
    print(f"Saved {FORECAST_RUNS_NPZ}: {len(run_times):,} IFS runs, {run_times[0]} -> {run_times[-1]}")


if __name__ == "__main__":
    build_actual_weather_with_surge()
    build_forecast_runs()

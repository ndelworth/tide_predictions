# Tide predictions: Pointe-du-Chêne

48-hour water-level forecasts for the CHS station at Pointe-du-Chêne, NB (01804).
Each forecast is a **harmonic tide** (fit by least squares on the training period) plus a
**storm surge** predicted by a small transformer from the last 48 h of observations and a
real ECMWF IFS weather forecast run, exactly as a live system would have had it.

## Pipeline

```sh
uv run download.py   # water level (DFO IWLS) + weather (Open-Meteo) into data/raw/
uv run dataset.py    # fit the harmonic tide -> models/tide_coefficients.json; build data/actual_weather_with_surge.csv and data/forecast_runs.npz
uv run train.py      # train the surge model on hours up to TRAIN_END -> models/surge_model.pt
uv run evaluate.py   # MAE by lead vs CHS, persistence and tide on the holdout; water-level plots -> results/
uv run predict.py    # live 48 h forecast issued this hour -> forecast.json
```

## Live forecasts

`predict.py` builds the same 96-hour window as `features.py`, from live data: the last 48 h of
top-of-hour water level from IWLS, the newest IFS run Open-Meteo has, and actual weather for the
hours before that run. It needs only `models/surge_model.pt` and `models/tide_coefficients.json`,
which are committed (the rest of `models/` and all of `data/` are ignored).
`uv run predict.py --replay 2026-09-05T06:00Z` rebuilds a past forecast from `data/raw/`
instead, and matches `evaluate.py` for the same hour.

[water-level-forecasts-data](https://github.com/ndelworth/water-level-forecasts-data) runs
`predict.py` from `main` every hour and stores the result next to the CHS forecasts, so after
retraining, commit the two model files and push.

Settings (station, dates, paths) are in `config.py`; each script's own settings are constants
at its top. `evaluate.py` reads CHS's forecasts from `../water-level-forecasts-data/docs/tides.db`.

## Modules

| File | What it holds |
|---|---|
| `features.py` | `ForecastDataset`: every hour of the dataset plus the IFS runs, cut into the 96-hour window the model sees for each forecast |
| `model.py` | The surge transformer |
| `predict.py` | Live (or replayed) forecast for the current hour |

import json
import requests
import pandas as pd
import numpy as np
from tidal_constituents import CONSTITUENTS as CONSTITUENTS
from matplotlib import pyplot as plt

EPOCH = pd.Timestamp("2020-01-01", tz="UTC")
STATION_ID = '64b6e5ec8027cb190816a0c0'
BASE_URL = "https://api-iwls.dfo-mpo.gc.ca/api/v1"

TRAINING_DATA = 'data/tide_pointe_du_chene_train_wlo.csv'
WEIGHTS_PATH = "model_weights.json"

def build_harmonic_features(df):
    df["bias"] = np.ones(len(df))
    for name, period in CONSTITUENTS.items():
        omega = 2 * np.pi / period
        df[f"{name}_cos"] = np.cos(omega * df["time"])
        df[f"{name}_sin"] = np.sin(omega * df["time"])
    return df

def hours_since_epoch(times: pd.Series) -> np.ndarray:
    t = pd.to_datetime(times)
    if t.dt.tz is None:
        t = t.dt.tz_localize("UTC")
    else:
        t = t.dt.tz_convert("UTC")
    result = (t - EPOCH).dt.total_seconds().values / 3600.0
    n_nan = np.isnan(result).sum()
    n_inf = np.isinf(result).sum()
    if n_nan or n_inf:
        print(f"  WARNING: t has {n_nan} NaN, {n_inf} Inf  (range {np.nanmin(result):.0f}–{np.nanmax(result):.0f}h)")
    return result

def load_data():    
    df = pd.read_csv(TRAINING_DATA)
    df['time'] = hours_since_epoch(df['time'])
    return df

def fit_harmonics(df, weights=None):
    y = np.array(df["wlo"], dtype=np.float64)
    X = build_harmonic_features(df)
    X.drop(columns=["time"], inplace=True)
    X.drop(columns=["wlo"], inplace=True)
    if weights is not None:
        weights = np.sqrt(weights)
        X = X * weights[:, None]
        y = y * weights
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs

def predict_wlo(coeffs, times):
    t = hours_since_epoch(pd.Series(times))
    y_pred = coeffs[0] * np.ones(len(t))   # bias term
    i = 1
    for name, period in CONSTITUENTS.items():
        omega = 2 * np.pi / period
        y_pred += coeffs[i]     * np.cos(omega * t)
        y_pred += coeffs[i + 1] * np.sin(omega * t)
        i += 2
    return y_pred

def save_weights(coeffs, path=WEIGHTS_PATH):
    weights = {
        "epoch": EPOCH.isoformat(),
        "bias": float(coeffs[0]),
        "constituents": {},
    }
    i = 1
    for name, period in CONSTITUENTS.items():
        weights["constituents"][name] = {
            "period_hours": period,
            "cos": float(coeffs[i]),
            "sin": float(coeffs[i + 1]),
        }
        i += 2
    with open(path, "w") as f:
        json.dump(weights, f, indent=2)
    return weights

def predict_chs(start_time):
    start_utc = start_time.tz_convert("UTC")
    params = {
        "time-series-code": "wlp",
        "from":             start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to":               end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resolution":       "ONE_MINUTE",
        "station-id":       STATION_ID,
    }
    url = f"{BASE_URL}/stations/{STATION_ID}/data"
    r = requests.get(url, params=params, timeout=30)
    df = pd.DataFrame(r.json())
    df["time"] = pd.to_datetime(df["eventDate"], utc=True)
    return df.set_index("time")["value"]
    

if __name__ == "__main__":
    df = load_data()
    coeffs = fit_harmonics(df)
    save_weights(coeffs)
    now = pd.Timestamp.now(tz="America/Halifax")
    forward_times = pd.date_range(now, now + pd.Timedelta(hours=71, minutes=59), freq="1min")
    predictions = predict_wlo(coeffs, forward_times)
    chs_predictions = predict_chs(now)
    plt.plot(forward_times, predictions, label='Local model predictions')
    plt.plot(chs_predictions.index, chs_predictions.values, label='CHS predictions')
    plt.legend()
    plt.show()
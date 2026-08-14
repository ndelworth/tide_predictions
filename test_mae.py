import json

import numpy as np
import pandas as pd

WEIGHTS_PATH = "model_weights.json"
TEST_WLO_PATH = "data/tide_pointe_du_chene_test_wlo.csv"
TEST_WLP_PATH = "data/tide_pointe_du_chene_test_wlp.csv"


def load_weights(path=WEIGHTS_PATH):
    with open(path) as f:
        return json.load(f)


def predict_from_weights(weights, times):
    epoch = pd.Timestamp(weights["epoch"])
    t = pd.to_datetime(pd.Series(times), utc=True)
    hours = (t - epoch).dt.total_seconds().to_numpy() / 3600.0
    y = np.full(len(hours), weights["bias"])
    for c in weights["constituents"].values():
        omega = 2 * np.pi / c["period_hours"]
        y += c["cos"] * np.cos(omega * hours) + c["sin"] * np.sin(omega * hours)
    return y


def mae(predicted, actual):
    return np.mean(np.abs(predicted - actual))


def _load_csv(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def load_test_data(path, col):
    df = _load_csv(path)
    return df["time"], df[col].to_numpy()


def load_chs_wlp_and_wlo():
    merged = pd.merge(_load_csv(TEST_WLP_PATH), _load_csv(TEST_WLO_PATH), on="time", how="inner")
    return merged["wlp"].to_numpy(), merged["wlo"].to_numpy()


def main():
    weights = load_weights()

    times, actual = load_test_data(TEST_WLO_PATH, "wlo")
    predictions = predict_from_weights(weights, times)
    print(f"MAE: {mae(predictions, actual):.4f} m")

    wlp, wlo = load_chs_wlp_and_wlo()
    print(f"CHS WLP vs WLO MAE: {mae(wlp, wlo):.4f} m")


if __name__ == "__main__":
    main()

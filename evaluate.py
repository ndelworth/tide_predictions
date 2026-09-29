import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import HORIZON, MODEL_PATH, ROOT, STATION, TRAIN_END
from features import LEADS, ForecastDataset
from model import SurgeTransformer, device

RESULTS_DIR      = ROOT / "results"
CHS_FORECASTS_DB = ROOT.parent / "water-level-forecasts-data" / "docs" / "tides.db"

SERIES = ["model", "CHS", "persistence", "tide only"]
LEAD_BUCKETS = [0, 1, 2, 3, 6, 12, 18, 24, 30, 36, 42, 48]
PEAK_WIND_BINS   = [0, 5, 7.5, np.inf]
PEAK_WIND_LABELS = ["calm (<5 m/s)", "moderate (5-7.5 m/s)", "windy (>7.5 m/s)"]

WATER_LEVEL_PLOTS_UTC = {
    "2026-09-noreaster": ("2026-09-04", "2026-09-08"),
}
PLOT_LEADS = (3, 6, 12, 24, 36, 48)


def model_surge(data, issue_rows):
    model = SurgeTransformer().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    with torch.no_grad():
        return torch.cat([model(weather.to(device), surge.to(device), time.to(device)).cpu()
                          for weather, surge, time, _ in DataLoader(data.windows(issue_rows), batch_size=256)]).numpy()


def load_chs_forecasts():
    chs = pd.read_sql_query("SELECT poll_time AS issued, valid_time, wlf AS CHS FROM wlf",
                            sqlite3.connect(f"file:{CHS_FORECASTS_DB}?mode=ro", uri=True),
                            parse_dates={"issued": {"utc": True}, "valid_time": {"utc": True}})
    chs["lead"] = ((chs["valid_time"] - chs["issued"]) / pd.Timedelta(hours=1)).round().astype(int)
    return chs[["issued", "lead", "CHS"]]


def forecast_table(data):
    issued = np.flatnonzero(data.usable)
    rows = issued[:, None] + LEADS
    tide = data.df["tide"].to_numpy()[rows]
    wind = np.hypot(data.df["wind_u_10m"], data.df["wind_v_10m"]).to_numpy()[rows]
    table = pd.DataFrame({
        "issued":      data.times[issued].repeat(HORIZON),
        "lead":        np.tile(LEADS, len(issued)),
        "valid":       data.times[rows.ravel()],
        "observed":    data.df["wlo"].to_numpy()[rows].ravel(),
        "model":       (tide + model_surge(data, issued)).ravel(),
        "persistence": (tide + data.last_surge[issued, None]).ravel(),
        "tide only":   tide.ravel(),
        "peak_wind":   np.nanmax(wind, axis=1).repeat(HORIZON),
    })
    return table.merge(load_chs_forecasts(), on=["issued", "lead"], how="left")


def mae_cm_by_lead(table, by=()):
    errors_cm = table[SERIES].sub(table["observed"], axis=0).abs() * 100
    buckets = pd.cut(table["lead"], LEAD_BUCKETS, labels=[f"{lo}-{hi}h" for lo, hi in zip(LEAD_BUCKETS, LEAD_BUCKETS[1:])])
    return errors_cm.groupby([*(table[c] for c in by), buckets], observed=True).mean().round(2)


def plot_water_level(data, table, name, start, end):
    hours = pd.date_range(start, end, freq="h", tz="UTC")
    fig, axes = plt.subplots(len(PLOT_LEADS), 1, figsize=(12, 2.6 * len(PLOT_LEADS)), sharex=True, sharey=True)
    for ax, lead in zip(axes, PLOT_LEADS):
        at_lead = table[table["lead"] == lead].set_index("valid").reindex(hours)
        ax.plot(hours, data.df["wlo"].reindex(hours), color="k", lw=1.6, label="observed")
        for series, color in [("model", "C2"), ("CHS", "C3")]:
            mae = (at_lead[series] - at_lead["observed"]).abs().mean() * 100
            ax.plot(hours, at_lead[series], color=color, lw=1.3, label=f"{series} (MAE {mae:.1f} cm)")
        ax.set_ylabel(f"{lead} h ahead\nwater level (m)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"{STATION['name']}, {name}: forecast vs observed water level")
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"water_level_{name}.png", dpi=150)
    plt.close(fig)


def main():
    data = ForecastDataset()
    table = forecast_table(data)
    RESULTS_DIR.mkdir(exist_ok=True)

    holdout = table[table["issued"] > TRAIN_END].dropna(subset=["observed", *SERIES])
    holdout = holdout.assign(wind=pd.cut(holdout["peak_wind"], PEAK_WIND_BINS, labels=PEAK_WIND_LABELS, right=False))
    print(f"{holdout['issued'].nunique():,} holdout forecasts with a CHS forecast, "
          f"issued {holdout['issued'].min()} -> {holdout['issued'].max()}")

    mae = mae_cm_by_lead(holdout)
    mae.to_csv(RESULTS_DIR / "mae_by_lead.csv")
    print(f"\nMAE (cm) by hours ahead:\n{mae.to_string()}")

    mae = mae_cm_by_lead(holdout, by=["wind"])
    mae.to_csv(RESULTS_DIR / "mae_by_lead_and_wind.csv")
    forecasts = holdout.groupby("wind", observed=True)["issued"].nunique()
    print(f"\nMAE (cm) by the strongest wind during the forecast "
          f"({', '.join(f'{w}: {n:,}' for w, n in forecasts.items())} forecasts):\n{mae.to_string()}")

    for name, (start, end) in WATER_LEVEL_PLOTS_UTC.items():
        plot_water_level(data, table, name, start, end)
    print("\nSaved results/mae_by_lead.csv, mae_by_lead_and_wind.csv, "
          + ", ".join(f"water_level_{name}.png" for name in WATER_LEVEL_PLOTS_UTC))


if __name__ == "__main__":
    main()

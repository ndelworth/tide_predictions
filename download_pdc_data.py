import requests
import pandas as pd
import os
import time as time_module
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta

STATION_ID   = "64b6e5ec8027cb190816a0c0" # pointe-du-chene 
BASE_URL     = "https://api-iwls.dfo-mpo.gc.ca/api/v1"

START = datetime(2023, 8, 1, tzinfo=timezone.utc)
TRAIN_END   = datetime(2025, 5, 31, 23, 59, tzinfo=timezone.utc)
TEST_START  = TRAIN_END + timedelta(minutes=1)
TEST_END    = datetime(2026, 5, 31, 23, 59, tzinfo=timezone.utc)

SLEEP_BETWEEN_CHUNKS = 2.5
MAX_RETRIES = 3

CHUNK_SIZE = relativedelta(weeks=1)
OUT_CSV    = "data/tide_pointe_du_chene"


def fetch_chunk(series: str, start: datetime, end: datetime) -> list:
    url = f"{BASE_URL}/stations/{STATION_ID}/data"
    for resolution in ["ONE_MINUTE", "FIFTEEN_MINUTES", "SIXTY_MINUTES"]:
        print(f" Trying resolution: {resolution}")
        params = {
            "time-series-code": series,
            "from":             start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to":               end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "resolution":       resolution,
            "station-id":       STATION_ID,
        }
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 400:
                    break  # this resolution isn't archived for this range — try coarser
                r.raise_for_status()
                return r.json()
            except requests.HTTPError:
                raise
            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = attempt * 5
                    print(f"    retry {attempt}/{MAX_RETRIES} in {wait}s ({e})")
                    time_module.sleep(wait)
                else:
                    raise
            time_module.sleep(SLEEP_BETWEEN_CHUNKS)
    print(f"    {series} {start.date()}: no resolution worked")
    return []


def fetch_series(series: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    rows    = []
    current = start_date
    while current < end_date:
        chunk_end = min(current + CHUNK_SIZE, end_date)
        try:
            data = fetch_chunk(series, current, chunk_end)
            rows.extend(data)
            print(f"    {series} {current.date()} — {len(data)} rows")
        except Exception as e:
            print(f"    {series} {current.date()} FAILED: {type(e).__name__}: {e}")
        current = chunk_end
        time_module.sleep(SLEEP_BETWEEN_CHUNKS)

    if not rows:
        print(f"  WARNING: no data retrieved for {series}")
        return pd.DataFrame(columns=["time", series])

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["eventDate"], utc=True)
    df = df.rename(columns={"value": series})
    return df[["time", series]]


def main(out_csv, start_date, end_date, series):
    print(f"Fetching CHS tide data for GTSM comparison period: {start_date.date()} → {end_date.date()}")
    print(f"Station: ({STATION_ID})\n")

    print(f"Fetching {series}...")
    df = fetch_series(series, start_date, end_date)
    print(f"  {len(df):,} 1-min rows")


    os.makedirs("data", exist_ok=True)
    df.to_csv(f"{out_csv}_{series}.csv", index=False)
    print(f"\nSaved {out_csv}_{series}.csv  ({len(df):,} rows)")
    print(f"Range: {df['time'].min()} → {df['time'].max()}")

if __name__ == "__main__":
    main(f"{OUT_CSV}_train", START, TRAIN_END, "wlo")
    main(f"{OUT_CSV}_train", START, TRAIN_END, "wlp")
    main(f"{OUT_CSV}_test", TEST_START, TEST_END, "wlo")
    main(f"{OUT_CSV}_test", TEST_START, TEST_END, "wlp")
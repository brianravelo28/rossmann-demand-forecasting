"""
Build the slim deploy_data/ bundle that the hosted dashboard loads (app.py uses it
automatically when present; otherwise it reads the full data/ + models/ outputs).

Run after day1_eda.py / day2_model.py:   python build_deploy_data.py

Contents (columns trimmed and downcast; parquet + snappy):
  predictions.parquet  Jan-Jul 2015 holdout: actual, predicted, and the model's features
  history.parquet      raw daily rows, full Jan 2013 - Jul 2015 history (the app loads only the tail)
  test.parquet         the Aug-Sep 2015 calendar (known promos / holidays / open days)
  store.parquet        store attributes
  model.txt            LightGBM booster (text format; predicts log1p(Sales))
  day2_metrics.json    headline metrics shown in the dashboard header
"""
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_COLS

BASE = Path(__file__).resolve().parent
OUT = BASE / "deploy_data"
HISTORY_START = "2013-01-01"  # full daily history; the app reads only the tail it needs


def _size(path):
    return path.stat().st_size / 1e6


def main():
    OUT.mkdir(exist_ok=True)

    # --- model -> text format (no pickle / scikit-learn needed at serve time)
    with open(BASE / "models/lightgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    model.booster_.save_model(str(OUT / "model.txt"))

    # --- holdout predictions + features
    pred = pd.read_csv(
        BASE / "data/test_predictions.csv",
        usecols=["Store", "Date", "Sales", "Predicted"] + FEATURE_COLS,
        low_memory=False,
    )
    pred["Date"] = pd.to_datetime(pred["Date"])
    pred["Store"] = pred["Store"].astype("int16")
    for col in FEATURE_COLS + ["Sales", "Predicted"]:
        if col.startswith(("DoW_", "Month_", "StoreType_", "Assortment_")) or col in (
            "IsStateHoliday", "IsSchoolHoliday", "Promo_active", "Promo_lag_1", "IsPromo2Active", "Promo2"
        ):
            pred[col] = pred[col].astype("int8")
        elif col in ("DaysToNextHoliday", "DaysSinceLastHoliday", "DaysToNextPromo", "DaysSinceLastPromo"):
            pred[col] = pred[col].astype("int8")
        elif col == "CompetitionYears":
            pred[col] = pred[col].astype("int16")
        else:
            pred[col] = pred[col].astype("float32")
    pred.to_parquet(OUT / "predictions.parquet", compression="snappy", index=False)

    # --- sanity: booster reproduces the sklearn model on the downcast features
    sample = pred.sample(5000, random_state=0)
    ref = model.predict(sample[FEATURE_COLS])
    import lightgbm as lgb

    got = lgb.Booster(model_file=str(OUT / "model.txt")).predict(sample[FEATURE_COLS])
    print("max |booster - sklearn| on 5,000 rows (log scale):", float(np.abs(ref - got).max()))

    # --- raw history tail
    hist = pd.read_csv(
        BASE / "data/train.csv",
        dtype={"StateHoliday": str},
        usecols=["Store", "Date", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday", "DayOfWeek"],
        low_memory=False,
    )
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist[hist["Date"] >= HISTORY_START].copy()
    hist["Store"] = hist["Store"].astype("int16")
    hist["Sales"] = hist["Sales"].astype("float32")
    for col in ("Open", "Promo", "SchoolHoliday", "DayOfWeek"):
        hist[col] = hist[col].astype("int8")
    hist.to_parquet(OUT / "history.parquet", compression="snappy", index=False)

    # --- future calendar + store attributes
    test = pd.read_csv(BASE / "data/test.csv", dtype={"StateHoliday": str}, low_memory=False)
    test["Date"] = pd.to_datetime(test["Date"])
    test.drop(columns=["Id"]).to_parquet(OUT / "test.parquet", compression="snappy", index=False)
    pd.read_csv(BASE / "data/store.csv").to_parquet(OUT / "store.parquet", compression="snappy", index=False)

    shutil.copy(BASE / "day2_metrics.json", OUT / "day2_metrics.json")

    total = 0.0
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:22s} {_size(f):6.2f} MB")
        total += _size(f)
    print(f"{'TOTAL':22s} {total:6.2f} MB")


if __name__ == "__main__":
    main()

"""
Multi-week backtest behind the numbers quoted in the README.

Question: how accurate is the dashboard's recursive forecast (it never sees real sales after the
cutoff) compared with (a) the same model predicting one day at a time with real recent sales, and
(b) a plain lookup of each store's average sales for that weekday and promo status?

Setup: 150 random stores (seed 0). Forecast Jul 1 - Jul 31 2015 starting from the data up to
Jun 30 2015, using the model trained on 2013-2014 only. Scored on open days with sales > 0.

Needs only deploy_data/ (no Kaggle download):   python analysis/backtest_recursive.py
Takes about a minute. Also writes analysis/backtest_results.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402  (loads deploy_data/ and exposes recursive_forecast)

CUTOFF = pd.Timestamp("2015-07-01")
N_STORES, SEED = 150, 0


def mape(actual, pred):
    return float((np.abs(actual - pred) / actual).mean() * 100)


stores = np.random.default_rng(SEED).choice(app.STORE_IDS, N_STORES, replace=False)

# (b) lookup baseline: store x weekday x promo mean from the training period (2013-2014)
hist = pd.read_parquet(ROOT / "deploy_data" / "history.parquet")
train = hist[(hist.Open == 1) & (hist.Sales > 0) & (hist.Date < "2015-01-01")]
lookup = train.groupby(["Store", "DayOfWeek", "Promo"])["Sales"].mean().rename("lookup").reset_index()

rec, look, one = [], [], []
for s in stores:
    raw = app.RAW_BY_STORE[s]
    before, after = raw[raw.Date < CUTOFF], raw[raw.Date >= CUTOFF]
    actual = after[(after.Open == 1) & (after.Sales > 0)]

    fc = app.recursive_forecast(s, after, before)
    rec.append(fc.merge(actual[["Date", "Sales"]], on="Date"))

    look.append(actual.merge(lookup, on=["Store", "DayOfWeek", "Promo"], how="left").dropna(subset=["lookup"]))

    p = app.predictions_df
    one.append(p[(p.Store == s) & (p.Date >= CUTOFF) & (p.Sales > 0)])

rec, look, one = pd.concat(rec), pd.concat(look), pd.concat(one)
inside = (rec.Sales >= rec.Lower_PI) & (rec.Sales <= rec.Upper_PI)

print(f"stores={N_STORES}  store-days scored={len(rec)}  (Jul 2015)")
print(f"LightGBM, recursive (no real sales after Jun 30): MAPE {mape(rec.Sales, rec.Forecast):.2f}%")
print(f"LightGBM, one-step (real recent sales each day):   MAPE {mape(one.Sales, one.Predicted):.2f}%")
print(f"Store x weekday x promo lookup (no recent sales):  MAPE {mape(look.Sales, look.lookup):.2f}%")
print(f"Recursive bias (mean signed % error): {((rec.Forecast - rec.Sales) / rec.Sales).mean() * 100:+.2f}%")
print(f"Share of actuals inside the +/-1.96 sigma band:    {inside.mean() * 100:.1f}%")

rec["bucket"] = ((rec.Date - CUTOFF).dt.days // 7).clip(upper=3)
weekly = rec.groupby("bucket").apply(lambda g: mape(g.Sales, g.Forecast), include_groups=False)
print("Recursive MAPE by 7-day bucket:", " / ".join(f"{v:.1f}%" for v in weekly))

results = {
    "stores": N_STORES,
    "seed": SEED,
    "store_days_scored": int(len(rec)),
    "window": "2015-07-01 to 2015-07-31",
    "recursive_mape": round(mape(rec.Sales, rec.Forecast), 2),
    "one_step_mape": round(mape(one.Sales, one.Predicted), 2),
    "lookup_mape": round(mape(look.Sales, look.lookup), 2),
    "recursive_mape_by_7day_bucket": [round(float(v), 1) for v in weekly],
    "recursive_bias_pct": round(float(((rec.Forecast - rec.Sales) / rec.Sales).mean() * 100), 2),
    "band_coverage_pct": round(float(inside.mean() * 100), 1),
}
(ROOT / "analysis" / "backtest_results.json").write_text(json.dumps(results, indent=2) + "\n")

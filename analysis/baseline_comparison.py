"""
Fair comparison of forecasting methods over the same month.

Same 150 stores and the same window as analysis/backtest_recursive.py: forecast 1-31 Jul 2015 from data up
to 30 Jun 2015, scored on open days with sales > 0. Every method gets the same known-in-advance inputs
(open flag, promotions, state and school holidays for July) and no real sales after 30 Jun.

  naive_4wk      mean of the same weekday's open days in the last four weeks before the cutoff
  lookup         store x weekday x promo average of 2013-14 sales
  sarimax        per-store SARIMAX on log1p(sales), daily calendar (closed days = 0), weekly seasonality,
                 with open / promo / holiday inputs; fitted through 30 Jun 2015
  lightgbm_recursive   the dashboard's recursive forecast (model trained on 2013-14 only)
  lightgbm_one_step    for reference: real recent sales known each day (not a month-ahead number)

SARIMAX sees six months more history than the LightGBM model (which was trained on 2013-14), which favors
SARIMAX. All methods are scored on the same rows (those every method produced a forecast for).

Needs only deploy_data/ plus statsmodels:   python analysis/baseline_comparison.py
Takes about ten to fifteen minutes. Writes analysis/baseline_comparison.json and baseline_comparison_by_store.csv.
"""
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CUTOFF = pd.Timestamp("2015-07-01")
END = pd.Timestamp("2015-07-31")
N_STORES, SEED = 150, 0
EXOG = ["Open", "Promo", "SchoolHoliday", "StateHoliday"]


def mape(actual, pred):
    return float((np.abs(actual - pred) / actual).mean() * 100)


def daily_frame(g):
    """One store's history on the full daily calendar through 31 Jul 2015, inputs filled with 0 where a day has no row."""
    g = g.assign(StateHoliday=(g["StateHoliday"].astype(str) != "0").astype(int))
    g = g.set_index("Date").sort_index()
    full = pd.date_range(g.index.min(), END, freq="D")
    return g.reindex(full)


def fit_forecast_sarimax(args):
    """Worker: fit on data before the cutoff, forecast July with the known July inputs. Returns (store, Series or None)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    store, g = args
    warnings.simplefilter("ignore")
    d = daily_frame(g)
    train, test = d[d.index < CUTOFF], d[d.index >= CUTOFF]
    # An input that never varies for this store (for example Open, if it is never closed) cannot be told apart from the intercept.
    cols = [c for c in EXOG if train[c].fillna(0).nunique() > 1]
    exog_train = train[cols].fillna(0).astype(float)
    exog_test = test[cols].fillna(0).astype(float)
    try:
        model = SARIMAX(
            np.log1p(train["Sales"].astype(float)),
            exog=exog_train,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 0, 7),
            trend="c",
        )
        res = model.fit(disp=False, maxiter=100)
        if not res.mle_retvals.get("converged", True):
            return store, None
        pred = np.expm1(res.forecast(steps=len(test), exog=exog_test)).clip(lower=0)
        return store, pd.Series(pred.values, index=test.index)
    except Exception:
        return store, None


def naive_4wk(g, actual_dates):
    """Mean of the same weekday's open days (sales > 0) in the 28 days before the cutoff."""
    before = g[(g.Date >= CUTOFF - pd.Timedelta(days=28)) & (g.Date < CUTOFF) & (g.Open == 1) & (g.Sales > 0)]
    by_dow = before.groupby("DayOfWeek")["Sales"].mean()
    dow = pd.Series(actual_dates.dt.day_of_week + 1, index=actual_dates.index)
    return dow.map(by_dow)


def main():
    import app  # loads deploy_data/ and exposes recursive_forecast

    hist = pd.read_parquet(ROOT / "deploy_data" / "history.parquet")
    stores = [int(s) for s in np.random.default_rng(SEED).choice(app.STORE_IDS, N_STORES, replace=False)]

    train = hist[(hist.Open == 1) & (hist.Sales > 0) & (hist.Date < "2015-01-01")]
    lookup = train.groupby(["Store", "DayOfWeek", "Promo"])["Sales"].mean().rename("lookup").reset_index()

    print(f"Fitting SARIMAX for {N_STORES} stores...", flush=True)
    jobs = [(s, hist[hist.Store == s]) for s in stores]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        sarimax = dict(pool.map(fit_forecast_sarimax, jobs, chunksize=2))
    failed = sorted(s for s, v in sarimax.items() if v is None)
    print(f"SARIMAX did not converge for {len(failed)} stores: {failed}", flush=True)

    rows = []
    for s in stores:
        if s in failed:
            continue
        h = hist[hist.Store == s]
        actual = h[(h.Date >= CUTOFF) & (h.Date <= END) & (h.Open == 1) & (h.Sales > 0)][["Store", "Date", "Sales", "DayOfWeek", "Promo"]].copy()
        actual["naive_4wk"] = naive_4wk(h, actual["Date"]).values
        actual = actual.merge(lookup, on=["Store", "DayOfWeek", "Promo"], how="left")
        actual["sarimax"] = actual["Date"].map(sarimax[s]).values

        raw = app.RAW_BY_STORE[s]
        fc = app.recursive_forecast(s, raw[raw.Date >= CUTOFF], raw[raw.Date < CUTOFF])
        actual = actual.merge(fc[["Date", "Forecast"]].rename(columns={"Forecast": "lightgbm_recursive"}), on="Date", how="left")
        one = app.predictions_df
        one = one[(one.Store == s) & (one.Date >= CUTOFF)][["Date", "Predicted"]].rename(columns={"Predicted": "lightgbm_one_step"})
        rows.append(actual.merge(one, on="Date", how="left"))
    df = pd.concat(rows, ignore_index=True)

    methods = ["naive_4wk", "lookup", "sarimax", "lightgbm_recursive", "lightgbm_one_step"]
    scored = df.dropna(subset=methods)
    print(f"stores={df.Store.nunique()}  store-days with a forecast from every method={len(scored)} of {len(df)}")
    results = {
        "window": "2015-07-01 to 2015-07-31",
        "stores": int(df.Store.nunique()),
        "store_days_scored": int(len(scored)),
        "store_days_dropped_no_forecast": int(len(df) - len(scored)),
        "sarimax_failed_stores": failed,
        "mape": {m: round(mape(scored.Sales, scored[m]), 2) for m in methods},
    }
    by_store = scored.groupby("Store").apply(lambda g: pd.Series({m: mape(g.Sales, g[m]) for m in methods}), include_groups=False)
    results["share_of_stores_where_lightgbm_recursive_beats"] = {
        m: round(float((by_store["lightgbm_recursive"] < by_store[m]).mean() * 100), 1) for m in ["naive_4wk", "lookup", "sarimax"]
    }
    for m, v in results["mape"].items():
        print(f"{m:20s} MAPE {v:.2f}%")
    print("LightGBM recursive beats it on this share of stores:", results["share_of_stores_where_lightgbm_recursive_beats"])
    (ROOT / "analysis" / "baseline_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    by_store.round(2).to_csv(ROOT / "analysis" / "baseline_comparison_by_store.csv")


if __name__ == "__main__":
    main()

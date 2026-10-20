"""
Feature ablations behind the "what changed the score" table in docs/METHODOLOGY.md.

Three views, all scored the same way as the main pipeline (train_model.py: same LightGBM settings, the
four 2014 walk-forward folds, then train on 2013-2014 and score Jan-Jul 2015; MAPE skips zero-sales days):

  ladder          start from a plain feature set and add one change at a time, ending at the final model.
  leave_one_out   start from the final model and remove or revert one change at a time.
  noise           the final model again with a different random seed or column order, to show how much the
                  score moves by chance (so differences smaller than that should not be read as real).

The ladder is a reconstruction in a sensible order, not a replay of the order the changes were made in.
The final rung should reproduce cv_results.csv / metrics.json.

Needs the raw Kaggle files in data/ (train.csv, store.csv). Writes analysis/ablations.csv.
Takes roughly twenty minutes:   python analysis/ablations.py

`python analysis/ablations.py bias` instead measures the average signed error on Fold 3 (Jul-Sep 2014)
with and without DaysSinceStart and writes analysis/ablations_fold3_bias.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_model import FEATURE_COLS, FOLDS, LGBM_PARAMS, _mape, engineer_features, load_raw_merged  # noqa: E402

OUT = ROOT / "analysis" / "ablations.csv"

BASE = (
    [f"DoW_{i}" for i in range(1, 7)]
    + [f"Month_{i}" for i in range(1, 12)]
    + ["IsStateHoliday", "IsSchoolHoliday", "Promo_active", "Promo2", "CompetitionYears", "CompetitionDistance"]
)
STORE_FORMAT = [f"StoreType_{t}" for t in "abcd"] + [f"Assortment_{a}" for a in "abc"]
HOLIDAY_DIST = ["DaysToNextHoliday", "DaysSinceLastHoliday"]
PROMO_DIST = ["DaysToNextPromo", "DaysSinceLastPromo", "IsPromo2Active"]
CAL_LAGS = ["Promo_lag_1", "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_28", "Sales_rolling_7d", "Sales_rolling_30d"]
ROW_LAGS = [c + "_row" for c in CAL_LAGS]


def add_row_shift_features(df):
    """The lag features as a plain row shift within each store's open days (no calendar alignment)."""
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    g = df.groupby("Store")
    for k in (1, 7, 14, 28):
        df[f"Sales_lag_{k}_row"] = g["Sales"].shift(k).fillna(0)
    df["Promo_lag_1_row"] = g["Promo_active"].shift(1).fillna(0)
    for w in (7, 30):
        df[f"Sales_rolling_{w}d_row"] = g["Sales"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean()).fillna(0)
    return df


CANONICAL_ORDER = {c: i for i, c in enumerate(FEATURE_COLS)}


def config(log=True, store_format=True, cal_lags=True, holiday=True, promo=True, days_since_start=False, seed=None, reverse=False):
    """Feature columns are put in the same order as FEATURE_COLS so the final model reproduces metrics.json:
    LightGBM's column subsampling depends on column order, so the order alone moves the score a little."""
    feats = list(BASE)
    feats += CAL_LAGS if cal_lags else ROW_LAGS
    feats += STORE_FORMAT if store_format else []
    feats += HOLIDAY_DIST if holiday else []
    feats += PROMO_DIST if promo else []
    feats += ["DaysSinceStart"] if days_since_start else []
    feats = sorted(feats, key=lambda c: CANONICAL_ORDER.get(c.removesuffix("_row"), len(CANONICAL_ORDER)))
    return {"log": log, "features": feats[::-1] if reverse else feats, "seed": seed}


LADDER = [
    ("0 start: row-shift lags, raw target, with DaysSinceStart", config(False, False, False, False, False, True)),
    ("1 + log1p target", config(True, False, False, False, False, True)),
    ("2 + store format (StoreType, Assortment)", config(True, True, False, False, False, True)),
    ("3 + calendar-day lags and rolling means", config(True, True, True, False, False, True)),
    ("4 + holiday distance", config(True, True, True, True, False, True)),
    ("5 + promo distance and IsPromo2Active", config(True, True, True, True, True, True)),
    ("6 - DaysSinceStart (final model)", config()),
]
LEAVE_ONE_OUT = [
    ("final model", config()),
    ("without log1p target", config(log=False)),
    ("without store format", config(store_format=False)),
    ("row-shift lags instead of calendar-day", config(cal_lags=False)),
    ("without holiday distance", config(holiday=False)),
    ("without promo distance and IsPromo2Active", config(promo=False)),
    ("with DaysSinceStart added", config(days_since_start=True)),
]
# Same final model, only the random seed or the column order changed: how much does the score move by chance?
NOISE = [
    ("final model, random_state=7", config(seed=7)),
    ("final model, random_state=123", config(seed=123)),
    ("final model, columns in reverse order", config(reverse=True)),
]


def fit_predict(train, test, cfg):
    y = np.log1p(train["Sales"]) if cfg["log"] else train["Sales"]
    params = dict(LGBM_PARAMS, **({"random_state": cfg["seed"]} if cfg["seed"] is not None else {}))
    model = LGBMRegressor(**params)
    model.fit(train[cfg["features"]], y)
    pred = model.predict(test[cfg["features"]])
    return (np.expm1(pred) if cfg["log"] else pred).clip(0)


def evaluate(df, cfg):
    fold_mapes = []
    for tr_start, tr_end, va_start, va_end in FOLDS:
        train = df[(df.Date >= tr_start) & (df.Date <= tr_end)]
        val = df[(df.Date >= va_start) & (df.Date <= va_end)]
        fold_mapes.append(_mape(val["Sales"].values, fit_predict(train, val, cfg)))
    train = df[(df.Date >= "2013-01-01") & (df.Date <= "2014-12-31")]
    hold = df[(df.Date >= "2015-01-01") & (df.Date <= "2015-07-31")]
    return fold_mapes, _mape(hold["Sales"].values, fit_predict(train, hold, cfg))


def fold3_bias(df):
    """Mean signed % error (prediction minus actual, over actual) on Fold 3, with and without DaysSinceStart."""
    tr_start, tr_end, va_start, va_end = FOLDS[2]
    train = df[(df.Date >= tr_start) & (df.Date <= tr_end)]
    val = df[(df.Date >= va_start) & (df.Date <= va_end)]
    val = val[val["Sales"] > 0]
    rows = []
    for name, cfg in (("final model", config()), ("with DaysSinceStart added", config(days_since_start=True))):
        pred = fit_predict(train, val, cfg)
        rows.append(
            {
                "config": name,
                "fold3_mape": round(_mape(val["Sales"].values, pred), 2),
                "fold3_mean_signed_pct_error": round(float(((pred - val["Sales"].values) / val["Sales"].values).mean() * 100), 2),
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "analysis" / "ablations_fold3_bias.csv", index=False)


def load_features():
    print("Loading and engineering features...", flush=True)
    return add_row_shift_features(engineer_features(load_raw_merged(str(ROOT / "data/train.csv"), str(ROOT / "data/store.csv"))))


def main():
    df = load_features()
    if len(sys.argv) > 1 and sys.argv[1] == "bias":
        return fold3_bias(df)
    rows = []
    for view, configs in (("ladder", LADDER), ("leave_one_out", LEAVE_ONE_OUT), ("noise", NOISE)):
        for name, cfg in configs:
            t0 = time.time()
            folds, hold = evaluate(df, cfg)
            rows.append(
                {
                    "view": view,
                    "config": name,
                    "n_features": len(cfg["features"]),
                    "cv_mean_mape": round(float(np.mean(folds)), 2),
                    **{f"fold{i}_mape": round(m, 2) for i, m in enumerate(folds, 1)},
                    "holdout_mape": round(hold, 2),
                }
            )
            print(f"[{view}] {name}: CV {rows[-1]['cv_mean_mape']}  holdout {rows[-1]['holdout_mape']}  ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

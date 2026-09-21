"""
Day 2: Feature engineering, ARIMA baseline, and LightGBM walk-forward CV
for Rossmann demand forecasting.

Outputs:
- models/lightgbm_model.pkl
- data/test_predictions.csv
- cv_results.csv
- arima_baseline.html

NOTE: the saved model is trained on log1p(Sales) (standard for retail %-error
metrics across stores of very different scale). Any code loading the pickle
must do np.expm1(model.predict(X)).clip(0) to get Sales back, not
model.predict(X) directly.
"""
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lightgbm import LGBMRegressor
from pmdarima import auto_arima

LGBM_PARAMS = dict(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    max_depth=7,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    verbose=-1,
    random_state=42,
)

from features import (  # noqa: F401  (re-exported for callers of day2_model)
    STORE_TYPES,
    ASSORTMENTS,
    HOLIDAY_MAX_DAYS,
    FEATURE_COLS,
    extract_event_calendar,
    add_event_distance_features,
    extract_holiday_calendar,
    add_holiday_distance_features,
    extract_promo_calendar,
    add_promo_distance_features,
    add_promo2_active_feature,
)


def load_raw_merged(train_path="data/train.csv", store_path="data/store.csv"):
    """Full per-store history, Open=0 rows INCLUDED (needed for calendar-correct lags)."""
    train = pd.read_csv(train_path, dtype={"StateHoliday": str}, low_memory=False)
    store = pd.read_csv(store_path)
    train["Date"] = pd.to_datetime(train["Date"])

    df = train.merge(store, on="Store", how="left")
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(999999)
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(1)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(2000)
    return df


def _calendar_day_lag_merge(df, lag_days, store_col, date_col, value_col, out_col):
    """Exact calendar-day lag via a date-shifted merge — robust to missing rows
    (data gaps) and to stores that don't log rows for days they're closed."""
    lookup = df[[store_col, date_col, value_col]].copy()
    lookup[date_col] = lookup[date_col] + pd.Timedelta(days=lag_days)
    lookup = lookup.rename(columns={value_col: out_col})
    return df.merge(lookup, on=[store_col, date_col], how="left")


def _calendar_complete_rolling(df, store_col, date_col, sales_col, windows):
    """Rolling mean (then shifted 1 day) computed on a per-store calendar-complete
    series, so a rolling window spans true calendar days even across data gaps or
    weekly closures — not just however many rows happen to precede it."""
    parts = {w: [] for w in windows}
    for store_id, g in df.groupby(store_col):
        s = g.set_index(date_col)[sales_col].sort_index()
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s_full = s.reindex(full_idx)
        for w in windows:
            roll = s_full.rolling(w, min_periods=1).mean().shift(1)
            parts[w].append(
                pd.DataFrame(
                    {store_col: store_id, date_col: roll.index, f"{sales_col}_rolling_{w}d": roll.values}
                )
            )
    out = None
    for w in windows:
        part = pd.concat(parts[w], ignore_index=True)
        out = part if out is None else out.merge(part, on=[store_col, date_col], how="outer")
    return out


def engineer_features(df, date_col="Date", store_col="Store", sales_col="Sales"):
    """`df` must be the FULL per-store history (Open=0 rows included) — see
    load_raw_merged(). Lag/rolling/promo-lag features are computed on true
    calendar-day offsets, then the result is filtered to Open==1 rows only,
    matching the original Day-1 "rows: Open=1 only" modeling contract."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([store_col, date_col]).reset_index(drop=True)

    for lag in [1, 7, 14, 28]:
        df = _calendar_day_lag_merge(
            df, lag, store_col, date_col, sales_col, f"{sales_col}_lag_{lag}"
        )
    df = _calendar_day_lag_merge(df, 1, store_col, date_col, "Promo", "Promo_lag_1")
    df["Promo_lag_1"] = df["Promo_lag_1"].fillna(0).astype(int)

    rolling_df = _calendar_complete_rolling(df, store_col, date_col, sales_col, windows=[7, 30])
    df = df.merge(rolling_df, on=[store_col, date_col], how="left")

    holiday_cal = extract_holiday_calendar(df, store_col, date_col, "StateHoliday")
    df = add_holiday_distance_features(df, holiday_cal, store_col, date_col)

    promo_cal = extract_promo_calendar(df, store_col, date_col, "Promo")
    df = add_promo_distance_features(df, promo_cal, store_col, date_col)

    df = add_promo2_active_feature(df, date_col)

    for lag in [1, 7, 14, 28]:
        df[f"{sales_col}_lag_{lag}"] = df[f"{sales_col}_lag_{lag}"].fillna(0)
    for window in [7, 30]:
        df[f"{sales_col}_rolling_{window}d"] = df[f"{sales_col}_rolling_{window}d"].fillna(0)

    dow = pd.get_dummies(df["DayOfWeek"], prefix="DoW")
    for i in range(1, 7):
        df[f"DoW_{i}"] = dow.get(f"DoW_{i}", 0).astype(int)

    month = pd.get_dummies(df[date_col].dt.month, prefix="Month")
    for i in range(1, 12):
        df[f"Month_{i}"] = month.get(f"Month_{i}", 0).astype(int)

    # Computed but deliberately NOT in FEATURE_COLS: every val/test fold is beyond the
    # training date range by construction, so a tree model can only extrapolate this
    # linear counter by reusing its rightmost leaf (~H1 2014's elevated sales level),
    # which measurably over-predicted the Q3 2014 seasonal dip. Dropping it improved
    # every walk-forward fold, not just the one that exposed the problem. Kept as a
    # column (unused by the model) since it's still useful for EDA/debugging.
    df["DaysSinceStart"] = (df[date_col] - df[date_col].min()).dt.days

    df["IsStateHoliday"] = (df["StateHoliday"].astype(str) != "0").astype(int)
    df["IsSchoolHoliday"] = df["SchoolHoliday"].astype(int)

    df["Promo_active"] = df["Promo"].astype(int)

    df["CompetitionYears"] = (
        df[date_col].dt.year - df["CompetitionOpenSinceYear"]
    ).clip(0, 100)

    for t in STORE_TYPES:
        df[f"StoreType_{t}"] = (df["StoreType"] == t).astype(int)
    for a in ASSORTMENTS:
        df[f"Assortment_{a}"] = (df["Assortment"] == a).astype(int)

    df = df[df["Open"] == 1].reset_index(drop=True)
    df = df.drop(columns=["DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"])

    return df


def _mape(actual, pred):
    mask = actual != 0
    return float(np.mean(np.abs(actual[mask] - pred[mask]) / actual[mask]) * 100)


def _rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))


def fit_arima_baseline(df, store_id=1, split_date="2015-01-01", output_path="arima_baseline.html"):
    d = df[df["Store"] == store_id].sort_values("Date")
    split = pd.Timestamp(split_date)
    train = d[d["Date"] < split]
    test = d[d["Date"] >= split]

    model = auto_arima(
        train["Sales"], seasonal=False, max_p=5, max_d=2, max_q=5, stepwise=True
    )
    order = model.order
    forecast = model.predict(n_periods=len(test))

    actual = test["Sales"].values
    pred = np.asarray(forecast)

    mape = _mape(actual, pred)
    rmse = _rmse(actual, pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Sales"], mode="lines", name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=test["Date"], y=pred, mode="lines", name="ARIMA Forecast", line=dict(color="green")))
    fig.update_layout(
        title=f"Store {store_id} ARIMA{order} Baseline (MAPE={mape:.1f}%)",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
    )
    fig.write_html(output_path)

    return {"order": order, "mape": mape, "rmse": rmse}


FOLDS = [
    ("2013-01-01", "2013-12-31", "2014-01-01", "2014-03-31"),
    ("2013-01-01", "2014-03-31", "2014-04-01", "2014-06-30"),
    ("2013-01-01", "2014-06-30", "2014-07-01", "2014-09-30"),
    ("2013-01-01", "2014-09-30", "2014-10-01", "2014-12-31"),
]


def walk_forward_cv(df, feature_cols, train_end_date="2015-01-01"):
    results = []
    for i, (tr_start, tr_end, val_start, val_end) in enumerate(FOLDS, start=1):
        train_fold = df[(df["Date"] >= tr_start) & (df["Date"] <= tr_end)]
        val_fold = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)]

        X_train, y_train = train_fold[feature_cols], train_fold["Sales"]
        X_val, y_val = val_fold[feature_cols], val_fold["Sales"]

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_train, np.log1p(y_train))
        preds = np.expm1(model.predict(X_val)).clip(0)

        results.append(
            {
                "Fold": i,
                "RMSE": _rmse(y_val.values, preds),
                "MAPE": _mape(y_val.values, preds),
                "MAE": _mae(y_val.values, preds),
            }
        )
        print(f"Fold {i}: RMSE={results[-1]['RMSE']:.1f} MAPE={results[-1]['MAPE']:.2f}% MAE={results[-1]['MAE']:.1f}")

    cv_results_df = pd.DataFrame(results)

    full_train = df[(df["Date"] >= "2013-01-01") & (df["Date"] <= "2014-12-31")]
    final_model = LGBMRegressor(**LGBM_PARAMS)
    final_model.fit(full_train[feature_cols], np.log1p(full_train["Sales"]))

    test_period = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2015-07-31")].copy()
    test_period["Predicted"] = np.expm1(final_model.predict(test_period[feature_cols])).clip(0)

    return cv_results_df, final_model, test_period


def save_model_and_predictions(model, predictions_df, model_path, csv_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    predictions_df.to_csv(csv_path, index=False)

    return {
        "model_saved": True,
        "predictions_rows": len(predictions_df),
        "predictions_shape": predictions_df.shape,
        "predicted_nan_count": int(predictions_df["Predicted"].isna().sum()),
    }


if __name__ == "__main__":
    print("Loading raw train.csv + store.csv (Open=0 rows included, for calendar-correct lags)...")
    df_raw = load_raw_merged("data/train.csv", "data/store.csv")
    print(f"Raw merged shape: {df_raw.shape}")

    print("Engineering features (calendar-day-aligned lags/rolling)...")
    df_feat = engineer_features(df_raw)
    print(f"Engineered shape: {df_feat.shape}")
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    print("Fitting ARIMA baseline on store 1...")
    arima_results = fit_arima_baseline(df_feat, store_id=1, split_date="2015-01-01")
    print(f"ARIMA order={arima_results['order']} MAPE={arima_results['mape']:.2f}% RMSE={arima_results['rmse']:.1f}")

    print("Running walk-forward CV + final LightGBM fit...")
    cv_results_df, final_model, test_with_predictions = walk_forward_cv(df_feat, FEATURE_COLS)
    cv_results_df.to_csv("cv_results.csv", index=False)
    print(cv_results_df)
    print(f"CV mean MAPE: {cv_results_df['MAPE'].mean():.2f}%")

    final_mape = _mape(test_with_predictions["Sales"].values, test_with_predictions["Predicted"].values)
    final_rmse = _rmse(test_with_predictions["Sales"].values, test_with_predictions["Predicted"].values)
    within_25 = float(
        np.mean(
            np.abs(test_with_predictions["Predicted"] - test_with_predictions["Sales"])
            / test_with_predictions["Sales"]
            <= 0.25
        )
        * 100
    )
    print(f"Final test MAPE: {final_mape:.2f}%  RMSE: {final_rmse:.1f}  within +/-25%: {within_25:.1f}%")

    print("Saving model and predictions...")
    save_info = save_model_and_predictions(
        final_model, test_with_predictions, "models/lightgbm_model.pkl", "data/test_predictions.csv"
    )
    print(save_info)

    import json
    final_metrics = {
        "arima": arima_results,
        "cv_mean_mape": float(cv_results_df["MAPE"].mean()),
        "cv_mean_rmse": float(cv_results_df["RMSE"].mean()),
        "cv_mean_mae": float(cv_results_df["MAE"].mean()),
        "final_mape": final_mape,
        "final_rmse": final_rmse,
        "within_25_pct": within_25,
        "feature_cols": FEATURE_COLS,
    }
    with open("day2_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)

    print("Day 2 complete.")

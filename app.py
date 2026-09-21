"""
Rossmann Demand Forecasting — Interactive Dash App

Tabs:
1. Forecast vs Actual      (one-step-ahead predictions on the Jan–Jul 2015 holdout)
2. Error Heatmap           (MAPE by store and ISO week, stores sorted by average error)
3. Promotion Simulator     (what-if: force promo on/off over a date range)
4. Forward Forecast        (recursive day-by-day forecast over test.csv's calendar)

Run: python app.py  ->  http://localhost:8050
"""
import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from day2_model import (
    FEATURE_COLS,
    STORE_TYPES,
    ASSORTMENTS,
    extract_holiday_calendar,
    add_holiday_distance_features,
    extract_promo_calendar,
    add_promo_distance_features,
    add_promo2_active_feature,
)

MODEL_PATH = BASE_DIR / "models/lightgbm_model.pkl"
PRED_PATH = BASE_DIR / "data/test_predictions.csv"
TEST_PATH = BASE_DIR / "data/test.csv"
STORE_PATH = BASE_DIR / "data/store.csv"
TRAIN_RAW_PATH = BASE_DIR / "data/train.csv"
METRICS_PATH = BASE_DIR / "day2_metrics.json"

# Design tokens (light surface; validated categorical slots 1-3 from the reference palette)
SURFACE = "#fcfcfb"
PAGE_BG = "#f4f4f2"
BORDER = "#e5e4df"
GRID = "#ecebe7"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#6b6a66"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4"]
MAX_FORECAST_STORES = 5

# ---------------------------------------------------------------- load ----

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict_sales(X):
    """Model predicts log1p(Sales); invert to real sales scale."""
    return np.expm1(model.predict(X)).clip(0)


predictions_df = pd.read_csv(
    PRED_PATH, usecols=["Store", "Date", "Sales", "Predicted"] + FEATURE_COLS, low_memory=False
)
predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
predictions_df["Week"] = predictions_df["Date"].dt.isocalendar().week.astype(int)

store_meta = pd.read_csv(STORE_PATH)
store_meta["CompetitionDistance"] = store_meta["CompetitionDistance"].fillna(999999)
store_meta["CompetitionOpenSinceMonth"] = store_meta["CompetitionOpenSinceMonth"].fillna(1)
store_meta["CompetitionOpenSinceYear"] = store_meta["CompetitionOpenSinceYear"].fillna(2000)
STORE_META = store_meta.set_index("Store")

test_future = pd.read_csv(TEST_PATH, dtype={"StateHoliday": str}, low_memory=False)
test_future["Date"] = pd.to_datetime(test_future["Date"])
test_future["Open"] = test_future["Open"].fillna(1)
FUTURE_START = test_future["Date"].min()
FUTURE_END = test_future["Date"].max()
FUTURE_DAYS = (FUTURE_END - FUTURE_START).days + 1

# Full raw history (closed days included: they are real zero-sales days the model's
# calendar-aligned lags were trained on).
_raw = pd.read_csv(
    TRAIN_RAW_PATH,
    dtype={"StateHoliday": str},
    usecols=["Store", "Date", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday", "DayOfWeek"],
    low_memory=False,
)
_raw["Date"] = pd.to_datetime(_raw["Date"])
RAW_BY_STORE = {s: g.sort_values("Date") for s, g in _raw.groupby("Store")}

# Holiday/promo calendars span history + test.csv's known future calendar, so the
# distance features can see events scheduled just past the end of the training data.
HOLIDAY_CAL = pd.concat(
    [
        extract_holiday_calendar(_raw, "Store", "Date", "StateHoliday"),
        extract_holiday_calendar(test_future, "Store", "Date", "StateHoliday"),
    ],
    ignore_index=True,
).drop_duplicates()
PROMO_CAL = pd.concat(
    [
        extract_promo_calendar(_raw, "Store", "Date", "Promo"),
        extract_promo_calendar(test_future, "Store", "Date", "Promo"),
    ],
    ignore_index=True,
).drop_duplicates()

STORE_IDS = sorted(predictions_df["Store"].unique().tolist())
STORE_OPTIONS = [{"label": f"Store #{s}", "value": s} for s in STORE_IDS]
# Kaggle's test.csv only covers a subset of stores; only those have a forward calendar.
FORWARD_STORE_IDS = sorted(set(test_future["Store"]) & set(RAW_BY_STORE))
FORWARD_STORE_OPTIONS = [{"label": f"Store #{s}", "value": s} for s in FORWARD_STORE_IDS]
DATE_MIN = predictions_df["Date"].min()
DATE_MAX = predictions_df["Date"].max()

# Per-store std of log-error on the holdout, for forecast bands (multiplicative, so
# bands scale with the store's sales level).
_nz = predictions_df[predictions_df["Sales"] > 0]
_log_err = np.log1p(_nz["Sales"]) - np.log1p(_nz["Predicted"])
GLOBAL_LOG_SIGMA = float(_log_err.std())
STORE_LOG_SIGMA = _log_err.groupby(_nz["Store"]).std().to_dict()

try:
    with open(METRICS_PATH) as f:
        METRICS = json.load(f)
except OSError:
    METRICS = {}


def mape(actual, pred):
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - pred[mask]) / actual[mask]) * 100)


def rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def within_pct(actual, pred, tol=0.25):
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(pred[mask] - actual[mask]) / actual[mask] <= tol) * 100)


# ------------------------------------------------------- tab2 precompute ----

_ape = predictions_df[predictions_df["Sales"] > 0].copy()
_ape["ape"] = (_ape["Sales"] - _ape["Predicted"]).abs() / _ape["Sales"] * 100
_hm = _ape.groupby(["Store", "Week"])["ape"].mean().unstack()
_hm = _hm.loc[_hm.mean(axis=1).sort_values(ascending=False).index]  # worst stores on top
HEATMAP = _hm
HEATMAP_CLIP = 40
HEATMAP_SHARE_UNDER_15 = float((_hm.values[~np.isnan(_hm.values)] <= 15).mean() * 100)
HEATMAP_SHARE_OVER_CLIP = float((_hm.values[~np.isnan(_hm.values)] > HEATMAP_CLIP).mean() * 100)
HEATMAP_WORST = HEATMAP.mean(axis=1).head(5)


# -------------------------------------------------- promotion simulator ----

def _counterfactual_promo_cal(store_id, start_date, end_date, promo_on):
    """PROMO_CAL for this store with the selected range replaced by a uniform
    on/off promo status, so DaysToNextPromo/DaysSinceLastPromo stay consistent
    with the toggled Promo_active instead of leaking the real historical pattern."""
    base = PROMO_CAL[PROMO_CAL["Store"] == store_id]
    outside = base[(base["Date"] < start_date) | (base["Date"] > end_date)]
    if not promo_on:
        return outside
    in_range = pd.DataFrame({"Store": store_id, "Date": pd.date_range(start_date, end_date, freq="D")})
    return pd.concat([outside, in_range], ignore_index=True)


def run_promo_scenario(store_id, start_date, end_date, promo_on):
    d = predictions_df[
        (predictions_df["Store"] == store_id)
        & (predictions_df["Date"] >= start_date)
        & (predictions_df["Date"] <= end_date)
    ].sort_values("Date").copy()

    if d.empty:
        return None

    scenario_a = d.copy()  # actual
    scenario_a["ScenarioSales"] = predict_sales(scenario_a[FEATURE_COLS])

    scenario_b = d.copy()
    toggle_val = 1 if promo_on else 0
    scenario_b["Promo_active"] = toggle_val
    scenario_b["Promo_lag_1"] = scenario_b["Promo_active"].shift(1).fillna(
        d["Promo_active"].iloc[0]
    ).astype(int)

    cf_promo_cal = _counterfactual_promo_cal(store_id, start_date, end_date, promo_on)
    cf_dist = add_promo_distance_features(
        scenario_b[["Store", "Date"]].copy(), cf_promo_cal, "Store", "Date"
    )
    scenario_b = scenario_b.drop(columns=["DaysToNextPromo", "DaysSinceLastPromo"]).merge(
        cf_dist[["Store", "Date", "DaysToNextPromo", "DaysSinceLastPromo"]], on=["Store", "Date"], how="left"
    )

    scenario_b["ScenarioSales"] = predict_sales(scenario_b[FEATURE_COLS])

    return d, scenario_a, scenario_b


# ------------------------------------------------- recursive forecasting ----

SALES_LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 30]


def recursive_forecast(store_id, fut, hist, log_sigma=None):
    """Day-by-day forecast for one store.

    `fut`: calendar rows to forecast (Date, DayOfWeek, Open, Promo, StateHoliday,
    SchoolHoliday) — these are known in advance. `hist`: this store's raw rows strictly
    before `fut` (Date, Sales, Promo). Each open day's prediction is written back into
    a per-date sales buffer, so later days' Sales_lag_* / Sales_rolling_* features are
    computed exactly as in training (calendar-day offsets, closed days = 0 sales,
    missing days ignored), but from the model's own earlier forecasts.
    Returns one row per open day: Date, Forecast, Lower_PI, Upper_PI.
    """
    meta = STORE_META.loc[store_id]
    fut = fut.sort_values("Date").reset_index(drop=True)
    dates = fut["Date"]

    hist_tail = hist.sort_values("Date").tail(60)
    buffer = dict(zip(hist_tail["Date"], hist_tail["Sales"].astype(float)))
    promo_by_date = dict(zip(hist_tail["Date"], hist_tail["Promo"].astype(int)))
    promo_by_date.update(zip(dates, fut["Promo"].astype(int)))

    feat = pd.DataFrame(index=fut.index)
    for i in range(1, 7):
        feat[f"DoW_{i}"] = (fut["DayOfWeek"] == i).astype(int)
    for m in range(1, 12):
        feat[f"Month_{m}"] = (dates.dt.month == m).astype(int)
    feat["IsStateHoliday"] = (fut["StateHoliday"].astype(str) != "0").astype(int)
    feat["IsSchoolHoliday"] = fut["SchoolHoliday"].astype(int)

    date_frame = pd.DataFrame({"Store": store_id, "Date": dates.values})
    hol = add_holiday_distance_features(date_frame.copy(), HOLIDAY_CAL, "Store", "Date")
    feat["DaysToNextHoliday"] = hol["DaysToNextHoliday"].values
    feat["DaysSinceLastHoliday"] = hol["DaysSinceLastHoliday"].values

    feat["Promo_active"] = fut["Promo"].astype(int)
    feat["Promo_lag_1"] = [promo_by_date.get(d - pd.Timedelta(days=1), 0) for d in dates]
    pr = add_promo_distance_features(date_frame.copy(), PROMO_CAL, "Store", "Date")
    feat["DaysToNextPromo"] = pr["DaysToNextPromo"].values
    feat["DaysSinceLastPromo"] = pr["DaysSinceLastPromo"].values

    promo2_input = pd.DataFrame(
        {
            "Date": dates.values,
            "Promo2": meta["Promo2"],
            "Promo2SinceYear": meta["Promo2SinceYear"],
            "Promo2SinceWeek": meta["Promo2SinceWeek"],
            "PromoInterval": meta["PromoInterval"],
        }
    )
    feat["IsPromo2Active"] = add_promo2_active_feature(promo2_input, "Date")["IsPromo2Active"].values

    feat["CompetitionYears"] = np.clip(dates.dt.year - meta["CompetitionOpenSinceYear"], 0, 100)
    feat["CompetitionDistance"] = meta["CompetitionDistance"]
    feat["Promo2"] = meta["Promo2"]
    for t in STORE_TYPES:
        feat[f"StoreType_{t}"] = int(meta["StoreType"] == t)
    for a in ASSORTMENTS:
        feat[f"Assortment_{a}"] = int(meta["Assortment"] == a)
    for col in [f"Sales_lag_{n}" for n in SALES_LAGS] + [f"Sales_rolling_{w}d" for w in ROLL_WINDOWS]:
        feat[col] = 0.0

    arr = feat[FEATURE_COLS].to_numpy(dtype=float)
    col_idx = {c: i for i, c in enumerate(FEATURE_COLS)}
    is_open = fut["Open"].to_numpy() == 1
    rows = []

    for i, d in enumerate(dates):
        if not is_open[i]:
            buffer[d] = 0.0  # a closed day is a real zero-sales day in training lags
            continue
        for n in SALES_LAGS:
            arr[i, col_idx[f"Sales_lag_{n}"]] = buffer.get(d - pd.Timedelta(days=n), 0.0)
        for w in ROLL_WINDOWS:
            vals = [buffer[d - pd.Timedelta(days=k)] for k in range(1, w + 1) if d - pd.Timedelta(days=k) in buffer]
            arr[i, col_idx[f"Sales_rolling_{w}d"]] = float(np.mean(vals)) if vals else 0.0
        x = pd.DataFrame(arr[i : i + 1], columns=FEATURE_COLS)
        pred = float(predict_sales(x)[0])
        buffer[d] = pred
        rows.append((d, pred))

    out = pd.DataFrame(rows, columns=["Date", "Forecast"])
    sigma = log_sigma if log_sigma is not None else STORE_LOG_SIGMA.get(store_id, GLOBAL_LOG_SIGMA)
    if pd.isna(sigma):
        sigma = GLOBAL_LOG_SIGMA
    out["Lower_PI"] = np.expm1(np.log1p(out["Forecast"]) - 1.96 * sigma)
    out["Upper_PI"] = np.expm1(np.log1p(out["Forecast"]) + 1.96 * sigma)
    return out


@lru_cache(maxsize=512)
def _forward_forecast_cached(store_id):
    fut = test_future[test_future["Store"] == store_id]
    hist = RAW_BY_STORE.get(store_id)
    if fut.empty or hist is None:
        return None
    return recursive_forecast(store_id, fut, hist)


@lru_cache(maxsize=256)
def _combined_log_sigma(store_key):
    """Std of the log-error of the *summed* daily sales of these stores on the 2015
    holdout. Measuring the aggregate directly captures how store errors offset (or
    move together) — no independence assumption. Returns None if too few shared days."""
    d = predictions_df[predictions_df["Store"].isin(store_key) & (predictions_df["Sales"] > 0)]
    per_day = d.groupby("Date").agg(n=("Store", "nunique"), actual=("Sales", "sum"), pred=("Predicted", "sum"))
    per_day = per_day[per_day["n"] == len(store_key)]
    if len(per_day) < 20:
        return None
    return float((np.log1p(per_day["actual"]) - np.log1p(per_day["pred"])).std())


def combine_forecasts(fc, store_ids):
    """Sum per-store forecasts by date into one combined trajectory with its own band."""
    g = fc.groupby("Date", as_index=False)[["Forecast", "Upper_PI", "Lower_PI"]].sum()
    sigma = _combined_log_sigma(tuple(sorted(store_ids)))
    if sigma is not None:
        g["Upper_PI"] = np.expm1(np.log1p(g["Forecast"]) + 1.96 * sigma)
        g["Lower_PI"] = np.expm1(np.log1p(g["Forecast"]) - 1.96 * sigma)
    g.insert(1, "Store", "Combined")
    return g


def build_forward_forecast(store_ids):
    frames = []
    for store_id in store_ids:
        fc = _forward_forecast_cached(store_id)
        if fc is None:
            continue
        frames.append(fc.assign(Store=store_id)[["Date", "Store", "Forecast", "Upper_PI", "Lower_PI"]])
    if not frames:
        return pd.DataFrame(columns=["Date", "Store", "Forecast", "Upper_PI", "Lower_PI"])
    return pd.concat(frames, ignore_index=True)


# ==================================================================== app ==

CSS = f"""
:root {{ color-scheme: light; }}
body {{ margin: 0; background: {PAGE_BG}; color: {INK};
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.wrap {{ font-size: 15px; max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }}
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 18px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 16px; }}
.kpi .v {{ font-size: 28px; font-weight: 650; letter-spacing: -0.01em; }}
.kpi .l {{ font-size: 15px; color: {INK_2}; margin-bottom: 2px; }}
.kpi .s {{ font-size: 15px; color: {INK_3}; margin-top: 2px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 16px 20px; align-items: flex-end; }}
.controls label {{ display: block; font-size: 15px; color: {INK_2}; margin-bottom: 4px; }}
.note {{ font-size: 15px; color: {INK_2}; line-height: 1.5; margin: 10px 2px 0; }}
details.about {{ margin-top: 12px; }}
details.about summary {{ cursor: pointer; font-weight: 600; color: {INK_2}; }}
details.about ul {{ margin: 8px 0 0; padding-left: 20px; color: {INK_2}; line-height: 1.55; font-size: 15px; }}
.dash-dropdown, .dash-dropdown *, .dash-datepicker-input, .DateInput_input, .Select-value-label, .Select-input input {{ font-size: 15px !important; }}
#t4-combine .dash-options-list-option {{ display: inline-flex !important; align-items: center; gap: 8px; margin: 0 !important; cursor: pointer; font-size: 15px; color: {INK_2}; }}
#t4-combine .dash-options-list-option-wrapper {{ display: inline-flex; }}
.dash-options-list-option-checkbox {{ width: 18px; height: 18px; margin: 0; cursor: pointer; }}
.dash-options-list:not(.dash-checklist) .dash-options-list-option {{ display: flex !important; align-items: center; gap: 10px; width: 100%; box-sizing: border-box; padding: 8px 12px; margin: 0; cursor: pointer; font-size: 15px; }}
.dash-options-list:not(.dash-checklist) .dash-options-list-option:hover {{ background: {PAGE_BG}; }}
button.primary {{ background: {BLUE}; color: #fff; border: 0; border-radius: 6px; padding: 9px 16px;
  font-size: 15px; font-weight: 600; cursor: pointer; }}
button.secondary {{ background: {SURFACE}; color: {INK}; border: 1px solid {BORDER}; border-radius: 6px;
  padding: 8px 14px; font-size: 15px; cursor: pointer; }}
"""

app = Dash(__name__)
app.title = "Rossmann Demand Forecasting"
app.index_string = f"""<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>{{%title%}}</title>
{{%favicon%}}
{{%css%}}
<style>{CSS}</style>
</head>
<body>
{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""


def kpi(label, value, sub=None):
    return html.Div(
        [html.Div(label, className="l"), html.Div(value, className="v")] + ([html.Div(sub, className="s")] if sub else []),
        className="kpi",
    )


def style_fig(fig, title=None, height=None):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0, font=dict(size=17, color=INK)) if title else None,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK_2, size=15),
        hoverlabel=dict(font_size=15),
        margin=dict(l=76, r=20, t=60 if title else 30, b=110),
        legend=dict(orientation="h", y=-0.2, x=0, xanchor="left", yanchor="top", font=dict(color=INK_2)),
        hovermode="x unified",
        height=height,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False, tickformat=",", hoverformat=",.2~f", automargin=True)
    return fig


def _fmt_pct(x):
    return f"{x:.1f}%" if isinstance(x, (int, float)) else "—"


def header():
    m = METRICS
    return html.Div(
        [
            html.H1("Rossmann Demand Forecasting", style={"margin": "0 0 4px", "fontSize": "28px", "letterSpacing": "-0.02em"}),
            html.Div(
                "Daily sales forecasts for 1,115 stores — one LightGBM model, validated forward in time.",
                style={"color": INK_2, "marginBottom": "16px"},
            ),
            html.Div(
                [
                    kpi("Holdout MAPE", _fmt_pct(m.get("final_mape")), "Jan–Jul 2015, never trained on"),
                    kpi("Within ±25% of actual", _fmt_pct(m.get("within_25_pct")), "share of store-days"),
                    kpi("Walk-forward CV MAPE", _fmt_pct(m.get("cv_mean_mape")), "4 expanding folds, 2014"),
                    kpi("ARIMA baseline, Store 1", _fmt_pct((m.get("arima") or {}).get("mape")), "per-store model, same holdout"),
                ],
                className="kpi-row",
            ),
            html.Details(
                [
                    html.Summary("About this model"),
                    html.Ul(
                        [
                            html.Li("One LightGBM regressor for all stores, trained on log(1 + daily sales) so small and large stores count equally in percentage terms."),
                            html.Li("42 features: weekday/month, state & school holidays and days to/since the nearest holiday, promo status and days to/since the nearest promo, the recurring Promo2 program, calendar-aligned sales lags (1/7/14/28 days) and rolling means (7/30 days), store type, assortment and competition."),
                            html.Li("Validated with 4 expanding-window folds across 2014, then a final holdout (Jan–Jul 2015) the model never saw."),
                            html.Li("Tabs 1–3 show one-step-ahead predictions: each day is predicted knowing the real sales of the days before it. Tab 4 has no real sales to lean on, so it forecasts recursively, feeding its own predictions forward — expect it to be less accurate than the headline numbers."),
                            html.Li("Sales are in euros. Closed days are excluded from accuracy metrics."),
                        ]
                    ),
                ],
                className="about",
            ),
        ],
        className="card",
        style={"marginBottom": "16px"},
    )


TAB_STYLE = {"padding": "10px 14px", "fontSize": "15px", "border": f"1px solid {BORDER}", "backgroundColor": PAGE_BG, "color": INK_2, "fontWeight": "500"}
TAB_SELECTED = {**TAB_STYLE, "backgroundColor": SURFACE, "color": INK, "fontWeight": "650", "borderTop": f"2px solid {BLUE}"}

app.layout = html.Div(
    className="wrap",
    children=[
        header(),
        dcc.Tabs(
            id="tabs",
            value="tab1",
            colors={"border": BORDER, "primary": BLUE, "background": PAGE_BG},
            children=[
                dcc.Tab(label="Forecast vs Actual", value="tab1", style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="Error Heatmap", value="tab2", style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="Promotion Simulator", value="tab3", style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label=f"Forward Forecast ({FUTURE_DAYS} days)", value="tab4", style=TAB_STYLE, selected_style=TAB_SELECTED),
            ],
        ),
        html.Div(id="tab-content", style={"marginTop": "16px"}),
    ],
)


# ------------------------------------------------------------- tab layouts

def store_dropdown(id_, value=1, multi=False, width="220px", options=None):
    return html.Div(
        [
            html.Label(f"Stores (up to {MAX_FORECAST_STORES})" if multi else "Store"),
            dcc.Dropdown(id=id_, options=options or STORE_OPTIONS, value=value, clearable=multi, multi=multi, style={"width": width}),
        ]
    )


def date_picker(id_):
    return html.Div(
        [
            html.Label("Date range"),
            dcc.DatePickerRange(id=id_, min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX, start_date=DATE_MIN, end_date=DATE_MAX),
        ]
    )


def tab1_layout():
    return html.Div(
        [
            html.Div([store_dropdown("t1-store"), date_picker("t1-dates")], className="controls card"),
            html.Div(id="t1-kpis", className="kpi-row", style={"margin": "12px 0"}),
            html.Div(dcc.Graph(id="t1-chart"), className="card"),
            html.Div("One-step-ahead predictions: each day is predicted using the real sales of the days before it.", className="note"),
        ]
    )


def tab2_layout():
    worst = ", ".join(f"Store #{s} ({v:.0f}%)" for s, v in HEATMAP_WORST.items())
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        f"Each row is a store, each column an ISO week of 2015. Stores are sorted by average error — hardest to forecast at the top. "
                        f"Color is capped at {HEATMAP_CLIP}% so the typical range stays readable; hover for exact values.",
                        className="note",
                        style={"margin": "0 0 8px"},
                    ),
                    dcc.Graph(id="t2-heatmap", style={"height": "640px"}),
                ],
                className="card",
            ),
            html.Div(
                f"{HEATMAP_SHARE_UNDER_15:.0f}% of store-weeks are within 15% error; {HEATMAP_SHARE_OVER_CLIP:.1f}% exceed the {HEATMAP_CLIP}% cap. "
                f"Highest average error: {worst}.",
                className="note",
            ),
        ]
    )


def tab3_layout():
    return html.Div(
        [
            html.Div(
                [
                    store_dropdown("t3-store"),
                    date_picker("t3-dates"),
                    html.Div(
                        [
                            html.Label("Scenario"),
                            dcc.RadioItems(
                                id="t3-toggle",
                                options=[{"label": " Promotion on", "value": "on"}, {"label": " Promotion off", "value": "off"}],
                                value="on",
                                inline=True,
                                inputStyle={"marginLeft": "10px"},
                            ),
                        ]
                    ),
                    html.Button("Run scenario", id="t3-run", n_clicks=0, className="primary"),
                ],
                className="controls card",
            ),
            html.Div(id="t3-kpi", style={"fontSize": "18px", "margin": "16px 2px 12px", "fontWeight": "600"}),
            html.Div(id="t3-table"),
            html.Div(dcc.Graph(id="t3-chart"), className="card", style={"marginTop": "12px"}),
            html.Div(
                "What-if only: the promotion is forced on (or off) for every day in the range, and the model's promo-timing features are recomputed to match. "
                "Sales-lag features keep their actual values, so this estimates the direct promo effect, not a full chain reaction.",
                className="note",
            ),
        ]
    )


def tab4_layout():
    return html.Div(
        [
            html.Div(
                [
                    store_dropdown("t4-stores", value=[1], multi=True, width="420px", options=FORWARD_STORE_OPTIONS),
                    dcc.Checklist(
                        id="t4-combine",
                        options=[{"label": "Combined forecast?", "value": "on"}],
                        value=[],
                        style={"paddingBottom": "5px"},
                    ),
                    html.Button("Download CSV", id="t4-download-btn", n_clicks=0, className="secondary"),
                    dcc.Download(id="t4-download"),
                ],
                className="controls card",
            ),
            html.Div(dcc.Graph(id="t4-chart"), className="card", style={"marginTop": "12px"}),
            html.Div(
                f"Recursive forecast for {FUTURE_START:%b %d}–{FUTURE_END:%b %d, %Y} ({FUTURE_DAYS} days — the calendar in Kaggle's test.csv, which covers {len(FORWARD_STORE_IDS)} of {len(STORE_IDS)} stores). "
                "Each day is predicted from the model's own earlier predictions; promotions and holidays come from the known calendar. "
                "Band: ±1.96σ of the log-error on the 2015 holdout (per store, or for the summed total when combined — measured on the actual combined sales, so offsetting store errors are reflected). "
                "It stays the same width across the horizon although recursive errors compound, so treat it as a lower bound on the real uncertainty.",
                className="note",
            ),
        ]
    )


TAB_LAYOUTS = {"tab1": tab1_layout, "tab2": tab2_layout, "tab3": tab3_layout, "tab4": tab4_layout}


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return TAB_LAYOUTS[tab]()


# --------------------------------------------------------------- tab1 cb ---

@app.callback(
    Output("t1-chart", "figure"),
    Output("t1-kpis", "children"),
    Input("t1-store", "value"),
    Input("t1-dates", "start_date"),
    Input("t1-dates", "end_date"),
)
def update_tab1(store_id, start_date, end_date):
    d = predictions_df[
        (predictions_df["Store"] == store_id)
        & (predictions_df["Date"] >= start_date)
        & (predictions_df["Date"] <= end_date)
    ].sort_values("Date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Sales"], mode="lines", name="Actual", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Predicted"], mode="lines", name="Predicted", line=dict(color=ORANGE, width=2)))
    style_fig(fig, f"Store #{store_id}: predicted vs actual daily sales (€)")
    fig.update_yaxes(title="Sales (€)")

    m = mape(d["Sales"].values, d["Predicted"].values)
    r = rmse(d["Sales"].values, d["Predicted"].values)
    w = within_pct(d["Sales"].values, d["Predicted"].values)
    kpis = [kpi("MAPE", f"{m:.1f}%"), kpi("RMSE", f"€{r:,.0f}"), kpi("Within ±25%", f"{w:.1f}%")]
    return fig, kpis


# --------------------------------------------------------------- tab2 cb ---

@app.callback(Output("t2-heatmap", "figure"), Input("tabs", "value"))
def update_tab2(tab):
    fig = go.Figure(
        data=go.Heatmap(
            z=HEATMAP.values,
            x=[int(w) for w in HEATMAP.columns],
            y=[str(s) for s in HEATMAP.index],
            zmin=0,
            zmax=HEATMAP_CLIP,
            colorscale=[[0, "#f6f1e7"], [0.5, "#f2a37c"], [1, "#a3350f"]],
            colorbar=dict(title=dict(text=f"MAPE % (capped at {HEATMAP_CLIP})", font=dict(size=15)), tickfont=dict(size=15), thickness=14),
            hovertemplate="Store #%{y} · week %{x}<br>MAPE %{z:.1f}%<extra></extra>",
            xgap=1,
        )
    )
    style_fig(fig, None)
    fig.update_layout(hovermode="closest", margin=dict(l=60, r=20, t=10, b=64))
    fig.update_xaxes(title="ISO week of 2015", dtick=2, showgrid=False)
    fig.update_yaxes(
        title="Stores — highest average error at top",
        type="category",
        autorange="reversed",
        showticklabels=False,
        showgrid=False,
    )
    return fig


# --------------------------------------------------------------- tab3 cb ---

@app.callback(
    Output("t3-kpi", "children"),
    Output("t3-table", "children"),
    Output("t3-chart", "figure"),
    Input("t3-run", "n_clicks"),
    State("t3-store", "value"),
    State("t3-dates", "start_date"),
    State("t3-dates", "end_date"),
    State("t3-toggle", "value"),
    prevent_initial_call=True,
)
def run_simulator(n_clicks, store_id, start_date, end_date, toggle):
    result = run_promo_scenario(store_id, start_date, end_date, promo_on=(toggle == "on"))
    if result is None:
        return "No data for this selection.", None, go.Figure()

    d, scenario_a, scenario_b = result
    total_a = scenario_a["ScenarioSales"].sum()
    total_b = scenario_b["ScenarioSales"].sum()
    delta = total_b - total_a
    delta_pct = (delta / total_a * 100) if total_a else 0
    sign = "+" if delta >= 0 else "−"
    label = "on" if toggle == "on" else "off"

    kpi_text = (
        f"Estimated effect: {sign}{abs(delta_pct):.1f}% ({sign}€{abs(delta):,.0f} sales) "
        f"if the promotion runs {label.upper()} every day of the period."
    )

    table_df = pd.DataFrame(
        {
            "Scenario": ["A · actual promo calendar", f"B · promo {label} every day"],
            "Total sales (€)": [f"{total_a:,.0f}", f"{total_b:,.0f}"],
            "Change (€)": ["—", f"{delta:+,.0f}"],
            "Change (%)": ["—", f"{delta_pct:+.1f}%"],
        }
    )
    table = dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in table_df.columns],
        style_cell={"textAlign": "left", "padding": "10px 14px", "fontSize": "15px", "backgroundColor": SURFACE, "color": INK, "border": f"1px solid {BORDER}"},
        style_header={"fontSize": "15px", "fontWeight": "600", "backgroundColor": PAGE_BG, "color": INK_2},
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=scenario_a["ScenarioSales"], mode="lines", name="A · actual promo calendar", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=d["Date"], y=scenario_b["ScenarioSales"], mode="lines", name=f"B · promo {label} every day", line=dict(color=ORANGE, width=2)))
    style_fig(fig, f"Store #{store_id}: predicted daily sales under each scenario (€)")
    fig.update_yaxes(title="Predicted sales (€)")

    return kpi_text, table, fig


# --------------------------------------------------------------- tab4 cb ---

def _hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _forecast_view(store_ids, combine):
    """Return [(label, forecast_df, recent_actual_df)] — one entry per store, or a
    single 'Combined' entry summing the selected stores."""
    fc = build_forward_forecast(store_ids)
    since = FUTURE_START - pd.Timedelta(days=42)
    per_store = []
    for s in store_ids:
        d = fc[fc["Store"] == s]
        if d.empty:
            continue
        h = RAW_BY_STORE[s]
        h = h[(h["Open"] == 1) & (h["Date"] > since)][["Date", "Sales"]]
        per_store.append((s, d, h))
    if combine and len(per_store) > 1:
        sids = [s for s, _, _ in per_store]
        recent = pd.concat([h for _, _, h in per_store]).groupby("Date", as_index=False)["Sales"].sum()
        return [(f"Combined ({len(sids)} stores)", combine_forecasts(fc[fc["Store"].isin(sids)], sids), recent)]
    return [(f"Store #{s}", d, h) for s, d, h in per_store]


@app.callback(Output("t4-chart", "figure"), Input("t4-stores", "value"), Input("t4-combine", "value"))
def update_tab4(store_ids, combine):
    store_ids = (store_ids or [])[:MAX_FORECAST_STORES]
    fig = go.Figure()
    if not store_ids:
        style_fig(fig, "Select at least one store")
        return fig

    for i, (label, d, recent) in enumerate(_forecast_view(store_ids, bool(combine))):
        color = SERIES[i % len(SERIES)]
        fig.add_trace(
            go.Scatter(
                x=recent["Date"], y=recent["Sales"], mode="lines", name=f"{label} · recent actual",
                line=dict(color=_hex_rgba(color, 0.45), width=2),
            )
        )
        fig.add_trace(go.Scatter(x=d["Date"], y=d["Upper_PI"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=d["Date"], y=d["Lower_PI"], mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=_hex_rgba(color, 0.14), showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_trace(go.Scatter(x=d["Date"], y=d["Forecast"], mode="lines", name=f"{label} · forecast", line=dict(color=color, width=2)))
    style_fig(fig, f"Forward forecast, {FUTURE_START:%b %d} – {FUTURE_END:%b %d, %Y} (€)")
    fig.update_yaxes(title="Sales (€)")
    return fig


@app.callback(
    Output("t4-download", "data"),
    Input("t4-download-btn", "n_clicks"),
    State("t4-stores", "value"),
    State("t4-combine", "value"),
    prevent_initial_call=True,
)
def download_tab4(n_clicks, store_ids, combine):
    store_ids = (store_ids or [])[:MAX_FORECAST_STORES]
    frames = [d.assign(Store=label)[["Date", "Store", "Forecast", "Upper_PI", "Lower_PI"]]
              for label, d, _ in _forecast_view(store_ids, bool(combine))]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Date", "Store", "Forecast", "Upper_PI", "Lower_PI"])
    out = out.round({"Forecast": 2, "Upper_PI": 2, "Lower_PI": 2})
    return dcc.send_data_frame(out.to_csv, "forward_forecast.csv", index=False)


if __name__ == "__main__":
    app.run(debug=False, port=8050)

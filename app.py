"""
Rossmann Demand Forecasting — Interactive Dash App

Tabs:
1. Forecast vs Actual
2. MAPE Heatmap (Store x Week)
3. Promotion Simulator
4. 8-Week Forward Forecast

Run: python app.py  ->  http://localhost:8050
"""
import pickle
import sys
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
TRAIN_PATH = BASE_DIR / "data/train_processed.csv"
TEST_PATH = BASE_DIR / "data/test.csv"
STORE_PATH = BASE_DIR / "data/store.csv"

# ---------------------------------------------------------------- load ----

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict_sales(X):
    """Model predicts log1p(Sales); invert to real sales scale."""
    return np.expm1(model.predict(X)).clip(0)


predictions_df = pd.read_csv(PRED_PATH, low_memory=False)
predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
predictions_df["Week"] = predictions_df["Date"].dt.isocalendar().week

train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
train_df["Date"] = pd.to_datetime(train_df["Date"])

store_meta = pd.read_csv(STORE_PATH)
store_meta["CompetitionDistance"] = store_meta["CompetitionDistance"].fillna(999999)
store_meta["CompetitionOpenSinceMonth"] = store_meta["CompetitionOpenSinceMonth"].fillna(1)
store_meta["CompetitionOpenSinceYear"] = store_meta["CompetitionOpenSinceYear"].fillna(2000)

test_future = pd.read_csv(TEST_PATH, dtype={"StateHoliday": str}, low_memory=False)
test_future["Date"] = pd.to_datetime(test_future["Date"])
test_future["Open"] = test_future["Open"].fillna(1)

# Holiday calendar for DaysToNextHoliday/DaysSinceLastHoliday: built from the RAW
# train.csv (Open=0 rows included — many holidays coincide with closures, so
# train_processed.csv alone would miss them) unioned with test.csv's known future
# holidays, so Tab 4's forecast window can see holidays just ahead of Jul 31 2015.
_raw_train_full = pd.read_csv(
    BASE_DIR / "data/train.csv",
    dtype={"StateHoliday": str},
    usecols=["Store", "Date", "StateHoliday", "Promo"],
    low_memory=False,
)
_raw_train_full["Date"] = pd.to_datetime(_raw_train_full["Date"])
HOLIDAY_CAL = pd.concat(
    [
        extract_holiday_calendar(_raw_train_full, "Store", "Date", "StateHoliday"),
        extract_holiday_calendar(test_future, "Store", "Date", "StateHoliday"),
    ],
    ignore_index=True,
).drop_duplicates()

# Promo calendar likewise spans history + test.csv's known future promo calendar, so
# Tab 4's DaysToNextPromo can see promos scheduled just ahead of Jul 31 2015.
PROMO_CAL = pd.concat(
    [
        extract_promo_calendar(_raw_train_full, "Store", "Date", "Promo"),
        extract_promo_calendar(test_future, "Store", "Date", "Promo"),
    ],
    ignore_index=True,
).drop_duplicates()

STORE_IDS = sorted(predictions_df["Store"].unique().tolist())
STORE_OPTIONS = [{"label": f"Store {s}", "value": s} for s in STORE_IDS]

DATE_MIN = predictions_df["Date"].min()
DATE_MAX = predictions_df["Date"].max()

# Residual std per store (for forward-forecast PI bands), with global fallback
_resid = predictions_df["Sales"] - predictions_df["Predicted"]
GLOBAL_RESID_STD = float(_resid.std())
STORE_RESID_STD = (
    predictions_df.assign(_resid=_resid).groupby("Store")["_resid"].std().to_dict()
)


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
    return float(
        np.mean(np.abs(pred[mask] - actual[mask]) / actual[mask] <= tol) * 100
    )


# ------------------------------------------------------- tab2 precompute ----

_week_group = predictions_df.groupby(["Store", "Week"])
_heatmap_records = []
for (store, week), g in _week_group:
    _heatmap_records.append({"Store": store, "Week": week, "MAPE": mape(g["Sales"].values, g["Predicted"].values)})
heatmap_df = pd.DataFrame(_heatmap_records)
heatmap_pivot = heatmap_df.pivot(index="Store", columns="Week", values="MAPE")


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


# ---------------------------------------------------- forward forecasting ----

def build_forward_forecast(store_ids, horizon_days=56):
    frames = []
    for store_id in store_ids:
        meta = store_meta[store_meta["Store"] == store_id]
        if meta.empty:
            continue
        meta = meta.iloc[0]

        future = test_future[
            (test_future["Store"] == store_id) & (test_future["Open"] == 1)
        ].sort_values("Date").head(horizon_days).copy()
        if future.empty:
            continue

        recent = train_df[train_df["Store"] == store_id].sort_values("Date").tail(7)
        recent_avg = float(recent["Sales"].mean()) if len(recent) else float(train_df["Sales"].mean())
        last_promo = int(recent["Promo"].iloc[-1]) if len(recent) else 0

        n = len(future)
        feat = pd.DataFrame(index=future.index)
        for i in range(1, 7):
            feat[f"DoW_{i}"] = (future["DayOfWeek"] == i).astype(int).values
        for m in range(1, 12):
            feat[f"Month_{m}"] = (future["Date"].dt.month == m).astype(int).values
        feat["IsStateHoliday"] = (future["StateHoliday"].astype(str) != "0").astype(int).values
        feat["IsSchoolHoliday"] = future["SchoolHoliday"].astype(int).values

        hol_input = pd.DataFrame({"Store": store_id, "Date": future["Date"].values})
        hol_out = add_holiday_distance_features(hol_input, HOLIDAY_CAL, "Store", "Date")
        feat["DaysToNextHoliday"] = hol_out["DaysToNextHoliday"].values
        feat["DaysSinceLastHoliday"] = hol_out["DaysSinceLastHoliday"].values

        feat["Promo_active"] = future["Promo"].astype(int).values
        promo_seq = [last_promo] + future["Promo"].astype(int).tolist()[:-1]
        feat["Promo_lag_1"] = promo_seq

        promo_input = pd.DataFrame({"Store": store_id, "Date": future["Date"].values})
        promo_out = add_promo_distance_features(promo_input, PROMO_CAL, "Store", "Date")
        feat["DaysToNextPromo"] = promo_out["DaysToNextPromo"].values
        feat["DaysSinceLastPromo"] = promo_out["DaysSinceLastPromo"].values

        promo2_input = pd.DataFrame(
            {
                "Date": future["Date"].values,
                "Promo2": meta["Promo2"],
                "Promo2SinceYear": meta["Promo2SinceYear"],
                "Promo2SinceWeek": meta["Promo2SinceWeek"],
                "PromoInterval": meta["PromoInterval"],
            }
        )
        feat["IsPromo2Active"] = add_promo2_active_feature(promo2_input, "Date")["IsPromo2Active"].values

        for col in ["Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_28", "Sales_rolling_7d", "Sales_rolling_30d"]:
            feat[col] = recent_avg
        feat["CompetitionYears"] = np.clip(
            future["Date"].dt.year.values - meta["CompetitionOpenSinceYear"], 0, 100
        )
        feat["CompetitionDistance"] = meta["CompetitionDistance"]
        feat["Promo2"] = meta["Promo2"]
        for t in STORE_TYPES:
            feat[f"StoreType_{t}"] = int(meta["StoreType"] == t)
        for a in ASSORTMENTS:
            feat[f"Assortment_{a}"] = int(meta["Assortment"] == a)

        feat = feat[FEATURE_COLS]
        forecast = predict_sales(feat)

        resid_std = STORE_RESID_STD.get(store_id, GLOBAL_RESID_STD)
        if pd.isna(resid_std):
            resid_std = GLOBAL_RESID_STD

        out = pd.DataFrame(
            {
                "Date": future["Date"].values,
                "Store": store_id,
                "Forecast": forecast,
                "Upper_PI": (forecast + 1.96 * resid_std).clip(0),
                "Lower_PI": (forecast - 1.96 * resid_std).clip(0),
            }
        )
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["Date", "Store", "Forecast", "Upper_PI", "Lower_PI"])
    return pd.concat(frames, ignore_index=True)


# ==================================================================== app ==

app = Dash(__name__)
app.title = "Rossmann Demand Forecasting"

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H2("Rossmann Demand Forecasting"),
        dcc.Tabs(
            id="tabs",
            value="tab1",
            children=[
                dcc.Tab(label="Forecast vs Actual", value="tab1"),
                dcc.Tab(label="MAPE Heatmap", value="tab2"),
                dcc.Tab(label="Promotion Simulator", value="tab3"),
                dcc.Tab(label="8-Week Forward Forecast", value="tab4"),
            ],
        ),
        html.Div(id="tab-content", style={"marginTop": "20px"}),
    ],
)


# ------------------------------------------------------------- tab layouts

def tab1_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Store"),
                    dcc.Dropdown(id="t1-store", options=STORE_OPTIONS, value=1, clearable=False, style={"width": "200px"}),
                    html.Label("Date range", style={"marginLeft": "20px"}),
                    dcc.DatePickerRange(
                        id="t1-dates",
                        min_date_allowed=DATE_MIN,
                        max_date_allowed=DATE_MAX,
                        start_date=DATE_MIN,
                        end_date=DATE_MAX,
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
            ),
            html.Div(id="t1-kpis", style={"display": "flex", "gap": "30px", "margin": "20px 0", "fontSize": "18px"}),
            dcc.Graph(id="t1-chart"),
        ]
    )


def tab2_layout():
    return html.Div(
        [
            html.P("MAPE (%) by store and ISO week — green (low error) to red (high error)."),
            dcc.Graph(id="t2-heatmap", style={"height": "800px"}),
        ]
    )


def tab3_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Store"),
                    dcc.Dropdown(id="t3-store", options=STORE_OPTIONS, value=1, clearable=False, style={"width": "200px"}),
                    html.Label("Date range", style={"marginLeft": "20px"}),
                    dcc.DatePickerRange(
                        id="t3-dates",
                        min_date_allowed=DATE_MIN,
                        max_date_allowed=DATE_MAX,
                        start_date=DATE_MIN,
                        end_date=DATE_MAX,
                    ),
                    dcc.RadioItems(
                        id="t3-toggle",
                        options=[{"label": "Promotion On", "value": "on"}, {"label": "Promotion Off", "value": "off"}],
                        value="on",
                        style={"marginLeft": "20px"},
                        inline=True,
                    ),
                    html.Button("Run Scenario", id="t3-run", n_clicks=0, style={"marginLeft": "20px"}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "10px", "flexWrap": "wrap"},
            ),
            html.Div(id="t3-kpi", style={"fontSize": "18px", "margin": "20px 0"}),
            html.Div(id="t3-table"),
            dcc.Graph(id="t3-chart"),
        ]
    )


def tab4_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Store(s)"),
                    dcc.Dropdown(id="t4-stores", options=STORE_OPTIONS, value=[1], multi=True, style={"width": "400px"}),
                    html.Button("Download CSV", id="t4-download-btn", n_clicks=0, style={"marginLeft": "20px"}),
                    dcc.Download(id="t4-download"),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
            ),
            dcc.Graph(id="t4-chart"),
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
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Sales"], mode="lines", name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Predicted"], mode="lines", name="Predicted", line=dict(color="green")))
    fig.update_layout(title=f"Store {store_id}: Forecast vs Actual", template="plotly_white", xaxis_title="Date", yaxis_title="Sales")

    m = mape(d["Sales"].values, d["Predicted"].values)
    r = rmse(d["Sales"].values, d["Predicted"].values)
    w = within_pct(d["Sales"].values, d["Predicted"].values)

    kpis = [
        html.Div([html.B("MAPE: "), f"{m:.1f}%"]),
        html.Div([html.B("RMSE: "), f"{r:.0f}"]),
        html.Div([html.B("% within ±25%: "), f"{w:.1f}%"]),
    ]
    return fig, kpis


# --------------------------------------------------------------- tab2 cb ---

@app.callback(Output("t2-heatmap", "figure"), Input("tabs", "value"))
def update_tab2(tab):
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale="RdYlGn_r",
            colorbar=dict(title="MAPE %"),
            hovertemplate="Store %{y}<br>Week %{x}<br>MAPE %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(title="MAPE by Store and Week", xaxis_title="ISO Week", yaxis_title="Store", template="plotly_white")
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

    kpi = html.Div(
        f"Estimated lift: {'+' if delta >= 0 else ''}{delta_pct:.1f}% "
        f"({'+' if delta >= 0 else ''}${delta:,.0f} revenue) if promotion is turned "
        f"{'ON' if toggle == 'on' else 'OFF'} for the full period."
    )

    table_df = pd.DataFrame(
        {
            "Scenario": ["A (actual promo)", f"B (promo {'on' if toggle=='on' else 'off'})"],
            "Total Sales": [round(total_a, 0), round(total_b, 0)],
            "Delta (units)": [0, round(delta, 0)],
            "Delta (%)": [0, round(delta_pct, 2)],
        }
    )
    table = dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in table_df.columns],
        style_cell={"textAlign": "center", "padding": "6px"},
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=scenario_a["ScenarioSales"], mode="lines", name="Scenario A (actual)", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=d["Date"], y=scenario_b["ScenarioSales"], mode="lines", name=f"Scenario B (promo {'on' if toggle=='on' else 'off'})", line=dict(color="orange")))
    fig.update_layout(title=f"Store {store_id}: Promotion Scenario Comparison", template="plotly_white", xaxis_title="Date", yaxis_title="Predicted Sales")

    return kpi, table, fig


# --------------------------------------------------------------- tab4 cb ---

@app.callback(Output("t4-chart", "figure"), Input("t4-stores", "value"))
def update_tab4(store_ids):
    if not store_ids:
        return go.Figure()
    fc = build_forward_forecast(store_ids)
    fig = go.Figure()
    for store_id in store_ids:
        d = fc[fc["Store"] == store_id]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(x=d["Date"], y=d["Upper_PI"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=d["Date"], y=d["Lower_PI"], mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(0,100,255,0.15)", showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_trace(go.Scatter(x=d["Date"], y=d["Forecast"], mode="lines", name=f"Store {store_id} forecast"))
    fig.update_layout(title="8-Week Forward Forecast (±95% PI)", template="plotly_white", xaxis_title="Date", yaxis_title="Forecast Sales")
    return fig


@app.callback(
    Output("t4-download", "data"),
    Input("t4-download-btn", "n_clicks"),
    State("t4-stores", "value"),
    prevent_initial_call=True,
)
def download_tab4(n_clicks, store_ids):
    fc = build_forward_forecast(store_ids or [])
    return dcc.send_data_frame(fc.to_csv, "forward_forecast.csv", index=False)


if __name__ == "__main__":
    app.run(debug=False, port=8050)

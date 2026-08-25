"""
Day 1: Data loading, merging, and exploratory analysis for Rossmann demand forecasting.

Outputs:
- data/train_processed.csv
- eda_summary.json
- store1_timeseries.html
- store1_acf_pacf.html
"""
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf


def load_and_merge_data(train_path, test_path, store_path):
    train = pd.read_csv(train_path, dtype={"StateHoliday": str}, low_memory=False)
    test = pd.read_csv(test_path, dtype={"StateHoliday": str}, low_memory=False)
    store = pd.read_csv(store_path)

    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    df_train = train.merge(store, on="Store", how="left")
    df_test = test.merge(store, on="Store", how="left")

    rows_before = len(df_train)
    df_train_clean = df_train[df_train["Open"] == 1].copy()
    rows_open0_removed = rows_before - len(df_train_clean)

    for df in (df_train_clean, df_test):
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(999999)
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(1)
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(2000)

    df_train_clean = df_train_clean.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train_clean.attrs["rows_open0_removed"] = rows_open0_removed
    return df_train_clean, df_test


def exploratory_analysis(df):
    promo1_mean = df.loc[df["Promo"] == 1, "Sales"].mean()
    promo0_mean = df.loc[df["Promo"] == 0, "Sales"].mean()
    promo_lift_pct = (promo1_mean - promo0_mean) / promo0_mean * 100

    mean_sales_by_dow = df.groupby("DayOfWeek")["Sales"].mean().round(2).to_dict()
    mean_sales_by_holiday = df.groupby("StateHoliday")["Sales"].mean().round(2).to_dict()

    sales_quantiles = df["Sales"].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(2).to_dict()

    eda = {
        "summary_stats": df[["Sales", "Customers", "CompetitionDistance"]].describe().round(2).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "sales_quantiles": {str(k): v for k, v in sales_quantiles.items()},
        "promo_lift_pct": round(promo_lift_pct, 2),
        "mean_sales_by_dow": {str(k): v for k, v in mean_sales_by_dow.items()},
        "mean_sales_by_holiday": {str(k): v for k, v in mean_sales_by_holiday.items()},
        "rows_open0_removed": int(df.attrs.get("rows_open0_removed", 0)),
        "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "store_count": int(df["Store"].nunique()),
    }
    return eda


def plot_store_timeseries(df, store_id=1, output_path="store1_timeseries.html"):
    d = df[df["Store"] == store_id].sort_values("Date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Sales"], mode="lines", name="Sales"))
    fig.update_layout(
        title=f"Store {store_id} Daily Sales (Training Period)",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
    )
    fig.write_html(output_path)
    return output_path


def plot_acf_pacf(df, store_id=1, lags=40, output_path="store1_acf_pacf.html"):
    d = df[df["Store"] == store_id].sort_values("Date")
    sales = d["Sales"].reset_index(drop=True)

    acf_vals = acf(sales, nlags=lags)
    pacf_vals = pacf(sales, nlags=lags)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF"), row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF"), row=1, col=2)
    fig.update_layout(
        title=f"Store {store_id} ACF / PACF (lags={lags})",
        template="plotly_white",
        showlegend=False,
    )
    fig.write_html(output_path)
    return output_path


def export_eda_summary(eda_dict, json_path="eda_summary.json"):
    with open(json_path, "w") as f:
        json.dump(eda_dict, f, indent=2, default=str)
    return json_path


if __name__ == "__main__":
    DATA_DIR = "data"

    print("Loading and merging data...")
    df_train_clean, df_test = load_and_merge_data(
        f"{DATA_DIR}/train.csv", f"{DATA_DIR}/test.csv", f"{DATA_DIR}/store.csv"
    )
    print(f"df_train_clean: {df_train_clean.shape}, df_test: {df_test.shape}")

    train_out_path = f"{DATA_DIR}/train_processed.csv"
    df_train_clean.to_csv(train_out_path, index=False)
    print(f"Saved {train_out_path}")

    print("Running exploratory analysis...")
    eda = exploratory_analysis(df_train_clean)
    export_eda_summary(eda, "eda_summary.json")
    print("Saved eda_summary.json")
    print(f"Promo lift: {eda['promo_lift_pct']}%")

    print("Plotting store 1 timeseries...")
    plot_store_timeseries(df_train_clean, store_id=1, output_path="store1_timeseries.html")

    print("Plotting store 1 ACF/PACF...")
    plot_acf_pacf(df_train_clean, store_id=1, lags=40, output_path="store1_acf_pacf.html")

    print("Day 1 complete.")

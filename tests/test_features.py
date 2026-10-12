"""Feature-engineering tests on small synthetic data (no Kaggle download needed).

What they pin down: lags are exact calendar-day offsets even across closed days and missing rows,
no feature for a date uses sales from that date or later, rolling windows span calendar days, and
the event-distance and Promo2 features behave as documented.
"""
import numpy as np
import pandas as pd
import pytest

from day2_model import engineer_features
from features import FEATURE_COLS, add_promo2_active_feature, add_promo_distance_features, extract_promo_calendar

SALES_FEATURES = ["Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_28", "Sales_rolling_7d", "Sales_rolling_30d"]


def make_history(stores=(1, 2), start="2014-01-01", days=90, seed=0):
    """Full daily history, closed days (Sundays) included with Sales = 0, like the Kaggle data."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in stores:
        d = pd.DataFrame({"Date": pd.date_range(start, periods=days, freq="D")})
        d["Store"] = s
        d["DayOfWeek"] = d["Date"].dt.dayofweek + 1
        d["Open"] = (d["DayOfWeek"] != 7).astype(int)
        d["Sales"] = np.where(d["Open"] == 1, rng.integers(1000, 9000, days), 0)
        d["Promo"] = ((d["Date"].dt.dayofweek < 5) & (d["Date"].dt.isocalendar().week % 2 == 0)).astype(int)
        d["StateHoliday"] = "0"
        d["SchoolHoliday"] = 0
        d["StoreType"], d["Assortment"] = "a", "a"
        d["CompetitionDistance"] = 500.0
        d["CompetitionOpenSinceMonth"], d["CompetitionOpenSinceYear"] = 1, 2010
        d["Promo2"], d["Promo2SinceWeek"], d["Promo2SinceYear"], d["PromoInterval"] = 0, np.nan, np.nan, np.nan
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def features_at(feats, store, date):
    row = feats[(feats.Store == store) & (feats.Date == pd.Timestamp(date))]
    assert len(row) == 1, f"expected one open-day row for store {store} on {date}"
    return row.iloc[0]


def test_lags_are_exact_calendar_day_offsets():
    df = make_history()
    feats = engineer_features(df)
    sales = df.set_index(["Store", "Date"])["Sales"]
    day = pd.Timestamp("2014-03-05")  # a Wednesday, well after the start of the data
    row = features_at(feats, 1, day)
    for lag in (1, 7, 14, 28):
        assert row[f"Sales_lag_{lag}"] == sales[(1, day - pd.Timedelta(days=lag))]


def test_lag_after_a_closed_day_sees_the_zero_not_the_previous_open_day():
    feats = engineer_features(make_history())
    monday = pd.Timestamp("2014-03-03")
    assert features_at(feats, 1, monday)["Sales_lag_1"] == 0  # Sunday: closed, zero sales


def test_a_missing_row_gives_zero_not_a_neighbouring_row():
    df = make_history()
    hole = pd.Timestamp("2014-02-12")  # Wednesday
    other = df[(df.Store == 2) & (df.Date == hole)]["Sales"].iloc[0]
    df = df[~((df.Store == 1) & (df.Date == hole))]
    feats = engineer_features(df)
    assert features_at(feats, 1, hole + pd.Timedelta(days=7))["Sales_lag_7"] == 0
    # the other store, same dates, is unaffected
    assert features_at(feats, 2, hole + pd.Timedelta(days=7))["Sales_lag_7"] == other


def test_rolling_mean_spans_calendar_days_including_closed_days():
    df = make_history()
    feats = engineer_features(df)
    day = pd.Timestamp("2014-03-05")
    window = df[(df.Store == 1) & (df.Date >= day - pd.Timedelta(days=7)) & (df.Date < day)]
    assert len(window) == 7  # includes the closed Sunday
    assert features_at(feats, 1, day)["Sales_rolling_7d"] == pytest.approx(window["Sales"].mean())


def test_no_feature_uses_sales_from_the_row_date_or_later():
    base = make_history()
    cutoff = pd.Timestamp("2014-02-20")
    changed = base.copy()
    changed.loc[changed.Date >= cutoff, "Sales"] *= 10  # rewrite the "future"

    key = ["Store", "Date"]
    a = engineer_features(base).sort_values(key).reset_index(drop=True)
    b = engineer_features(changed).sort_values(key).reset_index(drop=True)
    upto = a.Date <= cutoff
    pd.testing.assert_frame_equal(a.loc[upto, SALES_FEATURES], b.loc[upto, SALES_FEATURES])
    # sanity: the perturbation is visible the day after, so the check above can fail
    nxt = a.Date == cutoff + pd.Timedelta(days=1)
    assert not a.loc[nxt, "Sales_lag_1"].equals(b.loc[nxt, "Sales_lag_1"])


def test_output_has_open_days_only_and_all_model_features():
    feats = engineer_features(make_history())
    assert (feats["Open"] == 1).all()
    assert not feats.Date.dt.dayofweek.eq(6).any()  # Sundays are closed in the synthetic data
    missing = [c for c in FEATURE_COLS if c not in feats.columns]
    assert not missing
    assert not feats[FEATURE_COLS].isna().any().any()


def test_feature_list_contract():
    assert len(FEATURE_COLS) == 42 and len(set(FEATURE_COLS)) == 42
    for banned in ("Store", "Customers", "Sales", "DaysSinceStart"):
        assert banned not in FEATURE_COLS  # no ID, no same-day target proxy, no extrapolated counter


def test_promo_distance_features_and_clipping():
    dates = pd.date_range("2014-01-01", periods=40, freq="D")
    df = pd.DataFrame({"Store": 1, "Date": dates, "Promo": 0})
    df.loc[[9, 19], "Promo"] = 1  # promos on day index 9 and 19
    out = add_promo_distance_features(df, extract_promo_calendar(df), max_days=30).set_index("Date")

    def at(i):
        r = out.loc[dates[i]]
        return r.DaysToNextPromo, r.DaysSinceLastPromo

    assert at(9) == (0, 0)
    assert at(14) == (5, 5)
    assert at(0) == (9, 30)   # nothing earlier: clipped
    assert at(39) == (30, 20)  # nothing later: clipped


def test_promo_distance_is_per_store():
    dates = pd.date_range("2014-01-01", periods=10, freq="D")
    df = pd.concat([pd.DataFrame({"Store": s, "Date": dates, "Promo": 0}) for s in (1, 2)], ignore_index=True)
    df.loc[(df.Store == 1) & (df.Date == dates[3]), "Promo"] = 1  # only store 1 has a promo
    out = add_promo_distance_features(df, extract_promo_calendar(df), max_days=30)
    s2 = out[out.Store == 2]
    assert (s2.DaysToNextPromo == 30).all() and (s2.DaysSinceLastPromo == 30).all()


def test_promo2_active_handles_sept_spelling_and_participation():
    base = {"Promo2SinceYear": 2013, "Promo2SinceWeek": 1, "PromoInterval": "Mar,Jun,Sept,Dec"}
    rows = [
        {**base, "Promo2": 1, "Date": "2014-09-15"},  # interval month; the data spells it "Sept"
        {**base, "Promo2": 1, "Date": "2014-10-15"},  # not an interval month
        {**base, "Promo2": 0, "Date": "2014-09-15"},  # does not participate
        {**base, "Promo2": 1, "Date": "2012-09-15"},  # before Promo2 began (2013, week 1)
    ]
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    assert add_promo2_active_feature(df)["IsPromo2Active"].tolist() == [1, 0, 0, 0]

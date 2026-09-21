"""
Shared feature definitions and event-distance helpers.

Kept free of heavy dependencies (no pmdarima / plotly / statsmodels) so the
deployed dashboard can import them without the training stack installed.
"""
import numpy as np
import pandas as pd

STORE_TYPES = ["a", "b", "c", "d"]
ASSORTMENTS = ["a", "b", "c"]

HOLIDAY_MAX_DAYS = 30

FEATURE_COLS = (
    [f"DoW_{i}" for i in range(1, 7)]
    + [f"Month_{i}" for i in range(1, 12)]
    + [
        "IsStateHoliday",
        "IsSchoolHoliday",
        "DaysToNextHoliday",
        "DaysSinceLastHoliday",
        "Promo_active",
        "Promo_lag_1",
        "DaysToNextPromo",
        "DaysSinceLastPromo",
        "IsPromo2Active",
        "Sales_lag_1",
        "Sales_lag_7",
        "Sales_lag_14",
        "Sales_lag_28",
        "Sales_rolling_7d",
        "Sales_rolling_30d",
        "CompetitionYears",
        "CompetitionDistance",
        "Promo2",
    ]
    + [f"StoreType_{t}" for t in STORE_TYPES]
    + [f"Assortment_{a}" for a in ASSORTMENTS]
)


def extract_event_calendar(df, event_mask, store_col="Store", date_col="Date"):
    """(Store, Date) pairs where event_mask is True, deduplicated. Computed per store
    (rather than off one shared calendar) since both state holidays and promo timing
    vary store-to-store in this dataset."""
    return df.loc[event_mask, [store_col, date_col]].drop_duplicates()


def add_event_distance_features(df, event_cal, to_col, since_col, store_col="Store", date_col="Date", max_days=HOLIDAY_MAX_DAYS):
    """Generic days-to-next-event / days-since-last-event pair, per store, clipped to
    max_days. Both are 0 on the event date itself. Vectorized per store via
    searchsorted against that store's sorted event dates."""
    df = df.sort_values([store_col, date_col]).reset_index(drop=True)
    to_next = np.full(len(df), max_days, dtype=float)
    since_last = np.full(len(df), max_days, dtype=float)

    event_by_store = event_cal.groupby(store_col)[date_col].apply(lambda s: np.sort(s.values))

    for store_id, idx in df.groupby(store_col).groups.items():
        edays = event_by_store.get(store_id)
        if edays is None or len(edays) == 0:
            continue
        pos = df.index.get_indexer(idx)
        dates = df.loc[idx, date_col].values

        next_pos = np.searchsorted(edays, dates, side="left")
        has_next = next_pos < len(edays)
        next_days = np.full(len(dates), max_days, dtype=float)
        next_days[has_next] = (
            (edays[next_pos[has_next]] - dates[has_next]) / np.timedelta64(1, "D")
        )
        to_next[pos] = np.minimum(next_days, max_days)

        prev_pos = np.searchsorted(edays, dates, side="right") - 1
        has_prev = prev_pos >= 0
        prev_days = np.full(len(dates), max_days, dtype=float)
        prev_days[has_prev] = (
            (dates[has_prev] - edays[prev_pos[has_prev]]) / np.timedelta64(1, "D")
        )
        since_last[pos] = np.minimum(prev_days, max_days)

    df[to_col] = to_next
    df[since_col] = since_last
    return df


def extract_holiday_calendar(df, store_col="Store", date_col="Date", holiday_col="StateHoliday"):
    return extract_event_calendar(df, df[holiday_col].astype(str) != "0", store_col, date_col)


def add_holiday_distance_features(df, holiday_cal, store_col="Store", date_col="Date", max_days=HOLIDAY_MAX_DAYS):
    return add_event_distance_features(
        df, holiday_cal, "DaysToNextHoliday", "DaysSinceLastHoliday", store_col, date_col, max_days
    )


def extract_promo_calendar(df, store_col="Store", date_col="Date", promo_col="Promo"):
    return extract_event_calendar(df, df[promo_col] == 1, store_col, date_col)


def add_promo_distance_features(df, promo_cal, store_col="Store", date_col="Date", max_days=HOLIDAY_MAX_DAYS):
    return add_event_distance_features(
        df, promo_cal, "DaysToNextPromo", "DaysSinceLastPromo", store_col, date_col, max_days
    )


_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def add_promo2_active_feature(df, date_col="Date"):
    """IsPromo2Active: whether the recurring Promo2 program is running THIS MONTH for
    this store — distinct from the static `Promo2` participation flag. Promo2 renews
    every few months per `PromoInterval` (e.g. "Jan,Apr,Jul,Oct") starting from the
    ISO week/year in Promo2SinceWeek/Year."""
    df = df.copy()
    since_year = df["Promo2SinceYear"].fillna(2000).astype(int).astype(str)
    since_week = df["Promo2SinceWeek"].fillna(1).astype(int).astype(str).str.zfill(2)
    promo2_since = pd.to_datetime(since_year + since_week + "1", format="%Y%W%w", errors="coerce")

    month_abbr = df[date_col].dt.month.map(_MONTH_ABBR)
    # The source data spells September "Sept"; normalize so it matches _MONTH_ABBR.
    interval_lists = df["PromoInterval"].fillna("").str.replace("Sept", "Sep").str.split(",")
    in_interval = pd.Series(
        [abbr in lst for abbr, lst in zip(month_abbr, interval_lists)], index=df.index
    )

    df["IsPromo2Active"] = (
        (df["Promo2"] == 1) & (df[date_col] >= promo2_since) & in_interval
    ).astype(int)
    return df

# Methodology

How the Rossmann demand-forecasting model is built, validated and deployed, and where it falls short. The
[README](../README.md) has the summary. Every figure below comes from a file in this repository, named where it
appears.

**Terms.** MAPE (mean absolute percentage error) is the average of |prediction − actual| / actual, in percent; a
MAPE of 9% means the typical forecast is 9% away from actual sales. RMSE (root mean squared error) and MAE (mean
absolute error) are errors in euros. CV is cross-validation. LightGBM is a gradient-boosted decision-tree library.
SHAP (SHapley Additive exPlanations) measures how much each feature moves a prediction. ARIMA (AutoRegressive
Integrated Moving Average) is a classical single-series forecasting model. A *lag* is a past value: `Sales_lag_7` is
sales seven days earlier. Euros are the currency of the Kaggle data.

## 1. Data and cleaning

- **Source:** Kaggle *Rossmann Store Sales* (`train.csv`, `store.csv`, `test.csv`), provided by Dirk Rossmann GmbH.
  `train.csv` has 1,017,209 store-days for 1,115 stores, 1 Jan 2013 to 31 Jul 2015.
- **Modeling rows:** the 844,392 store-days on which the store was open. Closed days stay in the history that the
  lag features read from, with sales of 0. Of the open days, 54 have zero sales (3 in the 2015 holdout); the error
  metrics skip zero-sales days because a percentage error is undefined there.
- **Missing history:** 180 of the 1,115 stores have no rows at all for 1 Jul to 31 Dec 2014 (184 days); one more
  store has no row for 1 Jan 2013. The data does not record why. Features that look back into the gap read it as
  zero sales, and the two last validation folds cannot score those stores. In the 2015 holdout those 180 stores
  score 9.45% MAPE against 9.09% for the rest (January: 12.5% against 11.5%; computed from
  `deploy_data/predictions.parquet`).
- **Missing values:** `CompetitionDistance` is filled with 999,999 (effectively "no nearby competitor"), and a
  missing competition-open month or year is filled with January and 2000.
- **Exploration files:** `eda.py` writes `eda_summary.json`, a Store 1 time-series chart and `store1_acf_pacf.html`
  (autocorrelation and partial autocorrelation on the daily calendar; Store 1 peaks at lags 7, 14, 21 and 28, which
  is weekly seasonality).

## 2. Validation design

- **Walk-forward cross-validation:** four expanding-window folds over 2014. Each trains on everything before its
  validation quarter (fold 1: 2013 only; fold 4: 2013 through September 2014) and scores that quarter. Results are in
  `cv_results.csv`.

| Fold | Validation period | MAPE | RMSE (€) | MAE (€) |
|---|---|---|---|---|
| 1 | Q1 2014 | 10.67% | 1,055.3 | 670.2 |
| 2 | Q2 2014 | 9.51% | 978.2 | 661.4 |
| 3 | Q3 2014 | 8.88% | 900.6 | 595.3 |
| 4 | Q4 2014 | 11.09% | 1,293.4 | 805.1 |
| **Mean** | | **10.04%** | **1,056.9** | **683.0** |

- **Holdout:** a final model trained on 2013–2014 (648,360 rows) is scored on January to July 2015 (196,032 rows;
  `metrics.json`): MAPE 9.15%, RMSE €912.5, 95.9% of forecasts within ±25% of actual. By month the holdout MAPE is
  highest in January (11.6%) and February (10.2%) and lowest in March (7.3%); April to July are 8.4–8.9%.
- **The holdout is not a clean one-shot test.** The model never trained on it, but its score was checked after most
  feature changes while the model was being developed, so it influenced which changes were kept. The 9.15% is
  therefore mildly optimistic, by an amount that cannot be measured. The 2014 CV folds also guided those choices.
  No unused labelled data remains, because the data ends on 31 Jul 2015. I treat the CV mean as the primary
  estimate and the holdout as a second, similar-sized estimate.
- **Run-to-run noise:** retraining the final model with a different random seed or column order moves the CV mean by
  about ±0.1 points and the holdout MAPE by about 0.03 (section 4). Differences smaller than that are not evidence.

## 3. Features

One model serves all stores. It predicts `log1p(Sales)` and the result is inverted with `expm1` and clipped at zero.
Without the log transform, large stores dominate the squared-error loss and the model under-weights percentage error
on small stores. There are 42 inputs (`features.py: FEATURE_COLS`), and no store ID and no same-day `Customers`.

| Group | Features | Notes |
|---|---|---|
| Calendar | day of week (6 dummies), month (11 dummies) | |
| Holidays | `IsStateHoliday`, `IsSchoolHoliday`, `DaysToNextHoliday`, `DaysSinceLastHoliday` | Distances are per store, because state holidays are regional. Both are 0 on the holiday itself and are capped at 30 days. |
| Promotions | `Promo_active`, `Promo_lag_1`, `DaysToNextPromo`, `DaysSinceLastPromo`, `Promo2`, `IsPromo2Active` | Distances use the same helper as the holiday distances. `IsPromo2Active` is whether the recurring Promo2 programme runs in this calendar month for this store, derived from `Promo2SinceWeek/Year` and `PromoInterval`. |
| Recent sales | `Sales_lag_1`, `_7`, `_14`, `_28`; `Sales_rolling_7d`, `_30d` | Calendar-day offsets (below). Rolling means are shifted one day so a day's own sales are never included. |
| Store | `StoreType_*`, `Assortment_*`, `CompetitionDistance`, `CompetitionYears` | |

**Calendar-day lags.** A lag is the value exactly N calendar days earlier, found by a date-shifted merge on
`(Store, Date − N days)`. A plain row shift within a store's open days drifts off the true date wherever days are
missing, and treats the day after a Sunday closure as if Saturday were yesterday. Rolling means are computed on a
per-store series reindexed to every calendar day, so a 7-day window always spans seven days.

**Excluded: `DaysSinceStart`.** The linear days-since-start counter is computed in `engineer_features` but not used
by the model. Every validation period lies beyond the training dates, and a tree model can only reuse its
right-most leaf for dates it has not seen. Section 4 measures the effect.

**Model settings** (`train_model.py: LGBM_PARAMS`): 500 trees, 31 leaves, max depth 7, learning rate 0.05, L1 0.1,
L2 1.0, `colsample_bytree` 0.8, `min_child_samples` 20, seed 42. They come from the project brief and were not
tuned; there is no early stopping. `subsample=0.8` has no effect, because bagging is off (`bagging_freq` is 0).

## 4. What was tested and dropped

`analysis/ablations.py` retrains the model with one change at a time, using the same folds and settings as the
pipeline. Its final row reproduces `cv_results.csv` and `metrics.json` exactly. Results are in
`analysis/ablations.csv`.

**Leave one out** (the final model with one change removed; MAPE, lower is better):

| Change removed | CV mean | Holdout |
|---|---|---|
| none (final model) | 10.04% | 9.15% |
| `DaysSinceStart` added back | 12.58% | 11.41% |
| calendar-day lags replaced by row-shift lags | 11.45% | 10.31% |
| holiday distance features | 10.96% | 9.55% |
| promo distance features and `IsPromo2Active` | 10.62% | 9.12% |
| log target (raw sales) | 10.54% | 9.39% |
| store format (`StoreType`, `Assortment`) | 10.18% | 9.21% |
| *noise: final model, other seeds or column order* | *9.98–10.14%* | *9.15–9.18%* |

**Ladder** (start from row-shift lags, raw target and `DaysSinceStart`, then add one change at a time; this is an
ordering chosen for the table, not a replay of how development went):

| Step | CV mean | Holdout |
|---|---|---|
| 0 start | 23.68% | 16.83% |
| 1 + log target | 19.32% | 18.94% |
| 2 + store format | 18.27% | 17.92% |
| 3 + calendar-day lags and rolling means | 16.88% | 13.72% |
| 4 + holiday distance | 14.08% | 14.19% |
| 5 + promo distance and `IsPromo2Active` | 12.58% | 11.41% |
| 6 − `DaysSinceStart` (final) | 10.04% | 9.15% |

How to read them:
- **`DaysSinceStart` is the largest single effect.** Adding it back costs 2.5 points of CV and 2.3 of holdout. On
  Fold 3 (Jul–Sep 2014) it makes the model over-predict by 9.3% on average, against 1.5% without it, and Fold 3 MAPE
  goes from 8.88% to 12.41% (`analysis/ablations_fold3_bias.csv`).
- **Calendar-day lags and holiday distance** each help on every measure (about 1.4 and 0.9 CV points).
- **The log target** helps a little in the final model (0.5 CV, 0.24 holdout), but in the ladder it improves CV and
  worsens the holdout, so the evidence is mixed.
- **Promo distance and `IsPromo2Active`** help CV by about 0.6 points but do not change the holdout (9.12% without
  them, inside the noise). I cannot show that they help on the holdout.
- **Store format** is at the edge of the noise: +0.14 CV, +0.06 holdout when removed.
- Each effect depends on what else is in the model; the ladder and leave-one-out views are two partial pictures, not
  a decomposition of 23.7% into additive pieces.

## 5. Baselines

| Predictor | All stores | Store 1 |
|---|---|---|
| Store × weekday × promo average of 2013–14 sales (`static_lookup_baseline`) | 11.32% | 12.48% |
| LightGBM, one-step-ahead | 9.15% | 9.0% |
| ARIMA(0,0,5), Store 1 only | n/a | 17.15% |

The lookup uses no recent sales and no model. It is the bar to clear: LightGBM beats it by about 2 points, a 19% cut
in error (scored on the 196,029 holdout rows with positive sales). The ARIMA is a weak reference. It is
non-seasonal, so its forecast is constant after five days (170 of its 175 forecast points are identical); it is fit
on one store and forecasts the whole of 2015 with no new information, while LightGBM is given the real previous
days' sales. `arima_baseline.html` shows it. The ARIMA series is built from open days only, not calendar days.

## 6. Forecast procedure

- **One-step-ahead** (the headline MAPE, Tabs 1 and 2): each day is predicted knowing the real sales of the days
  before it.
- **Recursive** (Tab 4): each open day is predicted, the prediction is written into a per-date sales buffer, and later
  days' lag and rolling features are computed from those predictions. Promotions and holidays come from the known
  calendar in `test.csv`, which covers 856 of the 1,115 stores.
- **Backtest** (`analysis/backtest_recursive.py`, results in `analysis/backtest_results.json`): starting 1 Jul 2015
  with no later real sales, for 150 randomly chosen stores (seed 0; 4,066 store-days):

| | MAPE |
|---|---|
| LightGBM, recursive | 10.44% |
| LightGBM, one-step, same days | 8.89% |
| Store × weekday × promo lookup | 11.43% |

  By 7-day bucket the recursive error is 10.8 / 9.2 / 9.8 / 11.4%, so it does not blow up over the month. The mean
  signed error is +3.6% (the forecast runs slightly high). This is one month and a 150-store sample, not the full
  holdout.
- **Bands:** ±1.96 standard deviations of the log error on the holdout, constant across the horizon (for a combined
  total of several stores, measured on the actual summed sales). 93.0% of actuals fall inside them in the July
  backtest, but that month is part of the holdout the width was fitted on, so the coverage is not independent. Treat
  the bands as a lower bound on the true uncertainty.
- **Promotion Simulator** (Tab 3): to switch promotions on or off over a date range, `Promo_active`, `Promo_lag_1`,
  `DaysToNextPromo` and `DaysSinceLastPromo` are recomputed together, so the model never sees a contradictory
  combination. The sales-lag features keep their actual values.

## 7. Feature importance (SHAP)

Mean |SHAP value| on the log-sales scale (`shap_feature_importance.csv`, `shap_analysis.py`):

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `Sales_lag_14` | 0.098 |
| 2 | `DaysToNextPromo` | 0.066 |
| 3 | `Sales_rolling_7d` | 0.062 |
| 4 | `Sales_rolling_30d` | 0.054 |
| 5 | `Sales_lag_1` | 0.048 |
| 6 | `Sales_lag_28` | 0.045 |
| 7 | `Promo_active` | 0.038 |
| 8 | `Sales_lag_7` | 0.034 |
| 9 | `DaysSinceLastHoliday` | 0.024 |
| 10 | `DaysSinceLastPromo` | 0.022 |

The recent-sales features and the promo-timing features carry most of the signal. A continuous "days until the next
promotion" ranks above the plain same-day promo flag. SHAP shows how much a feature moves predictions, not whether
the movement helps accuracy; the ablations in section 4 are the check on that.

## 8. Dashboard and deployment

`app.py` is a Plotly Dash app with four tabs. It reads `deploy_data/` (about 9 MB: full daily history, the 2015
holdout with features and predictions, the Aug–Sep 2015 calendar, store attributes, the model as `model.txt`, and
`metrics.json`) and falls back to the full local files if `deploy_data/` is absent or `ROSSMANN_USE_LOCAL_DATA=1`.
`build_deploy_data.py` regenerates the bundle.

`render.yaml` defines a free-tier Render web service on Python 3.14.6 with the pinned `requirements-render.txt`
(runtime packages only, no training stack), served by `gunicorn wsgi:application` with one worker and four threads.
Loading the data takes about 25 seconds on the shared free CPU, and Render fails a deploy that does not answer
quickly, so `wsgi.py` answers `/healthz` and a self-refreshing "starting up" page at once and loads the dashboard on
a background thread. The thread is started on the first request inside the worker; starting it at import time
deadlocked the forked worker. `/debugz` reports whether the app has loaded and the process memory. The free tier has
512 MB of RAM (the app uses about 300 MB) and sleeps after about 15 minutes idle, so the first request after a pause
can take up to a minute.

Tests (`python -m pytest`) check, on small synthetic data, that lags are exact calendar-day offsets, that no feature
uses sales from the row's own date or later, that rolling windows span calendar days, that the event-distance and
Promo2 features behave as specified, and that the deployed `model.txt` reproduces the saved predictions. They do not
check dashboard layout or the live deployment.

## 9. Limitations

- **New stores:** with no history, the lag features fall back to 0. A store-type average would be a better cold start.
- **Forecast bands** are a constant-width log-error interval, not a quantile or conformal method, and do not widen
  with the horizon.
- **Forward forecast coverage:** only the 856 stores in Kaggle's `test.csv` have a forward calendar.
- **Evaluation:** one holdout window of seven months that also guided feature choices (section 2); the multi-week
  figure rests on one month and 150 stores.
- **Zero-filled gaps:** where a store has no rows, the lag features read zero instead of "unknown" (section 1).
- **Promo2** is a monthly on/off signal; it does not show which day of the month a Promo2 round started.
- **Untried:** an ensemble with a per-store seasonal model, and a school-holiday-distance feature.

Takeaways: validate with folds that move forward in time, since a feature that cannot extrapolate (`DaysSinceStart`)
only shows its cost when the validation window lies beyond the training dates; whenever a what-if scenario changes one
input, change every feature derived from the same event (the Promotion Simulator does); and test a suspected feature
by removing it rather than by reading its importance.

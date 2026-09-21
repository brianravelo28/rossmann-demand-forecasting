# Rossmann Demand Forecasting

**Live demo:** https://rossmann-demand-forecasting.onrender.com/ — hosted on Render's free tier, so if it has been idle the first load shows a "starting up" page for up to a minute.

## Problem Statement
Rossmann operates 1,115 stores across multiple regions with demand driven by promotions,
holidays, store type, and competition. Accurate forecasts reduce stockouts and excess inventory.

## Solution Overview
Time-series sales forecasting using LightGBM with walk-forward cross-validation.
Trained on 844,392 open-store-day records (Jan 2013–Jul 2015), 1,115 stores.

## Why LightGBM Over ARIMA
ARIMA assumes stationarity and linear dependencies but fails on retail data:
- **Promotions**: non-linear sales spikes (measured promo lift here: **+38.8%**)
- **Store heterogeneity**: StoreType b stores average ~48% higher sales than other types
- **Holidays**: discrete demand shifts violating differencing assumptions
- **Scale**: a single LightGBM model serves all 1,115 stores; ARIMA needs a separate fit per store

**ARIMA baseline (Store #1 only)**: 17.2% MAPE
**LightGBM (CV mean, all stores)**: 10.0% MAPE
**LightGBM (final holdout, all stores)**: 9.2% MAPE
**LightGBM (Store #1 only, same period as ARIMA)**: 9.0% MAPE

On Store #1 specifically, LightGBM (9.0% MAPE) clearly beats ARIMA (17.2%) — nearly halving its
error — and it does so while simultaneously serving all 1,115 stores' non-linear
promotion/holiday effects from a **single model** instead of 1,115 separate per-store fits.

## Data
- **Source**: Kaggle Rossmann Store Sales
- **Size**: 844,392 open-store-days after cleaning (1,115 stores, Jan 1, 2013–Jul 31, 2015)
- **Target**: Daily sales (units)
- **Features**: 42 engineered (temporal, lag, rolling, promotion, holiday, competition, store format)

## Feature Engineering
- **Temporal**: Day of week (6 dummies), month (11 dummies). `DaysSinceStart` (linear days-since-
  series-start) is computed but deliberately excluded from the model — see the note below.
- **Holiday**: State holiday, school holiday flags
- **Lag sales**: T-1, T-7, T-14, T-28, computed as true **calendar-day** offsets (via a
  date-shifted merge on `(Store, Date - N days)`), not a row-position shift. This matters: 181 of
  1,115 stores have interior gaps in their recorded history (missing days, not just weekly
  closures), so a plain `groupby('Store').shift(N)` silently drifts off the true N-days-ago value
  for those stores. The calendar-day version also correctly returns `Sales_lag_1 = 0` for a
  Monday whose Sunday was a closure, rather than borrowing Saturday's value as a proxy.
- **Rolling**: 7-day, 30-day rolling means, computed on a per-store calendar-complete
  reindexed series (so the window spans true calendar days even across gaps), then shifted by 1
  to avoid leakage.
- **Promotion**: Current, T-1 (also calendar-day-aligned, same fix as above)
- **Holiday distance**: `DaysToNextHoliday`, `DaysSinceLastHoliday` — per-store calendar-day
  distance to the nearest state holiday (clipped to 30 days), computed via `np.searchsorted`
  against each store's sorted holiday-date list. Both are 0 on the holiday date itself. State
  holidays in this dataset are regional (not every store observes every date), so this is
  computed per store rather than off one shared national calendar.
- **Promo distance**: `DaysToNextPromo`, `DaysSinceLastPromo` — same calendar-day-distance
  machinery as the holiday features (generalized into one shared `add_event_distance_features()`
  helper), applied to the primary `Promo` flag instead of `StateHoliday`. This turned out to be
  the single biggest feature added post-launch: `DaysToNextPromo` ranks 2nd in SHAP importance,
  ahead of the plain `Promo_active` flag it complements — a continuous "how close is the next
  promo" signal captures campaign timing patterns that a same-day binary flag can't.
- **Promo2 interval**: `IsPromo2Active` — whether the *recurring* Promo2 program (distinct from
  the day-to-day `Promo` flag) is running this calendar month for this store. Promo2 renews every
  few months per `PromoInterval` (e.g. `"Jan,Apr,Jul,Oct"`) starting from the ISO week/year in
  `Promo2SinceWeek`/`Promo2SinceYear` — the static `Promo2` column only says whether a store
  *participates* in the program at all, not whether it's active on a given date, so this is a
  materially different signal from just passing `Promo2` through.
- **Competition**: Distance, years active
- **Store format**: StoreType (a/b/c/d), Assortment (a/b/c) — one-hot. Not in the original
  spec but added after diagnosing the initial model: these two columns show a ~48% mean-sales
  gap by StoreType and are known strong predictors for this dataset, so leaving them out was
  clearly leaving signal on the table.
- **Target transform**: trained on `log1p(Sales)`, inverted with `expm1` at prediction time.
  Without this, a store's L2 loss contribution scales with its absolute sales level, so the
  model effectively ignores percentage error on smaller stores. This is standard practice for
  this competition and measurably improved CV MAPE (23.5% → 18.4%).
- **`DaysSinceStart` excluded from the model**: this linear "days since the training series
  started" counter was in the original feature list, but every walk-forward validation fold (and
  the final test period) is, by construction, entirely *beyond* the date range the model trained
  on. A tree model can't extrapolate a linear trend past its training range — it just reuses
  whatever leaf value corresponds to the rightmost dates it saw. For Fold 3 (train through Jun
  2014, validate Jul–Sep 2014) that leaf happened to encode H1 2014's relatively elevated sales
  level, so the model systematically over-predicted the Q3 seasonal dip by +14.8% on average.
  Dropping the feature fixed Fold 3 specifically (16.4% → 8.9% MAPE) and, more importantly,
  improved every other fold too (CV mean 13.6% → 10.0%) — it turned out to be net harmful
  everywhere, not just somewhere this happened to be visible. It's still computed as a DataFrame
  column (useful for EDA) but is not passed to the model.

## Model Performance

### Walk-Forward Cross-Validation (4 folds)
| Fold | Train Period | Validation Period | MAPE | RMSE | MAE |
|------|--------------|-------------------|------|------|-----|
| 1 | 2013 | Q1 2014 | 10.67% | 1055.3 | 670.2 |
| 2 | 2013–Q1'14 | Q2 2014 | 9.51% | 978.2 | 661.4 |
| 3 | 2013–Q2'14 | Q3 2014 | 8.88% | 900.6 | 595.3 |
| 4 | 2013–Q3'14 | Q4 2014 | 11.09% | 1293.4 | 805.1 |
| **Mean** | — | — | **10.04%** | **1056.9** | **683.0** |

### Final Test Set (Jan–Jul 2015, unseen)
- **MAPE**: 9.15%
- **RMSE**: 912.5
- **% predictions within ±25%**: 95.9%

Interpretation: 96% of forecasts land within 25% of actual sales, and final-test MAPE (9.2%) is
now comfortably past the original target range on every metric. Fold 3 (Q3 2014) — previously the
weakest fold at 16.36% MAPE — is now the *strongest* at 8.88%. Root cause: it had a systematic
+14.8% over-prediction bias traced to `DaysSinceStart`, a linear trend feature the model couldn't
extrapolate correctly past its training range (every fold's validation window is past that range,
by construction). Removing it fixed Fold 3 and improved every other fold too — see the Feature
Engineering note above. Fold 4 (holiday season) is now the weakest, though at 11.09% it's a much
smaller gap than the other folds show, not a specific weak spot the way Fold 3 or the original
Fold 4 (18.57%, pre-holiday-features) were.

## Feature Importance (SHAP)
Top 5 features by mean |SHAP value| (log-sales scale, since the model predicts log1p(Sales)):
1. Sales_lag_14: 0.098
2. DaysToNextPromo: 0.066
3. Sales_rolling_7d: 0.062
4. Sales_rolling_30d: 0.054
5. Sales_lag_1: 0.048

`DaysToNextPromo` still ranks 2nd, ahead of `Promo_active` — a continuous "how many days until
the next promo" signal captures more of the promo effect than a same-day binary flag alone.
`DaysSinceLastHoliday` (8th) and `DaysSinceLastPromo` (9th) both remain in the top 10.

**Insight**: the calendar-aligned lag features (especially `Sales_lag_14` — two weeks ago, same
weekday) and the promo-distance features dominate. Removing `DaysSinceStart` entirely (rather
than just watching its SHAP rank drift, as in earlier iterations of this README) turned out to
be the right call — a feature ranking 3rd in SHAP importance was nonetheless net harmful to
predictive accuracy, which is a useful reminder that SHAP importance measures how much a feature
*moves* the prediction, not whether it moves it in the *right direction*. See Feature Engineering
above and Limitations below.

## Dashboard
Interactive Plotly Dash app with 4 tabs:

A header shows the headline metrics (read from `day2_metrics.json`) and an "About this model" note.

1. **Forecast vs Actual**: One-step-ahead predictions per store on the Jan–Jul 2015 holdout, with
   KPIs (MAPE, RMSE in €, % within ±25%).
2. **Error Heatmap**: MAPE by store × ISO week. Stores are sorted by average error (worst on top) and
   the color scale is capped at 40% so the typical range stays readable; hover gives exact values.
3. **Promotion Simulator**: Force promotion on/off across a date range for a store; the promo-timing
   features are recomputed to match, and the sales-lag features keep their actual values.
4. **Forward Forecast (48 days)**: Recursive day-by-day forecast for Aug 1 – Sep 17, 2015 (the
   calendar in Kaggle's `test.csv`, which covers 856 of the 1,115 stores). Pick up to 5 stores,
   shown individually by default, or tick **Combined forecast?** for their summed total. CSV export.

**How the forward forecast works.** Each open day is predicted, and that prediction is written back
into a per-date sales buffer so later days' `Sales_lag_*` / `Sales_rolling_*` features are computed
exactly as in training (calendar-day offsets, closed days = 0) — but from the model's own earlier
forecasts. Promotions and holidays come from the known `test.csv` calendar. *Backtest:* started on
Jul 1 2015 for 150 random stores and scored against actuals, this recursive procedure gets **10.5%
MAPE vs 8.9% one-step-ahead** on the same days — a real but modest cost, with no blow-up over the
month (7-day buckets: 11.0 / 9.3 / 10.0 / 11.4%). Bands are ±1.96σ of log-error on the holdout (for
the combined total, measured on the actual summed sales, so offsetting store errors are reflected);
they don't widen with horizon, so treat them as a lower bound on the true uncertainty.

## How to Run
**Quickest (no Kaggle download):** the repo ships `deploy_data/`, a compact (~9.5 MB) copy of
everything the dashboard needs — the full daily history, the Jan–Jul 2015 holdout with
predictions and features, the Aug–Sep 2015 calendar, store attributes, and the trained model.

```bash
pip install -r requirements-render.txt
python app.py
# Navigate to http://localhost:8050
```

**Full pipeline (regenerate everything from the raw Kaggle files):**

```bash
pip install -r requirements.txt

# 1. Download the Kaggle "Rossmann Store Sales" competition data (train.csv, test.csv,
#    store.csv) and place the three files in data/
# 2. Run the pipeline in order:
python day1_eda.py    # -> data/train_processed.csv, eda_summary.json, store1_*.html
python day2_model.py  # -> models/lightgbm_model.pkl, data/test_predictions.csv, cv_results.csv
python day3_shap.py   # -> shap_feature_importance.csv, shap_*.png
python build_deploy_data.py   # -> deploy_data/ (optional; refreshes the compact bundle)

# 3. Launch the dashboard (uses deploy_data/ if present; set ROSSMANN_USE_LOCAL_DATA=1
#    to force the full data/ + models/ files instead)
python app.py
```

Data: *Rossmann Store Sales*, provided by Dirk Rossmann GmbH via Kaggle
([competition page](https://www.kaggle.com/c/rossmann-store-sales)).

## Deployment (Render)
`render.yaml` defines a free-tier web service that installs `requirements-render.txt` (runtime
dependencies only — no training stack) and serves the app with gunicorn. In Render:
**New + → Blueprint →** select this repository. The dashboard reads only `deploy_data/`, so the
service needs no data download or training step. Notes: the free tier has 512 MB of RAM (the app
uses roughly 350 MB) and sleeps after ~15 minutes idle, so the first request after a pause takes
a while; a paid instance removes both limits.

## Limitations & Future Work
1. ~~Forward-forecast lags proxied with a constant~~ — **fixed**: Tab 4 now forecasts recursively
   (see Dashboard). Remaining gap: forward bands don't widen with horizon, and only the 856 stores
   in Kaggle's `test.csv` have a forward calendar.
2. **New stores**: no historical data → lag features fall back to 0; a store-type aggregate would
   be a better cold-start estimate.
3. ~~Trend extrapolation: `DaysSinceStart` is a linear counter a tree model can't extrapolate past
   its training range~~ — **fixed** (see item 10 below): removed from the model entirely rather
   than replaced, since it turned out to be net harmful once actually measured.
4. ~~Row-index vs. calendar-day lags~~ — **fixed**: `Sales_lag_*`/`Sales_rolling_*d`/`Promo_lag_1`
   are now computed via calendar-day-offset merges (and a per-store calendar-complete reindex for
   the rolling means) instead of a row-position `groupby().shift()`. This mattered more than the
   weekly-closure case alone suggested: 181 of 1,115 stores have interior gaps in their recorded
   history (missing days, not just Sundays), so the row-shift version was drifting off the true
   N-days-ago value for ~16% of stores. Fixing it dropped final-test MAPE from 18.0% → **13.5%**
   and pushed % within ±25% from 73% → **87%**.
5. **Prediction intervals**: Tab 4's ±95% band uses ±1.96σ of the holdout log-error rather than a
   proper quantile/conformal method, and is constant across the forecast horizon.
6. ~~Feature scope: promo-interval / days-since/until-next-promo~~ — **fixed**: added
   `IsPromo2Active` (recurring Promo2 program, derived from `Promo2SinceWeek/Year` +
   `PromoInterval`) and `DaysToNextPromo`/`DaysSinceLastPromo` (calendar-day distance on the
   primary `Promo` flag, via the same generalized `add_event_distance_features()` helper used for
   holidays). This was the single largest improvement of the three feature passes: final-test
   MAPE 12.8% → **10.3%**, CV mean 14.2% → **13.6%**, within ±25% 89% → **94.2%**.
   `DaysToNextPromo` ranks 2nd in SHAP importance, ahead of `Promo_active` itself. One follow-on
   fix this required: the Promotion Simulator (Tab 3) now recomputes `DaysToNextPromo`/
   `DaysSinceLastPromo` for the counterfactual scenario too (previously only `Promo_active`/
   `Promo_lag_1` were toggled) — without that, the simulator was feeding the model a
   self-contradictory feature combination (promo toggled off, but the distance features still
   said a promo was imminent), which measurably understated the simulated lift.
7. **Ensemble**: combining LightGBM with a per-store seasonal model could further reduce error.
8. ~~Holiday season (Fold 4) was the weakest fold~~ — **fixed**: added `DaysToNextHoliday` /
   `DaysSinceLastHoliday` (per-store, calendar-day distance to the nearest state holiday, clipped
   to 30 days, 0 on the holiday itself). Fold 4 MAPE dropped 18.57% → 13.29%, and CV mean dropped
   17.11% → 14.15% — the improvement wasn't confined to Fold 4, since holiday timing correlates
   across the year. `DaysToNextHoliday`/`DaysSinceLastHoliday` don't use `SchoolHoliday` — a
   school-holiday-distance variant is a natural next feature to try if a future fold turns out to
   be school-calendar-driven.
9. **Promo2 vs. Promo**: `IsPromo2Active` is a coarse monthly on/off signal (Promo2 renews for
   the whole month named in `PromoInterval`) — it doesn't capture exactly which day within that
   month a Promo2 round started, unlike the day-level precision of `DaysToNextPromo` for the
   primary `Promo` flag.
10. ~~Fold 3 (Q3 2014) was the weakest fold~~ — **fixed, and the root cause was unrelated to
    Q3/summer/school-holidays despite what item 8 originally speculated**: Fold 3 had a
    systematic +14.8% over-prediction bias, traced (by an ablation, not a guess) to
    `DaysSinceStart`. Every walk-forward fold's validation window sits entirely beyond its
    training date range, so this linear trend feature was *always* extrapolating — for Fold 3
    specifically, the model's rightmost leaf apparently encoded H1 2014's elevated sales level
    and misapplied it to Q3's seasonal dip. Removing the feature (see Feature Engineering above)
    fixed Fold 3 (16.36% → 8.88%, now the *best* fold) and every other fold too (CV mean 13.58% →
    10.04%). Lesson: a feature ranking 3rd in SHAP importance was still net harmful — SHAP
    magnitude says how much a feature moves predictions, not whether the move is correct, so it's
    not a substitute for ablation when a specific fold or period looks anomalous.

## Files
```
Rossmann Project/
├── data/                     (raw Kaggle CSVs + generated files; not tracked)
├── models/                   (lightgbm_model.pkl; not tracked)
├── deploy_data/              compact bundle the dashboard loads (parquet + model.txt)
├── features.py               shared feature definitions + event-distance helpers
├── day1_eda.py, day2_model.py, day3_shap.py
├── build_deploy_data.py      builds deploy_data/ from the pipeline outputs
├── app.py                    Dash dashboard
├── render.yaml, requirements-render.txt   Render deployment
├── eda_summary.json, cv_results.csv, day2_metrics.json, shap_feature_importance.csv
├── shap_feature_importance_bar.png, shap_summary_beeswarm.png
├── README.md (this file)
└── requirements.txt          full (training) dependencies
```

## Learnings
- Log-transforming the target matters a lot when training one model across stores of very
  different sales scale — L2 loss on raw sales effectively ignores small stores' % error.
- Lag/rolling features computed on a row-filtered (Open=1) series don't correspond to true
  calendar-day lags — this was the biggest single lever early on. Fixing it (calendar-day-offset
  merges instead of `groupby().shift()`) took final-test MAPE from 18.0% to 13.5% and made
  LightGBM's win over ARIMA unambiguous on a like-for-like store comparison. The bug wasn't
  obvious from the code alone — it only showed up by checking whether every store had a complete
  calendar-day history, which 181 of 1,115 didn't.
- Once the lag/rolling machinery was calendar-correct and generalized (one
  `add_event_distance_features()` helper for "distance to nearest event"), extending it from
  holidays to promos was cheap and paid off more than the holiday feature did on its own
  (final-test MAPE 12.8% → 10.3% vs. 13.5% → 12.8% for holidays). A continuous "days until next
  promo" signal captured more than the binary same-day `Promo_active` flag alone — worth
  defaulting to distance features over binary flags for any recurring, schedulable event.
- A new feature can silently break a downstream consumer that only partially mirrors the training
  pipeline: adding `DaysToNextPromo` made the Promotion Simulator's counterfactual internally
  inconsistent (toggled `Promo_active` off, but the distance features still described an
  imminent promo) until it was updated to recompute those too. Worth an explicit check whenever a
  new feature is added: everywhere a "what-if" scenario mutates one input feature, make sure every
  other feature that's derived from the same underlying event gets mutated consistently.
- Store-format features (StoreType, Assortment) that weren't in the original feature spec turned
  out to carry real signal — worth checking "obvious" categorical columns before assuming an
  engineered feature list is complete.
- Walk-forward CV surfaces real weaknesses (Fold 4 / holiday season) that a single train/test
  split would hide — and per-fold breakdown pointed straight at the fix (a holiday-distance
  feature) rather than generic "tune the model more" advice.
- A targeted feature aimed at one weak fold (Fold 4) improved every fold, not just that one —
  holiday timing correlates across the calendar year, so the model generalized the signal rather
  than just memorizing Q4 2014's specific holidays.
- SHAP explainability is useful for catching model behavior (e.g. DaysSinceStart extrapolation
  risk) that raw accuracy metrics don't reveal — and for confirming a fix actually worked: after
  the calendar-alignment fix, all four `Sales_lag_*` features moved into the top 8 SHAP features,
  versus only two before. That said, SHAP rank and practical impact aren't the same thing:
  `DaysToNextHoliday` ranks 9th by SHAP magnitude but was decisive for Fold 4 specifically —
  don't use global SHAP rank alone to decide which features are worth keeping.
- ...and the same lesson cuts the other way, more sharply: `DaysSinceStart` ranked *3rd* in SHAP
  importance while being net harmful to accuracy on every single fold. High SHAP magnitude means
  a feature strongly influences the prediction — it says nothing about whether that influence
  points the right direction, especially for a feature (a linear trend counter) that's guaranteed
  to be extrapolating on every fold by construction. When a specific fold looks anomalous, treat
  that as a prompt to ablate the features most likely to behave differently in- vs. out-of-range,
  not just to add more features aimed at the symptom.

## Author
Brian | Data Scientist | August 2026

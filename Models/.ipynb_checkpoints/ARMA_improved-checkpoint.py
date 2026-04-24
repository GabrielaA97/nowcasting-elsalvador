# =============================================================================
# ARMA MODEL -- Monthly IVAE Nowcasting (El Salvador)
# =============================================================================
# Benchmark univariate model. Uses only the history of IVAE_TOT m-o-m
# growth rate to predict the current month's value.
#
# Role in the paper:
#   The ARMA serves as the univariate benchmark -- the simplest possible
#   model that uses no external predictors. If a multivariate model
#   (Ridge, XGBoost, DFM, etc.) cannot beat ARMA, it has no value added.
#
# Evaluation:
#   - Two windows: Crisis (2019M3-2021M6) and Post-Pandemic (2021M9-2024M6)
#   - Three vintages: lag -2, -1, 0
#     lag -2 = predict this month using data available 2 months ago
#     lag -1 = predict this month using data available 1 month ago
#     lag  0 = predict this month using all data available this month
#   - Metrics: RMSE, MAE, MAPE, R2
# =============================================================================

# -- 0. IMPORTS ----------------------------------------------------------------
# -- Setup: working directory ------------------------------------------------
import os
os.chdir("/Users/gabrielaaquino/Desktop/real-time-gdp-nowcasting-sv/Models")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

from utils import (
    CONFIG,
    load_data,
    gen_lagged_data,
    compute_metrics,
    get_monthly_test_dates,
    print_performance_table,
)

plt.rcParams["figure.figsize"] = [12, 5]
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# -- 1. CONFIGURATION ----------------------------------------------------------
TARGET     = CONFIG["target_variable"]   # "IVAE_TOT"
LAGS_USE   = [-2, -1, 0]                # ARMA only nowcasts and backcasts
                                         # +1/+2 not meaningful for univariate

# -- 2. LOAD DATA --------------------------------------------------------------
data, metadata = load_data()

print(f"\nIVAE_TOT last 6 months:")
print(data[["date", TARGET]].dropna().tail(6).to_string(index=False))

# -- 3. SELECT ARMA ORDER ON PRE-CRISIS TRAINING HISTORY ----------------------
print("\n" + "="*55)
print("  Selecting ARMA order on pre-crisis training data")
print("="*55)

train_series = (
    data.loc[data.date < CONFIG["window_1"]["start_date"], TARGET]
    .dropna()
    .values
)
print(f"Training observations: {len(train_series)}")

auto_model = pm.auto_arima(
    train_series,
    seasonal              = False,
    stationary            = True,
    information_criterion = "aic",
    max_p = 6, max_q = 6,
    stepwise          = True,
    suppress_warnings = True,
)
AR_ORDER = auto_model.order[0]
MA_ORDER = auto_model.order[2]
print(f"Selected order: ARMA({AR_ORDER},{MA_ORDER})")
print(auto_model.summary())

# -- 4. RESIDUAL DIAGNOSTICS ---------------------------------------------------
print("\n-- Residual Diagnostics (training fit) --")
residuals = auto_model.resid()

jb_stat, jb_p = stats.jarque_bera(residuals)
print(f"Jarque-Bera    : stat={jb_stat:.4f}  p={jb_p:.4f}  "
      f"({'normal' if jb_p > 0.05 else 'non-normal'})")

lb = acorr_ljungbox(residuals, lags=[6, 12], return_df=True)
print(f"Ljung-Box(6)   : p={lb['lb_pvalue'].iloc[0]:.4f}  "
      f"({'no autocorr' if lb['lb_pvalue'].iloc[0] > 0.05 else 'autocorr detected'})")
print(f"Ljung-Box(12)  : p={lb['lb_pvalue'].iloc[1]:.4f}  "
      f"({'no autocorr' if lb['lb_pvalue'].iloc[1] > 0.05 else 'autocorr detected'})")

dw = durbin_watson(residuals)
print(f"Durbin-Watson  : {dw:.4f}  (2=no autocorr)")

# -- 5. PREDICTION FUNCTION ----------------------------------------------------
def predict_arma(data, metadata, target_date, lag, ar_order, ma_order):
    """Predict IVAE_TOT for target_date with data available at lag months prior."""
    lagged = gen_lagged_data(metadata, data, target_date, lag)
    series = lagged[TARGET].dropna().values

    if len(series) < max(ar_order + ma_order + 5, 10):
        return np.nan
    try:
        fit  = ARIMA(series, order=(ar_order, 0, ma_order)).fit()
        return float(fit.forecast(steps=1)[0])
    except Exception:
        return np.nan

# -- 6. NOWCAST LOOP -----------------------------------------------------------
def run_window(window_cfg, window_label):
    print(f"\n{'='*55}\n  Window: {window_label}\n{'='*55}")

    dates_df     = get_monthly_test_dates(window=window_cfg)
    all_months   = dates_df["date"].tolist()
    pred_dict    = {lag: {} for lag in LAGS_USE}
    actuals_dict = {}

    for month in all_months:
        date_str   = str(month.date())
        actual_row = data.loc[data.date == month, TARGET]
        if actual_row.empty or actual_row.isna().all():
            continue
        actuals_dict[date_str] = float(actual_row.values[0])

        for lag in LAGS_USE:
            pred_dict[lag][date_str] = predict_arma(
                data, metadata, date_str, lag, AR_ORDER, MA_ORDER
            )

        print(f"  {date_str}: actual={actuals_dict[date_str]*100:.2f}%  "
              f"nowcast={pred_dict[0].get(date_str,np.nan)*100:.2f}%")

    # Metrics
    rows = []
    for lag in LAGS_USE:
        common = [d for d in actuals_dict if d in pred_dict[lag]
                  and not np.isnan(pred_dict[lag][d])]
        if not common:
            continue
        a = np.array([actuals_dict[d]   for d in common])
        f = np.array([pred_dict[lag][d] for d in common])
        m = compute_metrics(a, f, label=str(lag))
        m["Vintage"] = lag
        rows.append(m)

    perf = pd.DataFrame(rows)[["Vintage","RMSE","MAE","MAPE","R2","n"]]
    print_performance_table(perf.round(4), f"ARMA -- {window_label}")
    return perf, pred_dict, actuals_dict

perf_w1, preds_w1, actuals_w1 = run_window(CONFIG["window_1"], "Crisis (COVID-19)")
perf_w2, preds_w2, actuals_w2 = run_window(CONFIG["window_2"], "Post-Pandemic")

# -- 7. COMBINED TABLE ---------------------------------------------------------
perf_w1["Window"] = "Crisis"
perf_w2["Window"] = "Post-Pandemic"
combined = pd.concat([perf_w1, perf_w2], ignore_index=True)
combined["Model"] = f"ARMA({AR_ORDER},{MA_ORDER})"
print("\n-- Combined Performance --")
print(combined[["Model","Window","Vintage","RMSE","MAE","MAPE","R2","n"]]
      .to_string(index=False))

# -- 8. VISUALIZATIONS ---------------------------------------------------------
def plot_window(preds, actuals_dict, window_label):
    dates   = sorted(actuals_dict.keys())
    dt      = [pd.Timestamp(d) for d in dates]
    actual  = [actuals_dict[d]*100 for d in dates]
    pred_0  = [preds[0].get(d, np.nan)*100  for d in dates]
    pred_m1 = [preds[-1].get(d, np.nan)*100 for d in dates]
    pred_m2 = [preds[-2].get(d, np.nan)*100 for d in dates]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

    # Actual vs predictions
    ax1.plot(dt, actual,  color="black",   lw=1.6, label="Actual IVAE m-o-m")
    ax1.plot(dt, pred_0,  color="#2166ac", lw=1.0, ls="--", label="Nowcast (lag 0)")
    ax1.plot(dt, pred_m1, color="#f46d43", lw=0.8, ls=":",  label="Lag -1", alpha=0.85)
    ax1.plot(dt, pred_m2, color="#abdda4", lw=0.8, ls="-.", label="Lag -2", alpha=0.85)
    ax1.axhline(0, color="gray", lw=0.6, ls="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax1.set_title(f"ARMA({AR_ORDER},{MA_ORDER}) | {window_label}",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("IVAE m-o-m (%)")
    ax1.legend(fontsize=9)

    # Errors
    valid = [(d, t, a, p) for d, t, a, p in zip(dates, dt, actual, pred_0)
             if not np.isnan(p)]
    err_dt  = [v[1] for v in valid]
    errors  = [v[2] - v[3] for v in valid]
    colors  = ["#d73027" if e < 0 else "#4dac26" for e in errors]
    ax2.bar(err_dt, errors, color=colors, width=20, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax2.set_title("Forecast Errors (Actual - Predicted, lag 0)", fontsize=11)
    ax2.set_ylabel("Error (pp)")

    plt.suptitle(
        f"ARMA | El Salvador IVAE Monthly Nowcasting | {window_label}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

plot_window(preds_w1, actuals_w1, "Crisis (COVID-19)")
plot_window(preds_w2, actuals_w2, "Post-Pandemic Normalisation")

# -- 9. SAVE RESULTS -----------------------------------------------------------
def build_output(preds, actuals_dict, window_name):
    dates = sorted(actuals_dict.keys())
    return pd.DataFrame({
        "date"        : dates,
        "actual"      : [actuals_dict[d]        for d in dates],
        "pred_lag_m2" : [preds[-2].get(d,np.nan) for d in dates],
        "pred_lag_m1" : [preds[-1].get(d,np.nan) for d in dates],
        "pred_lag_0"  : [preds[0].get(d, np.nan) for d in dates],
        "model"       : f"ARMA({AR_ORDER},{MA_ORDER})",
        "window"      : window_name,
    })

results = pd.concat([
    build_output(preds_w1, actuals_w1, "Crisis"),
    build_output(preds_w2, actuals_w2, "Post-Pandemic"),
], ignore_index=True)

# Uncomment to save:
# import os; os.makedirs("outputs", exist_ok=True)
# results.to_csv("outputs/arma_predictions.csv", index=False)
# combined.to_csv("outputs/arma_performance.csv", index=False)

print(f"\nARMA complete. Results: {results.shape[0]} rows")
print(results.head(6).to_string(index=False))

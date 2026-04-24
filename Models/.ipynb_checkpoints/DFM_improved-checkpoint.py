# =============================================================================
# DYNAMIC FACTOR MODEL (DFM) -- Monthly IVAE Nowcasting (El Salvador)
# =============================================================================
# Implementation using statsmodels.tsa.statespace.dynamic_factor
# This replaces nowcastDFM (R) which was designed for quarterly targets.
#
# Why statsmodels instead of nowcastDFM:
#   - nowcastDFM was designed for quarterly targets with monthly indicators
#   - statsmodels handles monthly targets natively and efficiently
#   - Kalman filter runs once over full sample (not per prediction call)
#   - Based on the same Banbura & Modugno (2014) framework
#
# Model:
#   y_t = Lambda * f_t + e_t    (observation equation)
#   f_t = A * f_{t-1} + u_t     (state equation)
#
# References:
#   Banbura & Modugno (2014). JAE.
#   Stock & Watson (2002). JASA.
# =============================================================================

# -- Setup --------------------------------------------------------------------
import os
os.chdir("/Users/gabrielaaquino/Desktop/real-time-gdp-nowcasting-sv/Models")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.preprocessing import StandardScaler

from utils import (
    CONFIG, load_data, gen_lagged_data,
    compute_metrics, get_monthly_test_dates, print_performance_table,
)

plt.rcParams["figure.figsize"] = [12, 5]

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

TARGET    = CONFIG["target_variable"]   # IVAE_TOT
LAGS_USE  = CONFIG["lags"]              # [-2, -1, 0]
N_FACTORS = 4                           # one per economic block

# Parsimonious variable selection -- one representative per subcategory
# Following Stock & Watson (2002) and Bai & Ng (2002)
VARS_DFM = [
    "IVAE_TOT",                          # TARGET
    # block_s: real activity
    "IVAE_AG", "IVAE_IN", "IVAE_CO", "IVAE_CT", "IVAE_AF",
    "IPI", "PRO_ENER", "CON_ENER", "CON_APA_CEM",
    "ISSS_SEC", "ISSS_TER", "CE_TOT_VV",
    # block_r: external sector
    "EXP_IM", "EXP_CP", "EXP_MO",
    "IMP_IM", "IMP_D_IM", "IMP_D_BN",
    "REM", "IPI_US", "UNEM_US", "EFFR_US",
    # block_l: fiscal + financial
    "INGT_IVA", "INGT_ISR", "GPC", "GIP", "BFG",
    "M3", "CTC", "CCO", "CPI",
    "TIP_30", "TIP_180", "TPR1",
    # quarterly anchors
    "GDP", "GDP_IN", "GDP_CO",
]

print("=" * 55)
print("  DFM -- Monthly IVAE Nowcasting (Python)")
print("=" * 55)
print(f"Target    : {TARGET}")
print(f"Vintages  : {LAGS_USE}")
print(f"Factors   : {N_FACTORS}")
print(f"Variables : {len(VARS_DFM)}")

# =============================================================================
# 2. LOAD DATA
# =============================================================================

data, metadata = load_data()

available_vars = [v for v in VARS_DFM if v in data.columns]
data_dfm = data[["date"] + available_vars].copy()

print(f"\nVariables available : {len(available_vars)}")
print(f"Date range          : {data_dfm.date.min().date()} -> {data_dfm.date.max().date()}")

# =============================================================================
# 3. TRAINING DATA -- PRE-CRISIS
# =============================================================================

train_end = pd.Timestamp(CONFIG["window_1"]["start_date"]) - pd.DateOffset(months=1)

train_data = data_dfm.loc[
    (data_dfm.date >= pd.Timestamp(CONFIG["train_start_date"])) &
    (data_dfm.date <= train_end),
    available_vars
].copy().reset_index(drop=True)

print(f"\nTraining obs : {len(train_data)}")
print(f"Training end : {train_end.date()}")

# =============================================================================
# 4. ESTIMATE DFM ONCE ON TRAINING DATA
# =============================================================================

print("\n>>> Estimating DFM on pre-crisis training data...")

# Drop columns with >80% missing
missing_pct = train_data.isna().mean()
dfm_cols    = missing_pct[missing_pct < 0.8].index.tolist()
print(f"  Variables after missing filter: {len(dfm_cols)}")

# Standardize -- DFM requires standardized inputs
scaler     = StandardScaler()
X_train    = train_data[dfm_cols].fillna(train_data[dfm_cols].mean())
X_scaled   = pd.DataFrame(scaler.fit_transform(X_train), columns=dfm_cols)

# Fit DFM
model = DynamicFactor(
    X_scaled,
    k_factors      = N_FACTORS,
    factor_order   = 1,
    error_order    = 0,
    error_cov_type = "diagonal"
)

dfm_result = model.fit(
    method  = "em",
    maxiter = 200,
    disp    = True,
    em_initialization = True,
)

print(f"  Log-likelihood : {dfm_result.llf:.4f}")
print(f"  AIC            : {dfm_result.aic:.4f}")

# =============================================================================
# 5. PREDICTION FUNCTION
# =============================================================================

def predict_dfm_monthly(data_full, metadata, target_date, lag):
    """
    Predict IVAE_TOT for target_date using pre-estimated DFM.

    1. Generate ragged-edge dataset for this vintage
    2. Standardize using training scaler
    3. Apply Kalman smoother
    4. Extract and rescale prediction for target month
    """
    lagged = gen_lagged_data(metadata, data_full, target_date, lag)

    # Keep only DFM variables
    X_pred = lagged[[c for c in dfm_cols if c in lagged.columns]].copy()

    # Mask target to prevent leakage
    if TARGET in X_pred.columns:
        X_pred[TARGET] = np.nan

    # Fill NAs with training means for standardization
    X_filled = X_pred.fillna(X_train[dfm_cols].mean() if dfm_cols[0] in X_pred.columns
                             else X_pred.mean())

    # Align to training column order
    for c in dfm_cols:
        if c not in X_filled.columns:
            X_filled[c] = 0.0
    X_filled = X_filled[dfm_cols]

    # Standardize
    X_new_scaled = pd.DataFrame(
        scaler.transform(X_filled),
        columns = dfm_cols
    )

    try:
        # Apply model to new data
        pred_result   = dfm_result.apply(X_new_scaled)
        fitted_scaled = pred_result.fittedvalues

        if TARGET not in fitted_scaled.columns:
            return np.nan

        pred_scaled = float(fitted_scaled[TARGET].iloc[-1])

        # Rescale to original units
        idx           = dfm_cols.index(TARGET)
        pred_original = pred_scaled * scaler.scale_[idx] + scaler.mean_[idx]

        return float(pred_original)

    except Exception:
        return np.nan

# =============================================================================
# 6. NOWCAST LOOP -- TWO WINDOWS
# =============================================================================

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
            pred_dict[lag][date_str] = predict_dfm_monthly(
                data, metadata, date_str, lag
            )

        print(f"  {date_str}: "
              f"actual={actuals_dict[date_str]*100:+.2f}%  "
              f"lag-2={pred_dict[-2].get(date_str, np.nan)*100:+.2f}%  "
              f"lag-1={pred_dict[-1].get(date_str, np.nan)*100:+.2f}%  "
              f"lag 0={pred_dict[0].get(date_str, np.nan)*100:+.2f}%")

    # Metrics
    rows = []
    for lag in LAGS_USE:
        common = [d for d in actuals_dict
                  if d in pred_dict[lag] and not np.isnan(pred_dict[lag][d])]
        if not common:
            continue
        a = np.array([actuals_dict[d]   for d in common])
        f = np.array([pred_dict[lag][d] for d in common])
        m = compute_metrics(a, f, label=str(lag))
        m["Vintage"] = lag
        rows.append(m)

    perf = pd.DataFrame(rows)[["Vintage", "RMSE", "MAE", "R2", "n"]]
    print_performance_table(perf.round(4), f"DFM -- {window_label}")
    return perf, pred_dict, actuals_dict


print("\n>>> Running Window 1: Crisis (COVID-19)")
perf_w1, preds_w1, actuals_w1 = run_window(CONFIG["window_1"], "Crisis (COVID-19)")

print("\n>>> Running Window 2: Post-Pandemic")
perf_w2, preds_w2, actuals_w2 = run_window(CONFIG["window_2"], "Post-Pandemic Normalisation")

# =============================================================================
# 7. COMBINED PERFORMANCE TABLE
# =============================================================================

perf_w1["Window"] = "Crisis"
perf_w2["Window"] = "Post-Pandemic"
combined = pd.concat([perf_w1, perf_w2], ignore_index=True)
combined["Model"] = "DFM"

print("\n-- Combined Performance --")
print(combined[["Model", "Window", "Vintage", "RMSE", "MAE", "R2", "n"]]
      .to_string(index=False))

# =============================================================================
# 8. FACTOR LOADINGS
# =============================================================================

print("\n--- Factor Loadings (top 5 per factor) ---")
try:
    loadings = pd.DataFrame(
        dfm_result.loadings,
        index   = dfm_cols,
        columns = [f"F{i+1}" for i in range(N_FACTORS)]
    )
    for col in loadings.columns:
        top5 = loadings[col].abs().nlargest(5)
        print(f"\n  {col}:")
        for var, _ in top5.items():
            print(f"    {var:20}: {loadings.loc[var, col]:+.4f}")
except Exception:
    print("  Loadings not available.")

# =============================================================================
# 9. VISUALIZATIONS
# =============================================================================

def plot_window(preds, actuals_dict, window_label):
    dates   = sorted(actuals_dict.keys())
    dt      = [pd.Timestamp(d) for d in dates]
    actual  = [actuals_dict[d] * 100 for d in dates]
    pred_0  = [preds[0].get(d,  np.nan) * 100 for d in dates]
    pred_m1 = [preds[-1].get(d, np.nan) * 100 for d in dates]
    pred_m2 = [preds[-2].get(d, np.nan) * 100 for d in dates]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

    ax1.plot(dt, actual,  color="black",   lw=1.8, label="Actual IVAE m-o-m")
    ax1.plot(dt, pred_0,  color="#2166ac", lw=1.2, ls="--", label="Nowcast (lag 0)")
    ax1.plot(dt, pred_m1, color="#f46d43", lw=1.0, ls=":",  label="lag -1", alpha=0.85)
    ax1.plot(dt, pred_m2, color="#abdda4", lw=1.0, ls="-.", label="lag -2", alpha=0.85)
    ax1.axhline(0, color="gray", lw=0.6, ls="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax1.set_title(f"DFM | {window_label}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("IVAE m-o-m (%)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    valid  = [(t, a, p) for t, a, p in zip(dt, actual, pred_0) if not np.isnan(p)]
    err_dt = [v[0] for v in valid]
    errors = [v[1] - v[2] for v in valid]
    colors_bar = ["#d73027" if e < 0 else "#4dac26" for e in errors]
    ax2.bar(err_dt, errors, color=colors_bar, width=20, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax2.set_title("Forecast Errors -- Actual minus Predicted (lag 0)", fontsize=11)
    ax2.set_ylabel("Error (pp)")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"DFM | El Salvador IVAE Nowcasting | {window_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


plot_window(preds_w1, actuals_w1, "Crisis (COVID-19)")
plot_window(preds_w2, actuals_w2, "Post-Pandemic Normalisation")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================

def build_output(preds, actuals_dict, window_name):
    dates = sorted(actuals_dict.keys())
    return pd.DataFrame({
        "date"        : dates,
        "actual"      : [actuals_dict[d]          for d in dates],
        "pred_lag_m2" : [preds[-2].get(d, np.nan) for d in dates],
        "pred_lag_m1" : [preds[-1].get(d, np.nan) for d in dates],
        "pred_lag_0"  : [preds[0].get(d,  np.nan) for d in dates],
        "model"       : "DFM",
        "window"      : window_name,
    })

results = pd.concat([
    build_output(preds_w1, actuals_w1, "Crisis"),
    build_output(preds_w2, actuals_w2, "Post-Pandemic"),
], ignore_index=True)

os.makedirs("outputs", exist_ok=True)
results.to_csv("outputs/dfm_predictions.csv", index=False)
combined.to_csv("outputs/dfm_performance.csv", index=False)

print(f"\nDFM complete.")
print(f"Predictions : outputs/dfm_predictions.csv ({len(results)} rows)")
print(f"Performance : outputs/dfm_performance.csv")
print(f"\nNext: Ridge regression")

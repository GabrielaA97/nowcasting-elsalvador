# =============================================================================
# utils.py -- Shared utilities for GDP Nowcasting (El Salvador)
# =============================================================================
# All nowcasting models import from this single file.
# Ensures identical data handling, metrics, and parameters across:
# ARMA, Ridge, Lasso, Decision Tree, XGBoost, MLP, LSTM.
#
# Usage in any notebook:
#   from utils import (CONFIG, load_data, gen_lagged_data, flatten_data,
#                      mean_fill_dataset, compute_metrics,
#                      get_monthly_test_dates, build_results_df,
#                      compute_metrics_by_month, print_performance_table)
#
# Target:
#   Quarter-on-quarter (q-o-q) growth rate of the quarterly IVAE
#   (Indice de Volumen de Actividad Economica), BCR El Salvador.
#   Used as high-frequency proxy for real GDP.
#   Reference: Amaya & Rivas (2021); BCR (2024).
#
# Two evaluation windows:
#   Window 1 -- Crisis (COVID):        2019Q1-2021Q2
#   Window 2 -- Post-pandemic:         2021Q3-2024Q2
#
# References:
#   Baumeister, Leiva-Leon & Sims (2022), REStat
#   Banbura & Modugno (2014), JAE
# =============================================================================

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

CONFIG = {
    # -- Files
    "data_file"      : "../Data/data_tf.csv",
    "metadata_file"  : "../Data/meta_data_V2.csv",

    # -- Target
    # m-o-m growth rate of monthly IVAE_TOT
    # This IS the monthly indicator -- 12 values per year
    # Source: BCR El Salvador. Ref: Amaya & Rivas (2021)
    "target_variable": "IVAE_TOT",

    # BCR publishes monthly IVAE_TOT with ~1 month lag
    "target_lag"     : 1,

    # -- Vintages
    # -2 = 2 months before quarter end (least info)
    #  0 = at quarter end (baseline nowcast)
    # +2 = 2 months after quarter end (most info)
    "lags": list(range(-2, 3)),

    # -- Training history
    "train_start_date": "2005-01-01",

    # -- Evaluation windows
    # Window 1: COVID crisis -- extreme volatility
    #   IVAE q-o-q: -18.82% in 2020Q2, rebound +14.86% in 2020Q3
    "window_1": {
        "name"      : "Crisis (COVID-19)",
        "start_date": "2019-03-01",
        "end_date"  : "2021-06-01",
    },

    # Window 2: Post-pandemic normalisation
    "window_2": {
        "name"      : "Post-Pandemic Normalisation",
        "start_date": "2021-09-01",
        "end_date"  : "2024-06-01",
    },

    # Full sample (robustness / appendix)
    "full_window": {
        "name"      : "Full Sample",
        "start_date": "2019-03-01",
        "end_date"  : "2024-06-01",
    },

    # -- Quarter-end months
    "quarter_end_months": [3, 6, 9, 12],

    # -- Feature lags
    # Unified at 4 for ALL models (was inconsistent: 2, 3, or 4 per notebook)
    "n_lags": 4,

    # -- Reproducibility
    "random_state": 42,
}


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_data(
    data_file=None,
    metadata_file=None
):
    """
    Load and validate data_tf.csv and meta_data_V2.csv.

    Returns
    -------
    data : pd.DataFrame
        Monthly panel. Quarterly variables (IVAE/GDP) have NaN in
        non-quarter-end months -- intentional mixed-frequency design.
    metadata : pd.DataFrame
        One row per variable. Required columns: series, months_lag.
    """
    data_file     = data_file     or CONFIG["data_file"]
    metadata_file = metadata_file or CONFIG["metadata_file"]

    data = (
        pd.read_csv(data_file, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    metadata = pd.read_csv(metadata_file)

    assert "date" in data.columns,           "data_tf.csv must have a 'date' column"
    assert "series" in metadata.columns,     "metadata must have a 'series' column"
    assert "months_lag" in metadata.columns, "metadata must have a 'months_lag' column"

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    target = CONFIG["target_variable"]
    print(f"Data loaded  : {len(data)} rows x {len(data.columns)-1} variables")
    print(f"Date range   : {data.date.min().date()} -> {data.date.max().date()}")
    print(f"Target (IVAE): {data[target].notna().sum()} quarterly observations")

    return data, metadata


# =============================================================================
# 3. RAGGED EDGE -- REAL-TIME VINTAGE CONSTRUCTION
# =============================================================================

def gen_lagged_data(metadata, data, last_date, lag):
    """
    Simulate the information set available at a given point in time.

    For each variable, sets recent observations to NaN based on its
    publication delay (months_lag) plus a vintage shift (lag).
    Replicates the ragged-edge structure faced in real-time nowcasting.

    Parameters
    ----------
    metadata  : DataFrame with columns 'series' and 'months_lag'
    data      : Full monthly panel (date + variables)
    last_date : Upper date limit for this vintage (str, e.g. "2023-03-01")
    lag       : Vintage shift (-2 to +2 months)
                -2 -> earliest nowcast (least information available)
                 0 -> baseline nowcast at quarter end
                +2 -> most information available

    Returns
    -------
    lagged_data : DataFrame with NaN where data not yet published
    """
    lagged_data = (
        data.loc[data.date <= pd.Timestamp(last_date)]
        .copy()
        .reset_index(drop=True)
    )

    for col in lagged_data.columns[1:]:
        match = metadata.loc[metadata.series == col, "months_lag"]
        if match.empty:
            continue
        pub_lag = int(match.values[0])
        # lag-1: lag=0 means last month's data available (not current month)
        cutoff = len(lagged_data) - pub_lag + lag - 1
        if cutoff < len(lagged_data):
            lagged_data.loc[max(0, cutoff):, col] = np.nan

    return lagged_data


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def flatten_data(data, target_variable, n_lags):
    """
    Convert mixed-frequency panel to quarterly cross-sectional format.

    For each quarter-end row, appends n_lags additional columns per
    variable containing values from preceding months. Allows non-temporal
    models (Ridge, Lasso, XGBoost, MLP, DT) to exploit monthly information.

    Example with n_lags=4 for variable IVAE_TOT:
        IVAE_TOT_1 = 1 month before quarter end
        IVAE_TOT_2 = 2 months before quarter end
        IVAE_TOT_3 = 3 months before quarter end
        IVAE_TOT_4 = 4 months before quarter end

    Parameters
    ----------
    data            : Monthly panel (date + variables)
    target_variable : Name of the GDP/IVAE column
    n_lags          : Number of monthly lags to append per variable

    Returns
    -------
    flattened : Quarterly DataFrame (one row per quarter-end month)
    """
    flattened  = data.loc[~pd.isna(data[target_variable])].copy()
    orig_index = flattened.index

    for i in range(1, n_lags + 1):
        lagged_indices = orig_index - i
        lagged_indices = lagged_indices[lagged_indices >= 0]

        tmp = data.loc[lagged_indices].copy()
        tmp["date"] = tmp["date"] + pd.DateOffset(months=i)
        tmp = tmp.drop(columns=[target_variable])
        tmp.columns = [
            f"{col}_{i}" if col != "date" else col
            for col in tmp.columns
        ]
        flattened = flattened.merge(tmp, how="left", on="date")

    return flattened.reset_index(drop=True)


def mean_fill_dataset(training, test):
    """
    Impute NaN in test using column means from training set.

    Uses training means to prevent data leakage. Called AFTER
    gen_lagged_data() so the ragged edge is respected.

    Parameters
    ----------
    training : DataFrame to compute column means from
    test     : DataFrame to impute (may equal training)

    Returns
    -------
    filled : Copy of test with NaN replaced by training-set means
    """
    mean_dict = {
        col: np.nanmean(training[col])
        for col in training.columns[1:]
    }
    filled = test.copy()
    for col, mean_val in mean_dict.items():
        if col in filled.columns:
            filled.loc[pd.isna(filled[col]), col] = mean_val
    return filled


# =============================================================================
# 5. MONTHLY TEST DATES -- KEY CHANGE: QUARTERLY -> MONTHLY EVALUATION
# =============================================================================

def get_monthly_test_dates(test_start_date=None, test_end_date=None, window=None):
    """
    Generate monthly evaluation dates with vintage and quarter mapping.

    Enables MONTHLY nowcasting: every month gets one prediction of the
    IVAE q-o-q growth rate for its target quarter.

    Each month maps to:
      vintage_in_quarter = 1 -> lag -2 (Month 1, least info)
      vintage_in_quarter = 2 -> lag -1 (Month 2)
      vintage_in_quarter = 3 -> lag  0 (Month 3, quarter-end, most info)

    Example for Q1-2023:
      Jan 2023 -> quarter 2023-03, vintage 1, lag -2
      Feb 2023 -> quarter 2023-03, vintage 2, lag -1
      Mar 2023 -> quarter 2023-03, vintage 3, lag  0

    Usage:
      dates = get_monthly_test_dates(window=CONFIG["window_1"])  # Crisis
      dates = get_monthly_test_dates(window=CONFIG["window_2"])  # Post-pandemic
      dates = get_monthly_test_dates(window=CONFIG["full_window"])
      dates = get_monthly_test_dates("2020-01-01", "2022-12-01")  # custom

    Parameters
    ----------
    test_start_date : str (overridden by window if provided)
    test_end_date   : str (overridden by window if provided)
    window          : dict from CONFIG (e.g. CONFIG["window_1"])

    Returns
    -------
    dates_df : DataFrame [date, quarter_date, vintage_in_quarter,
                          lag_equivalent, month_label, window]
    """
    if window is not None:
        test_start_date = window["start_date"]
        test_end_date   = window["end_date"]
        window_name     = window.get("name", "Custom")
    else:
        test_start_date = test_start_date or CONFIG["full_window"]["start_date"]
        test_end_date   = test_end_date   or CONFIG["full_window"]["end_date"]
        window_name     = "Custom"

    all_months = pd.date_range(test_start_date, test_end_date, freq="MS")
    lag_map    = {1: -2, 2: -1, 3: 0}
    records    = []

    for month in all_months:
        m = month.month
        if m in [1, 2, 3]:
            quarter_end = month.replace(month=3,  day=1)
            vintage     = m
        elif m in [4, 5, 6]:
            quarter_end = month.replace(month=6,  day=1)
            vintage     = m - 3
        elif m in [7, 8, 9]:
            quarter_end = month.replace(month=9,  day=1)
            vintage     = m - 6
        else:
            quarter_end = month.replace(month=12, day=1)
            vintage     = m - 9

        records.append({
            "date"               : month,
            "quarter_date"       : quarter_end,
            "vintage_in_quarter" : vintage,
            "lag_equivalent"     : lag_map[vintage],
            "month_label"        : f"Month {vintage} of Q",
            "window"             : window_name,
        })

    dates_df = pd.DataFrame(records)
    n_quarters = len(dates_df["quarter_date"].unique())
    print(f"Window  : {window_name}")
    print(f"Months  : {len(dates_df)} ({n_quarters} quarters)")
    print(f"Period  : {dates_df.date.min().date()} -> {dates_df.date.max().date()}")
    return dates_df


# =============================================================================
# 6. EVALUATION METRICS
# =============================================================================

def compute_metrics(actuals, forecasts, label=""):
    """
    Compute RMSE, MAE, MAPE, R2.

    Parameters
    ----------
    actuals   : Realized IVAE q-o-q values
    forecasts : Model predictions
    label     : Optional identifier string

    Returns
    -------
    dict with keys: label, RMSE, MAE, MAPE, R2, n
    """
    actuals   = np.array(actuals,   dtype=float)
    forecasts = np.array(forecasts, dtype=float)
    mask      = ~(np.isnan(actuals) | np.isnan(forecasts))

    if mask.sum() == 0:
        return {"label": label, "RMSE": np.nan, "MAE": np.nan,
                "MAPE": np.nan, "R2": np.nan, "n": 0}

    a      = actuals[mask]
    f      = forecasts[mask]
    errors = a - f
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)

    return {
        "label": label,
        "RMSE" : float(np.sqrt(np.mean(errors ** 2))),
        "MAE"  : float(np.mean(np.abs(errors))),
        "MAPE" : float(np.mean(np.abs(errors / a)) * 100),
        "R2"   : float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "n"    : int(mask.sum()),
    }


def compute_metrics_by_vintage(actuals_dict, pred_dict, lags=None):
    """
    Main results table: one row per vintage.

    Parameters
    ----------
    actuals_dict : {date_str: actual_value}
    pred_dict    : {lag: {date_str: predicted_value}}
    lags         : List of vintages (default: CONFIG["lags"])

    Returns
    -------
    pd.DataFrame [Vintage, RMSE, MAE, MAPE, R2, n]
    """
    lags = lags or CONFIG["lags"]
    rows = []
    for lag in lags:
        common = [d for d in actuals_dict if d in pred_dict.get(lag, {})]
        if not common:
            continue
        a = np.array([actuals_dict[d]   for d in common])
        f = np.array([pred_dict[lag][d] for d in common])
        m = compute_metrics(a, f, label=str(lag))
        m["Vintage"] = lag
        rows.append(m)

    return (
        pd.DataFrame(rows)[["Vintage", "RMSE", "MAE", "MAPE", "R2", "n"]]
        .round(4)
    )


def compute_metrics_by_month(results_df):
    """
    Key paper table: metrics by month-within-quarter (1, 2, 3).

    Shows how forecast accuracy improves as more monthly data arrives.
    Month 1 (lag -2) has least info; Month 3 (lag 0) has most.

    Parameters
    ----------
    results_df : DataFrame with columns: vintage_in_quarter, actual, predicted

    Returns
    -------
    pd.DataFrame [vintage_in_quarter, lag_equivalent, RMSE, MAE, MAPE, R2, n]
    """
    lag_map = {1: -2, 2: -1, 3: 0}
    rows    = []
    for pos in [1, 2, 3]:
        subset = results_df[results_df["vintage_in_quarter"] == pos]
        if subset.empty:
            continue
        m = compute_metrics(
            subset["actual"].values,
            subset["predicted"].values,
            label=f"Month {pos}"
        )
        m["vintage_in_quarter"] = pos
        m["lag_equivalent"]     = lag_map[pos]
        rows.append(m)

    return (
        pd.DataFrame(rows)[
            ["vintage_in_quarter", "lag_equivalent",
             "RMSE", "MAE", "MAPE", "R2", "n"]
        ]
        .round(4)
    )


# =============================================================================
# 7. RESULTS ASSEMBLY
# =============================================================================

def build_results_df(dates_df, actuals_series, pred_dict):
    """
    Combine dates, actuals, and predictions into one clean DataFrame.

    Parameters
    ----------
    dates_df       : Output of get_monthly_test_dates()
    actuals_series : pd.Series of quarterly GDP values indexed by date
    pred_dict      : {lag: list of predictions aligned with dates_df}

    Returns
    -------
    results : DataFrame ready for metrics and plotting
    """
    results = dates_df.copy()
    results["actual"] = results["quarter_date"].map(actuals_series.to_dict())
    for lag in CONFIG["lags"]:
        if lag in pred_dict:
            results[f"pred_lag_{lag}"] = pred_dict[lag]
    return results


def print_performance_table(performance, model_name=""):
    """Print a formatted performance table to console."""
    title = f"  {model_name} -- Performance by Vintage  " if model_name else "  Performance  "
    line  = "=" * (len(title) + 4)
    print(f"\n{line}\n{title}\n{line}")
    print(performance.to_string(index=False))
    print("\nVintage key:")
    print("  -2 -> Month 1 of quarter  (least information)")
    print("  -1 -> Month 2 of quarter")
    print("   0 -> Month 3 / quarter-end  (most information)")


# =============================================================================
# 8. SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    print("utils.py -- Nowcasting utilities | El Salvador IVAE/GDP")
    print("=" * 56)

    # Test 1: monthly dates for both windows
    print("\n[1] Window 1 -- Crisis (first 9 months):")
    d1 = get_monthly_test_dates(window=CONFIG["window_1"])
    print(d1.head(9).to_string(index=False))

    print("\n[2] Window 2 -- Post-Pandemic (first 6 months):")
    d2 = get_monthly_test_dates(window=CONFIG["window_2"])
    print(d2.head(6).to_string(index=False))

    # Test 2: metrics
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    f = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    m = compute_metrics(a, f, label="test")
    print(f"\n[3] Sample metrics: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}")

    print("\nutils.py loaded correctly.")

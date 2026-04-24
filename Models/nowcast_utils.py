# =============================================================================
# nowcast_utils.py
# Shared configuration and utility functions for all nowcasting models
# in the El Salvador GDP nowcasting project.
#
# Author : Gabriela Aquino
# =============================================================================

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Vintages: months relative to the end of the target quarter.
# -2 = two months before, -1 = one month before, 0 = at quarter-end.
LAGS: List[int] = [-2, -1, 0]

# Target variable
TARGET_VARIABLE: str = "GDP"

# Training window start
# Training ends one quarter before each rolling nowcast date
# (see build_rolling_windows). This constant fixes where it begins.
TRAIN_START: str = "2005-01-01"   # 2005Q1

# Rolling-origin test window
# 2020Q1–2024Q2 = 18 quarters = 6 crisis + 12 normal
TEST_START: str = "2020-03-01"
TEST_END:   str = "2024-06-01"

# Crisis window: full COVID-19 episode (shock + recovery)
CRISIS_START: str = "2020-03-01"   # 2020Q1
CRISIS_END:   str = "2021-06-01"   # 2021Q2

# Normal window: post-pandemic normalisation
NORMAL_START: str = "2021-09-01"   # 2021Q3
NORMAL_END:   str = "2024-06-01"   # 2024Q2

# GDP publication lag (months after quarter-end)
GDP_PUB_LAG: int = 3

# Reproducibility
RANDOM_SEED: int = 42


# =============================================================================
# 1. Vintage simulation
# =============================================================================

def gen_lagged_data(
    metadata: pd.DataFrame,
    data: pd.DataFrame,
    nowcast_date: str,
    lag: int
) -> pd.DataFrame:
    """
    Simulate the data vintage available at a given nowcast date.

    Masks observations that would not yet have been published at nowcast
    time, based on each variable's publication lag (from metadata) plus
    the vintage offset.

    Parameters
    ----------
    metadata     : DataFrame with columns ['series', 'months_lag'].
    data         : Full monthly panel, sorted ascending by date.
                   Column 1 is 'date', others are series.
    nowcast_date : Reference date ('YYYY-MM-DD') for the target quarter.
    lag          : Vintage offset in months (lag=-2, -1, or 0).

    Returns
    -------
    DataFrame with future observations masked to NaN.
    """
    nowcast_dt = pd.to_datetime(nowcast_date)
    lagged_data = data.loc[data["date"] <= nowcast_dt].copy()

    for col in lagged_data.columns[1:]:
        pub_lag_rows = metadata.loc[metadata["series"] == col, "months_lag"]
        if pub_lag_rows.empty:
            warnings.warn(
                f"Variable '{col}' not found in metadata. Assuming pub_lag = 1.",
                stacklevel=2
            )
            pub_lag = 1
        else:
            pub_lag = int(pub_lag_rows.values[0])

        # Cutoff date = nowcast - (pub_lag - lag) months
        cutoff_dt = nowcast_dt - pd.DateOffset(months=pub_lag - lag)
        lagged_data.loc[lagged_data["date"] > cutoff_dt, col] = np.nan

    return lagged_data.reset_index(drop=True)


# =============================================================================
# 2. Mixed-frequency flattening
# =============================================================================

def flatten_data(
    data: pd.DataFrame,
    target_variable: str,
    n_lags: int
) -> pd.DataFrame:
    """
    Convert a monthly mixed-frequency panel into a quarterly wide dataset.

    For each quarterly target observation, appends n_lags lagged copies of
    every predictor. Lag alignment is date-based to handle data gaps.

    Parameters
    ----------
    data            : Monthly panel with 'date' column.
    target_variable : Quarterly target column name (e.g. 'GDP').
    n_lags          : Number of additional lag copies per predictor.

    Returns
    -------
    DataFrame with one row per quarter.
    """
    quarterly_rows = data.loc[~pd.isna(data[target_variable])].copy()
    quarterly_dates = quarterly_rows["date"].reset_index(drop=True)
    predictor_cols = [c for c in data.columns if c not in ["date", target_variable]]

    result = quarterly_rows.reset_index(drop=True)

    for i in range(1, n_lags + 1):
        lagged_dates = quarterly_dates - pd.DateOffset(months=i)
        lag_lookup = data.set_index("date")

        lag_values = []
        for d in lagged_dates:
            if d in lag_lookup.index:
                row = lag_lookup.loc[d, predictor_cols]
            else:
                row = pd.Series({c: np.nan for c in predictor_cols})
            lag_values.append(row)

        lag_df = pd.DataFrame(lag_values).reset_index(drop=True)
        lag_df.columns = [f"{c}_{i}" for c in predictor_cols]
        result = pd.concat([result, lag_df], axis=1)

    return result


# =============================================================================
# 3. Missing value imputation (training-mean fill)
# =============================================================================

def mean_fill_dataset(
    training: pd.DataFrame,
    test: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing values in `test` using column means from `training`.

    Means are computed from training data only — prevents look-ahead bias.
    Never call with mean_fill_dataset(test, test).
    """
    train_means = {
        col: np.nanmean(training[col])
        for col in training.columns
        if col != "date" and training[col].dtype in [float, np.float64]
    }

    filled = test.copy()
    for col, mean_val in train_means.items():
        if col in filled.columns:
            filled[col] = filled[col].fillna(mean_val)

    return filled


# =============================================================================
# 4. Evaluation metrics
# =============================================================================

def _pair_mask(y_true: np.ndarray, y_pred: np.ndarray):
    """Drop NaN pairs before computing any metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask], mask.sum()


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred, n = _pair_mask(y_true, y_pred)
    if n == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true, y_pred, n = _pair_mask(y_true, y_pred)
    if n == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_metrics(
    actuals: list,
    pred_dict: dict,
    lags: Optional[list] = None
) -> pd.DataFrame:
    """
    Compute RMSE and MAE for each vintage.

    MAPE is excluded: quarterly GDP growth is often near zero, which makes
    percentage errors economically uninterpretable.
    """
    if lags is None:
        lags = LAGS

    rows = []
    for lag in lags:
        y_hat = pred_dict.get(lag, [])
        rows.append({
            "Vintage": lag,
            "RMSE":    round(rmse(actuals, y_hat), 6),
            "MAE":     round(mae(actuals, y_hat), 6),
            "n":       sum(1 for v in y_hat if not pd.isna(v)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 5. Diebold-Mariano test
# =============================================================================

def diebold_mariano_test(
    actuals: list,
    pred_benchmark: list,
    pred_model: list,
    h: int = 1
) -> dict:
    """
    Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold (1997)
    small-sample correction for equal predictive accuracy.

    H0: E[d_t] = 0, where d_t = e_benchmark^2 - e_model^2.
    H1 (one-sided): model has lower MSPE than benchmark.

    Variance uses HAC (Newey-West, Bartlett weights) with truncation
    lag = h - 1. For h = 1 no HAC adjustment is needed.

    HLN correction: DM* = DM * sqrt((n + 1 - 2h + h(h-1)/n) / n).
    The corrected statistic is evaluated against t_{n-1}. This mitigates
    over-rejection of H0 in small samples and is the standard reference
    for nowcasting applications (see Harvey, Leybourne & Newbold 1997,
    IJF 13(2), 281–291).

    Returns both the classical DM statistic and the HLN-corrected one.
    The p_value reported is the HLN p-value.
    """
    from scipy import stats

    y  = np.asarray(actuals, dtype=float)
    yb = np.asarray(pred_benchmark, dtype=float)
    ym = np.asarray(pred_model, dtype=float)

    mask = ~np.isnan(y) & ~np.isnan(yb) & ~np.isnan(ym)
    y, yb, ym = y[mask], yb[mask], ym[mask]
    n = len(y)

    if n < 5:
        warnings.warn("Too few observations for reliable DM test (n < 5).", stacklevel=2)
        return {"DM_stat": np.nan, "DM_stat_HLN": np.nan,
                "p_value": np.nan, "n_obs": n,
                "RMSE_benchmark": np.nan, "RMSE_model": np.nan}

    e_b = y - yb
    e_m = y - ym
    d   = e_b**2 - e_m**2

    d_bar = d.mean()
    d_var = np.var(d, ddof=1)

    if h > 1:
        for k in range(1, h):
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            d_var  += 2.0 * (1.0 - k / h) * gamma_k

    dm_stat = d_bar / np.sqrt(d_var / n)

    # Harvey-Leybourne-Newbold (1997) small-sample correction.
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_hln = dm_stat * hln_factor

    # One-sided p-value: H1 says model beats benchmark (d_bar > 0).
    p_value = 1.0 - stats.t.cdf(dm_stat_hln, df=n - 1)

    return {
        "DM_stat":        round(float(dm_stat),     4),
        "DM_stat_HLN":    round(float(dm_stat_hln), 4),
        "p_value":        round(float(p_value),     4),
        "n_obs":          int(n),
        "RMSE_benchmark": round(float(np.sqrt(np.mean(e_b**2))), 6),
        "RMSE_model":     round(float(np.sqrt(np.mean(e_m**2))), 6),
    }


# =============================================================================
# 6. Period classification (Crisis vs Normal)
# =============================================================================

def classify_crisis_periods(
    dates: list,
    crisis_windows: Optional[List[tuple]] = None
) -> pd.Series:
    """
    Label each date as 'crisis' or 'normal' for stratified evaluation.

    Default partition:
        crisis : 2020Q1 – 2021Q2   (n=6)
        normal : 2021Q3 – 2024Q2   (n=12)

    The evaluation sample is fully covered by these two windows. Any date
    that falls outside both windows indicates a mismatch between TEST_START/
    TEST_END and the crisis/normal bounds, and raises a ValueError.
    """
    if crisis_windows is not None:
        labels = []
        for d in dates:
            d_dt = pd.to_datetime(d)
            is_crisis = any(
                pd.to_datetime(s) <= d_dt <= pd.to_datetime(e)
                for s, e in crisis_windows
            )
            labels.append("crisis" if is_crisis else "normal")
        return pd.Series(labels, index=range(len(dates)), name="period_type")

    crisis_block = (pd.to_datetime(CRISIS_START), pd.to_datetime(CRISIS_END))
    normal_block = (pd.to_datetime(NORMAL_START), pd.to_datetime(NORMAL_END))

    labels = []
    for d in dates:
        d_dt = pd.to_datetime(d)
        if crisis_block[0] <= d_dt <= crisis_block[1]:
            labels.append("crisis")
        elif normal_block[0] <= d_dt <= normal_block[1]:
            labels.append("normal")
        else:
            raise ValueError(
                f"Date {d_dt.date()} is outside both crisis "
                f"[{crisis_block[0].date()}, {crisis_block[1].date()}] and "
                f"normal [{normal_block[0].date()}, {normal_block[1].date()}] "
                f"windows. The evaluation sample should be fully covered by "
                f"crisis ∪ normal. Check TEST_START/TEST_END and "
                f"CRISIS_*/NORMAL_* constants in nowcast_utils.py."
            )
    return pd.Series(labels, index=range(len(dates)), name="period_type")


# =============================================================================
# 7. Rolling-origin evaluation scaffold
# =============================================================================

def build_rolling_windows(
    data: pd.DataFrame,
    target_variable: str = None,
    test_start_date: str = None,
    test_end_date: str = None,
    train_start_date: str = None
) -> List[Dict]:
    """
    Build rolling-origin window specifications.

    Each element defines one nowcast date, with the training cutoff one
    quarter before to prevent leakage.
    """
    if target_variable is None:
        target_variable = TARGET_VARIABLE
    if test_start_date is None:
        test_start_date = TEST_START
    if test_end_date is None:
        test_end_date = TEST_END

    test_dates = pd.date_range(test_start_date, test_end_date, freq="3MS")

    windows = []
    for d in test_dates:
        actual_rows = data.loc[
            (data["date"] == d) & (~pd.isna(data[target_variable])),
            target_variable
        ]
        actual_val = actual_rows.values[0] if not actual_rows.empty else np.nan
        train_cutoff = d - pd.DateOffset(months=3)
        train_start = (pd.to_datetime(train_start_date)
                       if train_start_date is not None
                       else pd.to_datetime(TRAIN_START))

        windows.append({
            "nowcast_date": d.strftime("%Y-%m-%d"),
            "train_cutoff": train_cutoff.strftime("%Y-%m-%d"),
            "train_start":  train_start.strftime("%Y-%m-%d"),
            "actual_value": actual_val,
        })

    return windows
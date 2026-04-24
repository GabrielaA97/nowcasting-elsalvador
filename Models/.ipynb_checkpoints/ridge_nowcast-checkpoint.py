# =============================================================================
# ridge_nowcast.py
# Purpose : Nowcast quarterly GDP for El Salvador using Ridge Regression
#           (L2-regularized OLS) with time-series cross-validation.
#           Implements a rolling-origin pseudo-real-time evaluation across
#           artificial data vintages.
# Author  : Gabriela Aquino
# Depends : nowcast_utils.py (in the same directory)
# Input   : data_tf.csv         — transformed panel (from prepare_data_v2.R)
#           meta_data_v2.csv    — variable metadata
# Output  : results/ridge_performance.csv
#           results/ridge_predictions.csv
#           results/ridge_feature_importance.csv
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

# Import shared utility functions
from nowcast_utils import (
    gen_lagged_data,
    flatten_data,
    mean_fill_dataset,
    compute_metrics,
    diebold_mariano_test,
    classify_crisis_periods,
    build_rolling_windows
)

warnings.filterwarnings("ignore")

# ── 0. Reproducibility and configuration ──────────────────────────────────────

RANDOM_SEED       = 42
np.random.seed(RANDOM_SEED)

# File paths
PATH_DATA         = "data_tf.csv"
PATH_META         = "meta_data_v2.csv"
PATH_OUT          = "results/"

# Model configuration
TARGET_VARIABLE   = "GDP"

# Nowcasting vintages: negative = months before quarter-end (real nowcast),
# zero = at quarter-end, positive = months after quarter-end (backcast)
LAGS              = list(range(-2, 3))   # [-2, -1, 0, 1, 2]

# Training and evaluation window
TRAIN_START       = "2005-01-01"
TEST_START        = "2019-03-01"   # Must be the same across ALL models for comparability
TEST_END          = "2024-06-01"   # Same for all models

# Number of lag copies of each predictor to include in the flattened dataset
N_LAGS_FEATURES   = 4             # Each predictor gets contemporaneous + 4 lags

# Ridge alpha (regularization strength) grid
# Extended to cover the range typical for p >> n nowcasting applications
ALPHAS            = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

# Time-series cross-validation splits for hyperparameter selection
N_CV_SPLITS       = 5

# ── 1. Load data ──────────────────────────────────────────────────────────────

print("Loading data...")
data     = pd.read_csv(PATH_DATA, parse_dates=["date"])
metadata = pd.read_csv(PATH_META)

# Sort chronologically — critical for all lag operations
data = data.sort_values("date").reset_index(drop=True)

print(f"Dataset: {data.shape[0]} rows x {data.shape[1]} columns")
print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
print(f"Quarterly GDP observations: {data[TARGET_VARIABLE].notna().sum()}")

# Verify test period GDP observations are available
test_actuals_check = data.loc[
    (data["date"] >= TEST_START) & (data["date"] <= TEST_END) &
    data[TARGET_VARIABLE].notna()
]
print(f"GDP observations in test period: {len(test_actuals_check)}")

# ── 2. Rolling-origin evaluation loop ─────────────────────────────────────────
#
# TEMPORAL VALIDITY AUDIT:
# ✓ No leakage: training always ends at date - 3 months (previous quarter).
# ✓ No look-ahead: gen_lagged_data masks observations after publication cutoff.
# ✓ No CV leakage: TimeSeriesSplit ensures CV folds are time-ordered.
# ✓ No imputation leakage: mean_fill_dataset uses training-period means only.

print("\nStarting rolling-origin evaluation...")

# Initialise prediction storage: one list per lag
pred_dict = {lag: [] for lag in LAGS}
dates     = []        # Nowcast dates in the test period
actuals   = []        # Actual GDP values for each test date
lambdas   = []        # Selected regularization strength at each date
feature_importance_last = None  # Feature importances from the final window

# Get all quarterly dates in the test period
test_quarterly_dates = (
    pd.date_range(TEST_START, TEST_END, freq="3MS")
    .strftime("%Y-%m-%d")
    .tolist()
)

for nowcast_date in test_quarterly_dates:
    print(f"\n{'='*60}")
    print(f"Nowcasting quarter: {nowcast_date}")

    # Retrieve the actual GDP value for this quarter
    actual_val = data.loc[
        data["date"].astype(str).str.startswith(nowcast_date[:7]),
        TARGET_VARIABLE
    ]
    if actual_val.empty or actual_val.isna().all():
        print(f"  ⚠ Actual GDP not available for {nowcast_date} — skipping.")
        for lag in LAGS:
            pred_dict[lag].append(np.nan)
        dates.append(nowcast_date)
        actuals.append(np.nan)
        lambdas.append(np.nan)
        continue

    dates.append(nowcast_date)
    actuals.append(float(actual_val.values[0]))

    # ── Step 1: Define training window ──────────────────────────────────────
    # Training ends one full quarter BEFORE the nowcast quarter.
    # This ensures the model never sees the outcome it is predicting.
    train_cutoff = (
        pd.to_datetime(nowcast_date) - pd.DateOffset(months=3)
    ).strftime("%Y-%m-%d")

    train_raw = data.loc[
        (data["date"] >= TRAIN_START) &
        (data["date"] <= train_cutoff)
    ].copy()

    print(f"  Training window: {TRAIN_START} → {train_cutoff} "
          f"({train_raw[TARGET_VARIABLE].notna().sum()} quarterly obs)")

    # ── Step 2: Build training feature matrix ───────────────────────────────
    # Impute training missing values with training-period means (no leakage)
    train_filled = mean_fill_dataset(train_raw, train_raw)

    # Flatten to quarterly observations with lagged predictor columns
    train_flat = flatten_data(train_filled, TARGET_VARIABLE, N_LAGS_FEATURES)

    # Keep only complete rows (dropping early rows where lags are unavailable)
    train_flat = train_flat.loc[
        train_flat["date"].dt.month.isin([3, 6, 9, 12])
    ].dropna(axis=0, how="any").reset_index(drop=True)

    if len(train_flat) < 10:
        print(f"  ⚠ Insufficient training observations ({len(train_flat)}) — skipping.")
        for lag in LAGS:
            pred_dict[lag].append(np.nan)
        lambdas.append(np.nan)
        continue

    X_train = train_flat.drop(columns=["date", TARGET_VARIABLE])
    y_train = train_flat[TARGET_VARIABLE]
    print(f"  Feature matrix: {X_train.shape[0]} obs × {X_train.shape[1]} features")

    # ── Step 3: Fit Ridge with time-series cross-validation ─────────────────
    # TimeSeriesSplit ensures CV folds respect temporal ordering.
    # This eliminates the look-ahead bias of standard k-fold CV.
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    pipe = Pipeline([
        ("scaler", StandardScaler()),       # Standardize features (within training data)
        ("ridge",  RidgeCV(
            alphas=ALPHAS,
            cv=tscv,                         # Time-ordered CV — NOT random k-fold
            scoring="neg_mean_squared_error",
            fit_intercept=True
        ))
    ])
    pipe.fit(X_train, y_train)

    best_alpha = pipe.named_steps["ridge"].alpha_
    lambdas.append(best_alpha)
    print(f"  Selected alpha: {best_alpha:.4f}")

    # ── Step 4: Generate predictions for each vintage ────────────────────────
    for lag in LAGS:
        # Simulate the data vintage available at (nowcast_date + lag months):
        # Mask all observations not yet published according to publication lags
        vintage_data = gen_lagged_data(metadata, data, nowcast_date, lag)

        # Impute missing values using TRAINING MEANS ONLY (no leakage)
        vintage_data = mean_fill_dataset(train_raw, vintage_data)

        # Flatten to quarterly format (same structure as training)
        vintage_flat = flatten_data(vintage_data, TARGET_VARIABLE, N_LAGS_FEATURES)

        # Select only the nowcast quarter's row
        target_row = vintage_flat.loc[
            vintage_flat["date"].astype(str).str.startswith(nowcast_date[:7])
        ]

        if target_row.empty:
            print(f"  ⚠ No feature row for lag={lag} at {nowcast_date}")
            pred_dict[lag].append(np.nan)
            continue

        # Align columns with training feature matrix
        X_test = target_row.drop(columns=["date", TARGET_VARIABLE], errors="ignore")
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

        y_pred = pipe.predict(X_test)[0]
        pred_dict[lag].append(float(y_pred))
        print(f"    lag={lag:+d}: predicted={y_pred:.4f}")

    # Save feature importances from the last (most data-rich) training window
    feature_importance_last = pd.DataFrame({
        "feature":     X_train.columns,
        "coefficient": pipe.named_steps["ridge"].coef_
    }).assign(abs_coef=lambda df: df["coefficient"].abs())

# ── 3. Performance evaluation ─────────────────────────────────────────────────

print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

performance = compute_metrics(actuals, pred_dict, LAGS)
print(performance.to_string(index=False))

# Stratified evaluation: crisis vs. normal periods
period_labels = classify_crisis_periods(dates)

print("\nPerformance by sub-period (lag=0, nowcast):")
results_df = pd.DataFrame({
    "date":       dates,
    "actual":     actuals,
    "prediction": pred_dict[0],
    "period":     period_labels
})
results_df["sq_error"] = (results_df["actual"] - results_df["prediction"]) ** 2
results_df["abs_error"] = np.abs(results_df["actual"] - results_df["prediction"])

for period in ["normal", "crisis"]:
    sub = results_df[results_df["period"] == period]
    if len(sub) == 0:
        continue
    rmse_sub = np.sqrt(sub["sq_error"].mean())
    mae_sub  = sub["abs_error"].mean()
    print(f"  {period.capitalize():8s}: n={len(sub):2d}, RMSE={rmse_sub:.4f}, MAE={mae_sub:.4f}")

# ── 4. Save results ───────────────────────────────────────────────────────────

import os
os.makedirs(PATH_OUT, exist_ok=True)

# Prediction table (actuals + all vintages)
pred_table = pd.DataFrame({
    "date":               dates,
    "actual":             actuals,
    **{f"lag_{lag:+d}":  pred_dict[lag] for lag in LAGS},
    "period_type":        period_labels,
    "selected_alpha":     lambdas
})
pred_table.to_csv(f"{PATH_OUT}ridge_predictions.csv", index=False)

# Performance table
performance.to_csv(f"{PATH_OUT}ridge_performance.csv", index=False)

# Feature importances (last training window)
if feature_importance_last is not None:
    feature_importance_last.sort_values("abs_coef", ascending=False).to_csv(
        f"{PATH_OUT}ridge_feature_importance.csv", index=False
    )

print(f"\nResults saved to {PATH_OUT}")

# ── 5. Visualisation ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("Ridge Regression — El Salvador GDP Nowcasting", fontsize=14, fontweight="bold")

# Panel A: Predictions vs actuals across vintages
ax = axes[0]
ax.plot(pred_table["date"], pred_table["actual"], "k-o", lw=2,
        markersize=5, label="Actual GDP", zorder=5)
vintage_colors = sns.color_palette("YlGnBu", n_colors=len(LAGS))
for i, lag in enumerate(LAGS):
    label = f"lag={lag:+d}" + (" (nowcast)" if lag == 0 else "")
    ax.plot(pred_table["date"], pred_table[f"lag_{lag:+d}"],
            "--o", color=vintage_colors[i], lw=1.2, markersize=3, label=label, alpha=0.8)

# Shade crisis periods
ax.axvspan("2020-03-01", "2021-06-01", alpha=0.12, color="red", label="COVID-19")
ax.axvspan("2008-09-01", "2009-12-01", alpha=0.10, color="orange", label="GFC")
ax.set_ylabel("GDP growth rate")
ax.set_title("Predictions vs Actuals by Vintage")
ax.legend(loc="lower left", fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)

# Panel B: RMSE by vintage
ax = axes[1]
ax.bar(performance["Vintage"].astype(str), performance["RMSE"],
       color=sns.color_palette("YlGnBu", n_colors=len(LAGS)),
       edgecolor="gray", linewidth=0.5)
ax.set_xlabel("Vintage (months relative to quarter-end)")
ax.set_ylabel("RMSE")
ax.set_title("RMSE by Vintage")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PATH_OUT}ridge_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved.")

# Panel C: Top feature importances (last training window)
if feature_importance_last is not None:
    top_n = 15
    top_features = (
        feature_importance_last
        .sort_values("abs_coef", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2a9d8f" if c >= 0 else "#e76f51"
              for c in top_features["coefficient"]]
    ax.barh(top_features["feature"][::-1], top_features["coefficient"][::-1],
            color=colors[::-1], edgecolor="gray", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Ridge Coefficient (last training window)")
    ax.set_title(f"Top {top_n} Features — Ridge Regression\n"
                 f"(Training window ending {train_cutoff})")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PATH_OUT}ridge_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\n✓ Ridge nowcasting complete.")

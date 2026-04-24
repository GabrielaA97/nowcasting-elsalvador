# =============================================================================
# comparison_final.py
# Purpose : Master comparison script — loads all model results and produces:
#           1. Unified predictions table (all models, all vintages)
#           2. Relative RMSE table (vs ARMA = 1) by period and vintage
#           3. Relative MAE  table (vs ARMA = 1) by period and vintage
#           4. Publication-ready summary table (crisis + normal combined)
#           5. Bar chart: RMSE by model at nowcast vintage (lag=0)
#           6. Heatmap: relative RMSE by model x vintage (crisis period)
#           7. Heatmap: relative RMSE by model x vintage (normal period)
#           8. Model ranking table (for paper)
#
# This script is the entry point for all subsequent extended analyses
# (Diebold-Mariano, fan charts, CRPS). Run this first.
#
# Author  : Gabriela Aquino
# Input   : ~/Desktop/ESA-gdp-nowcasting/Models/results/*/
# Output  : ~/Desktop/ESA-gdp-nowcasting/Models/results/comparison/
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# 0. PATHS
# =============================================================================

BASE_DIR  = Path.home() / "Desktop" / "ESA-gdp-nowcasting"
MODEL_DIR = BASE_DIR / "Models"
RES_DIR   = MODEL_DIR / "results"
OUT_DIR   = RES_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Vintages to include in the comparison
# Must match the lag column names in each model's predictions CSV
LAGS = [-2, -1, 0]

# Period definitions
CRISIS_START = "2019-01-01"
CRISIS_END   = "2021-06-01"
NORMAL_START = "2022-01-01"
NORMAL_END   = "2024-06-01"

# =============================================================================
# 2. MODEL REGISTRY
# =============================================================================
# Each entry defines:
#   folder   : subfolder inside results/
#   file     : predictions CSV filename
#   col_map  : mapping from lag integer to column name in that CSV
#   category : "econometric" or "ml"
#   color    : plot color
#
# ADD NEW MODELS HERE when their CSVs are ready.
# The script will skip models whose CSV does not exist yet.

MODEL_REGISTRY = {
    "ARMA": {
        "folder":   "arma",
        "file":     "arma_predictions.csv",
        "col_map":  {-2: "lag_-2", -1: "lag_-1", 0: "lag_+0"},
        "category": "econometric",
        "color":    "#6c757d",
    },
    "MIDAS": {
        "folder":   "midas",
        "file":     "midas_predictions.csv",
        "col_map":  {-2: "lag_-2", -1: "lag_-1", 0: "lag_+0"},
        "category": "econometric",
        "color":    "#2196F3",
    },
    "DFM": {
        "folder":   "dfm",
        "file":     "dfm_predictions.csv",
        "col_map":  {-2: "lag_-2", -1: "lag_-1", 0: "lag_+0"},
        "category": "econometric",
        "color":    "#00BCD4",
    },
    "Ridge": {
        "folder":   "ridge",
        "file":     "ridge_predictions.csv",
        "col_map":  {-2: "lag_-2", -1: "lag_-1", 0: "lag_+0"},
        "category": "ml_linear",
        "color":    "#4CAF50",
    },
    "Lasso": {
        "folder":   "lasso",
        "file":     "lasso_predictions.csv",
        "col_map":  {-2: "lag_-2", -1: "lag_-1", 0: "lag_+0"},
        "category": "ml_linear",
        "color":    "#8BC34A",
    },
    "XGBoost": {
        "folder":   "xgboost",
        "file":     "xgboost_predictions.csv",
        "col_map":  {-2: "lag_-2", -1: "lag_-1", 0: "lag_+0"},
        "category": "ml_nonlinear",
        "color":    "#FF5722",
    },
    "RandomForest": {
        "folder":   "randomforest",
        "file":     "rf_predictions.csv",
        "col_map":  {-2: "lag_-2", -1: "lag_-1", 0: "lag_+0"},
        "category": "ml_nonlinear",
        "color":    "#FF9800",
    },
}

# Display names for tables and plots
DISPLAY_NAMES = {
    "ARMA":         "ARMA",
    "MIDAS":        "MIDAS",
    "DFM":          "DFM",
    "Ridge":        "Ridge",
    "Lasso":        "Lasso",
    "XGBoost":      "XGBoost",
    "RandomForest": "Random Forest",
}

# =============================================================================
# 3. LOAD ALL MODEL PREDICTIONS
# =============================================================================

print("Loading model predictions...")
print()

all_preds  = {}   # model_name -> dict{lag -> Series of predictions}
actuals    = None
dates      = None
periods    = None

for model_name, cfg in MODEL_REGISTRY.items():
    csv_path = RES_DIR / cfg["folder"] / cfg["file"]
    if not csv_path.exists():
        print(f"  [{model_name}] NOT FOUND — skipping: {csv_path}")
        continue

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Load actuals from the first model that has them
    if actuals is None:
        actuals = df["actual"].values
        dates   = df["date"].values
        periods = df["period_type"].values if "period_type" in df.columns else None
        print(f"  [{model_name}] Loaded — {len(df)} quarters, actuals extracted")
    else:
        print(f"  [{model_name}] Loaded — {len(df)} quarters")

    # Extract predictions for each vintage
    lag_preds = {}
    for lag, col_name in cfg["col_map"].items():
        if col_name in df.columns:
            lag_preds[lag] = df[col_name].values
        else:
            # Try alternative column name formats
            alt_names = [
                f"lag_{lag}", f"lag_{'+' if lag>=0 else ''}{lag}",
                f"lag_+{abs(lag)}" if lag >= 0 else f"lag_{lag}"
            ]
            found = False
            for alt in alt_names:
                if alt in df.columns:
                    lag_preds[lag] = df[alt].values
                    found = True
                    break
            if not found:
                print(f"    WARNING: column for lag={lag} not found in {cfg['file']}")
                lag_preds[lag] = np.full(len(df), np.nan)

    all_preds[model_name] = lag_preds

loaded_models = list(all_preds.keys())
print(f"\nModels loaded: {loaded_models}")
print(f"Models missing: {[m for m in MODEL_REGISTRY if m not in loaded_models]}")

if not loaded_models:
    raise RuntimeError("No model CSVs found. Check paths in MODEL_REGISTRY.")

if actuals is None:
    raise RuntimeError("Could not load actuals. Check that at least one CSV has an 'actual' column.")

# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def _rmse(y, yhat):
    y, yhat = np.array(y, float), np.array(yhat, float)
    m = ~np.isnan(y) & ~np.isnan(yhat)
    return float(np.sqrt(np.mean((y[m] - yhat[m])**2))) if m.sum() > 0 else np.nan

def _mae(y, yhat):
    y, yhat = np.array(y, float), np.array(yhat, float)
    m = ~np.isnan(y) & ~np.isnan(yhat)
    return float(np.mean(np.abs(y[m] - yhat[m]))) if m.sum() > 0 else np.nan

def get_period_mask(period_name):
    """Return boolean mask for a given period."""
    dates_dt = pd.to_datetime(dates)
    if period_name == "crisis":
        return (dates_dt >= CRISIS_START) & (dates_dt <= CRISIS_END)
    elif period_name == "normal":
        return (dates_dt >= NORMAL_START) & (dates_dt <= NORMAL_END)
    else:
        return np.ones(len(dates), dtype=bool)

# =============================================================================
# 5. COMPUTE PERFORMANCE TABLES
# =============================================================================

# ---- 5A. Raw RMSE and MAE ---------------------------------------------------
# Compute for each model x vintage x period

rows = []
for period_name in ["full", "crisis", "normal"]:
    mask = get_period_mask(period_name)
    y    = actuals[mask]

    for model_name in loaded_models:
        for lag in LAGS:
            yhat = all_preds[model_name][lag][mask]
            rows.append({
                "model":   model_name,
                "display": DISPLAY_NAMES.get(model_name, model_name),
                "period":  period_name,
                "vintage": lag,
                "RMSE":    round(_rmse(y, yhat), 6),
                "MAE":     round(_mae(y,  yhat), 6),
                "n":       int((~np.isnan(y) & ~np.isnan(yhat)).sum()),
            })

perf_raw = pd.DataFrame(rows)

# ---- 5B. Relative RMSE and MAE (vs ARMA = 1) --------------------------------
if "ARMA" in loaded_models:
    arma_rmse = {}
    arma_mae  = {}
    for period_name in ["full", "crisis", "normal"]:
        mask = get_period_mask(period_name)
        y    = actuals[mask]
        # Use lag=0 as the reference vintage for ARMA (all lags identical for ARMA)
        arma_yhat = all_preds["ARMA"][0][mask]
        arma_rmse[period_name] = _rmse(y, arma_yhat)
        arma_mae[period_name]  = _mae(y,  arma_yhat)

    perf_raw["rel_RMSE"] = perf_raw.apply(
        lambda row: round(row["RMSE"] / arma_rmse[row["period"]], 4)
        if arma_rmse[row["period"]] and not np.isnan(row["RMSE"]) else np.nan,
        axis=1
    )
    perf_raw["rel_MAE"] = perf_raw.apply(
        lambda row: round(row["MAE"] / arma_mae[row["period"]], 4)
        if arma_mae[row["period"]] and not np.isnan(row["MAE"]) else np.nan,
        axis=1
    )
    print("\nARMA RMSE benchmark:")
    for p, v in arma_rmse.items():
        print(f"  {p}: {v:.6f}")
else:
    print("\nWARNING: ARMA not loaded. Relative RMSE cannot be computed.")
    perf_raw["rel_RMSE"] = np.nan
    perf_raw["rel_MAE"]  = np.nan

# ---- 5C. Pivot tables for paper ---------------------------------------------

def make_pivot_table(metric, period):
    """Create model x vintage pivot table for a given metric and period."""
    sub = perf_raw[perf_raw["period"] == period].copy()
    pivot = sub.pivot(index="display", columns="vintage", values=metric)
    # Rename columns to match paper format
    col_rename = {-2: "lag=-2", -1: "lag=-1", 0: "lag=0 (nowcast)"}
    pivot.columns = [col_rename.get(c, str(c)) for c in pivot.columns]
    # Add average across vintages
    pivot["Average"] = pivot.mean(axis=1)
    # Sort by average (best first)
    pivot = pivot.sort_values("Average")
    return pivot.round(4)

# Relative RMSE tables (paper format)
tbl_rmse_crisis = make_pivot_table("rel_RMSE", "crisis")
tbl_rmse_normal = make_pivot_table("rel_RMSE", "normal")
tbl_rmse_full   = make_pivot_table("rel_RMSE", "full")

# Relative MAE tables
tbl_mae_crisis  = make_pivot_table("rel_MAE", "crisis")
tbl_mae_normal  = make_pivot_table("rel_MAE", "normal")

print("\n" + "="*70)
print("RELATIVE RMSE (ARMA = 1) — CRISIS PERIOD (2019Q1-2021Q2)")
print("Values < 1 = better than ARMA benchmark")
print("="*70)
print(tbl_rmse_crisis.to_string())

print("\n" + "="*70)
print("RELATIVE RMSE (ARMA = 1) — NON-CRISIS PERIOD (2022Q1-2024Q2)")
print("="*70)
print(tbl_rmse_normal.to_string())

print("\n" + "="*70)
print("RELATIVE RMSE (ARMA = 1) — FULL PERIOD (2019Q1-2024Q2)")
print("="*70)
print(tbl_rmse_full.to_string())

# =============================================================================
# 6. SAVE TABLES
# =============================================================================

perf_raw.to_csv(OUT_DIR / "comparison_raw.csv", index=False)
tbl_rmse_crisis.to_csv(OUT_DIR / "rel_rmse_crisis.csv")
tbl_rmse_normal.to_csv(OUT_DIR / "rel_rmse_normal.csv")
tbl_rmse_full.to_csv(OUT_DIR   / "rel_rmse_full.csv")
tbl_mae_crisis.to_csv(OUT_DIR  / "rel_mae_crisis.csv")
tbl_mae_normal.to_csv(OUT_DIR  / "rel_mae_normal.csv")

# Unified predictions CSV (all models, all vintages, all dates)
unified = pd.DataFrame({"date": dates, "actual": actuals, "period": periods})
for model_name in loaded_models:
    for lag in LAGS:
        col = f"{DISPLAY_NAMES.get(model_name, model_name)}_lag{lag}"
        unified[col] = all_preds[model_name][lag]
unified.to_csv(OUT_DIR / "unified_predictions.csv", index=False)

print(f"\nCSVs saved to: {OUT_DIR}")

# =============================================================================
# 7. PLOTS
# =============================================================================

# ---- Plot settings ----------------------------------------------------------
MODEL_COLORS = {
    m: MODEL_REGISTRY[m]["color"]
    for m in loaded_models
}
VINTAGE_LABELS = {-2: "lag=-2", -1: "lag=-1", 0: "lag=0\n(nowcast)"}

def format_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ---- PLOT 1: RMSE bar chart at nowcast (lag=0), crisis vs normal -----------

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle(
    "Relative RMSE vs ARMA Benchmark (lag=0, nowcast vintage)\n"
    "Values below 1.0 indicate improvement over the ARMA benchmark",
    fontsize=13, fontweight="bold"
)

for ax, period, title in zip(
    axes,
    ["crisis", "normal"],
    ["Crisis Period (2019Q1-2021Q2)", "Non-Crisis Period (2022Q1-2024Q2)"]
):
    sub = perf_raw[(perf_raw["period"] == period) & (perf_raw["vintage"] == 0)]
    sub = sub.sort_values("rel_RMSE")
    colors = [MODEL_COLORS.get(m, "#888888") for m in sub["model"]]

    bars = ax.bar(sub["display"], sub["rel_RMSE"],
                  color=colors, edgecolor="gray", linewidth=0.5)
    ax.axhline(1.0, color="#e63946", lw=1.5, ls="--", label="ARMA = 1.0")

    # Add value labels on bars
    for bar, val in zip(bars, sub["rel_RMSE"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Relative RMSE (ARMA = 1)", fontsize=10)
    ax.set_ylim(0, max(sub["rel_RMSE"].max() * 1.2, 1.3))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "plot1_rmse_bar_crisis_normal.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plot1_rmse_bar_crisis_normal.png")

# ---- PLOT 2: Heatmap — relative RMSE by model x vintage (crisis) -----------

fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(loaded_models) * 0.7 + 1)))
fig.suptitle(
    "Relative RMSE (ARMA = 1) by Model and Vintage\n"
    "Green = better than ARMA | Red = worse than ARMA",
    fontsize=13, fontweight="bold"
)

for ax, period, title in zip(
    axes,
    ["crisis", "normal"],
    ["Crisis Period (2019Q1-2021Q2)", "Non-Crisis Period (2022Q1-2024Q2)"]
):
    sub = perf_raw[perf_raw["period"] == period][
        ["display", "vintage", "rel_RMSE"]
    ].copy()
    sub = sub[sub["vintage"].isin(LAGS)]
    pivot = sub.pivot(index="display", columns="vintage", values="rel_RMSE")
    pivot.columns = [VINTAGE_LABELS.get(c, str(c)) for c in pivot.columns]

    # Sort rows by average relative RMSE
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    # Use diverging colormap centered at 1.0
    vmin = min(0.5, pivot.min().min())
    vmax = max(1.5, pivot.max().max())
    center = 1.0

    sns.heatmap(pivot,
                annot=True, fmt=".3f",
                cmap="RdYlGn_r",
                center=center, vmin=vmin, vmax=vmax,
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Relative RMSE", "shrink": 0.8})
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Vintage")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
fig.savefig(OUT_DIR / "plot2_heatmap_rel_rmse.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plot2_heatmap_rel_rmse.png")

# ---- PLOT 3: Vintage profiles — RMSE across vintages for each model --------
# One line per model showing how relative RMSE changes from lag=-2 to lag=0

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Relative RMSE by Vintage — How Accuracy Improves as Information Arrives\n"
    "Each line shows one model's RMSE relative to ARMA across the 3 vintages",
    fontsize=12, fontweight="bold"
)

for ax, period, title in zip(
    axes,
    ["crisis", "normal"],
    ["Crisis Period (2019Q1-2021Q2)", "Non-Crisis Period (2022Q1-2024Q2)"]
):
    sub = perf_raw[perf_raw["period"] == period]

    for model_name in loaded_models:
        m_sub = sub[sub["model"] == model_name].sort_values("vintage")
        color = MODEL_COLORS.get(model_name, "#888888")
        lw    = 2.5 if model_name in ["Ridge", "MIDAS", "DFM"] else 1.2
        ls    = "-" if model_name in ["Ridge", "MIDAS", "DFM"] else "--"
        ax.plot(
            m_sub["vintage"],
            m_sub["rel_RMSE"],
            color=color, lw=lw, ls=ls,
            marker="o", markersize=6,
            label=DISPLAY_NAMES.get(model_name, model_name)
        )

    ax.axhline(1.0, color="#e63946", lw=1.5, ls=":", label="ARMA = 1.0")
    ax.set_xticks(LAGS)
    ax.set_xticklabels(["lag=-2\n(2 months before)", "lag=-1\n(1 month before)",
                         "lag=0\n(nowcast)"])
    ax.set_xlabel("Vintage")
    ax.set_ylabel("Relative RMSE (ARMA = 1)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "plot3_vintage_profiles.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plot3_vintage_profiles.png")

# ---- PLOT 4: Full-period actual vs predicted (nowcast, top 3 models) -------

fig, ax = plt.subplots(figsize=(14, 6))

dates_dt = pd.to_datetime(dates)

# Full GDP series for context
ax.plot(dates_dt, actuals,
        color="#333333", lw=2, marker="o", markersize=4,
        label="Actual GDP (QoQ)", zorder=5)

# Top models at nowcast vintage
top_models = ["Ridge", "MIDAS", "ARMA"]
for model_name in top_models:
    if model_name not in loaded_models:
        continue
    yhat  = all_preds[model_name][0]
    color = MODEL_COLORS.get(model_name, "#888888")
    lw    = 2.0 if model_name != "ARMA" else 1.2
    ls    = "-" if model_name != "ARMA" else "--"
    ax.plot(dates_dt, yhat,
            color=color, lw=lw, ls=ls,
            label=DISPLAY_NAMES.get(model_name, model_name) + " (lag=0)",
            alpha=0.85)

# Shade crisis and normal windows
ax.axvspan(pd.to_datetime(CRISIS_START), pd.to_datetime(CRISIS_END),
           alpha=0.10, color="#e63946", label="Crisis window")
ax.axvspan(pd.to_datetime(NORMAL_START), pd.to_datetime(NORMAL_END),
           alpha=0.08, color="#2a9d8f", label="Non-crisis window")

ax.set_xlabel("Date")
ax.set_ylabel("GDP growth rate (QoQ)")
ax.set_title(
    "Nowcasting Performance — Top Models at Nowcast Vintage (lag=0)\n"
    "El Salvador | 2019Q1-2024Q2 | Full evaluation period",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=9, ncol=3, loc="lower left")
ax.grid(True, alpha=0.25)
format_xaxis(ax)
plt.tight_layout()
fig.savefig(OUT_DIR / "plot4_actual_vs_predicted.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plot4_actual_vs_predicted.png")

# =============================================================================
# 8. MODEL RANKING TABLE (paper-ready)
# =============================================================================

# Rank models by: (1) crisis RMSE lag=0, (2) normal RMSE lag=0, (3) average
ranking_rows = []
for model_name in loaded_models:
    sub = perf_raw[perf_raw["model"] == model_name]

    crisis_rmse = sub[(sub["period"]=="crisis") & (sub["vintage"]==0)]["rel_RMSE"].values
    normal_rmse = sub[(sub["period"]=="normal") & (sub["vintage"]==0)]["rel_RMSE"].values
    full_rmse   = sub[(sub["period"]=="full")   & (sub["vintage"]==0)]["rel_RMSE"].values
    crisis_avg  = sub[sub["period"]=="crisis"]["rel_RMSE"].mean()
    normal_avg  = sub[sub["period"]=="normal"]["rel_RMSE"].mean()

    ranking_rows.append({
        "Model":              DISPLAY_NAMES.get(model_name, model_name),
        "Category":           MODEL_REGISTRY[model_name]["category"],
        "Crisis RMSE (lag=0)":  round(crisis_rmse[0], 4) if len(crisis_rmse) > 0 else np.nan,
        "Normal RMSE (lag=0)":  round(normal_rmse[0], 4) if len(normal_rmse) > 0 else np.nan,
        "Full RMSE (lag=0)":    round(full_rmse[0],   4) if len(full_rmse)   > 0 else np.nan,
        "Crisis avg (all lags)": round(crisis_avg, 4),
        "Normal avg (all lags)": round(normal_avg, 4),
    })

ranking = pd.DataFrame(ranking_rows).sort_values("Crisis RMSE (lag=0)")
ranking.to_csv(OUT_DIR / "model_ranking.csv", index=False)

print("\n" + "="*70)
print("MODEL RANKING — Relative RMSE vs ARMA (< 1.0 = better than ARMA)")
print("="*70)
print(ranking.to_string(index=False))

# =============================================================================
# DONE
# =============================================================================
print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)
print(f"Output: {OUT_DIR}")
print()
print("CSVs:")
print("  comparison_raw.csv       — all models x periods x vintages (raw)")
print("  unified_predictions.csv  — all model predictions in one file")
print("  rel_rmse_crisis.csv      — relative RMSE table, crisis period")
print("  rel_rmse_normal.csv      — relative RMSE table, normal period")
print("  rel_rmse_full.csv        — relative RMSE table, full period")
print("  rel_mae_crisis.csv       — relative MAE table, crisis period")
print("  rel_mae_normal.csv       — relative MAE table, normal period")
print("  model_ranking.csv        — summary ranking table")
print()
print("Plots:")
print("  plot1_rmse_bar_crisis_normal.png  — bar chart RMSE by model")
print("  plot2_heatmap_rel_rmse.png        — heatmap model x vintage")
print("  plot3_vintage_profiles.png        — RMSE improvement across vintages")
print("  plot4_actual_vs_predicted.png     — actual vs predicted top models")
print()
print("NEXT STEPS:")
print("  1. Run diebold_mariano_final.py   — significance tests")
print("  2. Run fanchart_final.py          — prediction intervals")
print("  3. Run crps_final.py              — density forecast evaluation")
print("  4. Run shap_final.py              — SHAP for XGBoost")

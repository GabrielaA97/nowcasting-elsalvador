# =============================================================================
# data_preparation.py -- Raw data -> data_tf.csv
# =============================================================================
# Transforms SLV_nowcasting_data_mq.csv into data_tf.csv, ready for all
# nowcasting models (ARMA, Ridge, Lasso, DT, XGBoost, MLP, LSTM, DFM, MIDAS).
#
# Transformation rules (based on metadata freq + economic type):
#
#   Monthly indices & flows  -> m-o-m growth rate: (x_t / x_{t-1}) - 1
#   Quarterly indices & flows-> q-o-q growth rate: (x_t / x_{t-1}) - 1
#                               where lag = previous quarter (not prev month)
#   Interest rates & %rates  -> kept in LEVELS (no transformation)
#                               TIP_30, TIP_180, TPR1, EFFR_US, MTB_6,
#                               UNEM_US, UNEM_US_LA
#
# Why this matters:
#   The original growth_rate() in R applied (x/lag(x))-1 to ALL variables
#   including interest rates. A rate going from 3.05% to 3.12% should NOT
#   be expressed as a +2.3% growth -- it should stay as 3.12%.
#   Applying growth rates to rates distorts the signal for models.
#
# Target variable for the paper:
#   IVAE_TOT m-o-m growth rate (monthly, 12 values per year)
#   This is the high-frequency proxy for economic activity in El Salvador.
#   Reference: Amaya & Rivas (2021), BCR (2024).
#
# Usage:
#   python data_preparation.py
#   -> produces data_tf.csv in the same folder
# =============================================================================

import pandas as pd
import numpy as np

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

INPUT_FILE  = "SLV_nowcasting_data_mq.csv"
OUTPUT_FILE = "data_tf.csv"
METADATA_FILE = "meta_data_V2.csv"

# Variables that are RATES or PERCENTAGES -- keep in levels, do NOT growth-rate
# Economic rationale: these are already expressed as % (e.g. 3.05% interest rate)
# Taking (3.12/3.05)-1 = +2.3% is meaningless -- the level IS the information
LEVEL_VARS = {
    "TIP_30",       # Tasa de interes pasiva 30 dias (El Salvador)
    "TIP_180",      # Tasa de interes pasiva 180 dias (El Salvador)
    "TPR1",         # Tasa de prestamos a un ano (El Salvador)
    "EFFR_US",      # Effective Federal Funds Rate (USA)
    "MTB_6",        # 6-Month Treasury Bill Rate (USA)
    "UNEM_US",      # Unemployment Rate USA (%)
    "UNEM_US_LA",   # Unemployment Rate Hispanic/Latino USA (%)
}

EPSILON = 1e-8  # avoid division by zero


# =============================================================================
# 2. LOAD DATA
# =============================================================================

def load_raw_data():
    data = (
        pd.read_csv(INPUT_FILE, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    metadata = pd.read_csv(METADATA_FILE)

    print(f"Raw data : {len(data)} rows x {len(data.columns)-1} variables")
    print(f"Metadata : {len(metadata)} variables")
    print(f"Date range: {data.date.min().date()} -> {data.date.max().date()}")
    return data, metadata


# =============================================================================
# 3. TRANSFORMATION FUNCTIONS
# =============================================================================

def growth_rate_monthly(series: pd.Series) -> pd.Series:
    """
    Month-on-month growth rate for a monthly variable.
    (x_t / x_{t-1}) - 1
    Applied only to non-NaN observations; NaN months stay NaN.
    """
    # Work only with non-null values, compute lag within that subset
    non_null = series.dropna()
    gr = (non_null / non_null.shift(1)) - 1
    # First observation becomes NaN (no lag available)
    return series.copy().where(series.isna(), gr.reindex(series.index))


def growth_rate_quarterly(series: pd.Series) -> pd.Series:
    """
    Quarter-on-quarter growth rate for a quarterly variable.
    The variable has NaN in non-quarter-end months.
    drop_na -> compute lag (= previous quarter) -> rejoin to full index.

    This ensures the lag of June is March (q-o-q), not the previous month.
    """
    non_null = series.dropna()
    gr = (non_null / non_null.shift(1)) - 1
    # Rejoin to full monthly index -- non-quarter months remain NaN
    return gr.reindex(series.index)


def keep_level(series: pd.Series) -> pd.Series:
    """
    No transformation -- return series as-is.
    Used for interest rates and percentage variables.
    """
    return series.copy()


# =============================================================================
# 4. APPLY TRANSFORMATIONS
# =============================================================================

def transform_data(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the correct transformation to each variable based on:
      1. Whether it is a level/rate variable (LEVEL_VARS set)
      2. Its frequency (freq column in metadata): 'm' or 'q'
    """
    # Build freq lookup from metadata
    freq_map = dict(zip(metadata["series"], metadata["freq"]))

    result = data[["date"]].copy()

    stats = {"monthly_growth": [], "quarterly_growth": [], "level": [], "skipped": []}

    for col in data.columns[1:]:   # skip date
        series = data[col].copy()

        # Convert to numeric (raw CSV may have string values)
        series = pd.to_numeric(series, errors="coerce")
        # Replace inf with NaN
        series.replace([np.inf, -np.inf], np.nan, inplace=True)

        if col in LEVEL_VARS:
            # Interest rates / percentage rates -> keep level
            result[col] = keep_level(series)
            stats["level"].append(col)

        elif freq_map.get(col) == "q":
            # Quarterly variable -> q-o-q growth rate
            result[col] = growth_rate_quarterly(series)
            stats["quarterly_growth"].append(col)

        elif freq_map.get(col) == "m":
            # Monthly variable -> m-o-m growth rate
            result[col] = growth_rate_monthly(series)
            stats["monthly_growth"].append(col)

        else:
            # Variable not in metadata -- apply monthly growth as default
            # and flag it
            result[col] = growth_rate_monthly(series)
            stats["skipped"].append(col)

    return result, stats


# =============================================================================
# 5. VALIDATION
# =============================================================================

def validate_output(data_raw: pd.DataFrame, data_tf: pd.DataFrame):
    """
    Run sanity checks on the transformed dataset.
    Prints a clear report so you can verify before using in models.
    """
    print("\n" + "="*60)
    print("  VALIDATION REPORT")
    print("="*60)

    target = "IVAE_TOT"

    # Check 1: IVAE_TOT is monthly and has reasonable growth rates
    ivae = data_tf[target].dropna()
    print(f"\n[1] Target variable: {target}")
    print(f"    Non-null obs : {len(ivae)} (expected ~233)")
    print(f"    Mean m-o-m   : {ivae.mean()*100:.3f}%")
    print(f"    Std m-o-m    : {ivae.std()*100:.3f}%")
    print(f"    Min m-o-m    : {ivae.min()*100:.3f}%")
    print(f"    Max m-o-m    : {ivae.max()*100:.3f}%")

    # Show COVID period
    covid = data_tf.loc[
        (data_tf.date >= "2020-03-01") & (data_tf.date <= "2020-09-01"),
        ["date", target]
    ]
    print(f"\n    COVID period (Mar-Sep 2020):")
    for _, row in covid.iterrows():
        val = f"{row[target]*100:.2f}%" if pd.notna(row[target]) else "NaN"
        print(f"      {row['date'].date()}: {val}")

    # Check 2: GDP quarterly structure preserved
    gdp = data_tf["GDP"].dropna()
    print(f"\n[2] GDP (quarterly)")
    print(f"    Non-null obs : {len(gdp)} (expected ~77)")
    print(f"    Sample values:")
    for _, row in data_tf.loc[data_tf["GDP"].notna()].head(4).iterrows():
        print(f"      {row['date'].date()}: {row['GDP']*100:.3f}%")

    # Check 3: Interest rates kept in levels (NOT growth rates)
    print(f"\n[3] Interest rates (should be in levels, NOT growth rates)")
    for var in ["TIP_30", "TIP_180", "TPR1", "EFFR_US"]:
        if var in data_tf.columns:
            sample = data_tf.loc[data_tf[var].notna(), var].head(3).values
            print(f"    {var:10}: {[round(v,4) for v in sample]}  <- should look like 3-7%")

    # Check 4: No all-NaN columns
    all_nan = [c for c in data_tf.columns[1:] if data_tf[c].isna().all()]
    if all_nan:
        print(f"\n[4] WARNING: All-NaN columns: {all_nan}")
    else:
        print(f"\n[4] No all-NaN columns -- OK")

    # Check 5: NaN count summary
    nan_pct = data_tf.iloc[:, 1:].isna().mean() * 100
    print(f"\n[5] Missing value summary:")
    print(f"    Overall missing: {nan_pct.mean():.1f}%")
    high_nan = nan_pct[nan_pct > 50]
    if len(high_nan) > 0:
        print(f"    Variables >50% missing:")
        for v, p in high_nan.items():
            print(f"      {v}: {p:.1f}%")
    else:
        print(f"    No variables with >50% missing -- OK")

    print("\n" + "="*60)


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    print("data_preparation.py -- El Salvador Nowcasting Data Pipeline")
    print("="*60)

    # Load
    data_raw, metadata = load_raw_data()

    # Transform
    print("\nApplying transformations...")
    data_tf, stats = transform_data(data_raw, metadata)

    # Report transformations applied
    print(f"\nTransformation summary:")
    print(f"  m-o-m growth rate : {len(stats['monthly_growth'])} variables")
    print(f"  q-o-q growth rate : {len(stats['quarterly_growth'])} variables")
    print(f"  Kept in levels    : {len(stats['level'])} variables -> {stats['level']}")
    if stats["skipped"]:
        print(f"  Not in metadata   : {len(stats['skipped'])} -> {stats['skipped']}")

    # Validate
    validate_output(data_raw, data_tf)

    # Save
    data_tf.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Shape: {data_tf.shape[0]} rows x {data_tf.shape[1]} columns")
    print("\ndata_tf.csv is ready to load in all models.")
    print("Target variable for models: IVAE_TOT (m-o-m growth rate)")


if __name__ == "__main__":
    main()

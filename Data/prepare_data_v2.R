# =============================================================================
# prepare_data_v2.R
# Purpose : Transform raw nowcasting panel for El Salvador into stationary,
#           model-ready features. Handles mixed-frequency, variable types,
#           and publication lags correctly.
# Author  : Gabriela Aquino
# Input   : SLV_nowcasting_data_mq.csv  — raw mixed-frequency panel
#           meta_data_v2.csv             — variable metadata (freq, blocks, lags)
# Output  : data_tf.csv                  — transformed panel (monthly frequency)
# Notes   :
#   - GDP and sub-components are quarterly (NaN in non-quarter-end months).
#     Their growth rate is quarter-on-quarter (QoQ).
#   - Monthly series are transformed to month-on-month (MoM) growth rates,
#     EXCEPT level/rate variables (interest rates, survey indices) which are
#     left in first differences or kept as levels.
#   - IMP_CO is read as character and must be coerced to numeric first.
#   - Interest rate variables (TIP_30, TIP_180, TPR1, EFFR_US, MTB_6) are
#     expressed as absolute first differences (Δr_t = r_t - r_{t-1}).
#   - Survey/confidence indices (CE_*) are already stationary and are kept
#     in levels. Their stationarity should be validated with ADF/KPSS.
#
# IMPORTANT — namespace conflict:
#   stats::filter() and dplyr::filter() share the same name. Loading
#   tidyverse last usually masks stats::filter, but other packages (e.g.
#   imputeTS, forecast, midasr) reload stats and can un-mask it mid-session.
#   Solution: use dplyr::filter() explicitly everywhere in this script.
# =============================================================================

library(tidyverse)   # loads dplyr, readr, lubridate, etc.

# Force dplyr::filter to be the active filter in this session.
# This prevents the "argument x is missing" error caused by stats::filter
# being loaded over dplyr::filter by other packages.
filter <- dplyr::filter

# ── 0. Configuration ──────────────────────────────────────────────────────────

# ── Paths — edit only this block ──────────────────────────────────────────────
PATH_DATA <- "~/Desktop/ESA-gdp-nowcasting/Data/SLV_nowcasting_data_mq.csv"
PATH_META <- "~/Desktop/ESA-gdp-nowcasting/Data/meta_data_v2.csv"

# Output goes to the same Data folder so all model scripts can find it
PATH_OUT  <- "~/Desktop/ESA-gdp-nowcasting/Data/data_tf.csv"

# Variables that should NOT be growth-rate transformed.
# Rationale: interest rates are level stationary (or near-stationary) and
# growth rates of near-zero interest rates are economically meaningless
# (e.g., a rate moving from 0.01% to 0.02% is a 100% "growth rate").
RATE_VARS <- c("TIP_30", "TIP_180", "TPR1", "EFFR_US", "MTB_6")

# Survey/confidence indices: already stationary levels, keep as-is.
# If ADF tests reject stationarity, switch these to first differences.
SURVEY_VARS <- c(
  "CE_TOT_VV", "CE_TOT_VI", "CE_IND_VV", "CE_IND_VI",
  "CE_CON_AG", "CE_CON_VI", "CE_COM_VV", "CE_COM_VI",
  "CE_SER_VV", "CE_SER_VI"
)

# Numerical precision guard: avoid dividing by values near zero
EPSILON <- 1e-8

# ── 1. Load data ──────────────────────────────────────────────────────────────

cat("Loading data...\n")

metadata <- read_csv(PATH_META, show_col_types = FALSE)

raw <- read_csv(PATH_DATA, show_col_types = FALSE) %>%
  # Ensure IMP_CO (stored as character due to formatting) is numeric
  mutate(IMP_CO = as.numeric(IMP_CO)) %>%
  # Sort chronologically (critical for lag operations)
  arrange(date)

cat(sprintf("Data loaded: %d rows x %d columns\n", nrow(raw), ncol(raw)))
cat(sprintf("Date range: %s to %s\n", min(raw$date), max(raw$date)))

# Sanity check: verify GDP is observed only at quarter-end months
# Using base R format() instead of lubridate::month() to avoid namespace issues
gdp_months <- raw %>%
  dplyr::filter(!is.na(GDP)) %>%
  dplyr::mutate(month = as.integer(format(as.Date(date), "%m"))) %>%
  dplyr::pull(month) %>%
  unique() %>%
  sort()

stopifnot("GDP should only appear in months 3, 6, 9, 12" =
            all(gdp_months %in% c(3, 6, 9, 12)))

cat(sprintf("GDP non-null observations: %d\n", sum(!is.na(raw$GDP))))

# ── 2. Transformation functions ───────────────────────────────────────────────

#' Compute period-on-period growth rates for a single series.
#'
#' For quarterly series (like GDP), the lag is computed on the compressed
#' non-NA series so that the denominator is always the previous quarter,
#' not the previous month.
#'
#' @param series  Numeric vector (may contain NA for non-quarterly rows).
#' @param dates   Date vector, same length as series.
#' @param epsilon Small guard against near-zero denominators.
#' @return        Numeric vector of growth rates; first obs is always NA.
compute_growth_rate <- function(series, dates, epsilon = EPSILON) {
  # Identify non-missing observations
  valid_idx  <- which(!is.na(series))
  valid_vals <- series[valid_idx]
  
  # Compute lag on the compressed (non-NA) sequence
  lagged_vals <- dplyr::lag(valid_vals)
  
  # Growth rate: (x_t - x_{t-1}) / |x_{t-1}| to handle negative values
  growth <- if_else(
    abs(lagged_vals) > epsilon,
    (valid_vals / lagged_vals) - 1,
    NA_real_
  )
  
  # Expand back to the full-length vector
  result <- rep(NA_real_, length(series))
  result[valid_idx] <- growth
  return(result)
}

#' Compute first differences for rate/level variables (e.g., interest rates).
#'
#' @param series  Numeric vector (may contain NA).
#' @return        Numeric vector of first differences; first obs is always NA.
compute_first_difference <- function(series) {
  valid_idx  <- which(!is.na(series))
  valid_vals <- series[valid_idx]
  diffs      <- c(NA_real_, diff(valid_vals))
  result     <- rep(NA_real_, length(series))
  result[valid_idx] <- diffs
  return(result)
}

# ── 3. Apply transformations column by column ─────────────────────────────────

cat("Applying transformations...\n")

data_tf <- raw

for (col in colnames(raw)[colnames(raw) != "date"]) {
  
  # --- Identify transformation type for this variable ---
  if (col %in% SURVEY_VARS) {
    # Keep as level; no transformation
    cat(sprintf("  %-20s → level (survey index)\n", col))
    next
  }
  
  if (col %in% RATE_VARS) {
    # Apply first difference to interest rates
    cat(sprintf("  %-20s → first difference (rate variable)\n", col))
    data_tf[[col]] <- compute_first_difference(raw[[col]])
    next
  }
  
  # Default: period-on-period growth rate
  cat(sprintf("  %-20s → growth rate\n", col))
  data_tf[[col]] <- compute_growth_rate(raw[[col]], raw[["date"]])
}

# ── 4. Post-transformation checks ─────────────────────────────────────────────

cat("\nPost-transformation checks:\n")

# Check for remaining Inf values (can occur if a level passes through zero)
inf_counts <- sapply(data_tf[, colnames(data_tf) != "date"],
                     function(x) sum(is.infinite(x)))
if (any(inf_counts > 0)) {
  cat("WARNING: Infinite values detected in:\n")
  print(inf_counts[inf_counts > 0])
  # Replace Inf with NA
  data_tf <- data_tf %>%
    mutate(across(where(is.numeric), ~ if_else(is.infinite(.), NA_real_, .)))
}

# Check GDP growth rates are sensible (between -50% and +50% quarterly)
gdp_growth <- data_tf %>% dplyr::filter(!is.na(GDP)) %>% dplyr::pull(GDP)
out_of_range <- sum(abs(gdp_growth) > 0.5, na.rm = TRUE)
if (out_of_range > 0) {
  cat(sprintf(
    "WARNING: %d GDP growth rate observations outside [-50%%, +50%%] range — review.\n",
    out_of_range
  ))
  cat("Extreme GDP growth values:\n")
  data_tf %>%
    dplyr::filter(!is.na(GDP), abs(GDP) > 0.5) %>%
    dplyr::select(date, GDP) %>%
    print()
}

cat(sprintf("\nFinal dataset: %d rows x %d columns\n",
            nrow(data_tf), ncol(data_tf)))
cat(sprintf("GDP non-null after transformation: %d observations\n",
            sum(!is.na(data_tf$GDP))))

# ── 5. Save output ────────────────────────────────────────────────────────────

write_csv(data_tf, PATH_OUT)
cat(sprintf("\nTransformed data saved to: %s\n", PATH_OUT))

# ── 6. Summary statistics on transformed data ─────────────────────────────────

cat("\nSummary of GDP growth rates (quarterly, transformed):\n")
data_tf %>%
  dplyr::filter(!is.na(GDP)) %>%
  dplyr::summarise(
    n     = n(),
    mean  = mean(GDP, na.rm = TRUE),
    sd    = sd(GDP, na.rm = TRUE),
    min   = min(GDP, na.rm = TRUE),
    max   = max(GDP, na.rm = TRUE),
    p25   = quantile(GDP, 0.25, na.rm = TRUE),
    p75   = quantile(GDP, 0.75, na.rm = TRUE)
  ) %>%
  print()

cat("\n✓ Data preparation complete.\n")

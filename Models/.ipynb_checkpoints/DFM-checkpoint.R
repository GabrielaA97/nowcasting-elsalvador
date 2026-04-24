setwd("/Users/gabrielaaquino/Desktop/real-time-gdp-nowcasting-sv/Models")
source("DFM.R")
# =============================================================================
# DYNAMIC FACTOR MODEL (DFM) -- Monthly IVAE Nowcasting (El Salvador)
# =============================================================================
# Description : Dynamic Factor Model for monthly IVAE_TOT nowcasting.
#               Uses the Kalman filter (nowcastDFM) to handle the ragged
#               edge of mixed-frequency data optimally.
#               Evaluated across 3 vintages and 2 windows -- directly
#               comparable to all Python models (ARMA, Ridge, etc.).
#
# Key methodological points:
#   [1] Target is IVAE_TOT m-o-m growth rate (monthly) -- same as all
#       other models. This ensures full comparability.
#   [2] NAs passed directly to dfm() -- Kalman filter handles missing
#       data. na_mean() is NOT used before estimation (would bias factors).
#   [3] Vintages: -2, -1, 0 (monthly, matching Python models)
#   [4] Two evaluation windows: Crisis (2019M3-2021M6) and
#       Post-Pandemic (2021M9-2024M6)
#   [5] Block structure validated against economic groupings.
#
# References:
#   Banbura & Modugno (2014). JAE.
#   Baumeister, Leiva-Leon & Sims (2022). REStat.
#   nowcastDFM: github.com/dhopp1/nowcastDFM
#
# Author      : Gabriela Aquino
# Last update : 2026
# =============================================================================

# =============================================================================
# 0. LIBRARIES
# =============================================================================

library(tidyverse)
library(nowcastDFM)
library(forecast)    # dm.test
library(car)         # linearHypothesis
library(ggplot2)
library(scales)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DATA_DIR        <- "../Data"
TARGET_VARIABLE <- "IVAE_TOT"       # monthly m-o-m growth rate
TRAIN_START     <- "2005-01-01"
DFM_MAX_ITER    <- 100
ROLLING_DFM     <- FALSE            # TRUE = re-estimate each month (slower)
LAGS            <- c(-2, -1, 0)     # 3 vintages, matching Python models

# Two evaluation windows -- identical to Python models
WINDOWS <- list(
  crisis = list(
    name       = "Crisis (COVID-19)",
    start_date = "2019-03-01",
    end_date   = "2021-06-01"
  ),
  post_pandemic = list(
    name       = "Post-Pandemic Normalisation",
    start_date = "2021-09-01",
    end_date   = "2024-06-01"
  )
)

cat("=======================================================\n")
cat("  DFM -- Monthly IVAE Nowcasting (El Salvador)\n")
cat("=======================================================\n")
cat(sprintf("Target   : %s (m-o-m growth rate)\n", TARGET_VARIABLE))
cat(sprintf("Vintages : %s\n", paste(LAGS, collapse = ", ")))
cat(sprintf("Window 1 : %s\n", WINDOWS$crisis$name))
cat(sprintf("Window 2 : %s\n", WINDOWS$post_pandemic$name))

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

#' Apply publication lags to simulate a real-time data vintage
#'
#' For each variable, sets the most recent observations to NA based on its
#' known publication delay (months_lag in metadata) plus the vintage
#' adjustment (lag). Replicates the ragged-edge structure in real time.
#'
#' @param metadata  Dataframe: series, months_lag
#' @param data      Full monthly panel (date + variables)
#' @param last_date Upper date limit for this vintage (string)
#' @param lag       Integer: -2, -1, or 0
#' @return Dataframe with NAs where data not yet published
gen_lagged_data <- function(metadata, data, last_date, lag) {
  lagged_data <- data %>%
    dplyr::filter(date <= as.Date(last_date))

  for (col in colnames(lagged_data)[-1]) {
    pub_lag <- metadata %>%
      dplyr::filter(series == col) %>%
      pull(months_lag)

    if (length(pub_lag) == 0 || is.na(pub_lag)) next

    condition <- nrow(lagged_data) - pub_lag + lag
    if (condition >= 1 && condition <= nrow(lagged_data)) {
      lagged_data[condition:nrow(lagged_data), col] <- NA
    }
  }

  lagged_data %>% dplyr::filter(!is.na(date))
}


#' Compute RMSE, MAE, R2 for one vintage
#'
#' @param actuals   Numeric vector of realized IVAE values
#' @param forecasts Numeric vector of model predictions
#' @param vintage   Vintage label (-2, -1, or 0)
#' @return One-row dataframe: Vintage, RMSE, MAE, R2, n
compute_metrics <- function(actuals, forecasts, vintage) {
  errors <- actuals - forecasts
  ss_res <- sum(errors^2, na.rm = TRUE)
  ss_tot <- sum((actuals - mean(actuals, na.rm = TRUE))^2, na.rm = TRUE)
  n      <- sum(!is.na(errors))

  data.frame(
    Vintage = vintage,
    RMSE    = round(sqrt(mean(errors^2, na.rm = TRUE)), 6),
    MAE     = round(mean(abs(errors),   na.rm = TRUE), 6),
    R2      = round(ifelse(ss_tot > 0, 1 - ss_res / ss_tot, NA), 6),
    n       = n
  )
}


#' Validate block structure and print summary
validate_blocks <- function(blocks, var_names) {
  cat("\n--- Block Structure ---\n")
  cat(sprintf("Variables: %d | Blocks: %d\n", nrow(blocks), ncol(blocks)))

  for (b in colnames(blocks)) {
    vars_in <- var_names[blocks[[b]] == 1]
    cat(sprintf("  %-12s: %d variables\n", b, length(vars_in)))
  }

  unassigned <- var_names[rowSums(blocks) == 0]
  if (length(unassigned) > 0) {
    cat(sprintf("\n  WARNING: %d unassigned variables: %s\n",
                length(unassigned), paste(unassigned, collapse = ", ")))
  } else {
    cat("  All variables assigned to at least one block.\n")
  }
}

# =============================================================================
# 3. DATA LOADING
# =============================================================================

metadata <- read_csv(
  file.path(DATA_DIR, "meta_data_V2.csv"),
  show_col_types = FALSE
)

data_full <- read_csv(
  file.path(DATA_DIR, "data_tf.csv"),
  show_col_types = FALSE
) %>%
  arrange(date) %>%
  mutate(date = as.Date(date)) %>%
  # Replace Inf/-Inf with NA -- Kalman will handle NAs optimally
  mutate(across(where(is.numeric), ~ifelse(is.infinite(.), NA, .)))

cat(sprintf("\nData loaded  : %d rows x %d variables\n",
            nrow(data_full), ncol(data_full) - 1))
cat(sprintf("Date range   : %s -> %s\n",
            min(data_full$date), max(data_full$date)))
cat(sprintf("Target (IVAE): %d monthly observations\n",
            sum(!is.na(data_full[[TARGET_VARIABLE]]))))

# Quick look at target
cat("\nIVAE_TOT last 6 months:\n")
data_full %>%
  select(date, !!TARGET_VARIABLE) %>%
  drop_na() %>%
  tail(6) %>%
  print()

# =============================================================================
# 4. PANEL CONSTRUCTION
# =============================================================================
# Single panel from TRAIN_START through end of last window.
# NAs are preserved -- Kalman filter handles the ragged edge.

panel <- data_full %>%
  dplyr::filter(date >= as.Date(TRAIN_START),
                date <= as.Date(WINDOWS$post_pandemic$end_date)) %>%
  data.frame()   # nowcastDFM requires data.frame, not tibble

# Replace any remaining Inf
for (col in colnames(panel)) {
  if (is.numeric(panel[[col]])) {
    panel[is.infinite(panel[[col]]), col] <- NA
  }
}

# =============================================================================
# 5. BLOCK STRUCTURE
# =============================================================================
# Block assignments from metadata columns prefixed "block_".
# Economic groupings (adapt labels to your metadata):
#   block_g = General / real activity
#   block_s = Supply / production side
#   block_r = External sector / trade
#   block_l = Fiscal / financial

var_names <- colnames(panel)[-1]

blocks <- metadata %>%
  dplyr::filter(series %in% var_names) %>%
  slice(match(var_names, series)) %>%
  select(starts_with("block_")) %>%
  select(where(~sum(., na.rm = TRUE) > 0)) %>%
  data.frame()

validate_blocks(blocks, var_names)

# =============================================================================
# 6. NOWCAST LOOP -- MONTHLY, ONE FUNCTION PER WINDOW
# =============================================================================

run_dfm_window <- function(window_cfg, output_dfm) {
  cat(sprintf("\n%s\n  Window: %s\n%s\n",
              strrep("=", 55), window_cfg$name, strrep("=", 55)))

  # Generate monthly test dates for this window
  start_dt  <- as.Date(window_cfg$start_date)
  end_dt    <- as.Date(window_cfg$end_date)
  all_months <- seq(start_dt, end_dt, by = "month")

  cat(sprintf("Months  : %d\n", length(all_months)))
  cat(sprintf("Period  : %s -> %s\n", start_dt, end_dt))

  # Pre-allocate results
  results <- data.frame(
    date        = all_months,
    actual      = NA_real_,
    pred_lag_m2 = NA_real_,
    pred_lag_m1 = NA_real_,
    pred_lag_0  = NA_real_
  )

  # DFM already estimated -- passed as argument
  if (is.null(output_dfm)) {
    cat("  ERROR: No DFM model provided.\n")
    return(results)
  }

  for (i in seq_along(all_months)) {
    eval_date <- all_months[i]
    date_str  <- as.character(eval_date)

    # Actual IVAE_TOT for this month
    actual_val <- panel %>%
      dplyr::filter(date == eval_date) %>%
      pull(!!TARGET_VARIABLE)

    if (length(actual_val) == 0 || is.na(actual_val)) next
    results$actual[i] <- actual_val

    # Predict for each vintage
    for (lag in LAGS) {
      # Generate ragged-edge dataset for this vintage
      lagged <- gen_lagged_data(metadata, panel, date_str, lag) %>%
        data.frame()

      # Mask target to prevent leakage
      lagged[lagged$date == eval_date, TARGET_VARIABLE] <- NA

      tryCatch({
        pred_raw <- predict_dfm(lagged, output_dfm)
        pred_val <- pred_raw %>%
          dplyr::filter(date == eval_date) %>%
          pull(!!TARGET_VARIABLE)

        if (length(pred_val) == 1 && !is.na(pred_val)) {
          col_name <- case_when(
            lag == -2 ~ "pred_lag_m2",
            lag == -1 ~ "pred_lag_m1",
            lag ==  0 ~ "pred_lag_0"
          )
          results[i, col_name] <- pred_val
        }
      }, error = function(e) {
        # Skip silently -- NAs will be excluded from metrics
      })
    }

    # Progress print
    cat(sprintf("  %s: actual=%+.2f%%  lag-2=%+.2f%%  lag-1=%+.2f%%  lag 0=%+.2f%%\n",
                date_str,
                actual_val * 100,
                ifelse(is.na(results$pred_lag_m2[i]), NA, results$pred_lag_m2[i] * 100),
                ifelse(is.na(results$pred_lag_m1[i]), NA, results$pred_lag_m1[i] * 100),
                ifelse(is.na(results$pred_lag_0[i]),  NA, results$pred_lag_0[i]  * 100)))
  }

  return(results %>% dplyr::filter(!is.na(actual)))
}

# =============================================================================
# 7. ESTIMATE DFM ONCE -- THEN APPLY TO BOTH WINDOWS
# =============================================================================
# KEY CHANGE: DFM estimated ONCE on full training data (pre-crisis).
# The same model is then used to predict across both windows.
# This is much faster than re-estimating for each month.
# Limitation: parameters fixed at pre-crisis values -- mention in paper.

cat("\n>>> Estimating DFM on full training data (pre-crisis)...\n")
train_full <- panel %>%
  dplyr::filter(date < as.Date(WINDOWS$crisis$start_date))

cat(sprintf("  Training obs: %d\n", nrow(train_full)))

output_dfm_global <- NULL
tryCatch({
  output_dfm_global <- dfm(
    data     = train_full,
    blocks   = blocks,
    max_iter = DFM_MAX_ITER
  )
  cat("  DFM estimated successfully.\n")
}, error = function(e) {
  cat(sprintf("  DFM estimation failed: %s\n", e$message))
})

cat("\n>>> Running Window 1: Crisis (COVID-19)\n")
results_w1 <- run_dfm_window(WINDOWS$crisis, output_dfm_global)

cat("\n>>> Running Window 2: Post-Pandemic\n")
results_w2 <- run_dfm_window(WINDOWS$post_pandemic, output_dfm_global)

# =============================================================================
# 8. PERFORMANCE METRICS
# =============================================================================

compute_window_metrics <- function(results, window_name) {
  lag_cols <- list(
    list(lag = -2, col = "pred_lag_m2"),
    list(lag = -1, col = "pred_lag_m1"),
    list(lag =  0, col = "pred_lag_0")
  )

  rows <- list()
  for (lc in lag_cols) {
    valid <- results %>%
      select(actual, pred = !!lc$col) %>%
      drop_na()
    if (nrow(valid) == 0) next
    m <- compute_metrics(valid$actual, valid$pred, lc$lag)
    m$Window <- window_name
    m$Model  <- "DFM"
    rows[[length(rows) + 1]] <- m
  }
  bind_rows(rows)
}

perf_w1 <- compute_window_metrics(results_w1, "Crisis")
perf_w2 <- compute_window_metrics(results_w2, "Post-Pandemic")

cat("\n=======================================================\n")
cat("  DFM -- Performance by Vintage\n")
cat("=======================================================\n")

cat("\n--- Window 1: Crisis (COVID-19) ---\n")
print(perf_w1[, c("Vintage", "RMSE", "MAE", "R2", "n")])

cat("\n--- Window 2: Post-Pandemic ---\n")
print(perf_w2[, c("Vintage", "RMSE", "MAE", "R2", "n")])

combined <- bind_rows(perf_w1, perf_w2)
cat("\n--- Combined Performance ---\n")
print(combined[, c("Model", "Window", "Vintage", "RMSE", "MAE", "R2", "n")])

cat("\nVintage key:\n")
cat("  -2 -> Month 1 (least information)\n")
cat("  -1 -> Month 2\n")
cat("   0 -> Month 3 / quarter-end (most information)\n")

# =============================================================================
# 9. DIEBOLD-MARIANO TEST (DFM lag 0 vs lag -2)
# =============================================================================

cat("\n--- Diebold-Mariano Test: lag 0 vs lag -2 ---\n")
cat("H0: equal forecast accuracy between lag 0 and lag -2\n\n")

for (w_name in c("Crisis", "Post-Pandemic")) {
  res <- if (w_name == "Crisis") results_w1 else results_w2
  valid <- res %>% select(actual, pred_lag_m2, pred_lag_0) %>% drop_na()
  if (nrow(valid) < 5) next

  e0  <- valid$actual - valid$pred_lag_0
  em2 <- valid$actual - valid$pred_lag_m2

  tryCatch({
    dm <- dm.test(em2, e0, alternative = "two.sided", h = 1)
    cat(sprintf("  %s: DM stat = %.4f  p = %.4f  %s\n",
                w_name,
                dm$statistic,
                dm$p.value,
                ifelse(dm$p.value < 0.05, "** Significant", "Not significant")))
  }, error = function(e) {
    cat(sprintf("  %s: DM test failed (insufficient obs)\n", w_name))
  })
}

# =============================================================================
# 10. FACTOR LOADINGS HEATMAP
# =============================================================================

if (!is.null(output_dfm) && !is.null(output_dfm$params$C)) {
  n_factors  <- ncol(output_dfm$params$C)
  n_vars_act <- min(length(var_names), nrow(output_dfm$params$C))

  loadings_df <- as.data.frame(output_dfm$params$C[1:n_vars_act, ]) %>%
    setNames(paste0("Factor ", seq_len(n_factors))) %>%
    mutate(Variable = var_names[1:n_vars_act]) %>%
    pivot_longer(cols = -Variable,
                 names_to  = "Factor",
                 values_to = "Loading")

  p_load <- ggplot(loadings_df, aes(x = Factor, y = Variable, fill = Loading)) +
    geom_tile(color = "white", linewidth = 0.3) +
    scale_fill_gradient2(
      low      = "#d73027",
      mid      = "white",
      high     = "#2166ac",
      midpoint = 0
    ) +
    labs(
      title    = "DFM -- Factor Loadings",
      subtitle = "Which variables drive each latent factor?",
      caption  = "Red = negative, Blue = positive loading"
    ) +
    theme_minimal(base_size = 9) +
    theme(
      axis.text.y  = element_text(size = 6),
      plot.title   = element_text(face = "bold")
    )
  print(p_load)
}

# =============================================================================
# 11. VISUALIZATIONS
# =============================================================================

plot_dfm_window <- function(results, window_label) {
  results <- results %>% mutate(date = as.Date(date))

  fig <- ggplot(results, aes(x = date)) +
    geom_line(aes(y = actual * 100), color = "black", linewidth = 1.5,
              linetype = "solid") +
    geom_line(aes(y = pred_lag_0  * 100, color = "lag 0"),
              linewidth = 0.9, linetype = "dashed") +
    geom_line(aes(y = pred_lag_m1 * 100, color = "lag -1"),
              linewidth = 0.8, linetype = "dotted", alpha = 0.85) +
    geom_line(aes(y = pred_lag_m2 * 100, color = "lag -2"),
              linewidth = 0.8, linetype = "dotdash", alpha = 0.85) +
    geom_hline(yintercept = 0, color = "gray40", linewidth = 0.5,
               linetype = "dashed") +
    scale_color_manual(
      values = c("lag 0" = "#2166ac", "lag -1" = "#f46d43", "lag -2" = "#abdda4"),
      name   = "Vintage"
    ) +
    scale_x_date(date_labels = "%Y-%m", date_breaks = "3 months") +
    labs(
      title    = sprintf("DFM | %s", window_label),
      subtitle = "IVAE_TOT m-o-m growth rate (%)",
      x        = "Month",
      y        = "m-o-m growth (%)",
      caption  = "Black = realized IVAE_TOT. Dashed = DFM nowcast by vintage."
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x     = element_text(angle = 45, hjust = 1),
      legend.position = "bottom",
      plot.title      = element_text(face = "bold")
    )

  print(fig)
}

plot_dfm_window(results_w1, "Crisis (COVID-19)")
plot_dfm_window(results_w2, "Post-Pandemic Normalisation")

# =============================================================================
# 12. SAVE RESULTS
# =============================================================================

dir.create("outputs", showWarnings = FALSE)

results_w1$Window <- "Crisis"
results_w1$Model  <- "DFM"
results_w2$Window <- "Post-Pandemic"
results_w2$Model  <- "DFM"

all_results <- bind_rows(results_w1, results_w2)

write_csv(all_results, "outputs/dfm_predictions.csv")
write_csv(combined[, c("Model", "Window", "Vintage", "RMSE", "MAE", "R2", "n")],
          "outputs/dfm_performance.csv")

cat("\n=======================================================\n")
cat("  DFM complete.\n")
cat("=======================================================\n")
cat(sprintf("Predictions : outputs/dfm_predictions.csv (%d rows)\n",
            nrow(all_results)))
cat("Performance : outputs/dfm_performance.csv\n")
cat("\nNext: compare all models in compare_models.R\n")

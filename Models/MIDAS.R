# =============================================================================
# MIDAS.R
# MIDAS (Mixed Data Sampling) nowcasting pipeline for El Salvador GDP.
# Single self-contained script — no other MIDAS file needed.
#
# What this script does (in order):
#   1.  Loads and validates data
#   2.  Runs rolling-origin evaluation across 3 vintages [-2,-1,0]
#       - Separate MIDAS model estimated per explanatory variable
#       - Models weighted by INVERSE training RMSE (standard in the model
#         averaging literature; avoids zero weights)
#       - Rolling: model re-estimated at each nowcast origin
#   3.  Computes RMSE and MAE for the full test period
#   4.  Computes RMSE and MAE stratified by period (Crisis / Normal)
#   5.  Saves all CSVs and plots
#
# Author  : Gabriela Aquino
#
# CONSTANTS are aligned with nowcast_utils.py (single source of truth):
#   TRAIN_START  = 2005-01-01
#   TEST_START   = 2020-03-01       (2020Q1 — first test quarter)
#   TEST_END     = 2024-06-01       (2024Q2 — last test quarter)
#   CRISIS       = 2020Q1 – 2021Q2  (n = 6)
#   NORMAL       = 2021Q3 – 2024Q2  (n = 12)
#   GDP_PUB_LAG  = 3 months
#
# Depends : data_tf.csv, meta_data_v2.csv (output of prepare_data_v2.R)
# Input   : ~/Desktop/ESA-gdp-nowcasting/Data/data_tf.csv
#           ~/Desktop/ESA-gdp-nowcasting/Data/meta_data_v2.csv
# Output  : ~/Desktop/ESA-gdp-nowcasting/Models/results/midas/
#
# Packages required:
#   install.packages(c("tidyverse", "midasr", "imputeTS"))
#
# =============================================================================
# HOW MIDAS WORKS IN THIS SCRIPT
# =============================================================================
# MIDAS (Mixed Data Sampling) directly models the relationship between
# high-frequency (monthly) predictors and a low-frequency (quarterly) target.
# Unlike Ridge/Lasso/XGBoost which flatten monthly data into quarterly rows,
# MIDAS uses polynomial lag weights (Almon / nealmon) to aggregate monthly
# information into the quarterly forecast in a parsimonious way.
#
# Implementation:
#   - One MIDAS model estimated per explanatory variable (univariate MIDAS)
#   - Monthly variables : mls(x, 0:3, 3, nealmon) — 3 quarterly lags
#   - Quarterly variables: mls(x, 0:1, 1, nealmon) — 1 lag
#   - Final forecast = weighted average of individual MIDAS predictions
#   - Weights = inverse training RMSE, normalised to sum to 1
#
# LIMITATION ON VINTAGE SENSITIVITY:
#   The model is estimated once per window and applied to all three vintages
#   with mean-filled ragged edges. This dilutes the contrast between
#   vintages (mean-filled months for -2 and -1 look similar to vintage 0).
#   This is standard in empirical MIDAS with mean-imputation, and is
#   documented as a known limitation in the thesis.
#
# TUNING PROTOCOL:
#   - Fully rolling: model re-estimated at each nowcast origin
#   - Training window expands as new data arrives (expanding window)
#   - No holdout needed: weights are determined by in-sample fit per variable
#
# VINTAGE LOGIC:
#   lag = -2 : 2 months before quarter-end — least monthly info available
#   lag = -1 : 1 month before quarter-end
#   lag =  0 : at quarter-end — all monthly data for the quarter available
#   (lag >= +1 excluded: consistent with ARMA, Ridge, Lasso, XGBoost, RF)
# =============================================================================

library(tidyverse)
library(midasr)
library(imputeTS)

# =============================================================================
# 0. PATHS
# =============================================================================

BASE_DIR   <- file.path(path.expand("~"), "Desktop", "ESA-gdp-nowcasting")
DATA_DIR   <- file.path(BASE_DIR, "Data")
MODEL_DIR  <- file.path(BASE_DIR, "Models")
OUT_DIR    <- file.path(MODEL_DIR, "results", "midas")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PATH_DATA  <- file.path(DATA_DIR, "data_tf.csv")
PATH_META  <- file.path(DATA_DIR, "meta_data_v2.csv")

for (p in c(PATH_DATA, PATH_META)) {
  if (!file.exists(p)) {
    stop("File not found: ", p,
         "\nRun prepare_data_v2.R first to generate data_tf.csv.")
  }
}

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# --- FIXED (thesis design — MUST match nowcast_utils.py) ---------------------
TARGET_VARIABLE <- "GDP"
LAGS            <- c(-2L, -1L, 0L)         # only genuine nowcast vintages
TRAIN_START     <- as.Date("2005-01-01")
TEST_START      <- as.Date("2020-03-01")   # 2020Q1
TEST_END        <- as.Date("2024-06-01")   # 2024Q2
CRISIS_START    <- as.Date("2020-03-01")   # 2020Q1
CRISIS_END      <- as.Date("2021-06-01")   # 2021Q2
NORMAL_START    <- as.Date("2021-09-01")   # 2021Q3
NORMAL_END      <- as.Date("2024-06-01")   # 2024Q2

# GDP publication lag in months. Used to set training cutoff explicitly
# (Option A, matches ARMA benchmark and DMF.R).
GDP_PUB_LAG <- 3L

# --- TUNABLE -----------------------------------------------------------------
# Variables selected for MIDAS — subset of the 123 indicators.
# Covers all economic blocks (activity, external, financial, fiscal).
# Rationale for subsetting: MIDAS estimates one model per variable.
# With 123 variables this becomes very slow and numerically unstable.
VARS_MIDAS <- c(
  "GDP",
  # Activity (IVAE family — main coincident indicator)
  "IVAE_TOT", "IVAE_AG", "IVAE_IN", "IVAE_CO", "IVAE_CT",
  "IVAE_IC",  "IVAE_AF", "IVAE_AI", "IVAE_AP", "IVAE_AA",
  # Industrial production and energy
  "IPI", "PRO_ENER", "CON_ENER",
  # Commerce and employment
  "CE_TOT_VV", "CE_TOT_VI", "CE_COM_VV", "CE_COM_VI",
  # Construction
  "CON_APA_CEM",
  # External sector — exports and imports
  "EXP_CP", "EXP_IM", "IMP_IM", "IMP_D_IM",
  # Remittances (key for El Salvador)
  "REM",
  # Financial conditions
  "CTC", "CCO", "CPI", "TIP_30", "TIP_180", "TPR1",
  # Fiscal revenues and expenditure
  "INGT_IVA", "INGT_ISR", "GPC", "GIP",
  # External conditions (US)
  "UNEM_US", "EFFR_US", "IPI_US", "MTB_6"
)

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

# gen_lagged_data: simulate data vintage available at nowcast_date + lag.
# Uses DATE-BASED cutoff (not positional index) to correctly handle data gaps.
gen_lagged_data <- function(metadata, data, nowcast_date, lag) {
  nowcast_dt <- as.Date(nowcast_date)
  
  lagged_data <- data %>%
    dplyr::filter(date <= nowcast_dt)
  
  for (col in colnames(lagged_data)[2:ncol(lagged_data)]) {
    pub_lag_row <- metadata %>%
      dplyr::filter(series == col) %>%
      dplyr::select(months_lag) %>%
      dplyr::pull()
    
    if (length(pub_lag_row) == 0) {
      pub_lag <- 1L
      warning("Variable '", col, "' not found in metadata. Assuming pub_lag = 1.")
    } else {
      pub_lag <- as.integer(pub_lag_row[1])
    }
    
    months_to_subtract <- pub_lag - lag
    cutoff_dt <- nowcast_dt - months(months_to_subtract)
    
    lagged_data[lagged_data$date > cutoff_dt, col] <- NA
  }
  
  return(lagged_data)
}

# classify_period: label dates as crisis / normal.
# The evaluation sample is fully covered by crisis ∪ normal. Any date
# outside both windows indicates a mismatch between TEST_* and
# CRISIS_*/NORMAL_* constants — fail explicitly rather than silently
# assigning a fallback label.
classify_period <- function(date_vec) {
  labels <- dplyr::case_when(
    date_vec >= CRISIS_START & date_vec <= CRISIS_END ~ "crisis",
    date_vec >= NORMAL_START & date_vec <= NORMAL_END ~ "normal",
    TRUE ~ NA_character_
  )
  bad <- which(is.na(labels))
  if (length(bad) > 0) {
    stop(
      "Date(s) outside both crisis [", CRISIS_START, ", ", CRISIS_END,
      "] and normal [", NORMAL_START, ", ", NORMAL_END, "] windows: ",
      paste(date_vec[bad], collapse = ", "),
      ". Check TEST_START/TEST_END and CRISIS/NORMAL constants."
    )
  }
  labels
}

# =============================================================================
# 3. LOAD DATA
# =============================================================================

cat("Loading data...\n")
metadata_full <- read_csv(PATH_META, show_col_types = FALSE)
data_full     <- read_csv(PATH_DATA, show_col_types = FALSE) %>%
  dplyr::mutate(date = as.Date(date)) %>%
  dplyr::arrange(date)

# Validate that all VARS_MIDAS exist in the data
missing_vars <- setdiff(VARS_MIDAS, colnames(data_full))
if (length(missing_vars) > 0) {
  stop("Variables not found in data_tf.csv: ",
       paste(missing_vars, collapse = ", "))
}

# Subset to selected variables
metadata <- metadata_full %>%
  dplyr::filter(series %in% VARS_MIDAS)
data <- data_full %>%
  dplyr::select(date, all_of(VARS_MIDAS))

# Replace infinites with NA
for (col in colnames(data)[2:ncol(data)]) {
  data[[col]][is.infinite(data[[col]])] <- NA
}

cat("  Rows         :", nrow(data), "\n")
cat("  Columns      :", ncol(data), "\n")
cat("  Date range   :", as.character(min(data$date)),
    "to", as.character(max(data$date)), "\n")
cat("  GDP obs      :",
    sum(!is.na(data[[TARGET_VARIABLE]])), "quarterly observations\n")
cat("  MIDAS vars   :", length(VARS_MIDAS) - 1, "(excl. GDP)\n")

# Test dates: quarterly sequence
test_dates <- seq(TEST_START, TEST_END, by = "3 months")
cat("  Test quarters:", length(test_dates), "\n\n")

# =============================================================================
# 4. ROLLING-ORIGIN EVALUATION LOOP
# =============================================================================
# Temporal validity checklist:
#   [ok] No leakage      - training ends GDP_PUB_LAG months before nowcast date
#   [ok] No look-ahead   - gen_lagged_data uses date-based cutoff per variable
#   [ok] Fully rolling   - model re-estimated at EACH nowcast origin
#   [ok] No impute leak  - na_mean() called on training data only;
#                          vintage data filled with training means separately
#   [ok] Target masked   - GDP for nowcast quarter set to NA before prediction

cat("Running rolling-origin MIDAS evaluation...\n")
cat("  Test window  :", as.character(TEST_START), "to", as.character(TEST_END), "\n")
cat("  Vintages     :", paste(LAGS, collapse = ", "), "\n")
cat("  Protocol     : fully rolling (model re-estimated each origin)\n")
cat("  Weighting    : inverse training RMSE (normalised)\n\n")

# Storage
pred_list        <- vector("list", length(LAGS))
names(pred_list) <- as.character(LAGS)
for (lag in LAGS) pred_list[[as.character(lag)]] <- rep(NA_real_, length(test_dates))

actuals_vec <- rep(NA_real_, length(test_dates))

for (i in seq_along(test_dates)) {
  nowcast_date <- test_dates[i]
  
  # ---- Actual GDP for this quarter ------------------------------------------
  # Direct date comparison — both sides are Date, safer than format().
  actual_val <- data %>%
    dplyr::filter(date == nowcast_date) %>%
    dplyr::pull(!!TARGET_VARIABLE)
  
  actuals_vec[i] <- if (length(actual_val) > 0 && !is.na(actual_val[1]))
    actual_val[1] else NA_real_
  
  # ---- Training window -------------------------------------------------------
  # Ends GDP_PUB_LAG months before the nowcast date (Option A).
  # Tied to GDP_PUB_LAG explicitly so changing the publication lag doesn't
  # silently introduce look-ahead.
  train_cutoff <- nowcast_date - months(GDP_PUB_LAG)
  
  train_raw <- data %>%
    dplyr::filter(date >= TRAIN_START & date <= train_cutoff) %>%
    data.frame()
  
  n_q <- sum(!is.na(train_raw[[TARGET_VARIABLE]]))
  
  if (n_q < 20) {
    cat("  [", as.character(nowcast_date), "] Only", n_q,
        "quarterly obs in training — skipping.\n")
    next
  }
  
  # ---- Impute training data with training means only ------------------------
  train_filled <- na_mean(train_raw)
  
  # ---- Quarterly target series for model estimation -------------------------
  y_train <- train_filled %>%
    dplyr::filter(format(date, "%m") %in% c("03", "06", "09", "12")) %>%
    dplyr::pull(!!TARGET_VARIABLE)
  
  if (length(y_train) < 10) {
    cat("  [", as.character(nowcast_date), "] Insufficient quarterly obs — skipping.\n")
    next
  }
  
  # ---- Estimate one MIDAS model per variable --------------------------------
  models   <- list()
  var_cols <- setdiff(colnames(train_filled), c("date", TARGET_VARIABLE))
  
  for (col in var_cols) {
    tryCatch({
      x_monthly <- train_filled[[col]]
      x_qtrly   <- train_filled %>%
        dplyr::filter(format(date, "%m") %in% c("03", "06", "09", "12")) %>%
        dplyr::pull(col)
      
      # Use quarterly specification only for variables recorded quarterly
      is_quarterly <- train_raw %>%
        dplyr::filter(!format(date, "%m") %in% c("03", "06", "09", "12")) %>%
        dplyr::pull(col) %>%
        is.na() %>%
        all()
      
      if (is_quarterly) {
        models[[col]] <- midas_r(
          y_train ~ mls(x_qtrly, 0:1, 1, nealmon),
          start = list(x_qtrly = c(1, -0.5))
        )
      } else {
        models[[col]] <- midas_r(
          y_train ~ mls(x_monthly, 0:3, 3, nealmon),
          start = list(x_monthly = c(1, -0.5))
        )
      }
    }, error = function(e) NULL)
  }
  
  # Remove variables where model estimation failed
  models <- Filter(Negate(is.null), models)
  
  if (length(models) == 0) {
    cat("  [", as.character(nowcast_date), "] All MIDAS models failed — skipping.\n")
    next
  }
  
  # ---- Compute variable weights (INVERSE training RMSE) --------------------
  # Standard model-averaging formulation: w_i ∝ 1 / RMSE_i, renormalised.
  # This is the literature-standard approach (e.g. Bates & Granger 1969;
  # Stock & Watson 2004). It avoids assigning zero weight to the worst model
  # and is more stable numerically than max_RMSE - RMSE_i schemes.
  rmse_in_sample <- sapply(names(models), function(col) {
    fitted_vals <- models[[col]]$fitted.values
    # midasr drops the first observation(s) used for initial lags; align
    actual_vals <- y_train[(length(y_train) - length(fitted_vals) + 1):length(y_train)]
    n_match     <- min(length(fitted_vals), length(actual_vals))
    if (n_match < 2) return(NA_real_)
    sqrt(mean((fitted_vals[1:n_match] - actual_vals[1:n_match])^2))
  })
  
  # Handle any NA RMSE: assign them the worst observed RMSE so they get
  # the lowest weight rather than breaking the normalisation.
  if (any(is.na(rmse_in_sample))) {
    rmse_in_sample[is.na(rmse_in_sample)] <- max(rmse_in_sample, na.rm = TRUE)
  }
  
  # Guard against exactly-zero RMSE (perfect in-sample fit) which would
  # blow up 1/RMSE. Replace with a small floor equal to 1% of the median.
  floor_rmse <- max(1e-12, 0.01 * median(rmse_in_sample))
  rmse_in_sample <- pmax(rmse_in_sample, floor_rmse)
  
  inv_rmse      <- 1 / rmse_in_sample
  weights_final <- inv_rmse / sum(inv_rmse)
  names(weights_final) <- names(models)
  
  # ---- Predict for each vintage --------------------------------------------
  for (lag in LAGS) {
    tryCatch({
      # Simulate vintage: mask data not yet published at this vintage
      vintage_raw <- gen_lagged_data(metadata, data, nowcast_date, lag)
      vintage_raw <- vintage_raw %>% data.frame()
      
      # Mask target variable for the nowcast quarter (never allow leakage)
      vintage_raw[vintage_raw$date == nowcast_date, TARGET_VARIABLE] <- NA
      
      # Impute vintage NAs using training means (no future data)
      train_means <- colMeans(train_filled[, -1], na.rm = TRUE)
      vintage_filled <- vintage_raw
      for (col in names(train_means)) {
        if (col %in% colnames(vintage_filled)) {
          na_idx <- is.na(vintage_filled[[col]])
          vintage_filled[na_idx, col] <- train_means[col]
        }
      }
      
      # Collect predictions from each individual MIDAS model
      preds_per_var <- sapply(names(models), function(col) {
        tryCatch({
          is_quarterly <- train_raw %>%
            dplyr::filter(!format(date, "%m") %in% c("03", "06", "09", "12")) %>%
            dplyr::pull(col) %>%
            is.na() %>%
            all()
          
          if (is_quarterly) {
            x_new <- vintage_filled %>%
              dplyr::filter(format(date, "%m") %in% c("03", "06", "09", "12")) %>%
              dplyr::pull(col)
            fc <- forecast(models[[col]], newdata = list(x_qtrly = x_new))$mean
          } else {
            x_new <- vintage_filled[[col]]
            fc <- forecast(models[[col]], newdata = list(x_monthly = x_new))$mean
          }
          as.numeric(fc[length(fc)])
        }, error = function(e) NA_real_)
      })
      
      # Remove failed predictions before computing weighted average
      valid_idx <- !is.na(preds_per_var)
      if (sum(valid_idx) == 0) {
        pred_list[[as.character(lag)]][i] <- NA_real_
      } else {
        w_valid <- weights_final[valid_idx]
        w_valid <- w_valid / sum(w_valid)   # renormalise if some dropped
        pred_list[[as.character(lag)]][i] <-
          weighted.mean(preds_per_var[valid_idx], w_valid)
      }
    }, error = function(e) {
      pred_list[[as.character(lag)]][i] <<- NA_real_
    })
  }
  
  cat("  [", as.character(nowcast_date), "]",
      "  n_train=",   n_q,
      "  n_models=",  length(models),
      "  n_q=",       length(y_train),
      "\n", sep = "")
}

# =============================================================================
# 5. PERFORMANCE METRICS
# =============================================================================

rmse_fn <- function(actual, predicted) {
  m <- !is.na(actual) & !is.na(predicted)
  if (sum(m) == 0) return(NA_real_)
  sqrt(mean((actual[m] - predicted[m])^2))
}

mae_fn <- function(actual, predicted) {
  m <- !is.na(actual) & !is.na(predicted)
  if (sum(m) == 0) return(NA_real_)
  mean(abs(actual[m] - predicted[m]))
}

period_labels <- classify_period(test_dates)

# ---- Full test period -------------------------------------------------------
perf_full <- data.frame(
  Vintage = LAGS,
  RMSE    = sapply(LAGS, function(lag)
    rmse_fn(actuals_vec, pred_list[[as.character(lag)]])),
  MAE     = sapply(LAGS, function(lag)
    mae_fn(actuals_vec,  pred_list[[as.character(lag)]])),
  n       = sapply(LAGS, function(lag)
    sum(!is.na(pred_list[[as.character(lag)]])))
)

# ---- Stratified: Crisis / Normal --------------------------------------------
strat_rows <- list()
for (period in c("crisis", "normal")) {
  label <- if (period == "crisis") "Crisis (2020Q1-2021Q2)"
  else "Normal (2021Q3-2024Q2)"
  idx   <- which(period_labels == period)
  y_sub <- actuals_vec[idx]
  for (lag in LAGS) {
    yhat <- pred_list[[as.character(lag)]][idx]
    strat_rows[[length(strat_rows) + 1]] <- data.frame(
      Period  = label,
      Vintage = lag,
      n       = sum(!is.na(yhat)),
      RMSE    = rmse_fn(y_sub, yhat),
      MAE     = mae_fn(y_sub,  yhat)
    )
  }
}
perf_strat <- do.call(rbind, strat_rows)

# ---- Print results ----------------------------------------------------------
cat("\n", strrep("=", 60), "\n", sep = "")
cat("FULL TEST PERIOD - MIDAS (2020Q1-2024Q2)\n")
cat(strrep("=", 60), "\n", sep = "")
print(
  perf_full %>%
    dplyr::mutate(
      dplyr::across(where(is.numeric), ~ round(., 6))
    )
)

cat("\n", strrep("=", 60), "\n", sep = "")
cat("STRATIFIED - Crisis vs Normal x all 3 vintages\n")
cat(strrep("=", 60), "\n", sep = "")
perf_strat_print <- perf_strat %>%
  dplyr::mutate(
    RMSE = round(RMSE, 6),
    MAE  = round(MAE, 6)
  )
print(perf_strat_print)

# =============================================================================
# 6. SAVE RESULTS
# =============================================================================

pred_table <- data.frame(date = test_dates, actual = actuals_vec)
for (lag in LAGS) {
  col_name <- paste0("lag_", if (lag < 0) lag else paste0("+", lag))
  pred_table[[col_name]] <- pred_list[[as.character(lag)]]
}
pred_table$period_type <- period_labels

write_csv(pred_table, file.path(OUT_DIR, "midas_predictions.csv"))
write_csv(perf_full,  file.path(OUT_DIR, "midas_performance_full.csv"))
write_csv(perf_strat, file.path(OUT_DIR, "midas_performance_stratified.csv"))

cat("\nCSVs saved to:", OUT_DIR, "\n")

# =============================================================================
# 7. PLOTS
# =============================================================================

library(scales)

LAG_COLORS <- c("-2" = "#a8dadc", "-1" = "#457b9d", "0" = "#1d3557")
LAG_LABELS <- c("-2" = "lag=-2", "-1" = "lag=-1", "0" = "lag=0 (nowcast)")

# GDP full series for context
gdp_full <- data %>%
  dplyr::filter(!is.na(!!sym(TARGET_VARIABLE))) %>%
  dplyr::select(date, GDP = !!sym(TARGET_VARIABLE))

# Predictions in long format
pred_long <- pred_table %>%
  tidyr::pivot_longer(
    cols      = starts_with("lag_"),
    names_to  = "vintage_raw",
    values_to = "prediction"
  ) %>%
  dplyr::mutate(
    vintage = dplyr::case_when(
      vintage_raw == "lag_-2" ~ "-2",
      vintage_raw == "lag_-1" ~ "-1",
      vintage_raw == "lag_+0" ~ "0",
      TRUE ~ vintage_raw
    )
  )

# ---- PLOT A: Crisis period ---------------------------------------------------
crisis_preds <- pred_long %>% dplyr::filter(period_type == "crisis")

p_crisis <- ggplot() +
  annotate("rect",
           xmin = CRISIS_START, xmax = CRISIS_END,
           ymin = -Inf, ymax = Inf,
           fill = "#e63946", alpha = 0.10) +
  geom_line(data = gdp_full,
            aes(x = date, y = GDP),
            color = "#444444", linewidth = 0.8, alpha = 0.7) +
  geom_point(data = gdp_full,
             aes(x = date, y = GDP),
             color = "#444444", size = 1.5, alpha = 0.7) +
  geom_line(data = crisis_preds,
            aes(x = date, y = prediction, color = vintage,
                linewidth = vintage, linetype = vintage)) +
  geom_point(data = crisis_preds,
             aes(x = date, y = prediction, color = vintage), size = 2.5) +
  scale_color_manual(values = LAG_COLORS, labels = LAG_LABELS, name = "Vintage") +
  scale_linewidth_manual(values = c("-2" = 0.8, "-1" = 0.8, "0" = 1.6),
                         labels = LAG_LABELS, name = "Vintage") +
  scale_linetype_manual(values = c("-2" = "dashed", "-1" = "dashed", "0" = "solid"),
                        labels = LAG_LABELS, name = "Vintage") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title    = "MIDAS - Crisis Period (2020Q1-2021Q2)",
       subtitle = "Nowcasting Vintages | El Salvador | full series shown for context",
       x = "Date", y = "GDP growth rate (QoQ)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title       = element_text(face = "bold"))

ggsave(file.path(OUT_DIR, "midas_plot_crisis.png"),
       p_crisis, width = 14, height = 6, dpi = 150)
cat("Saved: midas_plot_crisis.png\n")

# ---- PLOT B: Normal period ---------------------------------------------------
normal_preds <- pred_long %>% dplyr::filter(period_type == "normal")

p_normal <- ggplot() +
  annotate("rect",
           xmin = NORMAL_START, xmax = NORMAL_END,
           ymin = -Inf, ymax = Inf,
           fill = "#2a9d8f", alpha = 0.10) +
  annotate("rect",
           xmin = CRISIS_START, xmax = CRISIS_END,
           ymin = -Inf, ymax = Inf,
           fill = "#e63946", alpha = 0.06) +
  geom_line(data = gdp_full,
            aes(x = date, y = GDP),
            color = "#444444", linewidth = 0.8, alpha = 0.7) +
  geom_point(data = gdp_full,
             aes(x = date, y = GDP),
             color = "#444444", size = 1.5, alpha = 0.7) +
  geom_line(data = normal_preds,
            aes(x = date, y = prediction, color = vintage,
                linewidth = vintage, linetype = vintage)) +
  geom_point(data = normal_preds,
             aes(x = date, y = prediction, color = vintage), size = 2.5) +
  scale_color_manual(values = LAG_COLORS, labels = LAG_LABELS, name = "Vintage") +
  scale_linewidth_manual(values = c("-2" = 0.8, "-1" = 0.8, "0" = 1.6),
                         labels = LAG_LABELS, name = "Vintage") +
  scale_linetype_manual(values = c("-2" = "dashed", "-1" = "dashed", "0" = "solid"),
                        labels = LAG_LABELS, name = "Vintage") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title    = "MIDAS - Normal Period (2021Q3-2024Q2)",
       subtitle = "Nowcasting Vintages | El Salvador | full series shown for context",
       x = "Date", y = "GDP growth rate (QoQ)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title       = element_text(face = "bold"))

ggsave(file.path(OUT_DIR, "midas_plot_normal.png"),
       p_normal, width = 14, height = 6, dpi = 150)
cat("Saved: midas_plot_normal.png\n")

# ---- PLOT C: RMSE heatmap ---------------------------------------------------
heatmap_data <- perf_strat %>%
  dplyr::mutate(Vintage = factor(Vintage))

p_heatmap <- ggplot(heatmap_data,
                    aes(x = Vintage, y = Period, fill = RMSE)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = round(RMSE, 4)), size = 3.5) +
  scale_fill_gradient(low = "#fff7bc", high = "#d95f0e",
                      name = "RMSE") +
  labs(title    = "MIDAS - RMSE by Period and Vintage",
       subtitle = "rows = evaluation period | columns = vintage (months to quarter-end)",
       x = "Vintage (months relative to quarter-end)", y = "") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(OUT_DIR, "midas_plot_heatmap.png"),
       p_heatmap, width = 10, height = 3.5, dpi = 150)
cat("Saved: midas_plot_heatmap.png\n")

# =============================================================================
# DONE
# =============================================================================
cat("\n", strrep("=", 60), "\n", sep = "")
cat("MIDAS NOWCASTING COMPLETE\n")
cat(strrep("=", 60), "\n", sep = "")
cat("Output :", OUT_DIR, "\n\n")
cat("CSVs:\n")
cat("  midas_predictions.csv            - actuals + all vintage predictions\n")
cat("  midas_performance_full.csv       - RMSE/MAE full test period\n")
cat("  midas_performance_stratified.csv - RMSE/MAE crisis/normal x vintage\n\n")
cat("Plots:\n")
cat("  midas_plot_crisis.png  - predictions during crisis period\n")
cat("  midas_plot_normal.png  - predictions during normal period\n")
cat("  midas_plot_heatmap.png - RMSE heatmap period x vintage\n")
# =============================================================================
# DMF.R
# Purpose : Dynamic Factor Model nowcasting pipeline for El Salvador GDP.
#           Single self-contained script — no other DFM file needed.
#
# What this script does (in order):
#   1.  Loads and validates data
#   2.  Runs rolling-origin evaluation across 3 vintages [-2,-1,0]
#       - DFM estimated via nowcastDFM package (EM algorithm + Kalman filter)
#       - FULLY ROLLING: model re-estimated at each nowcast origin
#       - FALLBACK: if estimation fails, uses previous window's model
#       - Block structure from metadata (block_g, block_s, block_r, block_l)
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
# Output  : ~/Desktop/ESA-gdp-nowcasting/Models/results/dfm/
#
# Packages required:
#   devtools::install_github("dhopp1/nowcastDFM")
#   install.packages("tidyverse")
# =============================================================================

library(tidyverse)
library(nowcastDFM)

# =============================================================================
# 0. PATHS
# =============================================================================

BASE_DIR  <- file.path(path.expand("~"), "Desktop", "ESA-gdp-nowcasting")
DATA_DIR  <- file.path(BASE_DIR, "Data")
MODEL_DIR <- file.path(BASE_DIR, "Models")
OUT_DIR   <- file.path(MODEL_DIR, "results", "dfm")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PATH_DATA <- file.path(DATA_DIR, "data_tf.csv")
PATH_META <- file.path(DATA_DIR, "meta_data_v2.csv")

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
LAGS            <- c(-2L, -1L, 0L)    # only genuine nowcast vintages
TRAIN_START     <- as.Date("2005-01-01")
TEST_START      <- as.Date("2020-03-01")   # 2020Q1
TEST_END        <- as.Date("2024-06-01")   # 2024Q2
CRISIS_START    <- as.Date("2020-03-01")   # 2020Q1
CRISIS_END      <- as.Date("2021-06-01")   # 2021Q2
NORMAL_START    <- as.Date("2021-09-01")   # 2021Q3
NORMAL_END      <- as.Date("2024-06-01")   # 2024Q2

# GDP publication lag in months. Used to set training cutoff explicitly
# (Option A, matches ARMA benchmark).
GDP_PUB_LAG <- 3L

# --- TUNABLE -----------------------------------------------------------------

# Maximum EM iterations per estimation.
# 100 is sufficient for most DFMs — the log-likelihood change is already
# below 0.05% at iteration 100. Using 500 multiplies runtime by 5x.
# Compare RMSE at 100 vs 500 iterations — if difference < 0.001, keep 100.
DFM_MAX_ITER <- 100

# ROLLING=TRUE  : re-estimate DFM at each origin (methodologically correct)
# ROLLING=FALSE : estimate once on first training window, apply to all origins
#                 (faster, document as computational constraint in paper)
ROLLING <- TRUE

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

# gen_lagged_data: simulate data vintage available at nowcast_date + lag.
# Uses DATE-BASED cutoff — robust to data gaps (original used positional index
# which fails when rows are not evenly spaced).
#
# Parameters:
#   metadata     : data.frame with columns 'series' and 'months_lag'
#   data         : full monthly panel sorted by date ascending
#   nowcast_date : Date — the quarter being nowcast
#   lag          : integer — months relative to quarter-end
#                  lag=0  → at quarter-end
#                  lag=-2 → 2 months before quarter-end
gen_lagged_data <- function(metadata, data, nowcast_date, lag) {
  nowcast_dt  <- as.Date(nowcast_date)
  lagged_data <- data %>%
    dplyr::filter(date <= nowcast_dt)
  
  for (col in colnames(lagged_data)[2:ncol(lagged_data)]) {
    pub_lag_row <- metadata %>%
      dplyr::filter(series == col) %>%
      dplyr::select(months_lag) %>%
      dplyr::pull()
    
    pub_lag <- if (length(pub_lag_row) == 0) {
      warning("Variable '", col, "' not found in metadata. Assuming pub_lag=1.")
      1L
    } else {
      as.integer(pub_lag_row[1])
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

# Replace infinites with NA
for (col in colnames(data_full)[2:ncol(data_full)]) {
  data_full[[col]][is.infinite(data_full[[col]])] <- NA
}

cat("  Rows         :", nrow(data_full), "\n")
cat("  Columns      :", ncol(data_full), "\n")
cat("  GDP obs      :",
    sum(!is.na(data_full[[TARGET_VARIABLE]])), "quarterly observations\n")

# ---- Build block structure from metadata ------------------------------------
# nowcastDFM requires blocks matrix: one row per variable, one col per block.
# Entry = 1 if variable loads on that block (from block_g/s/r/l in metadata).
# CRITICAL: nrow(block_cols) must equal ncol(data_model) - 1 (excl. date).

var_in_data      <- colnames(data_full)[2:ncol(data_full)]
metadata_matched <- metadata_full %>%
  dplyr::filter(series %in% var_in_data)

# Reorder metadata to match column order in data_full
col_order        <- match(var_in_data, metadata_matched$series)
metadata_ordered <- metadata_matched[col_order[!is.na(col_order)], ]

# Keep only block columns with at least one variable assigned
block_cols <- metadata_ordered %>%
  dplyr::select(starts_with("block_")) %>%
  dplyr::select_if(~ sum(., na.rm = TRUE) > 0) %>%
  data.frame()

cat("  DFM blocks   :", ncol(block_cols), "\n")
cat("  DFM vars     :", nrow(block_cols), "\n")

# Variables in model (intersection of data and metadata)
vars_in_model <- metadata_ordered$series
data_model    <- data_full %>%
  dplyr::select(date, all_of(vars_in_model))

# Verify dimensions match before running
stopifnot(
  "block_cols rows must equal data_model columns minus date" =
    nrow(block_cols) == ncol(data_model) - 1
)
cat("  Dimension check: PASSED (", nrow(block_cols),
    "vars x", ncol(block_cols), "blocks)\n")

# Test dates: quarterly sequence
test_dates <- seq(TEST_START, TEST_END, by = "3 months")
cat("  Test quarters :", length(test_dates), "\n\n")

# =============================================================================
# 4. ROLLING-ORIGIN EVALUATION LOOP
# =============================================================================
# Temporal validity:
#   [ok] No leakage    - training ends GDP_PUB_LAG months before nowcast date
#   [ok] No look-ahead - gen_lagged_data uses date-based cutoff per variable
#   [ok] Rolling       - DFM re-estimated each origin (if ROLLING=TRUE)
#   [ok] Fallback      - if estimation fails, previous model is used
#   [ok] Target masked - GDP for nowcast quarter set to NA before prediction
#   [ok] Ragged edges  - nowcastDFM handles NAs via Kalman filter natively
#
# NOTE on the Lapack 'dgesdd' error:
#   When the COVID shock enters the training set (2020Q2), the covariance
#   matrix can become singular and the EM algorithm fails. The fallback
#   approach uses the previous window's model, which is methodologically
#   sound — central banks do the same in real-time systems when estimation
#   fails. We additionally record model_age_quarters per window so the
#   degradation (model K quarters behind) can be inspected ex-post.

cat("Running rolling-origin DFM evaluation...\n")
cat("  Test window  :", as.character(TEST_START),
    "to", as.character(TEST_END), "\n")
cat("  Vintages     :", paste(LAGS, collapse = ", "), "\n")
cat("  Rolling      :", ROLLING, "\n")
cat("  Max EM iter  :", DFM_MAX_ITER, "\n\n")

# Storage
pred_list        <- vector("list", length(LAGS))
names(pred_list) <- as.character(LAGS)
for (lag in LAGS) {
  pred_list[[as.character(lag)]] <- rep(NA_real_, length(test_dates))
}

actuals_vec                  <- rep(NA_real_, length(test_dates))
model_age_q_vec              <- rep(NA_integer_, length(test_dates))
output_dfm                   <- NULL   # current DFM model object
last_successful_train_cutoff <- NA_Date_  # cutoff date of model in output_dfm
n_fallback                   <- 0L      # count of fallback uses

for (i in seq_along(test_dates)) {
  nowcast_date <- test_dates[i]
  
  # ---- Actual GDP for this quarter ------------------------------------------
  # Direct date comparison (O1): both sides are Date, safer than format()
  actual_val <- data_model %>%
    dplyr::filter(date == nowcast_date) %>%
    dplyr::pull(!!TARGET_VARIABLE)
  
  actuals_vec[i] <- if (length(actual_val) > 0 && !is.na(actual_val[1]))
    actual_val[1] else NA_real_
  
  # ---- Training window -------------------------------------------------------
  # Ends GDP_PUB_LAG months before the nowcast date (Option A).
  # Tied to GDP_PUB_LAG explicitly so changing the publication lag doesn't
  # silently introduce look-ahead.
  train_cutoff <- nowcast_date - months(GDP_PUB_LAG)
  
  train_raw <- data_model %>%
    dplyr::filter(date >= TRAIN_START & date <= train_cutoff) %>%
    data.frame()
  
  n_q <- sum(!is.na(train_raw[[TARGET_VARIABLE]]))
  
  if (n_q < 20) {
    cat("  [", as.character(nowcast_date), "] Only", n_q,
        "quarterly obs — skipping.\n")
    next
  }
  
  # ---- Estimate DFM ---------------------------------------------------------
  # Re-estimate if ROLLING=TRUE or if no model exists yet.
  # FALLBACK LOGIC:
  #   If estimation succeeds  → update output_dfm AND last_successful_train_cutoff
  #   If estimation fails     → keep previous output_dfm (fallback)
  
  should_estimate <- ROLLING || is.null(output_dfm)
  
  if (should_estimate) {
    output_dfm_new <- tryCatch({
      dfm(
        data     = train_raw,
        blocks   = block_cols,
        max_iter = DFM_MAX_ITER
      )
    }, error = function(e) {
      cat("    DFM estimation failed [", as.character(nowcast_date),
          "] :", conditionMessage(e), "\n")
      cat("    -> Using previous window model as fallback.\n")
      NULL
    })
    
    if (!is.null(output_dfm_new)) {
      # Estimation succeeded — update model and its cutoff
      output_dfm                   <- output_dfm_new
      last_successful_train_cutoff <- train_cutoff
    } else {
      # Estimation failed — keep previous model (fallback)
      n_fallback <- n_fallback + 1L
    }
  }
  
  # If no model available at all (first iteration failed), skip
  if (is.null(output_dfm)) {
    cat("  [", as.character(nowcast_date),
        "] No DFM model available — skipping.\n")
    next
  }
  
  # ---- Record model age in quarters (C6) ------------------------------------
  # Quarters between the cutoff of the model currently loaded and
  # this window's intended cutoff. 0 = model just estimated this window.
  model_age_q_vec[i] <- as.integer(round(
    as.numeric(train_cutoff - last_successful_train_cutoff) / (365.25 / 4)
  ))
  
  # ---- Predict for each vintage ---------------------------------------------
  for (lag in LAGS) {
    tryCatch({
      # Simulate vintage: mask data not yet published at this vintage
      vintage_data <- gen_lagged_data(
        metadata_full, data_model, nowcast_date, lag
      )
      vintage_data <- vintage_data %>% data.frame()
      
      # Mask GDP for the nowcast quarter — prevent any leakage
      vintage_data[
        vintage_data$date == nowcast_date,
        TARGET_VARIABLE
      ] <- NA
      
      # nowcastDFM handles ragged edges internally via Kalman filter
      # No manual imputation needed — that is the key advantage of DFM
      dfm_output <- predict_dfm(vintage_data, output_dfm)
      
      # Extract prediction for the nowcast quarter (O1: direct date compare)
      prediction <- dfm_output %>%
        dplyr::filter(date == nowcast_date) %>%
        dplyr::pull(!!TARGET_VARIABLE)
      
      pred_list[[as.character(lag)]][i] <-
        if (length(prediction) > 0 && !is.na(prediction[1]))
          as.numeric(prediction[1]) else NA_real_
      
    }, error = function(e) {
      cat("    Prediction error [", as.character(nowcast_date),
          "] lag=", lag, ":", conditionMessage(e), "\n")
      pred_list[[as.character(lag)]][i] <<- NA_real_
    })
  }
  
  cat("  [", as.character(nowcast_date), "]",
      "  n_train_q=",           n_q,
      "  model_age_q=",         model_age_q_vec[i],
      "  fallbacks_so_far=",    n_fallback,
      "\n", sep = "")
}

cat("\nTotal fallbacks used:", n_fallback, "out of", length(test_dates),
    "windows\n")

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
  RMSE = sapply(LAGS, function(lag)
    rmse_fn(actuals_vec, pred_list[[as.character(lag)]])),
  MAE  = sapply(LAGS, function(lag)
    mae_fn(actuals_vec, pred_list[[as.character(lag)]])),
  n    = sapply(LAGS, function(lag)
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
cat("FULL TEST PERIOD - DFM (2020Q1-2024Q2)\n")
cat(strrep("=", 60), "\n", sep = "")
print(round(perf_full, 6))

cat("\n", strrep("=", 60), "\n", sep = "")
cat("STRATIFIED - Crisis vs Normal x all 3 vintages\n")
cat(strrep("=", 60), "\n", sep = "")
print(perf_strat %>%
        dplyr::mutate(RMSE = round(RMSE, 6), MAE = round(MAE, 6)))

# NOTE: Relative RMSE vs ARMA is computed in comparison_final.py
# which loads all model CSVs together after all models have been run.

# =============================================================================
# 6. SAVE RESULTS
# =============================================================================

pred_table <- data.frame(date = test_dates, actual = actuals_vec)
for (lag in LAGS) {
  col_name           <- paste0("lag_", if (lag < 0) lag else paste0("+", lag))
  pred_table[[col_name]] <- pred_list[[as.character(lag)]]
}
pred_table$period_type    <- period_labels
pred_table$model_age_q    <- model_age_q_vec

write_csv(pred_table, file.path(OUT_DIR, "dfm_predictions.csv"))
write_csv(perf_full,  file.path(OUT_DIR, "dfm_performance_full.csv"))
write_csv(perf_strat, file.path(OUT_DIR, "dfm_performance_stratified.csv"))

cat("\nCSVs saved to:", OUT_DIR, "\n")

# =============================================================================
# 7. PLOTS
# =============================================================================

LAG_COLORS <- c("-2" = "#a8dadc", "-1" = "#457b9d", "0" = "#1d3557")
LAG_LABELS <- c("-2" = "lag=-2", "-1" = "lag=-1", "0" = "lag=0 (nowcast)")

gdp_full <- data_full %>%
  dplyr::filter(!is.na(!!sym(TARGET_VARIABLE))) %>%
  dplyr::select(date, GDP = !!sym(TARGET_VARIABLE))

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
  geom_line(data  = gdp_full,
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
  scale_color_manual(values = LAG_COLORS, labels = LAG_LABELS,
                     name = "Vintage") +
  scale_linewidth_manual(values = c("-2" = 0.8, "-1" = 0.8, "0" = 1.6),
                         labels = LAG_LABELS, name = "Vintage") +
  scale_linetype_manual(
    values = c("-2" = "dashed", "-1" = "dashed", "0" = "solid"),
    labels = LAG_LABELS, name = "Vintage") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title    = "DFM - Crisis Period (2020Q1-2021Q2)",
       subtitle = "Nowcasting Vintages | El Salvador | full series for context",
       x = "Date", y = "GDP growth rate (QoQ)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title       = element_text(face = "bold"))

ggsave(file.path(OUT_DIR, "dfm_plot_crisis.png"),
       p_crisis, width = 14, height = 6, dpi = 150)
cat("Saved: dfm_plot_crisis.png\n")

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
  geom_line(data  = gdp_full,
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
  scale_color_manual(values = LAG_COLORS, labels = LAG_LABELS,
                     name = "Vintage") +
  scale_linewidth_manual(values = c("-2" = 0.8, "-1" = 0.8, "0" = 1.6),
                         labels = LAG_LABELS, name = "Vintage") +
  scale_linetype_manual(
    values = c("-2" = "dashed", "-1" = "dashed", "0" = "solid"),
    labels = LAG_LABELS, name = "Vintage") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title    = "DFM - Normal Period (2021Q3-2024Q2)",
       subtitle = "Nowcasting Vintages | El Salvador | full series for context",
       x = "Date", y = "GDP growth rate (QoQ)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title       = element_text(face = "bold"))

ggsave(file.path(OUT_DIR, "dfm_plot_normal.png"),
       p_normal, width = 14, height = 6, dpi = 150)
cat("Saved: dfm_plot_normal.png\n")

# ---- PLOT C: RMSE heatmap ---------------------------------------------------
heatmap_data <- perf_strat %>%
  dplyr::mutate(Vintage = factor(Vintage))

p_heatmap <- ggplot(heatmap_data,
                    aes(x = Vintage, y = Period, fill = RMSE)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = round(RMSE, 4)), size = 3.5) +
  scale_fill_gradient(low = "#fff7bc", high = "#d95f0e", name = "RMSE") +
  labs(title    = "DFM - RMSE by Period and Vintage",
       subtitle = "rows = evaluation period | columns = vintage",
       x = "Vintage (months relative to quarter-end)", y = "") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(OUT_DIR, "dfm_plot_heatmap.png"),
       p_heatmap, width = 10, height = 3.5, dpi = 150)
cat("Saved: dfm_plot_heatmap.png\n")

# =============================================================================
# DONE
# =============================================================================
cat("\n", strrep("=", 60), "\n", sep = "")
cat("DFM NOWCASTING COMPLETE\n")
cat(strrep("=", 60), "\n", sep = "")
cat("Output :", OUT_DIR, "\n\n")
cat("CSVs:\n")
cat("  dfm_predictions.csv\n")
cat("  dfm_performance_full.csv\n")
cat("  dfm_performance_stratified.csv\n\n")
cat("Plots:\n")
cat("  dfm_plot_crisis.png\n")
cat("  dfm_plot_normal.png\n")
cat("  dfm_plot_heatmap.png\n\n")
cat("NEXT: Run comparison_final.py to compare all models vs ARMA.\n")
cat("\nNOTE on fallbacks: if DFM_MAX_ITER=100 produces many fallbacks,\n")
cat("consider ROLLING=FALSE (estimate once) and document in the paper.\n")




# =============================================================================
# INTERPRETABILITY BLOCK — DFM Factor Loadings
#
# What this block does:
#   1. Saves the DFM model object to disk (saveRDS) with the cutoff in name
#   2. Extracts factor loadings from output_dfm (in memory after main loop)
#   3. Produces one plot per factor (top 15 variables)
#   4. Produces a heatmap of all loadings (top 30 variables)
#   5. Saves CSVs with complete loadings and top 10 per factor
#
# NOTE: The loaded output_dfm corresponds to the LAST window where
# estimation succeeded (see last_successful_train_cutoff). This is the
# model actually used to produce the final forecast(s).
#
# Output: ~/Desktop/ESA-gdp-nowcasting/Models/results/interpretability/
# =============================================================================

OUT_DIR_INTERP <- file.path(MODEL_DIR, "results", "interpretability")
dir.create(OUT_DIR_INTERP, recursive = TRUE, showWarnings = FALSE)

# ---- STEP 1: Save DFM object with cutoff date in filename -------------------

cutoff_tag <- format(last_successful_train_cutoff, "%Y%m")
rds_path   <- file.path(OUT_DIR,
                        paste0("dfm_model_object_", cutoff_tag, ".rds"))

saveRDS(output_dfm, file = rds_path)
cat("DFM model object saved to:", rds_path, "\n")
cat("(Cutoff of this model:", as.character(last_successful_train_cutoff), ")\n")
cat("(Reload in new session: readRDS('", rds_path, "'))\n\n", sep = "")

# ---- STEP 2: Verify object structure ----------------------------------------
cat("DFM object structure:\n")
print(names(output_dfm))
cat("Factor loadings matrix C — dimensions:", dim(output_dfm$C), "\n")
cat("  Rows (variables) :", nrow(output_dfm$C), "\n")
cat("  Cols (factors)   :", ncol(output_dfm$C), "\n\n")

# ---- STEP 3: Extract factor loadings ----------------------------------------
# C matrix has n_blocks * p_lags columns. Extract the first column of
# each block = contemporaneous loading for each factor.

n_blocks <- ncol(block_cols)
cat("Number of blocks:", n_blocks, "\n")
cat("C matrix dimensions:", nrow(output_dfm$C), "x", ncol(output_dfm$C), "\n")

p_lags   <- ncol(output_dfm$C) / n_blocks
cat("Lags per block (p):", p_lags, "\n")

block_names  <- c("Factor G (Global/Real)",
                  "Factor S (External Sector)",
                  "Factor R (Financial/Monetary)",
                  "Factor L (Labour/Fiscal)")

block_starts <- seq(1, ncol(output_dfm$C), by = p_lags)
cat("Block start indices:", block_starts, "\n")

loadings_df <- data.frame(series = as.character(vars_in_model))

for (b in seq_len(n_blocks)) {
  col_idx <- block_starts[b]
  loadings_df[[block_names[b]]] <- output_dfm$C[, col_idx]
}

loadings_df <- loadings_df %>%
  dplyr::left_join(
    metadata_full %>%
      dplyr::select(series, name, freq) %>%
      dplyr::distinct(series, .keep_all = TRUE),
    by = "series"
  ) %>%
  dplyr::relocate(series, name, freq)

loadings_df <- loadings_df %>%
  dplyr::mutate(
    mean_abs = rowMeans(abs(dplyr::across(all_of(block_names))))
  )

cat("\nLoadings extracted successfully!\n")
cat("Variables:", nrow(loadings_df), "\n")
cat("Columns:", paste(names(loadings_df), collapse=", "), "\n\n")

# Dynamic subtitle: uses last_successful_train_cutoff (C3 + C4 + O4)
subtitle_cutoff <- paste0("Last successful training window (ending ",
                          format(last_successful_train_cutoff, "%YQ"),
                          ceiling(as.integer(format(last_successful_train_cutoff,
                                                    "%m")) / 3),
                          " — ",
                          last_successful_train_cutoff, ")")

# ---- Plot: one per block (top 15 variables by absolute loading) -------------

for (b in seq_len(n_blocks)) {
  fct_col <- block_names[b]
  
  top_vars <- loadings_df %>%
    dplyr::mutate(abs_load = abs(!!sym(fct_col))) %>%
    dplyr::arrange(dplyr::desc(abs_load)) %>%
    dplyr::slice_head(n = 15) %>%
    dplyr::mutate(
      label     = ifelse(!is.na(name) & name != "",
                         paste0(series, " — ", substr(name, 1, 25)),
                         series),
      direction = ifelse(!!sym(fct_col) >= 0, "Positive", "Negative")
    )
  
  p <- ggplot(top_vars,
              aes(x    = reorder(label, abs_load),
                  y    = !!sym(fct_col),
                  fill = direction)) +
    geom_col(width = 0.72, show.legend = TRUE) +
    scale_fill_manual(
      values = c("Positive" = "#2a9d8f", "Negative" = "#e63946"),
      name   = "Direction"
    ) +
    coord_flip() +
    geom_hline(yintercept = 0, color = "#333333",
               linewidth = 0.8, linetype = "dashed") +
    labs(
      title    = paste0("DFM Factor Loadings — ", fct_col),
      subtitle = paste0(
        "Top 15 variables by absolute loading | ",
        subtitle_cutoff, "\n",
        "Green = positive (variable co-moves with factor) | ",
        "Red = negative (inverse)"
      ),
      x = NULL,
      y = "Factor Loading (contemporaneous)"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title      = element_text(face = "bold", size = 12),
      plot.subtitle   = element_text(size = 9, color = "gray40"),
      axis.text.y     = element_text(size = 9),
      legend.position = "bottom"
    )
  
  fname <- file.path(OUT_DIR_INTERP,
                     paste0("dfm_loadings_block", b, ".png"))
  ggsave(fname, p, width = 11, height = 7, dpi = 150)
  cat("Saved:", fname, "\n")
}

# ---- Heatmap: top 30 variables x 4 blocks -----------------------------------

top30 <- loadings_df %>%
  dplyr::arrange(dplyr::desc(mean_abs)) %>%
  dplyr::slice_head(n = 30)

heat_long <- top30 %>%
  dplyr::select(series, all_of(block_names)) %>%
  tidyr::pivot_longer(
    cols      = all_of(block_names),
    names_to  = "Factor",
    values_to = "Loading"
  ) %>%
  dplyr::mutate(
    Factor = dplyr::recode(Factor,
                           "Factor G (Global/Real)"       = "F1: Global",
                           "Factor S (External Sector)"   = "F2: External",
                           "Factor R (Financial/Monetary)"= "F3: Financial",
                           "Factor L (Labour/Fiscal)"     = "F4: Labour"
    )
  )

# Order rows by dominant factor
series_order <- top30 %>%
  dplyr::select(series, all_of(block_names)) %>%
  tidyr::pivot_longer(-series, names_to="f", values_to="v") %>%
  dplyr::group_by(series) %>%
  dplyr::summarise(max_abs = max(abs(v))) %>%
  dplyr::arrange(dplyr::desc(max_abs)) %>%
  dplyr::pull(series)

heat_long$series <- factor(heat_long$series, levels = rev(series_order))

p_heat <- ggplot(heat_long,
                 aes(x = Factor, y = series, fill = Loading)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = round(Loading, 2)),
            size = 2.8, color = "white", fontface = "bold") +
  scale_fill_gradient2(
    low      = "#e63946",
    mid      = "#f1faee",
    high     = "#2a9d8f",
    midpoint = 0,
    name     = "Loading"
  ) +
  labs(
    title    = "DFM Factor Loadings — Top 30 Variables",
    subtitle = paste0(
      "Contemporaneous loadings only | ",
      "Green = positive | Red = negative | ",
      subtitle_cutoff
    ),
    x = "Latent Factor", y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title    = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9, color = "gray40"),
    axis.text.y   = element_text(size = 8),
    axis.text.x   = element_text(size = 9)
  )

ggsave(file.path(OUT_DIR_INTERP, "dfm_loadings_heatmap.png"),
       p_heat, width = 10, height = 10, dpi = 150)
cat("Saved: dfm_loadings_heatmap.png\n")

# ---- Save CSVs --------------------------------------------------------------

write_csv(loadings_df,
          file.path(OUT_DIR_INTERP, "dfm_factor_loadings_full.csv"))

# Top 10 per block
top10_all <- do.call(rbind, lapply(block_names, function(b) {
  loadings_df %>%
    dplyr::mutate(abs_load = abs(!!sym(b)), factor = b) %>%
    dplyr::arrange(dplyr::desc(abs_load)) %>%
    dplyr::slice_head(n = 10) %>%
    dplyr::select(factor, series, name, loading = !!sym(b), abs_load)
}))

write_csv(top10_all,
          file.path(OUT_DIR_INTERP, "dfm_factor_loadings_top10.csv"))

cat("Saved: dfm_factor_loadings_top10.csv\n")
cat("Saved: dfm_factor_loadings_full.csv\n")
cat("\nAll DFM loading files saved to:", OUT_DIR_INTERP, "\n")
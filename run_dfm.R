# =============================================================================
# run_dfm.R — Run the DFM on the November 2026 data (107 series, 2026M6)
#
# From this folder:   Rscript run_dfm.R
# (RStudio: open this file, then Source)
#
# Options (edit below):
#   REESTIMATE = "annual" : re-estimate once a year (each Q1, data through the
#                           previous December); Kalman update in between.
#                           Expected runtime: ~6-8 EM estimations.
#                "every"  : re-estimate at every quarter (thesis design; slow)
#                "once"   : estimate once (fastest; for quick checks)
# =============================================================================

REESTIMATE <- "annual"
TEST_END   <- "2026-06-01"   # last quarter nowcast (2026Q2, GDP not yet published)

# Folder that contains this script = project base (Data/, Models/)
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
base_dir <- if (length(file_arg)) dirname(normalizePath(file_arg)) else
            if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable())
              dirname(rstudioapi::getSourceEditorContext()$path) else getwd()

# Packages (installed only if missing)
need <- c("tidyverse", "remotes")
for (p in need) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
if (!requireNamespace("nowcastDFM", quietly = TRUE))
  remotes::install_github("dhopp1/nowcastDFM", upgrade = "never")

Sys.setenv(ESA_BASE_DIR       = base_dir,
           ESA_TEST_END       = TEST_END,
           ESA_DFM_REESTIMATE = REESTIMATE)

cat("Base dir   :", base_dir, "\nRe-estimate:", REESTIMATE, "\nStart      :", format(Sys.time()), "\n")
source(file.path(base_dir, "Models", "DMF.R"), echo = FALSE)
cat("Finished   :", format(Sys.time()), "\nResults in : Models/results/dfm/\n")

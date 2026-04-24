# Nowcasting GDP in Times of Crisis and Non-Crisis: A Comparative Study of Econometric and Machine Learning Models in El Salvador

**Author:** Gabriela Sulamita Aquino Calderón
**Affiliation:** MSc Data Science for Economics, University of Liverpool
**Status:** Under revision for journal submission

---

## Overview

This repository contains the full replication code for a nowcasting study of quarterly GDP growth in El Salvador using a harmonized mixed-frequency panel of 123 macroeconomic indicators (2005Q1–2024Q2).

The study evaluates seven models from three methodological families — classical time series, mixed-frequency econometric, and machine learning — under a unified rolling-origin protocol with 18 out-of-sample test quarters and three publication-lag vintages. Performance is stratified into a **crisis period** (2020Q1–2021Q2, six quarters encompassing the COVID shock and its immediate rebound) and a **normal period** (2021Q3–2024Q2, twelve quarters of post-crisis stabilization).

The central empirical finding is a **regime-dependent reversal of the model ranking**: the Dynamic Factor Model dominates during crisis by exploiting cross-indicator co-movement, while regularized linear models and tree ensembles dominate during normal periods by exploiting sparse and interactive signal structure.

### Models implemented

| Model | Type | Script |
|---|---|---|
| ARMA | Econometric benchmark | `Models/ARMA.ipynb` |
| MIDAS | Mixed-frequency econometric | `Models/MIDAS.R` |
| DFM | Dynamic Factor Model | `Models/DMF.R` (+ `dfm_postprocess.py`) |
| Ridge | Penalized linear (L2) | `Models/Ridge.py` |
| Lasso | Penalized linear (L1) | `Models/Lasso.py` |
| XGBoost | Gradient boosted trees | `Models/XGBoost.py` |
| Random Forest | Bagged trees | `Models/RandomForest.py` |

### Comparison and density analysis

| Notebook | Purpose |
|---|---|
| `Models/01_point_forecast_evaluation.ipynb` | RMSE/MAE tables, vintage profiles, Diebold-Mariano tests with HLN small-sample correction |
| `Models/02_density_and_interpretability.ipynb` | Bootstrap prediction intervals, coverage tables, fan charts, CRPS, Ridge / Lasso / XGBoost interpretability |

### Key methodological features

- **Rolling-origin evaluation**: model re-estimated at each forecast origin using only data available at that point in time.
- **Three publication-lag vintages**: lag = -2 (two months before quarter-end), lag = -1 (one month before), lag = 0 (at quarter-end).
- **Fixed holdout tuning**: hyperparameters selected on the last 8 quarters of each training window (identical protocol across Ridge, Lasso, XGBoost, Random Forest).
- **Ragged-edge simulation**: publication lags applied per variable using the metadata release calendar.
- **Single source of truth**: all constants and shared utilities live in `Models/nowcast_utils.py`. All scripts import from it.
- **No data leakage**: scaling, imputation, and hyperparameter selection strictly inside the training window at each origin.
- **Diebold-Mariano**: one-sided with Harvey-Leybourne-Newbold (1997) small-sample correction.
- **MIDAS**: inverse-RMSE weighting across univariate models (Bates & Granger 1969; Stock & Watson 2004).
- **DFM**: handles missing values natively via Kalman filter; fallback to previous window if EM fails to converge.

---

## Repository Structure

```
nowcasting-elsalvador/
│
├── README.md
├── requirements.txt                          # Python dependencies
├── .gitignore
│
├── Data/
│   ├── README_data.md                        # Data description and access instructions
│   └── meta_data_v2.csv                      # Metadata: 123 indicators with publication lags
│                                             # and DFM block structure
│
└── Models/
    ├── prepare_data_v2.R                     # Data preparation: growth rates, transformations
    ├── nowcast_utils.py                      # Shared constants and utilities (canonical)
    │
    ├── ARMA.ipynb                            # Univariate benchmark
    ├── MIDAS.R                               # Mixed-frequency with inverse-RMSE weighting
    ├── DMF.R                                 # Dynamic Factor Model
    ├── dfm_postprocess.py                    # Post-processes DFM output to canonical test window
    ├── Ridge.py                              # L2 penalized regression
    ├── Lasso.py                              # L1 penalized regression + rolling selection frequency
    ├── XGBoost.py                            # Gradient boosting + SHAP interpretability
    ├── RandomForest.py                       # Bagged trees
    │
    ├── 01_point_forecast_evaluation.ipynb    # Accuracy evaluation + DM tests
    └── 02_density_and_interpretability.ipynb # Density forecasts + interpretability
```

---

## Data

### Why raw data are not included

The raw macroeconomic panel (`SLV_nowcasting_data_mq.csv`) is not included in this repository due to institutional data agreements with the Banco Central de Reserva de El Salvador (BCR) and other official statistical sources.

### How to obtain the data

The dataset covers 123 monthly and quarterly indicators for El Salvador (2005–2024) from the following sources:

| Source | Variables | Access |
|---|---|---|
| Banco Central de Reserva de El Salvador (BCR) | IVAE, monetary aggregates, exchange rates | [www.bcr.gob.sv](https://www.bcr.gob.sv) |
| DIGESTYC | National accounts GDP | [www.digestyc.gob.sv](https://www.digestyc.gob.sv) |
| ISSS | Social security employment | [www.isss.gob.sv](https://www.isss.gob.sv) |
| Ministerio de Hacienda | Tax revenues, government expenditure | [www.transparenciafiscal.gob.sv](https://www.transparenciafiscal.gob.sv) |
| Federal Reserve (FRED) | US indicators (EFFR, IPI_US, UNEM_US) | [fred.stlouisfed.org](https://fred.stlouisfed.org) |

Upon reasonable request and for replication purposes, the processed dataset (`data_tf.csv`) can be provided by the author subject to the terms of the original data providers.

### What IS included

`Data/meta_data_v2.csv` — metadata file describing all 123 indicators (safe to share):

| Column | Description |
|---|---|
| `series` | Variable code |
| `name` | Description (Spanish) |
| `freq` | `M` monthly or `Q` quarterly |
| `block_g`, `block_s`, `block_r`, `block_l` | DFM block membership (Global/Real, External Sector, Financial/Monetary, Labour/Fiscal) |
| `months_lag` | Publication lag in months |

---

## Requirements

### Python (≥ 3.9)

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `statsmodels`, `matplotlib`, `seaborn`, `shap`, `nbformat`, `jupyter`, `Pillow`.

### R (≥ 4.2)

```r
install.packages(c("tidyverse", "midasr", "imputeTS", "scales"))
devtools::install_github("dhopp1/nowcastDFM")
```

---

## How to Reproduce Results

### Step 1 — Prepare data

Place the raw data file in `Data/SLV_nowcasting_data_mq.csv`, then:

```bash
Rscript Models/prepare_data_v2.R
```

This generates `Data/data_tf.csv` — the transformed panel with growth rates and seasonal adjustments applied.

### Step 2 — Run models (any order)

Each script is self-contained and writes results to `Models/results/<model_name>/`.

**Python / Jupyter:**

```bash
jupyter nbconvert --to notebook --execute Models/ARMA.ipynb
python Models/Ridge.ipynb
python Models/Lasso.ipynb
python Models/XGBoost.ipynb
python Models/RandomForest.ipynb
```

**R:**

```bash
Rscript Models/MIDAS.R
Rscript Models/DMF.R
python Models/dfm_postprocess.py    # Filters DFM output to canonical test window
```

**Approximate runtimes (standard laptop):**

| Model | Runtime |
|---|---|
| ARMA | ~2 min |
| Ridge | ~3 min |
| Lasso | ~3 min |
| XGBoost (includes SHAP) | ~3 min |
| Random Forest | ~3 min |
| MIDAS | ~5 min |
| DFM (rolling) | up to 2 days depending on config |

### Step 3 — Comparison, DM tests, density forecasting

Run the two notebooks in order:

```bash
jupyter nbconvert --to notebook --execute Models/01_point_forecast_evaluation.ipynb
jupyter nbconvert --to notebook --execute Models/02_density_and_interpretability.ipynb
```

**Notebook 01** produces:
- `results/comparison/` — unified predictions, relative RMSE/MAE tables, ranking, comparison plots
- `results/diebold_mariano/` — DM results vs ARMA / Ridge / DFM, pairwise heatmaps

**Notebook 02** produces:
- `results/fanchart/` — coverage table, fan charts per model × period
- `results/crps/` — CRPS scores, skill table, RMSE-vs-CRPS comparison
- `results/interpretability/` — Ridge coefficients, Lasso selection frequency, XGBoost SHAP, cross-model importance

---

## Reproducibility Notes

- **Random seed**: `42` fixed across all Python models via `np.random.seed(42)`, `random_state=42`.
- **R seed**: `set.seed(42)` at the start of all R scripts.
- **Hyperparameter tuning**: fixed 8-quarter holdout (not random k-fold) — results are deterministic given the same data.
- **Single-source-of-truth constants**: `TRAIN_START`, `TEST_START`, `TEST_END`, `CRISIS_*`, `NORMAL_*`, `LAGS`, `GDP_PUB_LAG`, `RANDOM_SEED` defined once in `Models/nowcast_utils.py` and imported by every script.
- **DFM fallback**: if the EM algorithm fails to converge in a given window (e.g., singular matrix during COVID shock), the previous successful window's model is reused. Number of fallbacks reported in output.
- **SHAP**: requires `pip install shap`. If absent, `XGBoost.py` skips the SHAP block gracefully and produces the rest of its outputs.

---

## Results Summary

Full results are reported in the paper. Key finding:

> **Performance is strongly regime-dependent.** During the crisis period (2020Q1–2021Q2), the Dynamic Factor Model reduces RMSE by 63% relative to ARMA, followed by Ridge (47% reduction). During the normal period (2021Q3–2024Q2), the ranking inverts: Random Forest, MIDAS, Lasso, XGBoost, and Ridge all reduce RMSE by 68–76%, while the DFM reduces it by 39%. Diebold-Mariano tests with HLN small-sample correction confirm the inversions at the 5% level. CRPS Skill Scores relative to ARMA range from 0.24 (tree ensembles) to 0.63 (DFM in crisis), confirming all seven models produce better density forecasts than the univariate benchmark.

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{aquino2025nowcasting,
  title   = {Nowcasting GDP in Times of Crisis and Non-Crisis:
             A Comparative Study of Econometric and Machine Learning
             Models in El Salvador},
  author  = {Aquino Calderón, Gabriela Sulamita},
  journal = {[Journal name — to be confirmed]},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

Code is released under the MIT License. Data are subject to the terms of the original data providers — see `Data/README_data.md`.

---

## Contact

For questions about the replication or data access:
**Gabriela Aquino**
MSc Data Science for Economics — University of Liverpool

# Series-selection robustness check (November 2026)

Full set (123 indicators) vs reduced set (107 indicators, `Data/series_selection.csv`),
original evaluation design: rolling origin 2020Q1–2024Q2, vintages −2/−1/0.

**DFM caveat:** estimated once on the first training window (`ESA_DFM_ROLLING=FALSE`)
in both runs, not re-estimated at each origin as in the thesis specification.

RMSE ratio reduced / full, vintage 0 (< 1 = reduced set more accurate):

| Model | Full sample | Crisis (2020Q1–2021Q2) | Normal (2021Q3–2024Q2) |
|---|---|---|---|
| ARMA | 1.000 | 1.000 | 1.000 |
| DFM | 1.303 | 1.333 | 0.927* |
| MIDAS | 0.971 | 0.971 | 0.996 |
| Ridge | 0.991 | 0.993 | 0.928 |
| Lasso | 1.299 | 1.308 | 0.931* |
| Random Forest | 0.994 | 0.993 | 1.080 |
| XGBoost | 0.999 | 1.002 | 0.898 |

\* Two-sided HLN Diebold-Mariano p < 0.10. No crisis-period difference is
significant (n = 6).

Findings:

- In the normal period, dropping the 16 series does not hurt any model; DFM and
  Lasso improve slightly (≈7% lower RMSE).
- In the crisis, DFM and Lasso lose accuracy, concentrated in 2020Q2–Q3: the
  FUSADES confidence indices captured the April–May 2020 collapse (Lasso
  2020Q2 nowcast −6.3% with FUSADES vs 0.0% without; actual −18.7%).
- Sensitivity: setting the FUSADES publication lag to 3 months (instead of 1
  in `meta_data_v2.csv`) leaves the Lasso 2020Q2 nowcast unchanged, because the
  March 2020 reading already reflected the shock.

`series_selection_robustness.csv` has all periods, vintages, RMSEs and DM tests.
Reproduce with `Evaluation/03_series_selection_robustness.py`.

"""
03_series_selection_robustness.py
Robustness check: does excluding the series flagged in
Data/series_selection.csv (FUSADES, ISSS and three BCR series) change
nowcast accuracy?

Both runs use the original evaluation design (2020Q1-2024Q2 rolling origin,
vintages -2/-1/0). Model scripts are run twice, once per data folder, by
pointing ESA_BASE_DIR at each one:

    full    : all 123 indicators (thesis specification)
    reduced : 107 indicators (series_selection.csv, include == 1)

For each model, period and vintage this script reports the RMSE in both runs,
the ratio reduced/full, and a two-sided Diebold-Mariano test (HLN-corrected)
of equal accuracy between the two runs.

Usage
-----
python Evaluation/03_series_selection_robustness.py <full_base_dir> <reduced_base_dir> <out_dir>

Author : Gabriela Aquino
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models"))
from nowcast_utils import LAGS, rmse  # noqa: E402

MODELS = {
    "ARMA":          ("arma",         "arma_predictions.csv"),
    "DFM":           ("dfm",          "dfm_predictions.csv"),
    "MIDAS":         ("midas",        "midas_predictions.csv"),
    "Ridge":         ("ridge",        "ridge_predictions.csv"),
    "Lasso":         ("lasso",        "lasso_predictions.csv"),
    "Random Forest": ("randomforest", "rf_predictions.csv"),
    "XGBoost":       ("xgboost",      "xgb_predictions.csv"),
}
LAG_COL = {-2: "lag_-2", -1: "lag_-1", 0: "lag_+0"}
PERIODS = ["full", "crisis", "normal"]


def load(base: Path, folder: str, fname: str) -> pd.DataFrame | None:
    path = base / "Models" / "results" / folder / fname
    if not path.exists():
        alt = list((base / "Models" / "results" / folder).glob("*predictions*.csv"))
        if not alt:
            return None
        path = alt[0]
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def dm_two_sided(y, e1_pred, e2_pred):
    """HLN-corrected DM test, h = 1, two-sided. d = e_full^2 - e_reduced^2."""
    y, p1, p2 = (np.asarray(a, dtype=float) for a in (y, e1_pred, e2_pred))
    m = ~np.isnan(y) & ~np.isnan(p1) & ~np.isnan(p2)
    y, p1, p2 = y[m], p1[m], p2[m]
    n = len(y)
    d = (y - p1) ** 2 - (y - p2) ** 2
    if n < 5 or np.var(d, ddof=1) == 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(np.var(d, ddof=1) / n)
    dm_hln = dm * np.sqrt((n - 1) / n)
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return dm_hln, p


def main(full_base: Path, red_base: Path, out_dir: Path) -> None:
    rows = []
    for model, (folder, fname) in MODELS.items():
        f, r = load(full_base, folder, fname), load(red_base, folder, fname)
        if f is None or r is None:
            print(f"  [{model}] predictions missing in one run - skipped")
            continue
        if not f["date"].equals(r["date"]):
            raise ValueError(f"[{model}] date grids differ between runs")
        for period in PERIODS:
            mask = np.ones(len(f), bool) if period == "full" else (f["period_type"] == period).values
            y = f.loc[mask, "actual"].values
            for lag in LAGS:
                pf = f.loc[mask, LAG_COL[lag]].values
                pr = r.loc[mask, LAG_COL[lag]].values
                rf, rr = rmse(y, pf), rmse(y, pr)
                dm, p = dm_two_sided(y, pf, pr)
                rows.append({
                    "model": model, "period": period, "vintage": lag,
                    "n": int(mask.sum()),
                    "RMSE_full": rf, "RMSE_reduced": rr,
                    "ratio_reduced_full": rr / rf if rf else np.nan,
                    "DM_HLN": dm, "DM_pvalue_2s": p,
                })

    res = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.round(6).to_csv(out_dir / "series_selection_robustness.csv", index=False)

    nowcast = res[res["vintage"] == 0].pivot(
        index="model", columns="period", values="ratio_reduced_full"
    )[PERIODS].reindex([m for m in MODELS if m in res["model"].unique()])
    nowcast.round(3).to_csv(out_dir / "series_selection_ratio_lag0.csv")

    print("\nRMSE ratio reduced / full (vintage 0; < 1 = reduced set is more accurate)")
    print(nowcast.round(3).to_string())
    sig = res[res["DM_pvalue_2s"] < 0.10]
    print(f"\nCases with DM p < 0.10: {len(sig)} of {res['DM_pvalue_2s'].notna().sum()}")
    if len(sig):
        print(sig[["model", "period", "vintage", "ratio_reduced_full", "DM_pvalue_2s"]]
              .round(3).to_string(index=False))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

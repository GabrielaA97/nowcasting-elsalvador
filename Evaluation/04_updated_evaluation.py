"""
04_updated_evaluation.py
Evaluation of the updated nowcasts (data through 2026M6, 107 indicators).

Reads the outputs of Models/ml_nowcast_v2.py (all cv/imputation
configurations), the benchmarks, and, when present, MIDAS.R and DMF.R
predictions, then reports:

  1. RMSE by period and vintage, and relative to the AR benchmark
       crisis : 2020Q1-2021Q2
       normal : 2021Q3-2024Q2   (thesis evaluation window)
       recent : 2024Q3-2026Q1   (new quarters)
       full   : 2020Q1-2026Q1
  2. Effect of each methodological change versus the original protocol
     (holdout + mean fill): RMSE ratio and two-sided HLN Diebold-Mariano test
  3. Real-time nowcast for the quarter without published GDP (2026Q2)

Usage
-----
python Evaluation/04_updated_evaluation.py <v2_pred_dir> <ESA_BASE_DIR> <out_dir>

Author : Gabriela Aquino
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PERIODS = {
    "crisis": ("2020-03-01", "2021-06-01"),
    "normal": ("2021-09-01", "2024-06-01"),
    "recent": ("2024-09-01", "2026-03-01"),
    "full":   ("2020-03-01", "2026-03-01"),
}
VINTAGES = ["lag_-2", "lag_-1", "lag_+0"]
BASELINE = "holdout_mean"
NAMES = {"ridge": "Ridge", "lasso": "Lasso", "rf": "Random Forest", "xgb": "XGBoost"}


def rmse(y, p):
    m = ~np.isnan(y) & ~np.isnan(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.any() else np.nan


def dm_two_sided(y, p_base, p_new):
    m = ~np.isnan(y) & ~np.isnan(p_base) & ~np.isnan(p_new)
    y, a, b = y[m], p_base[m], p_new[m]
    n = len(y)
    d = (y - a) ** 2 - (y - b) ** 2
    if n < 5 or np.var(d, ddof=1) == 0:
        return np.nan
    dm = d.mean() / np.sqrt(np.var(d, ddof=1) / n) * np.sqrt((n - 1) / n)
    return float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))


def load_all(pred_dir: Path, base: Path) -> pd.DataFrame:
    frames = []
    for f in pred_dir.glob("*_predictions.csv"):
        if f.name == "benchmarks_predictions.csv":
            b = pd.read_csv(f)
            b["config"] = "-"
            frames.append(b)
            continue
        model, cv, imp = f.stem.replace("_predictions", "").split("_")
        d = pd.read_csv(f).assign(model=NAMES[model], config=f"{cv}_{imp}")
        frames.append(d)
    for name, folder, fname in [("MIDAS", "midas", "midas_predictions.csv"),
                                ("DFM", "dfm", "dfm_predictions.csv")]:
        p = base / "Models" / "results" / folder / fname
        if p.exists():
            frames.append(pd.read_csv(p).assign(model=name, config="-"))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[["model", "config", "date", "actual"] + VINTAGES]


def main(pred_dir: Path, base: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = load_all(pred_dir, base)

    # 1. RMSE table
    rows = []
    for (model, cfg), g in df.groupby(["model", "config"]):
        for per, (s, e) in PERIODS.items():
            h = g[(g["date"] >= s) & (g["date"] <= e)]
            for v in VINTAGES:
                rows.append({"model": model, "config": cfg, "period": per, "vintage": v,
                             "n": int(h["actual"].notna().sum()),
                             "RMSE": rmse(h["actual"].values, h[v].values)})
    res = pd.DataFrame(rows)
    ar = res[res["model"] == "AR"].set_index(["period", "vintage"])["RMSE"]
    res["rel_AR"] = res.apply(lambda r: r["RMSE"] / ar[(r["period"], r["vintage"])], axis=1)
    res.round(6).to_csv(out / "rmse_updated.csv", index=False)

    lag0 = res[res["vintage"] == "lag_+0"].pivot_table(
        index=["model", "config"], columns="period", values="RMSE")[list(PERIODS)]
    print("RMSE, vintage 0 (percentage points of quarterly growth)")
    print((lag0 * 100).round(2).to_string())

    # 2. Effect of each change vs original protocol
    eff = []
    for model in NAMES.values():
        g = df[df["model"] == model]
        base_g = g[g["config"] == BASELINE].set_index("date")
        for cfg in sorted(set(g["config"]) - {BASELINE}):
            new_g = g[g["config"] == cfg].set_index("date")
            for per, (s, e) in PERIODS.items():
                idx = base_g.loc[s:e].index
                y = base_g.loc[idx, "actual"].values
                for v in VINTAGES:
                    pb, pn = base_g.loc[idx, v].values, new_g.loc[idx, v].values
                    eff.append({"model": model, "config": cfg, "period": per, "vintage": v,
                                "ratio_vs_original": rmse(y, pn) / rmse(y, pb),
                                "DM_p_2s": dm_two_sided(y, pb, pn)})
    eff = pd.DataFrame(eff)
    eff.round(4).to_csv(out / "effect_vs_original.csv", index=False)
    e0 = eff[eff["vintage"] == "lag_+0"].pivot_table(
        index=["model", "config"], columns="period", values="ratio_vs_original")[list(PERIODS)]
    print("\nRMSE ratio vs original protocol (holdout + mean fill), vintage 0; < 1 = better")
    print(e0.round(3).to_string())
    sig = eff[(eff["DM_p_2s"] < 0.10)]
    print(f"\nDM p < 0.10: {len(sig)} of {eff['DM_p_2s'].notna().sum()}")
    if len(sig):
        print(sig.round(3).to_string(index=False))

    # 3. Real-time nowcast
    nc = df[df["actual"].isna() & (df["date"] > pd.Timestamp(PERIODS["full"][1]))]
    if len(nc):
        nc = nc.copy()
        nc[VINTAGES] = nc[VINTAGES] * 100
        nc.round(2).to_csv(out / "nowcast_current.csv", index=False)
        print(f"\nNowcast {nc['date'].max().date()} (q/q growth, %)")
        print(nc[["model", "config"] + VINTAGES].round(2).to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

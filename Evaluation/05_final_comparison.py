"""
05_final_comparison.py
Final model comparison on the latest data: year-on-year nowcasts, RMSE ranking,
Model Confidence Set (Hansen, Lunde & Nason 2011, as recommended in the IMFNow
working paper), and comparison with an external nowcast (e.g. IMF NowHub).

Year-on-year nowcasts are obtained by chaining each model's q/q nowcast
(vintage 0) with the three previously published q/q growth rates of the
seasonally adjusted GDP index.

Usage
-----
python Evaluation/05_final_comparison.py <ml_pred_dir> <ESA_BASE_DIR> <external_csv|-> <out_dir>

external_csv columns: date (quarter-end month, e.g. 2026-06-01), nowcast_yoy,
actual_yoy (percent). Use "-" to skip.

Author : Gabriela Aquino
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import MCS

NAMES = {"ridge": "Ridge", "lasso": "Lasso", "rf": "Random Forest", "xgb": "XGBoost"}
CFG = {"holdout_mean": "original", "tscv_regression": "nueva"}
PERIODS = {
    "Total 2020T1-": ("2020-03-01", None),
    "Sin COVID 2021T3-": ("2021-09-01", None),
    "Recientes 2024T3-": ("2024-09-01", None),
}


def load(pred_dir, base):
    fr = []
    for f in pred_dir.glob("*_predictions.csv"):
        d = pd.read_csv(f, parse_dates=["date"])
        if f.name == "benchmarks_predictions.csv":
            for m, g in d.groupby("model"):
                fr.append(g.assign(label={"AR": "AR (BIC)", "HistMean": "Promedio histórico",
                                          "RandomWalk": "Random walk"}[m]))
            continue
        m, cv, imp = f.stem.replace("_predictions", "").split("_")
        fr.append(d.assign(label=f"{NAMES[m]} ({CFG.get(f'{cv}_{imp}', cv + '_' + imp)})"))
    for lab, folder, fn in [("MIDAS", "midas", "midas_predictions.csv"), ("DFM", "dfm", "dfm_predictions.csv")]:
        p = base / "Models" / "results" / folder / fn
        if p.exists():
            fr.append(pd.read_csv(p, parse_dates=["date"]).assign(label=lab))
    df = pd.concat(fr, ignore_index=True)[["label", "date", "actual", "lag_+0"]]
    return df.rename(columns={"lag_+0": "qq"})


def main(pred_dir, base, ext_csv, out):
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(base / "Data" / "data_tf.csv", parse_dates=["date"]).set_index("date")
    gdp = data["GDP"].dropna()
    df = load(pred_dir, base)
    last_actual = gdp.index.max()

    def chain(date, q):
        prev = gdp.loc[date - pd.DateOffset(months=9): date - pd.DateOffset(months=3)]
        return (np.prod(1 + prev.values) * (1 + q) - 1) * 100 if len(prev) == 3 else np.nan

    df["yoy"] = [chain(d, q) for d, q in zip(df["date"], df["qq"])]
    df["yoy_actual"] = [chain(d, gdp.get(d, np.nan)) for d in df["date"]]
    df.to_csv(out / "all_nowcasts_yoy.csv", index=False)

    # RMSE table (y/y and q/q, pp)
    rows = []
    for lab, g in df.groupby("label"):
        r = {"Modelo": lab}
        for per, (s, _) in PERIODS.items():
            h = g[(g["date"] >= s) & (g["date"] <= last_actual)]
            r[f"RMSE a/a {per}"] = np.sqrt(np.mean((h["yoy"] - h["yoy_actual"]) ** 2))
        h = g[g["date"] <= last_actual]
        r["RMSE t/t total"] = np.sqrt(np.mean((h["qq"] - h["actual"]) ** 2)) * 100
        rows.append(r)
    rm = pd.DataFrame(rows).set_index("Modelo")

    # Model Confidence Set on squared q/q errors (90%)
    mcs_cols = {}
    for per, (s, _) in PERIODS.items():
        w = df[(df["date"] >= s) & (df["date"] <= last_actual)]
        L = w.pivot(index="date", columns="label", values="qq").sub(
            w.groupby("date")["actual"].first(), axis=0) ** 2
        L = L.dropna(axis=1)
        # identical loss series (e.g. AR with 0 lags = historical mean) make the
        # MCS statistic undefined: keep one, give the duplicates its p-value
        dup = L.T.duplicated()
        twins = {c: L.columns[(L.T == L[c]).all(axis=1).values][0] for c in L.columns[dup.values]}
        L = L.loc[:, ~dup.values]
        mcs = MCS(L, size=0.10, reps=5000, block_size=4, bootstrap="stationary",
                  method="R", seed=2026)
        mcs.compute()
        pv = mcs.pvalues["Pvalue"].copy()
        for c, t in twins.items():
            pv[c] = pv[t]
        mcs_cols[per] = pv
    for per, pv in mcs_cols.items():
        rm[f"MCS p {per}"] = pv
    rm = rm.sort_values("RMSE a/a Sin COVID 2021T3-")
    rm.round(3).to_csv(out / "ranking_rmse_mcs.csv")

    # Recent quarters table (y/y)
    qs = sorted(df["date"].unique())[-5:]
    tab = df[df["date"].isin(qs)].pivot(index="label", columns="date", values="yoy")
    tab.columns = [f"{c.year}T{(c.month - 1)//3 + 1}" for c in tab.columns]
    act = df[df["date"].isin(qs)].groupby("date")["yoy_actual"].first()
    tab.loc["Observado (índice desest.)"] = act.values
    tab = tab.reindex(list(rm.index) + ["Observado (índice desest.)"])
    tab.round(2).to_csv(out / "recent_nowcasts_yoy.csv")

    pd.set_option("display.width", 250)
    print(rm.round(2).to_string())
    print("\nNowcasts y/y (%)\n", tab.round(2).to_string())

    # External comparison
    if ext_csv != "-":
        ext = pd.read_csv(ext_csv, parse_dates=["date"]).set_index("date")
        common = [d for d in ext.index if d <= last_actual and not np.isnan(ext.loc[d, "actual_yoy"])]
        rows = [{"Modelo": "Externo (NowHub)",
                 **{f"{d.year}T{(d.month-1)//3+1}": ext.loc[d, "nowcast_yoy"] for d in ext.index},
                 "RMSE vs oficial": np.sqrt(np.mean((ext.loc[common, "nowcast_yoy"] - ext.loc[common, "actual_yoy"]) ** 2))}]
        for lab, g in df.groupby("label"):
            g = g.set_index("date")
            r = {"Modelo": lab}
            for d in ext.index:
                r[f"{d.year}T{(d.month-1)//3+1}"] = g["yoy"].get(d, np.nan)
            r["RMSE vs oficial"] = np.sqrt(np.mean((g.loc[common, "yoy"] - ext.loc[common, "actual_yoy"]) ** 2))
            rows.append(r)
        ec = pd.DataFrame(rows).set_index("Modelo").sort_values("RMSE vs oficial")
        ec.loc["Oficial (a/a publicado)"] = [ext.loc[d, "actual_yoy"] for d in ext.index] + [np.nan]
        ec.round(2).to_csv(out / "external_comparison.csv")
        print("\nComparison with external nowcast, same quarters, vs official y/y\n", ec.round(2).to_string())


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))

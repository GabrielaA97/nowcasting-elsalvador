"""
06_shap_current_nowcast.py
Which indicators drive the nowcasts? SHAP decomposition for the ML models.

For each model (Ridge, Lasso, Random Forest, XGBoost) the script re-fits the
model at the latest origin exactly as Models/ml_nowcast_v2.py does (same
hyperparameter selection, imputation and vintage), and computes:

  1. Local SHAP values for the current nowcast (vintage 0): contribution of
     each indicator, in percentage points of q/q GDP growth, relative to the
     model's average prediction on the training sample (the "base value").
  2. Global importance: mean |SHAP| over the training sample.

Linear models use exact linear SHAP (coef * (x - mean)), tree models use
TreeExplainer. Contributions are summed over the current value and the four
monthly lags of each indicator, and then over indicator groups.

Usage
-----
python Evaluation/06_shap_current_nowcast.py <ESA_BASE_DIR> <nowcast_date> <cv> <impute> <out_dir>

Author : Gabriela Aquino
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models"))
import ml_nowcast_v2 as ml  # noqa: E402
from nowcast_utils import TARGET_VARIABLE, TRAIN_START, flatten_data, gen_lagged_data  # noqa: E402

GROUPS = [
    (r"^GDP_", "PIB sectorial (trimestral)"),
    (r"^IVAE_|^IPI$", "Actividad (IVAE, IPI)"),
    (r"^EXP_", "Exportaciones"),
    (r"^IMP_", "Importaciones"),
    (r"^REM$", "Remesas"),
    (r"_US$|_US_LA$|^MTB_6$", "Estados Unidos"),
    (r"^ING|^GPC$|^GIP$|^BFG$", "Finanzas públicas"),
    (r"^M3$|^C[A-Z]+$|^TIP_|^TPR", "Dinero, crédito y tasas"),
    (r"ENER|^TRAN_|_PAS$|CEM", "Energía, transporte y construcción"),
]


def group_of(series):
    for pat, g in GROUPS:
        if re.search(pat, series):
            return g
    return "Otros"


def base_series(feature):
    return re.sub(r"_[1-4]$", "", feature)


def main(base, nowcast_date, cv, impute, out):
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(base / "Data" / "data_tf.csv", parse_dates=["date"]).sort_values("date")
    meta = pd.read_csv(base / "Data" / "meta_data_v2.csv")
    nd = pd.to_datetime(nowcast_date)
    cutoff = nd - pd.DateOffset(months=3)
    train_raw = data[(data["date"] >= TRAIN_START) & (data["date"] <= cutoff)].copy()
    imp = ml.Imputer(impute, train_raw, meta)
    tf = flatten_data(imp.fill(train_raw), TARGET_VARIABLE, ml.N_LAGS_FEATURES)
    tf = ml.quarterly_rows(tf).dropna().reset_index(drop=True)
    X, y = tf.drop(columns=["date", TARGET_VARIABLE]), tf[TARGET_VARIABLE]
    vint = gen_lagged_data(meta, data, nowcast_date, 0)
    vf = flatten_data(imp.fill(vint), TARGET_VARIABLE, ml.N_LAGS_FEATURES)
    Xt = vf.loc[vf["date"] == nd].drop(columns=["date", TARGET_VARIABLE], errors="ignore") \
           .reindex(columns=X.columns, fill_value=0.0)

    local_rows, global_rows, summary = [], [], []
    for name in ["ridge", "lasso", "rf", "xgb"]:
        params = ml.select_params(name, X, y, cv)
        if ml.SCALE[name]:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X)
            Xs, Xts = sc.transform(X), sc.transform(Xt)
        else:
            Xs, Xts = X.values, Xt.values
        model = ml.make_model(name, params).fit(Xs, y)
        if name in ("ridge", "lasso"):
            mu = Xs.mean(axis=0)
            sv_local = model.coef_ * (Xts[0] - mu)
            sv_train = model.coef_ * (Xs - mu)
            base_val = float(model.predict(Xs).mean())
        else:
            expl = shap.TreeExplainer(model)
            sv_local = expl.shap_values(Xts)[0]
            sv_train = expl.shap_values(Xs)
            base_val = float(np.ravel(expl.expected_value)[0])
        pred = float(model.predict(Xts)[0])
        summary.append({"model": name, "base_value_pp": base_val * 100, "nowcast_pp": pred * 100,
                        "sum_shap_pp": float(sv_local.sum()) * 100, "params": str(params)})
        for f, v, g in zip(X.columns, sv_local, np.abs(sv_train).mean(axis=0)):
            s = base_series(f)
            local_rows.append({"model": name, "series": s, "group": group_of(s), "shap_pp": v * 100})
            global_rows.append({"model": name, "series": s, "group": group_of(s), "mean_abs_shap_pp": g * 100})
        print(f"[{name}] base {base_val*100:.2f} + shap {sv_local.sum()*100:.2f} = nowcast {pred*100:.2f} (q/q, pp)")

    loc = pd.DataFrame(local_rows).groupby(["model", "group", "series"], as_index=False)["shap_pp"].sum()
    glo = pd.DataFrame(global_rows).groupby(["model", "group", "series"], as_index=False)["mean_abs_shap_pp"].sum()
    loc.to_csv(out / "shap_local_by_series.csv", index=False)
    glo.to_csv(out / "shap_global_by_series.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "shap_summary.csv", index=False)
    lg = loc.groupby(["group", "model"])["shap_pp"].sum().unstack().round(3)
    gg = glo.groupby(["group", "model"])["mean_abs_shap_pp"].sum().unstack().round(3)
    lg.to_csv(out / "shap_local_by_group.csv"); gg.to_csv(out / "shap_global_by_group.csv")
    print("\nContribution to the current nowcast by group (pp of q/q growth)\n", lg.to_string())
    print("\nGlobal importance by group (mean |SHAP|, pp)\n", gg.to_string())
    for m in ["ridge", "lasso", "rf", "xgb"]:
        t = glo[glo.model == m].nlargest(8, "mean_abs_shap_pp")
        print(f"\nTop 8 series, {m}:", ", ".join(f"{r.series} ({r.mean_abs_shap_pp:.2f})" for r in t.itertuples()))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5]))

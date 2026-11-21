"""
ml_nowcast_v2.py
Rolling-origin nowcasting for the machine-learning models (Ridge, Lasso,
Random Forest, XGBoost) and simple benchmarks, with two methodological
options added to the original notebook pipelines:

1. Hyperparameter selection (--cv)
   holdout : original protocol. Fixed holdout = last 8 quarters of the
             training window (Ridge.ipynb, Lasso.ipynb, ...).
   tscv    : time-series cross-validation with successive validation windows.
             K blocks of 4 consecutive quarters, evenly spaced between the
             first block that leaves MIN_TRAIN_Q quarters for training and the
             end of the training window. Each block is predicted by a model
             trained only on the quarters before it (expanding window), and the
             candidate with the lowest pooled validation RMSE is selected.
             Everything stays inside the training window of each origin.

2. Ragged-edge imputation (--impute)
   mean       : original protocol. Missing values are filled with the
                training-window mean (nowcast_utils.mean_fill_dataset).
   regression : each missing monthly predictor is imputed by a regression on
                all other monthly predictors (sklearn IterativeImputer with
                BayesianRidge), fitted on the training window of each origin
                and then applied to the training panel and to every vintage.
                Quarterly series and GDP keep the mean fill.

Everything else follows the notebooks: features = current month plus 4 monthly
lags of every predictor (flatten_data), vintages -2/-1/0 simulated with
gen_lagged_data, expanding training window from 2005Q1 with a one-quarter gap,
same hyperparameter grids.

If --nowcast-date is a quarter without published GDP, the script also
produces the real-time nowcast for that quarter (training on all published
quarters).

Usage
-----
python Models/ml_nowcast_v2.py --base <ESA_BASE_DIR> --out <out_dir> \
    --cv tscv --impute regression --models ridge,lasso,rf,xgb,bench \
    --test-start 2020-03-01 --test-end 2026-03-01 --nowcast-date 2026-06-01

Author : Gabriela Aquino
"""

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nowcast_utils import (  # noqa: E402
    LAGS, TARGET_VARIABLE, TRAIN_START, RANDOM_SEED,
    gen_lagged_data, flatten_data, mean_fill_dataset, rmse,
)

warnings.filterwarnings("ignore")

N_LAGS_FEATURES = 4
N_HOLDOUT_QUARTERS = 8
TSCV_K = 5            # number of validation blocks
TSCV_BLOCK = 4        # quarters per validation block
MIN_TRAIN_Q = 16      # minimum quarters before the first validation block

GRIDS = {
    "ridge": [{"alpha": a} for a in np.logspace(-6, 6, 40)],
    "lasso": [{"alpha": a} for a in np.logspace(-6, 0, 40)],
    "rf": [dict(zip(k, v)) for k, v in [(
        ("n_estimators", "max_depth", "min_samples_leaf", "max_features"), c)
        for c in itertools.product([200, 500], [3, 5, None], [3, 5], ["sqrt", 0.1])]],
    "xgb": [dict(zip(k, v)) for k, v in [(
        ("max_depth", "n_estimators", "learning_rate", "reg_lambda"), c)
        for c in itertools.product([2, 3], [50, 100, 150], [0.05, 0.10], [1.0, 5.0])]],
}
SCALE = {"ridge": True, "lasso": True, "rf": False, "xgb": False}


def make_model(name, params):
    if name == "ridge":
        return Ridge(fit_intercept=True, **params)
    if name == "lasso":
        return Lasso(fit_intercept=True, max_iter=50000, tol=1e-5, **params)
    if name == "rf":
        return RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **params)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                            objective="reg:squarederror", random_state=RANDOM_SEED,
                            n_jobs=-1, verbosity=0, **params)
    raise ValueError(name)


def fit_predict(name, params, X_tr, y_tr, X_te):
    if SCALE[name]:
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    return make_model(name, params).fit(X_tr, y_tr).predict(X_te)


def validation_splits(n, cv):
    """Return list of (train_idx_end, val_start, val_end) on quarterly rows."""
    if cv == "holdout":
        return [(n - N_HOLDOUT_QUARTERS, n - N_HOLDOUT_QUARTERS, n)]
    last_start = n - TSCV_BLOCK
    starts = np.unique(np.linspace(MIN_TRAIN_Q, last_start, TSCV_K).astype(int))
    return [(s, s, s + TSCV_BLOCK) for s in starts]


def select_params(name, X, y, cv):
    splits = validation_splits(len(X), cv)
    best, best_score = None, np.inf
    for params in GRIDS[name]:
        err = []
        for tr_end, v0, v1 in splits:
            pred = fit_predict(name, params, X.iloc[:tr_end], y.iloc[:tr_end], X.iloc[v0:v1])
            err.append((y.iloc[v0:v1].values - pred) ** 2)
        score = np.sqrt(np.mean(np.concatenate(err)))
        if score < best_score:
            best, best_score = params, score
    return best


class Imputer:
    """Mean fill (original) or regression fill of monthly predictors."""

    def __init__(self, method, train_raw, metadata):
        self.method, self.train_raw = method, train_raw
        q = set(metadata.loc[metadata["freq"].str.lower() == "q", "series"])
        self.reg_cols = [c for c in train_raw.columns
                         if c not in ("date", TARGET_VARIABLE) and c not in q]
        if method == "regression":
            self.imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                                        initial_strategy="mean", random_state=RANDOM_SEED,
                                        skip_complete=True, keep_empty_features=True)
            self.imp.fit(np.array(train_raw[self.reg_cols], dtype=float))

    def fill(self, df):
        out = df.copy()
        if self.method == "regression":
            vals = np.array(out[self.reg_cols], dtype=float)
            all_nan = np.isnan(vals).all(axis=0)
            filled = self.imp.transform(vals)
            out[self.reg_cols] = filled
            # columns entirely missing in df fall back to the mean fill below
            out.loc[:, np.array(self.reg_cols)[all_nan]] = np.nan
        return mean_fill_dataset(training=self.train_raw, test=out)


def quarterly_rows(flat):
    return flat.loc[flat["date"].dt.month.isin([3, 6, 9, 12])]


def run_ml(name, data, metadata, origins, cv, impute):
    rows = []
    for nowcast_date in origins:
        t0 = time.time()
        nd = pd.to_datetime(nowcast_date)
        actual = data.loc[data["date"] == nd, TARGET_VARIABLE]
        actual = float(actual.values[0]) if len(actual) and not pd.isna(actual.values[0]) else np.nan
        cutoff = nd - pd.DateOffset(months=3)
        train_raw = data.loc[(data["date"] >= TRAIN_START) & (data["date"] <= cutoff)].copy()

        imp = Imputer(impute, train_raw, metadata)
        train_flat = flatten_data(imp.fill(train_raw), TARGET_VARIABLE, N_LAGS_FEATURES)
        train_flat = quarterly_rows(train_flat).dropna(axis=0, how="any").reset_index(drop=True)
        X = train_flat.drop(columns=["date", TARGET_VARIABLE])
        y = train_flat[TARGET_VARIABLE]

        params = select_params(name, X, y, cv)
        row = {"date": nowcast_date, "actual": actual, "params": str(params)}
        if SCALE[name]:
            sc = StandardScaler().fit(X)
            model = make_model(name, params).fit(sc.transform(X), y)
        else:
            sc, model = None, make_model(name, params).fit(X, y)

        for lag in LAGS:
            vint = gen_lagged_data(metadata, data, nowcast_date, lag)
            vflat = flatten_data(imp.fill(vint), TARGET_VARIABLE, N_LAGS_FEATURES)
            target = vflat.loc[vflat["date"] == nd]
            if target.empty:
                row[f"lag_{lag:+d}" if lag == 0 else f"lag_{lag}"] = np.nan
                continue
            Xt = target.drop(columns=["date", TARGET_VARIABLE], errors="ignore") \
                       .reindex(columns=X.columns, fill_value=0.0)
            Xt = sc.transform(Xt) if sc is not None else Xt
            row["lag_+0" if lag == 0 else f"lag_{lag}"] = float(model.predict(Xt)[0])
        rows.append(row)
        print(f"  [{name}|{cv}|{impute}] {nowcast_date}  {params}  {time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(rows)


def run_benchmarks(data, origins):
    """AR(p) on GDP (p by BIC, max 4), historical mean, random walk."""
    gdp = data.loc[data[TARGET_VARIABLE].notna(), ["date", TARGET_VARIABLE]]
    rows = []
    for nowcast_date in origins:
        nd = pd.to_datetime(nowcast_date)
        actual = gdp.loc[gdp["date"] == nd, TARGET_VARIABLE]
        actual = float(actual.values[0]) if len(actual) else np.nan
        # GDP of the previous quarter is published ~3 months after its end,
        # so at the end of quarter t the last known GDP is t-1 (same as ARMA.ipynb).
        hist = gdp.loc[(gdp["date"] >= TRAIN_START) &
                       (gdp["date"] <= nd - pd.DateOffset(months=3)), TARGET_VARIABLE].values
        sel = ar_select_order(hist, maxlag=4, ic="bic", trend="c")
        lags = sel.ar_lags if sel.ar_lags else 0
        ar = AutoReg(hist, lags=lags, trend="c").fit()
        ar_fc = float(ar.predict(start=len(hist), end=len(hist))[0])
        for name, fc in [("AR", ar_fc), ("HistMean", float(np.mean(hist))), ("RandomWalk", float(hist[-1]))]:
            rows.append({"model": name, "date": nowcast_date, "actual": actual,
                         "lag_-2": fc, "lag_-1": fc, "lag_+0": fc})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv", choices=["holdout", "tscv"], default="tscv")
    ap.add_argument("--impute", choices=["mean", "regression"], default="regression")
    ap.add_argument("--models", default="ridge,lasso,rf,xgb,bench")
    ap.add_argument("--test-start", default="2020-03-01")
    ap.add_argument("--test-end", default="2024-06-01")
    ap.add_argument("--nowcast-date", default=None)
    a = ap.parse_args()

    base, out = Path(a.base), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(base / "Data" / "data_tf.csv", parse_dates=["date"]) \
             .sort_values("date").reset_index(drop=True)
    metadata = pd.read_csv(base / "Data" / "meta_data_v2.csv")
    origins = pd.date_range(a.test_start, a.test_end, freq="3MS").strftime("%Y-%m-%d").tolist()
    if a.nowcast_date and a.nowcast_date not in origins:
        origins.append(a.nowcast_date)

    tag = f"{a.cv}_{a.impute}"
    for m in a.models.split(","):
        if m == "bench":
            run_benchmarks(data, origins).to_csv(out / "benchmarks_predictions.csv", index=False)
            continue
        res = run_ml(m, data, metadata, origins, a.cv, a.impute)
        res.to_csv(out / f"{m}_{tag}_predictions.csv", index=False)


if __name__ == "__main__":
    main()

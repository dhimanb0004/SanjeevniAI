"""
train_engine2.py - Training pipeline for Sanjeevni Engine 2.

Loads the standalone 2000-2026 climate time series (SeasonalClimaticData_
Engine2.csv) -- NOT the engineered BiologicalMatrix dataset Engine 1 trains
on; Engine 2 has no crop information at all. Fits one Prophet (or Huber
fallback, for sparse groups) model per State+District+Season+target, uses
it immediately to produce a forecast, then discards it -- no model object
is ever saved. The only output is a single flat CSV of precomputed forecast
values (engine2_climate_forecast.csv).

This script is only meant for retraining/refreshing forecasts occasionally;
it is not run during deployment or per farmer query. There is no separate
serving module for Engine 2 -- engine1_serving.py loads this script's CSV
output directly as a plain pandas DataFrame at import time, with no model
object involved anywhere at serving time.
"""

import os
import warnings
import logging

# Prophet and cmdstanpy print a lot of noise by default, silencing both so the
# actual progress prints below are the only thing showing up in the console.
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").disabled = True

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from joblib import Parallel, delayed


# ---------------- Configuration ----------------

DATA_DIR = os.environ.get("SANJEEVNI_DATA_DIR", r"C:\Sanjeevni_App\data")
ARTIFACT_DIR = os.environ.get("SANJEEVNI_ARTIFACT_DIR", r"C:\Sanjeevni_App\artifacts")

INPUT_PATH = os.path.join(DATA_DIR, "SeasonalClimaticData_Engine2.csv")
OUT_PATH = os.path.join(ARTIFACT_DIR, "engine2_climate_forecast.csv")
PARTIAL_OUT_PATH = OUT_PATH + ".partial"  # saving progress here as we go, in case the run gets interrupted partway through

GROUP_COLS = ["State", "District", "Season"]
TARGETS = ["T_Avg", "R_Total"]
MIN_YEARS = 15          # need at least 15 years before we trust Prophet's trend detection, otherwise fall back to Huber+poly2
FORECAST_H = 1          # only forecasting 1 year ahead, that's all serving actually needs
ANOMALY_WIN = 10        # comparing the forecast against the last 10 years to see if it's unusual
Z_THRESH = 1.0          # how many standard deviations off before we call it Hotter/Cooler/Drier/Wetter
CHECKPOINT_EVERY = 500  # saving progress every 500 groups so a crash doesn't mean starting over from scratch


# -------------------- Fitting one target (temperature or rainfall) --------------------
# Prophet is the default because it can actually pick up on a real shift in trend --
# like a district steadily getting hotter -- instead of just averaging everything out.
# But it needs a decent stretch of history to trust that kind of pattern, so below
# 15 years we fall back to something simpler: Huber regression on a degree-2 curve.

def fit_prophet(sub: pd.DataFrame, target_col: str, periods: int = FORECAST_H) -> np.ndarray:
    pdf = sub[["Year", target_col]].rename(columns={"Year": "ds", target_col: "y"})
    pdf["ds"] = pd.to_datetime(pdf["ds"], format="%Y")
    # We only have one data point per year, so there's no weekly/daily/yearly
    # seasonality to find -- turning all of that off.
    m = Prophet(
        changepoint_prior_scale=0.5,  # bumped this up so the model reacts to recent shifts instead of smoothing them away
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
    )
    m.fit(pdf)
    future = m.make_future_dataframe(periods=periods, freq="YS")
    return m.predict(future)["yhat"].tail(periods).values


def fit_huber(sub: pd.DataFrame, target_col: str, periods: int = FORECAST_H) -> np.ndarray:
    base = sub["Year"].min()  # centering years around 0 so the polynomial fit stays numerically stable instead of working with raw years like 2015
    years_rel = (sub["Year"].values - base).reshape(-1, 1)
    poly = PolynomialFeatures(degree=2).fit(years_rel)
    model = HuberRegressor(max_iter=500).fit(poly.transform(years_rel), sub[target_col].values)
    last_rel = sub["Year"].max() - base
    future_rel = np.arange(last_rel + 1, last_rel + 1 + periods).reshape(-1, 1)
    return model.predict(poly.transform(future_rel))


def fit_target(sub: pd.DataFrame, target_col: str, periods: int = FORECAST_H, min_years: int = MIN_YEARS):
    """Uses Prophet if there's enough history, otherwise falls back to Huber+poly2.
    Never lets an exception escape -- if Prophet fails for whatever reason, we just
    fall back to Huber instead of taking down the whole batch. Same idea as the
    cold-start fallback tiers in Engine 1."""
    if len(sub) >= min_years:
        try:
            return fit_prophet(sub, target_col, periods), "prophet", len(sub)
        except Exception:
            pass
    return fit_huber(sub, target_col, periods), "huber_poly2", len(sub)


# -------------------- Anomaly scoring and climate label --------------------
# Checking how far off the forecast is from what's been normal recently, then
# turning that into a plain-English label a farmer can actually read.

def anomaly_z(forecast_val: float, actual_series: pd.Series, window: int = ANOMALY_WIN) -> float:
    """How many standard deviations is the forecast from the trailing `window`-year
    average of actual values?"""
    recent = actual_series.tail(window)
    mu, sigma = recent.mean(), recent.std(ddof=0)
    return 0.0 if sigma == 0 or np.isnan(sigma) else (forecast_val - mu) / sigma


def climate_label(t_z: float, r_z: float, thresh: float = Z_THRESH) -> str:
    tags = []
    if t_z >= thresh:
        tags.append("Hotter")
    if t_z <= -thresh:
        tags.append("Cooler")
    if r_z <= -thresh:
        tags.append("Drier")
    if r_z >= thresh:
        tags.append("Wetter")
    return (" & ".join(tags) + " than normal") if tags else "Normal"


# -------------------- Turning one State+District+Season group into one output row --------------------
# Wrapping this in its own try/except so if one group has a problem -- bad data, Prophet
# failing to converge, whatever -- it doesn't take the entire batch down with it.

def process_group(key: tuple, sub: pd.DataFrame) -> dict:
    logging.getLogger("cmdstanpy").disabled = True  # each parallel worker starts fresh, so this needs to be silenced again here
    state, district, season = key
    sub = sub.sort_values("Year")
    try:
        t_val, t_method, n = fit_target(sub, "T_Avg")
        r_val, r_method, _ = fit_target(sub, "R_Total")
        t_z = anomaly_z(t_val[0], sub["T_Avg"])
        r_z = anomaly_z(r_val[0], sub["R_Total"])
        return {
            "State": state, "District": district, "Season": season,
            "forecast_T_Avg": round(float(t_val[0]), 3),
            "forecast_R_Total": round(float(r_val[0]), 2),
            "T_Avg_anomaly_z": round(float(t_z), 3),
            "R_Total_anomaly_z": round(float(r_z), 3),
            "Climate_Label": climate_label(t_z, r_z),
            "method_T_Avg": t_method, "method_R_Total": r_method,
            "history_years": n, "status": "ok",
        }
    except Exception as e:
        return {
            "State": state, "District": district, "Season": season,
            "forecast_T_Avg": np.nan, "forecast_R_Total": np.nan,
            "T_Avg_anomaly_z": np.nan, "R_Total_anomaly_z": np.nan,
            "Climate_Label": np.nan, "method_T_Avg": "failed", "method_R_Total": "failed",
            "history_years": len(sub), "status": f"error: {e}",
        }


# -------------------- Backtesting (optional, run this by hand) --------------------
# main() never calls any of this. It's just here for when I want to check how
# accurate the forecasts actually are -- not something that needs to run every
# time we train.

def backtest_group(sub: pd.DataFrame, target_col: str, hold_out: int = 2):
    sub = sub.sort_values("Year")
    train, test = sub.iloc[:-hold_out], sub.iloc[-hold_out:]
    preds, method, _ = fit_target(train, target_col, periods=hold_out)
    err = test[target_col].values - preds
    return np.abs(err).mean(), float(np.sqrt((err ** 2).mean())), method


def run_backtest(df: pd.DataFrame, n_sample: int = 100, seed: int = 42) -> pd.DataFrame:
    """Hides the last 2 years of each group, forecasts them, and checks how close
    we actually got. n_sample=100 keeps this quick for a spot-check -- pass the full
    group count if you want to backtest everything. Run this manually when you want
    to double-check accuracy, it's not part of the normal training run."""
    groups = list(df.groupby(GROUP_COLS, observed=True))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(groups), size=min(n_sample, len(groups)), replace=False)

    bt_rows = []
    for idx in sample_idx:
        key, sub = groups[idx]
        if len(sub) < 5:
            continue
        for tgt in TARGETS:
            mae, rmse, method = backtest_group(sub, tgt, hold_out=2)
            bt_rows.append({
                "State": key[0], "District": key[1], "Season": key[2],
                "target": tgt, "MAE": mae, "RMSE": rmse, "method": method,
            })

    bt_df = pd.DataFrame(bt_rows)
    print(bt_df.groupby(["target", "method"])[["MAE", "RMSE"]].mean())
    return bt_df


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(df.shape, df.columns.tolist())
    assert df.isnull().sum().sum() == 0, "unexpected nulls in input climate data"
    print(df.groupby(GROUP_COLS, observed=True).ngroups, "State+District+Season groups")
    print(df["Year"].min(), "-", df["Year"].max())

    groups = list(df.groupby(GROUP_COLS, observed=True))
    all_rows = []

    for i in range(0, len(groups), CHECKPOINT_EVERY):
        batch = groups[i: i + CHECKPOINT_EVERY]
        batch_rows = Parallel(n_jobs=-1)(delayed(process_group)(k, g) for k, g in batch)
        all_rows.extend(batch_rows)
        pd.DataFrame(all_rows).to_csv(PARTIAL_OUT_PATH, index=False)
        print(f"{min(i + CHECKPOINT_EVERY, len(groups))}/{len(groups)} groups done")

    engine2_output = pd.DataFrame(all_rows)
    engine2_output.to_csv(OUT_PATH, index=False)
    print("Saved ->", OUT_PATH)
    print("Failures:", (engine2_output["status"] != "ok").sum())

    print(engine2_output["method_T_Avg"].value_counts())
    print(engine2_output["Climate_Label"].value_counts())


if __name__ == "__main__":
    main()

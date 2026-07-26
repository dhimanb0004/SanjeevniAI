"""
engine1_serving.py — Sanjeevni Engine 1 serving path.

This is what actually runs when a farmer submits a query -- everything
train_engine1.py already computed gets loaded here once, and generate_
shortlist()/generate_shortlist_forecast() just filter and look things up
against that, plus a handful of cheap single-row model.predict() calls
(one per shortlisted crop, never a full national pass). No training, no
retraining, no recomputing the whole country's scores happens anywhere
in this file.

Ported over from engine1.ipynb -- specifically Cells 25-29 (resolve_season),
33 (irrigation multiplier), 45-46 (CATEGORY_AGGREGATES, extrapolation risk),
48 (generate_shortlist), and Cell 56 -- and deliberately NOT Cells 57 or 58.
Those last two were broken debug leftovers I never meant to keep: Cell 58
hardcoded T_Avg=45/R_Total=5000 into every single prediction, completely
overriding whatever Engine 2 actually forecasted. Since it happened to be
the last cell in the notebook, running the whole thing top-to-bottom always
left that broken version active without any error telling you so. What's
below is Cell 56's version -- the one that was actually correct.
"""

import os
import joblib
import pandas as pd
from scipy.stats import percentileofscore

# ---------------- Config ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.environ.get("SANJEEVNI_ARTIFACT_DIR", os.path.join(BASE_DIR, "artifacts"))
DATA_DIR = os.environ.get("SANJEEVNI_DATA_DIR", os.path.join(BASE_DIR, "data"))

MODEL_PATH = os.path.join(ARTIFACT_DIR, "engine1_lightgbm_model.pkl")
SEASON_LOOKUP_PATH = os.path.join(ARTIFACT_DIR, "season_lookup.pkl")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "engine1_metadata.pkl")
LATEST_SCORED_PATH = os.path.join(ARTIFACT_DIR, "engine1_latest_scored.csv")
WATER_NEED_PATH = os.path.join(DATA_DIR, "crop_water_need.csv")
CLIMATE_BOUNDS_PATH = os.path.join(DATA_DIR, "crop_climate_bounds.csv")
ENGINE2_FORECAST_PATH = os.path.join(ARTIFACT_DIR, "engine2_climate_forecast.csv")  # this is train_engine2.py's output, not something that lives in data/

FOOTPRINT_THRESHOLD = 0.01  # a crop needs at least 1% of the district's cultivated area to even be considered
IRRIGATION_LEVEL_MAP = {"Good": 1.0, "Average": 0.66, "Poor": 0.33}
CATEGORY_AGGREGATES = {"Cereals", "Kharif Pulse", "Rabi Pulse", "Oilseed", "Millet"}


# -------------------- Loading everything once, at import time --------------------
# This is the whole reason serving is fast -- none of this gets redone per farmer query. Nothing below this point retrains anything or recomputes scores nationally.

print("[engine1_serving] Loading persisted artifacts ...")

model = joblib.load(MODEL_PATH)
season_lookup = joblib.load(SEASON_LOOKUP_PATH)
_metadata = joblib.load(METADATA_PATH)
FEATURE_COLS = _metadata["feature_cols"]
CAT_COLS = _metadata["cat_cols"]
CAT_DTYPES = _metadata["cat_dtypes"]

latest = pd.read_csv(LATEST_SCORED_PATH)
# Saving to CSV strips pandas' category dtype back down to plain text, so reading it back in loses the exact category codes the model was trained on. Restoring it here with the schema saved at training time so it actually matches what the model expects.
for col in CAT_COLS:
    latest[col] = latest[col].astype(CAT_DTYPES[col])

water_need = pd.read_csv(WATER_NEED_PATH)
climate_bounds = pd.read_csv(CLIMATE_BOUNDS_PATH)
engine2_forecast = pd.read_csv(ENGINE2_FORECAST_PATH)  # Engine 2's own precomputed output, also just loaded once here

print(f"[engine1_serving] Ready. latest_scored: {latest.shape}, season_lookup: {len(season_lookup)} entries")


# -------------------- resolve_season() --------------------
# This figures out what "season" actually means for a given crop, without me hardcoding any agronomic knowledge into it -- it works purely off which season labels a crop is actually recorded under. There are two genuinely different situations it has to tell apart:

# 1. It's just a naming quirk. Assam/WB/Kerala Rice, for example, is only ever recorded under Autumn/Summer/Winter -- never Kharif or Rabi at all. Since the crop never touches the standard vocabulary in the first place, falling back to whatever labels it does use is safe -- it's the same growing window, just called something else.

# 2. It's a genuinely real seasonal restriction. Urad in Assam is a good example -- consistently Rabi, confirmed across 470 rows, with zero Kharif rows anywhere in the state. Falling back here would be wrong --it would recommend Urad for a season it has never actually been grown in. So instead this returns an empty list, and the caller just drops the crop from this particular query rather than quietly substituting a season that doesn't reflect reality. 


STANDARD_SEASONS = {"Kharif", "Rabi"}

def resolve_season(state: str, crop: str, farmer_season: str, lookup: dict = season_lookup) -> list:
    available = lookup.get((state, crop))
    if not available:
        return [farmer_season]   # never seen this State+Crop combo before -- just pass the season through as-is

    if farmer_season in available:
        return [farmer_season]   # farmer's season already matches directly, nothing to resolve

    uses_standard_vocabulary = bool(available & STANDARD_SEASONS)
    if uses_standard_vocabulary:
        return []   # this crop really is season-specific -- don't recommend it for a season it's never grown in

    return sorted(available)   # just a naming quirk -- safe to fall back to whatever labels this crop actually uses


# -------------------- irrigation_multiplier() --------------------
# No penalty if the farmer's irrigation already meets or beats what the crop needs. Otherwise the penalty scales with how big the shortfall is, but never drops below 0.5x -- a crop should never be treated as flat-out impossible, just less favourable.

def irrigation_multiplier(farmer_level_numeric: float, crop_need_numeric: float) -> float:
    gap = crop_need_numeric - farmer_level_numeric
    if gap <= 0:
        return 1.0
    return max(0.5, 1.0 - (gap / 0.66) * 0.5)


# -------------------- check_extrapolation_risk() --------------------
# Just a disclosure flag -- this never touches the Score itself. All it does is check whether Engine 2's forecast falls outside the range of climate conditions this crop actually saw during training, so we're honest about it instead of quietly presenting a prediction the model is basically guessing at.

def check_extrapolation_risk(crop: str, forecast_T_avg: float, forecast_R_total: float,
                              bounds_df: pd.DataFrame = climate_bounds) -> bool:
    row = bounds_df[bounds_df.Crop == crop]
    if row.empty:
        return False  # every one of the 53 crops should have bounds recorded, so this really shouldn't happen
    row = row.iloc[0]
    t_out = not (row.T_Avg_p05 <= forecast_T_avg <= row.T_Avg_p95)
    r_out = not (row.R_Total_p05 <= forecast_R_total <= row.R_Total_p95)
    return bool(t_out or r_out)


# -------------------- generate_shortlist() --------------------
# The historical-only shortlist -- no live model.predict() calls happen here at all. This just filters and queries the already-scored `latest` table, then layers the irrigation multiplier and extrapolation check on top.

def generate_shortlist(state: str, district: str, season: str, irrigation_level: str = "Average") -> pd.DataFrame:
    farmer_num = IRRIGATION_LEVEL_MAP[irrigation_level]
    rows = []
    for crop in latest["Crop"].unique():
        seasons = resolve_season(state, crop, season, season_lookup)
        sub = latest[
            (latest.State == state) & (latest.District == district) &
            (latest.Crop == crop) & (latest.Season.isin(seasons)) &
            (latest.historical_area_share >= FOOTPRINT_THRESHOLD)
        ]
        if not sub.empty:
            rows.append(sub.loc[sub["Score"].idxmax()])

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).merge(water_need[["Crop", "WaterNeedNumeric"]], on="Crop", how="left")
    out["Irrigation_Multiplier"] = out["WaterNeedNumeric"].apply(
        lambda need: irrigation_multiplier(farmer_num, need)
    )
    out["Final_Score"] = out["Score"] * out["Irrigation_Multiplier"]
    out["climate_extrapolation_risk"] = out.apply(
        lambda r: check_extrapolation_risk(r["Crop"], r["T_Avg"], r["R_Total"]), axis=1
    )
    return (
        out[~out["Crop"].isin(CATEGORY_AGGREGATES)]
        .sort_values("Final_Score", ascending=False)
        .drop(columns="WaterNeedNumeric")
    )


# -------------------- generate_shortlist_forecast() --------------------
# This is Cell 56's version specifically, the one that actually works right.
# Takes the historical shortlist above and swaps in Engine 2's real forecasted
# T_Avg/R_Total for each crop's resolved season, re-predicts with the trained model (cheap -- one row per shortlisted crop, not a national pass), and recomputes Score using where that new forecast-based prediction lands within the crop's national distribution -- which is already sitting in `latest`,so there's no need to recompute it from scratch.

def generate_shortlist_forecast(state: str, district: str, season: str, irrigation_level: str = "Average") -> pd.DataFrame:
    base = generate_shortlist(state, district, season, irrigation_level)
    if base.empty:
        return base

    merged = base.merge(
        engine2_forecast[["State", "District", "Season", "forecast_T_Avg", "forecast_R_Total"]],
        on=["State", "District", "Season"], how="left",
    )

    # Merging against engine2_forecast's plain-string Season column quietly drops the category dtype on these join columns -- putting it back using the exact dtypes saved at training time, otherwise model.predict() below breaks on a categorical mismatch.
    for col in CAT_COLS:
        merged[col] = merged[col].astype(CAT_DTYPES[col])

    results = []
    for idx, row in merged.iterrows():
        f_t, f_r = row["forecast_T_Avg"], row["forecast_R_Total"]

        if pd.isna(f_t) or pd.isna(f_r):
            results.append({
                "Predicted_Yield_capped_forecast": row["Predicted_Yield_capped"],
                "YieldPercentile_forecast": row["YieldPercentile"],
                "Score_forecast": row["Score"],
                "Final_Score_forecast": row["Final_Score"],
                "climate_extrapolation_risk_forecast": row["climate_extrapolation_risk"],
                "no_engine2_forecast": True,
            })
            continue

        # Using .loc[[idx], ...] with double brackets on purpose -- keeps this a 1-row DataFrame instead of a Series. A Series would silently lose the category dtype here, and the model call below would break.
        swapped_row = merged.loc[[idx], FEATURE_COLS].copy()
        swapped_row["T_Avg"] = f_t
        swapped_row["R_Total"] = f_r
        pred = model.predict(swapped_row)[0]

        crop_dist = latest.loc[latest.Crop == row["Crop"], "Predicted_Yield_capped"]
        yp = percentileofscore(crop_dist, pred, kind="mean") / 100
        score = 0.53 * yp + 0.27 * row["AreaSharePercentile"] + 0.20 * row["StabilityPercentile"]
        final = score * row["Irrigation_Multiplier"]
        risk = check_extrapolation_risk(row["Crop"], f_t, f_r)

        results.append({
            "Predicted_Yield_capped_forecast": pred,
            "YieldPercentile_forecast": yp,
            "Score_forecast": score,
            "Final_Score_forecast": final,
            "climate_extrapolation_risk_forecast": risk,
            "no_engine2_forecast": False,
        })

    merged = pd.concat([merged, pd.DataFrame(results, index=merged.index)], axis=1)
    return merged.sort_values("Final_Score_forecast", ascending=False)

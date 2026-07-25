"""
integration.py - Integration layer for Sanjeevni.

This file brings together the outputs from all three engines and combines
them into a single structured report that is finally sent to Gemini. It
contains no training code and doesn't make the Gemini API call itself.

This was originally built from MSP_integrationJSON.ipynb. Moving everything
here removed the need to %run multiple notebooks, which used to retrain all
three engines every time I wanted to test the integration flow.
"""

import os
import numpy as np
import pandas as pd

from engine1_serving import generate_shortlist_forecast, engine2_forecast
from engine3_serving import rerank_with_soil


# ---------------- Configuration ----------------
DATA_DIR = os.environ.get("SANJEEVNI_DATA_DIR", r"C:\Sanjeevni_App\data")
MSP_PATH = os.path.join(DATA_DIR, "MSP.csv")
CROPYIELD_PATH = os.path.join(DATA_DIR, "CropYield_mergeReady_v2.csv")


# ---------------- Loading the required datasets once ----------------
msp_df = pd.read_csv(MSP_PATH)
assert {"Crop", "Price", "Unit", "Price_Type"}.issubset(set(msp_df.columns)), \
    f"MSP.csv columns unexpected: {msp_df.columns.tolist()}"
assert msp_df["Crop"].is_unique, "Expected exactly one MSP row per crop"


def _load_known_districts(path: str = CROPYIELD_PATH) -> set:
    """One-time load of every (State, District) pair present in the training data."""
    df = pd.read_csv(path, usecols=["State", "District"])
    return set(zip(df["State"].str.strip(), df["District"].str.strip()))


KNOWN_DISTRICTS = _load_known_districts()
print(f"[integration] Loaded {len(KNOWN_DISTRICTS)} known State+District pairs, "
      f"MSP.csv: {msp_df.shape}")


def is_district_covered(state: str, district: str) -> bool:
    """True -> ML pipeline (Engines 1/2/3) can run for this farmer.
    False -> route to the direct-Gemini fallback path (Section 11.4)."""
    return (state.strip(), district.strip()) in KNOWN_DISTRICTS



# ---------------- Resolving dataframe column names ----------------
#
# Instead of hardcoding column names everywhere, I look them up through a
# list of possible names. If a column name changes later, updating the
# candidate list is enough instead of modifying the logic throughout the file.
def _resolve_col(columns, candidates, label):
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(
        f"Could not resolve column for '{label}'. "
        f"Tried {candidates}, available columns: {list(columns)}"
    )


def sanitize_for_json(obj):
    """Recursively convert numpy/pandas scalar types to native Python types
    so json.dumps() doesn't choke on np.float64 / np.bool_ / np.int64."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj



# ---------------- Possible column names ----------------

# Different versions of Engine 1 have used slightly different column names.
# Instead of assuming a fixed name, I check a list of possible names and use whichever one is available.
COL_CANDIDATES = {
    "crop": ["Crop", "crop"],
    "season": ["Season", "season"],
    "predicted_yield": ["Predicted_Yield_capped_forecast", "Predicted_Yield_capped"],
    "score": ["Final_Score_forecast", "Score_forecast", "Final_Score", "Score"],
    "yield_pct": ["YieldPercentile_forecast", "YieldPercentile"],
    "area_share_pct": ["AreaSharePercentile_forecast", "AreaSharePercentile"],
    "historical_area_share": ["historical_area_share"],
    "stability": ["StabilityPercentile_forecast", "StabilityPercentile",
                  "StabilityScore_forecast", "StabilityScore"],
    "climate_risk": ["climate_extrapolation_risk_forecast", "climate_extrapolation_risk"],
    "no_engine2_forecast": ["no_engine2_forecast"],
}

ENGINE2_COL_CANDIDATES = {
    "climate_label": ["Climate_Label", "climate_label"],
    "forecast_T_Avg": ["forecast_T_Avg", "Forecast_T_Avg"],
    "forecast_R_Total": ["forecast_R_Total", "Forecast_R_Total", "forecast_Rainfall_Total"],
    "anomaly_z_T_Avg": ["T_Avg_anomaly_z", "anomaly_z_T_Avg"],
    "anomaly_z_R_Total": ["R_Total_anomaly_z", "anomaly_z_R_Total"],
}


# ---------------- Building Engine 2 output ----------------

# A farmer's shortlist can contain crops from different seasons. Instead of creating one climate summary for the whole report, I store the forecast separately for each season so every crop can use the correct climate data.
def build_engine2_output(engine2_forecast_df: pd.DataFrame, seasons_needed: list,
                          state: str, district: str) -> dict:
    cols = engine2_forecast_df.columns
    label_col = _resolve_col(cols, ENGINE2_COL_CANDIDATES["climate_label"], "climate_label")
    t_avg_col = _resolve_col(cols, ENGINE2_COL_CANDIDATES["forecast_T_Avg"], "forecast_T_Avg")
    r_total_col = _resolve_col(cols, ENGINE2_COL_CANDIDATES["forecast_R_Total"], "forecast_R_Total")
    anom_t_col = _resolve_col(cols, ENGINE2_COL_CANDIDATES["anomaly_z_T_Avg"], "anomaly_z_T_Avg")
    anom_r_col = _resolve_col(cols, ENGINE2_COL_CANDIDATES["anomaly_z_R_Total"], "anomaly_z_R_Total")

    by_season = {}
    for season in seasons_needed:
        match = engine2_forecast_df.loc[
            (engine2_forecast_df["State"] == state) &
            (engine2_forecast_df["District"] == district) &
            (engine2_forecast_df["Season"] == season)
        ]
        if match.empty:
            print(f"WARNING: no Engine 2 forecast row for {state}/{district}/{season} - skipping")
            continue
        row = match.iloc[0]
        by_season[str(season)] = {
            "climate_label": row[label_col],
            "forecast_T_Avg": round(float(row[t_avg_col]), 2),
            "forecast_Rainfall_Total": round(float(row[r_total_col]), 2),
            "anomaly_z_T_Avg": round(float(row[anom_t_col]), 2),
            "anomaly_z_R_Total": round(float(row[anom_r_col]), 2),
        }
    return by_season



# MSP revenue layer -- predicted_yield_tonnes_per_hectare exposed alongside the rate figures, so Gemini can report it verbatim instead of it being silently discarded after the revenue_per_hectare multiplication.

def compute_revenue_estimate(shortlist_df: pd.DataFrame, msp_df: pd.DataFrame = msp_df) -> dict:
    cols = shortlist_df.columns
    crop_col = _resolve_col(cols, COL_CANDIDATES["crop"], "crop")
    yield_col = _resolve_col(cols, COL_CANDIDATES["predicted_yield"], "predicted_yield")

    revenue = {}
    for _, row in shortlist_df.iterrows():
        crop = row[crop_col]
        msp_row = msp_df.loc[msp_df["Crop"] == crop]

        if msp_row.empty:
            print(f"WARNING: no MSP row found for crop '{crop}' - skipping revenue calc")
            continue

        price = float(msp_row["Price"].iloc[0])
        price_type = str(msp_row["Price_Type"].iloc[0])
        predicted_yield = row.get(yield_col, np.nan)

        revenue_per_ha = round(float(predicted_yield) * price, 2) if pd.notna(predicted_yield) else None

        revenue[crop] = {
            "price_per_tonne": price,
            "price_type": price_type,
            "predicted_yield_tonnes_per_hectare": round(float(predicted_yield), 3) if pd.notna(predicted_yield) else None,
            "revenue_per_hectare": revenue_per_ha,
            "low_confidence_price": price_type == "Derived_Average",
        }
    return revenue



# assemble_farmer_report() -- builds the exact object Gemini consumes. soil_card: dict with keys N, P, K, OC, pH, or None if farmer didn't supply a Soil Health Card. If None, engine3_output is None and Engine 1's shortlist passes through untouched.
def assemble_farmer_report(state: str, district: str, season: str, irrigation_level: str,
                            area_hectares: float,
                            soil_card: dict = None,
                            msp_df: pd.DataFrame = msp_df,
                            engine2_forecast_df: pd.DataFrame = engine2_forecast) -> dict:
    shortlist_df = generate_shortlist_forecast(
        state=state, district=district, season=season, irrigation_level=irrigation_level
    )
    cols = shortlist_df.columns

    crop_col = _resolve_col(cols, COL_CANDIDATES["crop"], "crop")
    season_col = _resolve_col(cols, COL_CANDIDATES["season"], "season")
    score_col = _resolve_col(cols, COL_CANDIDATES["score"], "score")
    yieldpct_col = _resolve_col(cols, COL_CANDIDATES["yield_pct"], "yield_pct")
    areapct_col = _resolve_col(cols, COL_CANDIDATES["area_share_pct"], "area_share_pct")
    hist_share_col = _resolve_col(cols, COL_CANDIDATES["historical_area_share"], "historical_area_share")
    stability_col = _resolve_col(cols, COL_CANDIDATES["stability"], "stability")
    risk_col = _resolve_col(cols, COL_CANDIDATES["climate_risk"], "climate_risk")
    no_e2_col = _resolve_col(cols, COL_CANDIDATES["no_engine2_forecast"], "no_engine2_forecast")

    shortlist_records = []
    for _, row in shortlist_df.iterrows():
        shortlist_records.append({
            "crop": row[crop_col],
            "season": row[season_col],
            "score": round(float(row[score_col]), 4),
            "yield_pct": round(float(row[yieldpct_col]), 4),
            "area_share_pct": round(float(row[areapct_col]), 4),
            "historical_area_share": round(float(row[hist_share_col]), 4),
            "stability": round(float(row[stability_col]), 4),
            "climate_extrapolation_risk": bool(row[risk_col]),
            "no_engine2_forecast": bool(row[no_e2_col]),
        })

    seasons_needed = shortlist_df[season_col].unique().tolist()
    engine2_output = build_engine2_output(engine2_forecast_df, seasons_needed, state, district)

    # NOTE on why this is None rather than "wired in but unused": rerank_with_soil() requires an existing shortlist to restrict/rerank against (Engine 1's shortlist, location-dependent). There is no fallback-mode equivalent yet -- a standalone "soil-only candidate list" function (predict_proba over all 53 crops, no shortlist restriction) would need to be built separately for "Engine 3 can run independently" case. Not built yet -- documented debt, not a simple wiring gap.
    engine3_output = None
    if soil_card is not None:
        engine3_output = rerank_with_soil(
            shortlist_df,
            N=soil_card.get("N"), P=soil_card.get("P"), K=soil_card.get("K"),
            OC=soil_card.get("OC"), pH=soil_card.get("pH"),
        )

    revenue_estimate = compute_revenue_estimate(shortlist_df, msp_df)

    report = {
        "farmer_input": {
            "state": state,
            "district": district,
            "season": season,
            "irrigation": irrigation_level,
            "area_hectares": area_hectares,
            "soil_health_card": soil_card,
        },
        "engine1_output": {"shortlist": shortlist_records},
        "engine2_output": engine2_output,
        "engine3_output": engine3_output,
        "revenue_estimate": revenue_estimate,
        "note_to_gemini": (
            "irrigation adjustment is already applied inside engine1_output scores; "
            "select and explain the final top 3 from the shortlist; "
            "each shortlist entry's 'season' may differ from the farmer's literal season "
            "input due to state-specific season-label handling (e.g. Assam/WB/Kerala Rice) - "
            "engine2_output is keyed by that resolved season, look up the matching entry "
            "for that crop's actual climate context; "
            "if climate_extrapolation_risk is true for a crop, disclose plainly that its "
            "ranking rests on climate conditions outside anything the model was trained on; "
            "if a high-footprint crop (historical_area_share notably higher than another "
            "shortlisted crop) scores below a low-footprint one, flag that explicitly rather "
            "than silently presenting the ranking"
        ),
    }

    return sanitize_for_json(report)

"""
engine3_serving.py — Sanjeevni Engine 3 serving path.

Loads the persisted RandomForest model once at import time. No training
code lives here.

Worth calling out the fix vs. the original engine3.ipynb: that notebook's
Cell 7 got generate_shortlist_forecast() into its namespace by literally
running `%run engine1.ipynb` -- which meant every single Engine 3 session
also triggered a full Engine 1 retrain (LightGBM fit from scratch), just to
borrow one function. I'd actually flagged this exact problem in a comment in
that notebook and suggested a shared module as the fix. engine1_serving.py
is that shared module now, so this file just imports the function directly,
with zero retraining cost.
"""

import os
import pickle
import pandas as pd

from engine1_serving import generate_shortlist_forecast  # noqa: F401 -- keeping this around so callers can grab it from either engine1_serving or engine3_serving, whichever makes more sense in context

# ---------------- Config ----------------

ARTIFACT_DIR = os.environ.get("SANJEEVNI_ARTIFACT_DIR", r"C:\Sanjeevni_App\artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "engine3_soil_rf_model.pkl")

FEATURES = ["Nitrogen", "Phosphorus", "Potassium", "OC", "pH"]

print("[engine3_serving] Loading persisted RF model ...")
with open(MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)
print(f"[engine3_serving] Ready. Classes: {len(rf_model.classes_)}")


# -------------------- rerank_with_soil() --------------------
# Brought this over as-is from engine3.ipynb Cell 9 -- specifically the version that renormalizes within the shortlist, not an earlier draft that didn't do that.

def rerank_with_soil(shortlist_forecast, N: float = None, P: float = None,
                      K: float = None, OC: float = None, pH: float = None):
    """
    Takes Engine 1's shortlist and re-ranks it using the farmer's own Soil
    Health Card values. If any of N/P/K/OC/pH is missing, that just means
    the farmer never gave us a soil card, so this returns None and Engine
    1's shortlist goes straight to Gemini untouched.

    rf_model.predict_proba() spreads probability across all 53 crops, so
    even a crop that clearly dominates a 9-crop shortlist can end up looking
    artificially tiny, like 0.04. Renormalizing just the shortlisted crops
    back up to sum to 1 doesn't change who wins -- it just reports the
    number honestly relative to what Gemini is actually choosing between,
    instead of relative to all 53 crops nationally.
    """
    if any(v is None for v in [N, P, K, OC, pH]):
        return None

    # shortlist could come in as a DataFrame or as a list of dicts, handling both
    if isinstance(shortlist_forecast, pd.DataFrame):
        shortlist_crops = shortlist_forecast["Crop"].tolist()
    else:
        shortlist_crops = [entry.get("Crop", entry.get("crop")) for entry in shortlist_forecast]

    farmer_soil = pd.DataFrame([[N, P, K, OC, pH]], columns=FEATURES)
    proba = rf_model.predict_proba(farmer_soil)[0]
    crop_proba_map = dict(zip(rf_model.classes_, proba))

    # only keeping crops that are already on Engine 1's shortlist -- never introducing anything new here
    raw_scores = {crop: float(crop_proba_map.get(crop, 0.0)) for crop in shortlist_crops}

    total = sum(raw_scores.values())
    if total > 0:
        normalized = {crop: score / total for crop, score in raw_scores.items()}
    else:
        # if RF genuinely found no signal for any of these crops, splitting the probability evenly is more honest than showing a ranking that looks meaningful when it really isn't
        normalized = {crop: 1.0 / len(raw_scores) for crop in raw_scores}

    reranked = [
        {"crop": crop, "probability": round(prob, 4)}
        for crop, prob in normalized.items()
    ]
    reranked.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "reranked_within_shortlist": reranked,
        "note": "only computed if farmer supplied Soil Health Card values; re-ranks Engine 1's shortlist only, introduces no new crops",
    }


# -------------------- predict_soil_candidates() --------------------
# This is the fallback-mode version of Engine 3, added after the decision to make the soil card mandatory -- what used to be a deferred edge case suddenly needed a real answer. It's genuinely different from rerank_with_soil() above: that one narrows down an existing shortlist, but in fallback mode there's no shortlist to narrow in the first place, since Engine 1 never had local district data to build one from. So this just runs the RF model on its own and surfaces its own top candidates nationally,without any of the filtering Engine 1 would normally apply -- no footprint threshold, no climate check, none of it.

def predict_soil_candidates(N: float = None, P: float = None, K: float = None,
                              OC: float = None, pH: float = None, top_k: int = 10) -> dict:
    """
    Engine 3 doesn't need a location, just N/P/K/OC/pH, so it can still run
    even when a farmer's district has no local ML data at all. Returns the
    top_k crops nationally by predict_proba(), with nothing filtered out.

    Same convention as rerank_with_soil() -- missing any of N/P/K/OC/pH means
    no signal, so this returns None. In the live app the soil sliders always
    have some value though, so this should rarely actually come up.
    """
    if any(v is None for v in [N, P, K, OC, pH]):
        return None

    farmer_soil = pd.DataFrame([[N, P, K, OC, pH]], columns=FEATURES)
    proba = rf_model.predict_proba(farmer_soil)[0]
    ranked = sorted(zip(rf_model.classes_, proba), key=lambda x: -x[1])[:top_k]

    return {
        "soil_only_candidates": [
            {"crop": crop, "probability": round(float(p), 4)} for crop, p in ranked
        ],
        "note": (
            "Computed independently of location/climate data -- no local ML shortlist "
            "was available for this district. Reflects only how well this farmer's soil "
            "chemistry matches each crop nationally; does not account for climate, market "
            "access, or local viability the way Engine 1's shortlist normally would."
        ),
    }

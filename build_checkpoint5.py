"""
build_checkpoint5.py -- builds the canonical training dataset for Sanjeevni.

Takes the raw BiologicalMatrix.csv and turns it into
BiologicalMatrix_checkpoint5_areasharefixed.csv, carrying every engineered
column train_engine1.py actually needs: Yield_capped, the corrected
historical_area_share, yield_lag1/2, yield_roll_mean3, and data_confidence.

Only run this rarely -- when raw BiologicalMatrix.csv itself changes, or if
the historical_area_share season-collapse limitation (more on that below)
ever actually gets fixed properly. train_engine1.py just loads whatever
comes out of this file directly; none of this feature-engineering runs
inside train_engine1.py anymore.

This consolidates what used to be five separate notebooks: yield_capped.ipynb,
then hist_area_share.ipynb (superseded, see below), then lag_feature.ipynb,
then data_confidence.ipynb, then the area-share fix that originally lived
inside engine1.ipynb's Cell 1.

One deliberate deviation from hist_area_share.ipynb, not a bug: this skips
that notebook's original season-scoped historical_area_share computation
entirely. That version is confirmed broken for states where a crop's season
labels are exclusive to it -- Assam/WB/Kerala Rice under Autumn/Summer/Winter
being the example -- since the denominator collapses down to basically the
crop's own area, making the share degenerate to ~1.0. The corrected, crop-level
version below replaces it completely. Since lag_feature.ipynb and
data_confidence.ipynb never actually read historical_area_share themselves,
there was no reason to compute the broken version at all just to throw it away.

One limitation carried forward on purpose, not fixed here: the corrected
historical_area_share merges back on State+District+Crop only, with Season
dropped from the merge key. That fixes the cross-crop national comparison,
but it does mean every season-variant of the same crop in the same district
(Nalbari's Autumn/Summer/Winter Rice, for instance) ends up sharing one
identical area-share value. Telling those season-variants apart from each
other is left entirely to yield and stability for now. This was a conscious
call to defer, not something I missed -- see the Day 9 catchup notes, bug #1.
"""

import os
import numpy as np
import pandas as pd

# ---------------- Config ----------------

RAW_DIR = os.environ.get("SANJEEVNI_RAW_DIR", r"C:\Sanjeevni_App\data\raw")
DATA_DIR = os.environ.get("SANJEEVNI_DATA_DIR", r"C:\Sanjeevni_App\data")

BIOLOGICAL_MATRIX_PATH = os.path.join(RAW_DIR, "BiologicalMatrix.csv")
CHECKPOINT5_OUT = os.path.join(DATA_DIR, "BiologicalMatrix_checkpoint5_areasharefixed.csv")


# -------------------- Step 1: Yield_capped --------------------
# Winsorizing at the 99.5th percentile, per crop -- never one global cutoff for everything, since one extreme row shouldn't get to single-handedly decide what "perfect" looks like for a crop.

def add_yield_capped(df: pd.DataFrame) -> pd.DataFrame:
    cap = df.groupby("Crop")["Yield"].transform(lambda s: s.quantile(0.995))
    df["Yield_capped"] = np.minimum(df["Yield"], cap)
    return df


# -------------------- Step 2: historical_area_share (the corrected version) --------------------
# Numerator is a crop's total area in a State+District+Year, summed across every season-variant that crop has, not just one. Denominator is the district's entire cultivated area that year, across every crop and every season -- never scoped down to a single season label, since some states record certain crops under season names nobody else uses.

def add_historical_area_share(df: pd.DataFrame) -> pd.DataFrame:
    crop_year_area = df.groupby(["State", "District", "Crop", "Year"], observed=True)["Area"].sum()
    total_year_area = df.groupby(["State", "District", "Year"], observed=True)["Area"].sum()

    raw_share = (crop_year_area / total_year_area).rename("raw_share").reset_index()
    fixed_share = (
        raw_share.groupby(["State", "District", "Crop"], observed=True)["raw_share"]
        .mean()
        .rename("historical_area_share")
        .reset_index()
    )

    df = df.merge(fixed_share, on=["State", "District", "Crop"], how="left")
    return df


# -------------------- Step 3: lag features --------------------
# Grouping on the full State+District+Crop+Season key -- this matches the 21.6% cold-start figure, which was always computed on this exact same grouping.

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["State", "District", "Crop", "Season", "Year"]).reset_index(drop=True)
    group_cols = ["State", "District", "Crop", "Season"]

    df["yield_lag1"] = df.groupby(group_cols, observed=True)["Yield_capped"].shift(1)
    df["yield_lag2"] = df.groupby(group_cols, observed=True)["Yield_capped"].shift(2)

    def _roll_mean3(s):
        return s.shift(1).rolling(window=3, min_periods=1).mean()

    df["yield_roll_mean3"] = df.groupby(group_cols, observed=True)["Yield_capped"].transform(_roll_mean3)

    # 3-tier cold-start fallback -- never filling with 0, that would make a genuine data gap look like an actual crop failure
    tier1 = df.groupby(["State", "District", "Crop"], observed=True)["Yield_capped"].transform("mean")
    tier2 = df.groupby(["State", "Crop"], observed=True)["Yield_capped"].transform("mean")
    tier3 = df.groupby(["Crop"], observed=True)["Yield_capped"].transform("mean")

    for col in ["yield_lag1", "yield_lag2", "yield_roll_mean3"]:
        df[col] = df[col].fillna(tier1).fillna(tier2).fillna(tier3)

    return df


# -------------------- Step 4: data_confidence --------------------

def add_data_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df["data_confidence"] = df.groupby(
        ["State", "District", "Crop", "Season"], observed=True
    )["Year"].transform("count")
    return df


def main():
    print(f"Loading {BIOLOGICAL_MATRIX_PATH} ...")
    df = pd.read_csv(BIOLOGICAL_MATRIX_PATH)
    print("Loaded:", df.shape)

    df = add_yield_capped(df)
    df = add_historical_area_share(df)
    df = add_lag_features(df)
    df = add_data_confidence(df)

    df.to_csv(CHECKPOINT5_OUT, index=False)
    print("Saved ->", CHECKPOINT5_OUT)
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()

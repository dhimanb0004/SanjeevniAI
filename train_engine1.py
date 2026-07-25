"""
Training script for Sanjeevni Engine 1.

This script trains the LightGBM yield prediction model using the
preprocessed BiologicalMatrix_checkpoint5_areasharefixed.csv dataset.

After training, it saves:
- trained LightGBM model
- season lookup dictionary
- metadata (feature list and categorical dtypes)
- latest scored crop dataset

We shall run this script only when the model needs to be retrained.
"""

import os   # The os module provides functions to interact with the operating system
import numpy as np  # NumPy is the fundamental numerical computing library, fast and consumes less memory
import pandas as pd # Pandas is the library used for tabular data manipulation 
import lightgbm as lgb  # the ML algorithm used for training 
from sklearn.metrics import mean_squared_error, r2_score    # evaluation metrics
import joblib   #joblib is used to serialize (save) Python objects to disk and load them back later - it save/load model artifacts



# ---------------- Configuration ----------------


# Getting the data directory path (using the default if no environment variable is set)
DATA_DIR = os.environ.get("SANJEEVNI_DATA_DIR", r"C:\Sanjeevni_App\data")
ARTIFACT_DIR = os.environ.get("SANJEEVNI_ARTIFACT_DIR", r"C:\Sanjeevni_App\artifacts")


# Joining the folder and file name into a complete file path
CHECKPOINT5_PATH = os.path.join(DATA_DIR, "BiologicalMatrix_checkpoint5_areasharefixed.csv")

MODEL_OUT = os.path.join(ARTIFACT_DIR, "engine1_lightgbm_model.pkl")
SEASON_LOOKUP_OUT = os.path.join(ARTIFACT_DIR, "season_lookup.pkl")
METADATA_OUT = os.path.join(ARTIFACT_DIR, "engine1_metadata.pkl")  # feature_cols + cat_dtypes bundled together
LATEST_SCORED_OUT = os.path.join(ARTIFACT_DIR, "engine1_latest_scored.csv")

CAT_COLS = ["State", "District", "Crop", "Season", "USDA_Texture_0_15cm", "Dominant_Soil_Type_(WRB)"]

FEATURE_COLS = [
    "State", "District", "Crop", "Season", "Year", "Area",
    "yield_lag1", "yield_lag2", "yield_roll_mean3",
    "T_Max", "T_Min", "T_Avg", "R_Total", "R_Max", "R_Min", "R_Avg", "R_CV",
    "pH_0_15cm", "Sand_%_0_15cm", "Silt_%_0_15cm", "Clay_%_0_15cm",
    "USDA_Texture_0_15cm", "Dominant_Soil_Type_(WRB)",
    "historical_area_share", "data_confidence",
]
TARGET_COL = "Yield_capped"

# Setting the contribution of each factor (weights) towards the final recommendation score
W_YIELD, W_AREA_SHARE, W_STABILITY = 0.53, 0.27, 0.20



# ------------ Categorical dtype fix (must happen before train/val split, per Master ------------------

# Converting the categorical columns before splitting the dataset. This ensures that pandas creates a single, consistent category mapping for both the training and validation data. Since LightGBM internally works with category codes instead of the original strings, converting them separately after splitting could result in different code mappings for the same category, leading to incorrect model training or evaluation.

def fix_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    return df




# --------- Training the LightGBM model using a time-based train-validation split -------------------

def train_lightgbm(df: pd.DataFrame):
    train_df = df[df["Year"] <= 2016].copy()
    val_df = df[df["Year"].isin([2017, 2018])].copy()
    
    # TIME-BASED train-validation split instead of a random split:
    
    # Since this dataset contains historical records, using a random train/test split could leak future information into the training data. To mimic the real deployment scenario, the model is trained only on data up to 2016 and validated on the later years (2017-2018).
    
    # One limitation of this approach is that some categorical values (such as a new Crop, District, or Season) may appear for the first time only in the validation years. The model genuinely cannot learn patterns for categories absent from training -- no amount of preprocessing fixes that. What CAN be fixed is a subtler failure mode: converting categorical columns to the 'category' dtype only after splitting would let train and validation independently assign different numeric codes to the same value, corrupting predictions even on categories both sets DO share. Converting to 'category' dtype on the full dataset before splitting avoids that specific failure -- it does not teach the model anything new, it just guarantees train and validation agree on what each code means.

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]

    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    model.fit(
        X_train, y_train,
        categorical_feature=CAT_COLS,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²:   {r2:.4f}")

    return model



# ---------------- season_lookup (data-driven State+Crop -> season labels present) ----------------------
# Creating a lookup table that stores all valid growing seasons for each State-Crop combination
def build_season_lookup(df: pd.DataFrame) -> dict:
    return (
        df.groupby(["State", "Crop"], observed=True)["Season"]
        .apply(lambda s: set(s.unique()))
        .to_dict()
    )



# ------------------ National scoring pass: builds the `latest` snapshot table with Score --------------------

# Creating the latest scored dataset that will be used during inference. This computes the final recommendation score once during training, so the serving script can directly use the precomputed results without repeating the scoring process for every farmer query.

def build_latest_scored(df: pd.DataFrame, model) -> pd.DataFrame:
    latest = (
        df.sort_values("Year")
        .groupby(["State", "District", "Crop", "Season"], observed=True)
        .tail(1)
        .copy()
    )
    latest["Predicted_Yield_capped"] = model.predict(latest[FEATURE_COLS])

    latest["YieldPercentile"] = latest.groupby("Crop", observed=True)["Predicted_Yield_capped"].rank(pct=True)
    latest["AreaSharePercentile"] = latest.groupby("Crop", observed=True)["historical_area_share"].rank(pct=True)

    grp = df.groupby(["State", "District", "Crop", "Season"], observed=True)["Yield_capped"]
    stats = grp.agg(mean="mean", std="std", count="count").reset_index()
    stats["CV"] = stats["std"] / stats["mean"]

    crop_mean_cv = stats[stats["count"] >= 3].groupby("Crop", observed=True)["CV"].mean()
    stats["CV_final"] = stats.apply(
        lambda r: r["CV"] if r["count"] >= 3 else crop_mean_cv.get(r["Crop"], stats["CV"].mean()),
        axis=1,
    )
    stats["StabilityScore"] = 1 / (stats["CV_final"] + 0.05)

    latest = latest.merge(
        stats[["State", "District", "Crop", "Season", "StabilityScore"]],
        on=["State", "District", "Crop", "Season"], how="left",
    )

    # StabilityScore is an unbounded raw ratio (can exceed 1 for low-CV crops) --
    # percentile-rank it too, same as YieldPercentile/AreaSharePercentile, so it can't swamp the other two bounded 0-1 terms.
    latest["StabilityPercentile"] = latest.groupby("Crop", observed=True)["StabilityScore"].rank(pct=True)

    latest["Score"] = (
        W_YIELD * latest["YieldPercentile"]
        + W_AREA_SHARE * latest["AreaSharePercentile"]
        + W_STABILITY * latest["StabilityPercentile"]
    )

    return latest


def main():
    print(f"Loading {CHECKPOINT5_PATH} ...")
    df = pd.read_csv(CHECKPOINT5_PATH)
    print("Loaded:", df.shape)

    df = fix_categorical_dtypes(df)

    print("Training LightGBM ...")
    model = train_lightgbm(df)

    print("Building season_lookup ...")
    season_lookup = build_season_lookup(df)

    print("Building national scored snapshot (this replaces per-query recomputation at serving time) ...")
    latest_scored = build_latest_scored(df, model)

    # Saving the feature list and categorical dtype information so the serving script uses the same feature schema as the training script
    metadata = {
        "feature_cols": FEATURE_COLS,
        "cat_cols": CAT_COLS,
        "cat_dtypes": {col: df[col].dtype for col in CAT_COLS},
    }

    print("Persisting artifacts ...")
    joblib.dump(model, MODEL_OUT)
    joblib.dump(season_lookup, SEASON_LOOKUP_OUT)
    joblib.dump(metadata, METADATA_OUT)
    latest_scored.to_csv(LATEST_SCORED_OUT, index=False)

    print("Done. Persisted:")
    for p in [MODEL_OUT, SEASON_LOOKUP_OUT, METADATA_OUT, LATEST_SCORED_OUT]:
        print(" -", p)


if __name__ == "__main__":
    main()

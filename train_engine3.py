"""
train_engine3.py — Sanjeevni Engine 3 (Soil Suitability Engine) training.

Trains a RandomForestClassifier on SoilProfile_Crop.csv to predict which
crop a given N/P/K/OC/pH soil reading best matches, across 53 crops. The
hyperparameters below aren't a first guess -- I went through a whole
size-vs-accuracy tradeoff earlier: the original setup (300 estimators, no
depth limit) came out to about 2.4GB and was clearly overfitting. What's
here now lands at roughly 410MB with accuracy and macro-F1 actually a bit
better than that bloated version, so it's a straight improvement, not just
a smaller file.

Only run this when retraining -- e.g. if SoilProfile_Crop.csv changes.
Never run this from the Streamlit app. Training itself takes seconds on
this dataset, so splitting this into its own file isn't about speed -- it's
so engine3_serving.py doesn't have to drag in engine1.ipynb's whole %run
chain just to get a model (see engine3_serving.py for why that mattered).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

RANDOM_STATE = 42

# ---------------- Config ----------------

DATA_DIR = os.environ.get("SANJEEVNI_DATA_DIR", r"C:\Sanjeevni_App\data")
ARTIFACT_DIR = os.environ.get("SANJEEVNI_ARTIFACT_DIR", r"C:\Sanjeevni_App\artifacts")

SOIL_PROFILE_PATH = os.path.join(DATA_DIR, "SoilProfile_Crop.csv")
MODEL_OUT_PATH = os.path.join(ARTIFACT_DIR, "engine3_soil_rf_model.pkl")

FEATURES = ["Nitrogen", "Phosphorus", "Potassium", "OC", "pH"]
TARGET = "Crop"


def load_and_validate(path: str = SOIL_PROFILE_PATH) -> pd.DataFrame:
    soil_df = pd.read_csv(path)
    print("Shape:", soil_df.shape)
    print("Nulls:", soil_df.isnull().sum().sum())
    print("Duplicates:", soil_df.duplicated().sum())
    print("Unique crops:", soil_df["Crop"].nunique())
    print("Class balance (should be exactly 500/crop):")
    print(soil_df["Crop"].value_counts().describe())
    # Should be 26,500 rows, 0 nulls, 0 dupes, 53 crops, exactly 500 of each.
    # If any of that looks off, stop right here -- no point training on bad data.
    return soil_df


def train(soil_df: pd.DataFrame):
    X, y = soil_df[FEATURES], soil_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

    # Not scaling anything -- Random Forest doesn't care about feature scale.
    # Not using a LabelEncoder either -- sklearn's classifier is fine with the
    # crop names as plain strings. And no class_weight needed since the data
    # is already perfectly balanced at 500 rows per crop.
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_model.fit(X_train, y_train)
    print("Classes learned:", len(rf_model.classes_))

    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {macro_f1:.4f}")
    print(classification_report(y_test, y_pred))

    return rf_model, X_test, y_test, y_pred, acc, macro_f1


# -------------------- Confusion matrix (optional, run by hand) --------------------
# main() doesn't call this. It's here for when I actually want to look at
# where the model gets confused -- pulling out the pulse and oilseed crops
# specifically since those are the ones agronomically similar enough to trip
# each other up.

def plot_confusion_matrix(soil_df: pd.DataFrame, y_test, y_pred):
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = sorted(soil_df["Crop"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(18, 15))
    sns.heatmap(cm_df, cmap="Blues", cbar=True)
    plt.title("Engine 3 — Full Confusion Matrix (53 crops)")
    plt.tight_layout()
    plt.show()

    pulse_crops = ["Arhar", "Cowpea", "Gram", "Horse Gram", "Khesari", "Masoor",
                   "Moong", "Moth", "Pea", "Urad", "Kharif Pulse", "Rabi Pulse"]
    oilseed_crops = ["Castor Seed", "Groundnut", "Guar Seed", "Linseed", "Mustard",
                      "Niger Seed", "Safflower", "Sesamum", "Soyabean", "Sunflower", "Oilseed"]

    print("=== Pulse cluster confusion sub-matrix ===")
    print(cm_df.loc[pulse_crops, pulse_crops])
    print("=== Oilseed cluster confusion sub-matrix ===")
    print(cm_df.loc[oilseed_crops, oilseed_crops])
    # If the diagonal dominates within each cluster, that means the model is
    # actually telling agronomically-similar crops apart, not just guessing --
    # which matters since these are exactly the crops most likely to get mixed up.


def main():
    print(f"Loading {SOIL_PROFILE_PATH} ...")
    soil_df = load_and_validate()

    print("Training RandomForestClassifier ...")
    rf_model, X_test, y_test, y_pred, acc, macro_f1 = train(soil_df)

    print("Persisting model ...")
    with open(MODEL_OUT_PATH, "wb") as f:
        pickle.dump(rf_model, f)

    size_mb = os.path.getsize(MODEL_OUT_PATH) / 1e6
    print("=" * 60)
    print("ENGINE 3 -- TRAINING SUMMARY")
    print("=" * 60)
    print(f"Accuracy:       {acc:.4f}")
    print(f"Macro-F1:       {macro_f1:.4f}")
    print(f"Model size:     {size_mb:.1f} MB")
    print(f"Saved to:       {MODEL_OUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

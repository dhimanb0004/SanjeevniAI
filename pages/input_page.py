"""
input_page.py — "Farm Details" input form (Step 2).

Collects all 6 farmer inputs per the locked UI spec and calls
gemini_client.get_sanjeevni_report() FOR REAL on "Generate Report" --
this spends one live Gemini API call per click. Report rendering itself
is still a placeholder (plus a raw JSON dump for verification); Step 3
builds the real display.
"""

import os
import pandas as pd
import streamlit as st

from gemini_client import get_sanjeevni_report

DATA_DIR = os.environ.get("SANJEEVNI_DATA_DIR", r"C:\Sanjeevni_App\data")
STATE_DIST_SEASON_PATH = os.path.join(DATA_DIR, "State_Dist_Season_UI.csv")

OTHERS_LABEL = "Others (type below)"
IRRIGATION_OPTIONS = ["Good", "Average", "Poor"]


@st.cache_data
def load_state_dist_season():
    return pd.read_csv(STATE_DIST_SEASON_PATH)


sds_df = load_state_dist_season()

st.title("Farm Details")


# State -- closed-set searchable dropdown, no custom entry allowed

states = sorted(sds_df["State"].unique().tolist())
state = st.selectbox("State", options=states, index=None, placeholder="Select your state...")


# District -- filtered by State. "Others" fallback: the literal typed text is what gets submitted -- "Others" is just a UI indicator, never the actual stored value (per the locked spec).

district = None
if state:
    districts = sorted(sds_df.loc[sds_df["State"] == state, "District"].unique().tolist())
    district_choice = st.selectbox(
        "District", options=districts + [OTHERS_LABEL], index=None,
        placeholder="Select your district...",
    )
    if district_choice == OTHERS_LABEL:
        district = st.text_input("Enter your district name")
    else:
        district = district_choice
else:
    st.selectbox("District", options=[], index=None, placeholder="Select a state first...", disabled=True)


# Season -- filtered by (State, District) when the district is a real, known one. Falls back to State-level season options if the districtwas custom-typed, since no district-specific season data exists for it.

season = None
if state and district:
    known_districts = sds_df.loc[sds_df["State"] == state, "District"].values
    if district in known_districts:
        seasons = sorted(sds_df.loc[
            (sds_df["State"] == state) & (sds_df["District"] == district), "Season"
        ].unique().tolist())
    else:
        seasons = sorted(sds_df.loc[sds_df["State"] == state, "Season"].unique().tolist())

    season_choice = st.selectbox(
        "Season", options=seasons + [OTHERS_LABEL], index=None,
        placeholder="Select the season...",
    )
    if season_choice == OTHERS_LABEL:
        season = st.text_input("Enter the season name")
    else:
        season = season_choice
else:
    st.selectbox("Season", options=[], index=None, placeholder="Select a state and district first...", disabled=True)


# ----------------------------- Irrigation -------------------------

irrigation = st.radio("Irrigation level", options=IRRIGATION_OPTIONS, horizontal=True)


# --------------------Soil Health Card -----------------------------
# 5 sliders, exact bounds/defaults from the locked spec (sourced from SoilProfile_Crop.csv's training range). Always have a value -- never treated as "missing" for validation purposes.

st.markdown("#### Soil Health Card")
col1, col2 = st.columns(2)
with col1:
    N = st.slider("Nitrogen (N) -- kg/ha", min_value=60, max_value=500, value=195, step=5)
    P = st.slider("Phosphorus (P) -- kg/ha", min_value=10, max_value=80, value=36, step=1)
    K = st.slider("Potassium (K) -- kg/ha", min_value=60, max_value=450, value=162, step=5)
with col2:
    OC = st.slider("Organic Carbon (OC) -- %", min_value=0.2, max_value=2.5, value=0.73, step=0.05)
    pH = st.slider("pH", min_value=5.0, max_value=9.0, value=6.7, step=0.1)


# Area -- text box (not a number spinner), decimals required
area_text = st.text_input("Area under cultivation (hectares)", placeholder="e.g. 1.50")


# Nav row -- Back (left, direct switch) + Generate Report (right)
col_back, _sp, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("⮜ Back", key="back_btn"):
        st.switch_page("pages/instructions_page.py")

with col_next:
    generate_clicked = st.button("Generate Report", type="primary")


# Validation, then the REAL backend call (only on click)
if generate_clicked:
    errors = []
    if not state:
        errors.append("Please select your State.")
    if not district:
        errors.append("Please select or enter your District.")
    if not season:
        errors.append("Please select or enter the Season.")

    area_hectares = None
    if not area_text or not area_text.strip():
        errors.append("Please enter your cultivation Area (in hectares).")
    else:
        try:
            area_hectares = float(area_text.strip())
            if area_hectares <= 0:
                errors.append("Area must be a positive number.")
        except ValueError:
            errors.append("Area must be a valid number (e.g. 1.5).")

    if errors:
        st.toast("⚠️ Missing Input", icon="⚠️")
        for e in errors:
            st.error(e)
    else:
        soil_card = {"N": N, "P": P, "K": K, "OC": OC, "pH": pH}
        with st.spinner("Sanjeevni is analyzing your farm profile -- this typically takes 30-40 seconds..."):
            report = get_sanjeevni_report(
                state=state, district=district, season=season,
                irrigation=irrigation, area_hectares=area_hectares,
                soil_card=soil_card,
            )
        st.session_state.farmer_report = report
        st.session_state.report_generated = True
        st.session_state["_pending_switch"] = "pages/report_page.py"
        st.rerun()

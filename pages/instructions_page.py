"""
instructions_page.py — "User's Journey" step-by-step overview.
"""

import streamlit as st

st.markdown("## User's Journey")
st.write("")

steps = [
    ("Step 1", "Tell Us About Your Farm",
     "Your state, district, season, and how much irrigation you have available."),
    ("Step 2", "Share Your Soil Health Card",
     "Your Nitrogen, Phosphorus, Potassium, Organic Carbon, and pH values -- "
     "straight off your physical Soil Health Card, no conversion needed."),
    ("Step 3", "Tell Us Your Land Size",
     "How much land, in hectares, you're planning to cultivate."),
    ("Step 4", "Get Your Personalized Report",
     "Your top 3 recommended crops -- seed variety, cultivation steps, fertilizer "
     "guidance, irrigation guidance, and expected revenue for each -- generated from "
     "real yield data, climate forecasting, and soil science."),
    ("Step 5", "Ask Follow-Up Questions",
     "Use the built-in assistant to ask anything about your report -- variety "
     "substitutions, fertilizer timing, or anything else on your mind."),
]

for step_label, title, desc in steps:
    st.markdown(f"""
    <div class="sanjeevni-step-card">
        <div class="sanjeevni-step-number">{step_label}</div>
        <h3 style="margin: 0.2rem 0 0.4rem 0;">{title}</h3>
        <p style="margin:0; opacity:0.9;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
col_back, _sp, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("⮜ Back", key="back_btn"):
        st.switch_page("pages/home_page.py")
with col_next:
    if st.button("Next ⮞", type="primary"):
        if "input" not in st.session_state.visited_sections:
            st.session_state.visited_sections.append("input")
        st.session_state["_pending_switch"] = "pages/input_page.py"
        st.rerun()

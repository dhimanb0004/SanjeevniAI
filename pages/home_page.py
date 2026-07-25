"""
home_page.py — "Know About Sanjeevni AI" section.
"""

import streamlit as st

st.write("")
st.markdown("""
<div class="sanjeevni-about-box">
    <h2>Know About Sanjeevni AI</h2>
    <br>
    <h3>🌾 Why This Exists</h3>
    <ul>
        <li>Millions of smallholder farmers decide what to grow using <strong>tradition and guesswork</strong> — not because they lack skill, but because personalized guidance has simply never existed.</li>
        <li>A government advisory can tell an entire <strong>district</strong> what typically grows there. It cannot tell <strong>you</strong> — on your plot, with your soil, this season.</li>
    </ul>
    <br>
    <h3>🔍 What Sanjeevni Actually Does</h3>
    <ul>
        <li>📊 <strong>Real yield data</strong> — years of historical performance from your own district.</li>
        <li>🌦️ <strong>Climate forecasting</strong> — tailored to your exact growing season.</li>
        <li>🧪 <strong>Your own soil chemistry</strong> — if you share your Soil Health Card.</li>
        <li>📋 <strong>One complete report</strong> — top crops, seed variety, cultivation steps, fertilizer correction, and an honest revenue estimate.</li>
    </ul>
    <br>
    <h3>💡 The Thinking Behind It</h3>
    <ul>
        <li>Raw yield numbers <strong>lie</strong>. A crop can top the charts on paper and still be the wrong choice — ignoring water needs, capital intensity, market access, and hard-won local knowledge.</li>
        <li>We weigh what farmers <strong>actually grow</strong>, how <strong>stable</strong> it's been over time, and how well it fits <strong>your</strong> soil — not just which number looks biggest.</li>
    </ul>
    <br>
    <h3>🤝 A Note To You</h3>
    <ul>
        <li>This is a <strong>second opinion</strong>, not a replacement for what you already know about your own land.</li>
        <li>Every report is honest about its limitations — and you can always ask follow-up questions once it's ready.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.write("")
_sp, col_next = st.columns([4, 1])
with col_next:
    if st.button("Next ⮞", type="primary"):
        if "instructions" not in st.session_state.visited_sections:
            st.session_state.visited_sections.append("instructions")
        st.session_state["_pending_switch"] = "pages/instructions_page.py"
        st.rerun()

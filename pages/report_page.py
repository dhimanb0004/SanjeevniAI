"""
report_page.py — "Your Crop Recommendations" (Step 3: real rendering).

Minimalist, plain-language, farmer-facing display. Never shows raw schema
field names (is_category_aggregate, climate_extrapolation_risk_disclosed,
etc.) directly -- those are read internally and rendered as friendly
badges/callouts instead.
"""

import streamlit as st
from ui_common import get_theme_colors

st.title("Your Crop Recommendations")

report = st.session_state.get("farmer_report")

if not report:
    st.info("No report yet -- go to Farm Details and click Generate Report first.")
    st.stop()

c = get_theme_colors()


def format_inr(amount):
    """Indian digit-grouping (e.g. 1,12,500.90) -- reads naturally for an
    Indian audience, unlike standard Western 3-digit grouping."""
    if amount is None:
        return "N/A"
    amount = round(float(amount), 2)
    whole = int(amount)
    frac = round(amount - whole, 2)
    s = str(whole)
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        formatted = ",".join(parts) + "," + last3
    if frac > 0:
        formatted += f".{int(round(frac * 100)):02d}"
    return formatted



# Shared card CSS -- same visual language as the step-cards elsewhere (rounded, left accent border) for site-wide consistency.
st.markdown(f"""
<style>
.sanjeevni-summary-box {{
    background-color: {c['card_bg']};
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    border-left: 6px solid {c['accent']};
}}
.sanjeevni-fallback-banner {{
    background-color: {c['accent2']};
    color: #FFFFFF;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    font-weight: 500;
}}
.sanjeevni-crop-card {{
    background-color: {c['card_bg']};
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    border-left: 6px solid {c['accent']};
}}
.sanjeevni-crop-card.top-pick {{
    border-left-width: 10px;
}}
.sanjeevni-crop-badge {{
    display: inline-block;
    background-color: {c['accent']};
    color: {c['bg']};
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}}
.sanjeevni-category-note {{
    background-color: {c['accent2']};
    color: #FFFFFF;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin: 0.6rem 0 1rem 0;
    font-size: 0.95rem;
}}
.sanjeevni-numbers-row {{
    display: flex;
    gap: 1.5rem;
    margin: 1.2rem 0;
    flex-wrap: wrap;
}}
.sanjeevni-number-box {{
    background-color: {c['bg']};
    border-radius: 12px;
    padding: 1rem 1.5rem;
    flex: 1;
    min-width: 200px;
}}
.sanjeevni-number-label {{
    font-size: 0.85rem;
    opacity: 0.75;
    margin-bottom: 0.2rem;
}}
.sanjeevni-number-value {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {c['accent']};
}}
.sanjeevni-note-line {{
    font-size: 0.9rem;
    opacity: 0.85;
    margin: 0.3rem 0;
    padding-left: 0.8rem;
    border-left: 3px solid {c['accent2']};
}}
</style>
""", unsafe_allow_html=True)


# Fallback-mode banner -- friendly, plain framing (not "fallback_mode: true")

if report.get("fallback_mode"):
    st.markdown("""
    <div class="sanjeevni-fallback-banner">
        📍 Your exact area isn't in our detailed local records yet, so these
        suggestions are based on general farming knowledge for your region
        rather than your district's own history. Still worth reading --
        just a bit less precise than usual.
    </div>
    """, unsafe_allow_html=True)


# Location + farmer-facing summary -- the most important, most readable content, shown first and prominently.

st.markdown(f"""
<div class="sanjeevni-summary-box">
    <p style="margin:0 0 0.5rem 0; opacity:0.75; font-size:0.95rem;">📍 {report.get('location_summary', '')}</p>
    <p style="margin:0; font-size:1.1rem; line-height:1.6;">{report.get('farmer_facing_summary', '')}</p>
</div>
""", unsafe_allow_html=True)

if report.get("climate_context_note"):
    st.markdown(f"""
    <div class="sanjeevni-note-line">🌦️ {report['climate_context_note']}</div>
    """, unsafe_allow_html=True)

st.write("")

# The 3 recommendations -- full depth each, #1 visually marked as the top pick, #2/#3 as strong alternatives.
BADGES = ["🏆 Top Recommendation", "🌟 Strong Alternative", "🌟 Strong Alternative"]

for i, rec in enumerate(report.get("top_recommendations", [])):
    badge = BADGES[i] if i < len(BADGES) else "🌟 Strong Alternative"
    card_class = "sanjeevni-crop-card top-pick" if i == 0 else "sanjeevni-crop-card"

    category_note_html = ""
    if rec.get("is_category_aggregate") and rec.get("category_aggregate_note"):
        category_note_html = f'<div class="sanjeevni-category-note">🌾 {rec["category_aggregate_note"]}</div>'

    production = rec.get("total_expected_production_tonnes")
    revenue = rec.get("total_expected_revenue_rupees")

    disclosure_lines = ""
    for key, icon in [
        ("season_mismatch_disclosed", "📅"),
        ("climate_extrapolation_risk_disclosed", "🌡️"),
        ("footprint_flag_disclosed", "📊"),
        ("stale_climate_context_disclosed", "🕐"),
    ]:
        if rec.get(key):
            disclosure_lines += f'<div class="sanjeevni-note-line">{icon} {rec[key]}</div>'

    # Built as ONE continuous string, no embedded newlines -- a blank line from an empty conditional piece (like category_note_html for non-aggregate crops) makes Streamlit's markdown parser treat everything after it as literal text instead of HTML. Single-line concatenation sidesteps that failure mode entirely.
    card_html = (
        f'<div class="{card_class}">'
        f'<div class="sanjeevni-crop-badge">{badge}</div>'
        f'<h2 style="margin:0.3rem 0;">{rec.get("crop_name", "")}</h2>'
        f'{category_note_html}'
        f'<p style="margin:0.8rem 0;"><strong>Why we suggest this:</strong> {rec.get("rank_rationale", "")}</p>'
        f'<p style="margin:0.6rem 0;">🌱 <strong>Seed to use:</strong> {rec.get("seed_variety", "")}</p>'
        f'<p style="margin:0.6rem 0;">🚜 <strong>How to grow it:</strong> {rec.get("cultivation_practices", "")}</p>'
        f'<p style="margin:0.6rem 0;">🧪 <strong>Fertilizer:</strong> {rec.get("fertilizer_guidance", "")}</p>'
        f'<p style="margin:0.6rem 0;">💧 <strong>Watering:</strong> {rec.get("irrigation_guidance", "")}</p>'
        f'<div class="sanjeevni-numbers-row">'
        f'<div class="sanjeevni-number-box"><div class="sanjeevni-number-label">Expected Harvest</div>'
        f'<div class="sanjeevni-number-value">{production} tonnes</div></div>'
        f'<div class="sanjeevni-number-box"><div class="sanjeevni-number-label">Expected Earnings</div>'
        f'<div class="sanjeevni-number-value">₹{format_inr(revenue)}</div></div>'
        f'</div>'
        f'<p style="margin:0.4rem 0; font-size:0.9rem; opacity:0.8;">{rec.get("price_confidence_note", "")}</p>'
        f'{disclosure_lines}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# Limitations -- tucked into a collapsible section so it doesn't clutter the main view, but still easy to find.

if report.get("limitations_disclosed"):
    with st.expander("ℹ️ Good to Know"):
        for note in report["limitations_disclosed"]:
            st.markdown(f"- {note}")


# Chatbot -- Step 4, first pass: a normal section at the bottom of the page using native Streamlit chat widgets. This is deliberately NOT the floating bottom-right popup from the original spec yet -- that's a separate, riskier CSS-positioning attempt to try only after this confirmed-working version exists, so a fancy layout attempt can never take down a working chatbot.
from chatbot import send_chat_message
from ui_common import CHATBOT_LOGO_PATH

st.divider()
st.markdown("## 💬 Ask a Follow-Up Question")

# Streamlit's chat widgets have their own internal text styling that doesn't automatically follow custom theme colors -- force it explicitly, since light mode was otherwise rendering invisible (white-on-white) text.
st.markdown(f"""
<style>
[data-testid="stChatMessage"], [data-testid="stChatMessage"] * {{
    color: {c['text']} !important;
}}
[data-testid="stChatInput"], [data-testid="stChatInput"] textarea {{
    color: {c['text']} !important;
}}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_previous_interaction_id" not in st.session_state:
    st.session_state.chat_previous_interaction_id = None

for msg in st.session_state.chat_history:
    avatar = CHATBOT_LOGO_PATH if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask about your recommendations, fertilizer, timing, anything...")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar=CHATBOT_LOGO_PATH):
        try:
            with st.spinner("Thinking..."):
                answer, new_id = send_chat_message(
                    user_question, report,
                    previous_interaction_id=st.session_state.chat_previous_interaction_id,
                )
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.chat_previous_interaction_id = new_id
        except TimeoutError as e:
            st.error(str(e))

"""
ui_common.py — shared state, theme, and CSS for Sanjeevni's multipage app.

Imported by app.py and every individual page file. No navbar CSS anymore --
Streamlit's native st.navigation sidebar handles that now, which is the
whole point of this pivot.
"""

import os
import base64
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.environ.get("SANJEEVNI_ASSETS_DIR", os.path.join(BASE_DIR, "assets"))
LOGO_PATH = os.path.join(ASSETS_DIR, "Logo_sanjeevni.png")
CHATBOT_LOGO_PATH = os.path.join(ASSETS_DIR, "SanjeevniChatbotLogo.png")  # not used yet, this is for the chatbot logo coming in Step 4


def encode_image_b64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_floating_logo():
    """Shows the logo top-right on every page except the welcome gate --
    only call this from app.py's post-welcome branch, never from the
    welcome-gate branch itself. Using position:fixed here since it pins
    the logo to the actual browser viewport, not to whatever Streamlit's
    nested containers happen to be doing -- none of the ancestor-width
    guessing the earlier navbar attempts needed."""
    logo_b64 = encode_image_b64(LOGO_PATH)
    if not logo_b64:
        return
    st.markdown(f"""
    <style>
    .sanjeevni-floating-logo {{
        position: fixed;
        top: 75px;
        right: 28px;
        z-index: 999;
    }}
    .sanjeevni-floating-logo img {{
        height: 210px;
        width: auto;
        display: block;
    }}
    </style>
    <div class="sanjeevni-floating-logo">
        <img src="data:image/png;base64,{logo_b64}">
    </div>
    """, unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "theme": "light",
        "welcomed": False,
        "visited_sections": ["home"],
        "report_generated": False,
        "farmer_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def start_over():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_session_state()


def get_theme_colors():
    if st.session_state.theme == "dark":
        return {"bg": "#1A1A1A", "text": "#FFFFFF", "accent": "#76C51C",
                "accent2": "#1DA9F0", "card_bg": "#242424"}
    return {"bg": "#FFFFFF", "text": "#000000", "accent": "#1B4332",
            "accent2": "#1DA9F0", "card_bg": "#F3F7F4"}


def inject_page_css():
    """The shared CSS every page gets: theme background/text, the boxed
    About section, hoverable step cards, primary buttons, and the welcome
    card too -- that last one's only actually visible pre-login, but
    there's no harm in defining it globally."""
    c = get_theme_colors()
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700;800;900&display=swap');

    .stApp, .stApp * {{
        font-family: 'Poppins', sans-serif !important;
    }}
    /* Putting Streamlit's own icon font back for anything icon-related --
       the broad font-family rule above was overriding it, so icons like
       the sidebar collapse arrow were showing up as literal text
       ("keyboard_double_arrow_left") instead of the actual glyph. */
    [data-testid="stIconMaterial"],
    [data-testid="collapsedControl"] *,
    [data-testid="stSidebarCollapsedControl"] *,
    span[class*="material-symbols"],
    span[class*="material-icons"] {{
        font-family: 'Material Symbols Outlined', 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-weight: 800 !important;
    }}
    h1 {{ font-size: 2.6rem !important; }}
    h2 {{ font-size: 2.1rem !important; }}
    h3 {{ font-size: 1.6rem !important; }}
    p, li, span {{
        font-weight: 400;
    }}

    .stApp {{ background-color: {c['bg']}; color: {c['text']}; }}

    /* None of the native form widgets were styled before this -- they
       were just showing Streamlit's raw defaults, including that default
       red accent color, nothing to do with our actual theme. */

    /* Widget labels (State, District, Nitrogen, etc.) */
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
        color: {c['text']} !important;
        opacity: 1 !important;
    }}

    /* Selectbox -- the visible closed box */
    div[data-baseweb="select"] > div {{
        background-color: {c['card_bg']} !important;
        color: {c['text']} !important;
        border-color: {c['accent']} !important;
    }}
    /* Making sure everything inside the selectbox follows the theme, no
       exceptions -- this specifically catches the live search/typing
       input, which was rendering invisible (white text on a white
       background) since it's a completely different element from the
       final selected-value display and wasn't caught by the rule above. */
    div[data-baseweb="select"], div[data-baseweb="select"] * {{
        color: {c['text']} !important;
    }}
    /* Selectbox -- the open dropdown menu of options */
    ul[data-baseweb="menu"] {{
        background-color: {c['card_bg']} !important;
    }}
    li[data-baseweb="menu-item"] {{
        color: {c['text']} !important;
    }}

    /* Text input (Area field) */
    [data-testid="stTextInput"] input {{
        background-color: {c['card_bg']} !important;
        color: {c['text']} !important;
        border-color: {c['accent']} !important;
    }}

    /* Radio buttons (Irrigation level) -- option text plus the accent
       color. Same nested-tag issue I ran into with the button labels
       earlier: the visible text sits in its own tag inside the label,
       not directly on it. */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label span,
    [data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {{
        color: {c['text']} !important;
    }}
    [data-testid="stRadio"] input {{
        accent-color: {c['accent']} !important;
    }}

    /* Sliders (Soil Health Card) -- the thumb (using the ARIA role
       selector since that one's reliable), the current-value readout,
       and the proportional fill. Went one level deeper on the fill
       selector than my first attempt, which was actually hitting the
       full-width outer wrapper instead of the real value-proportional
       fill -- that's why it looked like a solid box instead of an
       actual slider. */
    [data-testid="stSlider"] [role="slider"] {{
        background-color: {c['accent']} !important;
        border-color: {c['accent']} !important;
    }}
    [data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] * {{
        color: {c['accent']} !important;
        background-color: transparent !important;
    }}
    /* Min/max range boundary labels -- making the box fully transparent,
       not just background-matched, since background-matching wasn't
       reaching the actual element rendering Streamlit's default colored
       badge. Just the accent-colored number should show. Applying this
       to both the element and its children since I wasn't sure which
       nested level actually held the visible background. */
    [data-testid="stTickBarMin"], [data-testid="stTickBarMin"] *,
    [data-testid="stTickBarMax"], [data-testid="stTickBarMax"] * {{
        background-color: transparent !important;
        color: {c['accent']} !important;
    }}
    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
        background-color: {c['accent']} !important;
    }}

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {{
        background-color: {c['accent']} !important; color: {c['bg']} !important; border: none !important;
        border-radius: 20px !important; padding: 0.75rem 2rem !important; font-size: 1.4rem !important; font-weight: 700 !important;
        white-space: nowrap !important;
    }}
    div.stButton > button[kind="primary"] p,
    div.stButton > button[data-testid="stBaseButton-primary"] p {{
        font-size: 1.4rem !important; font-weight: 700 !important;
    }}

    /* Back button -- outline style so it reads as secondary next to the
       solid Next/forward button, same size and rounding otherwise. */
    .st-key-back_btn button {{
        background-color: transparent !important;
        color: {c['accent']} !important;
        border: 2px solid {c['accent']} !important;
        border-radius: 20px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
    }}
    .st-key-back_btn button p {{
        font-size: 1.4rem !important; font-weight: 700 !important;
    }}

    .sanjeevni-about-box {{
        margin: 0 10%; background-color: {c['card_bg']}; border-radius: 16px; padding: 2.5rem 3rem;
    }}

    .sanjeevni-step-card {{
        background-color: {c['card_bg']}; border-radius: 14px; padding: 1.5rem 2rem;
        margin-bottom: 1rem; transition: transform 0.15s ease, box-shadow 0.15s ease;
        border-left: 5px solid {c['accent']};
    }}
    .sanjeevni-step-card:hover {{ transform: scale(1.015); box-shadow: 0 6px 20px rgba(0,0,0,0.15); }}
    .sanjeevni-step-number {{
        font-size: 0.85rem; font-weight: 700; letter-spacing: 1px;
        color: {c['accent']}; text-transform: uppercase;
    }}

    /* Welcome card -- always dark grey no matter the theme, that's locked
       in by design. Using max-width plus margin:auto gives precise,
       predictable sizing regardless of the outer st.columns ratio --
       more reliable than trying to guess the right ratio. */
    .st-key-welcome_card {{
        background-color: #1A1A1A !important;
        border-radius: 20px !important;
        padding: 2.5rem 3rem !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.35) !important;
        border: none !important;
        max-width: 600px !important;
        margin: 0 auto !important;
    }}
    /* Forcing width and centering across every wrapper level Streamlit
       might insert here -- if even one of these ends up shrink-wrapped
       to the button's own width, neither margin:auto nor inherited
       text-align has any actual room left to center anything. */
    .st-key-welcome_continue,
    .st-key-welcome_continue > div,
    .st-key-welcome_continue [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-welcome_continue [data-testid="stVerticalBlock"],
    .st-key-welcome_continue [data-testid="stElementContainer"],
    .st-key-welcome_continue .stButton {{
        width: 100% !important;
        text-align: center !important;
    }}
    .st-key-welcome_continue button {{
        display: inline-block !important;
        width: auto !important;
        background-color: #1DA9F0 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.6rem !important;
        min-width: 120px !important;
        white-space: nowrap !important;
    }}
    .st-key-welcome_continue button p {{
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }}
    .st-key-welcome_continue button:hover {{ background-color: #1697d9 !important; }}
    </style>
    """, unsafe_allow_html=True)


def inject_theme_heading_styles():
    """Sets every heading (h1/h2/h3) to the theme's accent color and bumps
    the weight up to 900 -- one step past the site-wide 800 default. Only
    called from app.py's post-welcome branch on purpose: this loads after
    inject_page_css()'s general rule, so it overrides both color and
    weight through source order alone, without ever touching the welcome
    card's own fixed styling."""
    c = get_theme_colors()
    st.markdown(f"""
    <style>
    h1, h2, h3 {{ color: {c['accent']} !important; font-weight: 900 !important; }}
    </style>
    """, unsafe_allow_html=True)


def render_theme_toggle_sidebar():
    label = "🌙 Switch to Dark Mode" if st.session_state.theme == "light" else "☀️ Switch to Light Mode"
    if st.sidebar.button(label, key="theme_toggle_sidebar", use_container_width=True):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()


def render_start_over_sidebar():
    if st.session_state.report_generated:
        st.sidebar.divider()
        if st.sidebar.button("🔁 Start Over", key="start_over_sidebar", use_container_width=True):
            start_over()
            st.rerun()

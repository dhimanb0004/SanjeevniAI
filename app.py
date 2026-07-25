"""
app.py — Sanjeevni AI entry point (multipage architecture).

Handles ONLY: page config, session state, the welcome gate, and building
the dynamic st.navigation() page list based on what's unlocked. All actual
section content lives in its own file: home_page.py, instructions_page.py,
input_page.py, report_page.py.

Run with: streamlit run app.py
"""

import streamlit as st
from ui_common import init_session_state, inject_page_css, render_theme_toggle_sidebar, render_start_over_sidebar, render_floating_logo, inject_theme_heading_styles

st.set_page_config(
    page_title="Sanjeevni AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
inject_page_css()

# -------------------- Welcome gate --------------------
# Shown once per session, before any real navigation exists.

if not st.session_state.welcomed:
    # Hiding the sidebar just for this branch. Since we never call st.navigation() here, Streamlit falls back to auto-discovering every file in pages/ and shows them all in the sidebar -- CSS is the simplest way to suppress that without touching how navigation actually works elsewhere.
    st.markdown(
        '<style>[data-testid="stSidebar"] { display: none !important; }</style>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height: 28vh;"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.container(key="welcome_card"):
            st.markdown(
                "<div style='text-align:center;'>"
                "<h1 style='color:#76C51C; margin:0 0 0.4rem 0; font-size:1.9rem; white-space:nowrap;'>"
                "Welcome to SanjeevniAI</h1>"
                "<p style='color:#76C51C; font-size:1.3rem; font-weight:400; margin:0 0 1.6rem 0;'>"
                "AI powered AG-Tech Platform</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            with st.container(key="welcome_continue"):
                if st.button("Continue"):
                    st.session_state.welcomed = True
                    st.rerun()

else:
    # -------------------- Sidebar extras --------------------
    # Theme toggle is always there, but Start Over only shows up once there's an actual report to start over from.
    render_theme_toggle_sidebar()
    render_start_over_sidebar()
    render_floating_logo()
    inject_theme_heading_styles()

    # -------------------- Dynamic page list --------------------
    # This is the actual progressive-unlock mechanism. Paths need to poin into pages/ specifically -- st.switch_page() only accepts the main script or files under pages/, it won't take arbitrary paths the way st.Page() will. Learned this one the hard way after switch_page kept silently failing with a path that worked fine for st.Page().
    pages = [st.Page("pages/home_page.py", title="Home", default=True)]

    if "instructions" in st.session_state.visited_sections:
        pages.append(st.Page("pages/instructions_page.py", title="Instructions"))

    if "input" in st.session_state.visited_sections:
        pages.append(st.Page("pages/input_page.py", title="Farm Details"))

    if st.session_state.report_generated:
        pages = [st.Page("pages/report_page.py", title="Your Report")]

    nav = st.navigation(pages)

    # Handling any cross-page navigation a page queued up through its own "Next ->" button. This has to be two steps -- queue it, rerun, THEN switch here -- because st.navigation()'s set of active pages gets locked in at the top of each run. Trying to switch straight from inside a page to a target that was only just added to session_state in that same run doesn't work, since that target was never part of this run's registered page set to begin with.
    pending = st.session_state.pop("_pending_switch", None)
    if pending:
        st.switch_page(pending)

    nav.run()

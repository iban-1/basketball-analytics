"""Glass-style theme: a full-page photo behind translucent, blurred panels with pill navigation.

The photo is optional and local: put an image at app/static/background.webp (git-ignored, never
committed). Without it the page uses a dark gradient with the same glass panels.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

BACKGROUND = Path(__file__).resolve().parent / "static" / "background.webp"   # page background and banner
ACCENT = "#ff9a3c"
NAV = [("Home", "Home.py"), ("Team analysis", "pages/1_Team_analysis.py"),
       ("Player analysis", "pages/2_Player_analysis.py"), ("Tracking", "pages/3_Tracking.py"),
       ("Video analysis", "pages/4_Video_analysis.py")]


def _css() -> str:
    photo = "url('app/static/background.webp') center / cover no-repeat fixed," if BACKGROUND.exists() else ""
    return f"""
<style>
/* ---- page background: photo under a dark veil so text stays readable ---- */
.stApp {{
  background:
    linear-gradient(180deg, rgba(8,10,16,.12) 0%, rgba(8,10,16,.45) 100%),
    {photo}
    radial-gradient(circle at 20% 10%, #2a1f3d 0%, #0b0d12 70%) fixed;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
/* the header bar spans the top of the page and would swallow clicks meant for the navigation words:
   let clicks fall through it, but keep its own buttons (Deploy, menu) clickable */
header[data-testid="stHeader"], [data-testid="stToolbar"] {{ pointer-events: none; }}
[data-testid="stToolbar"] button, [data-testid="stToolbar"] a, [data-testid="stMainMenu"],
[data-testid="stAppDeployButton"] {{ pointer-events: auto; }}

/* ---- glass panels ---- */
.block-container {{
  max-width: 1120px; margin-left: auto; margin-right: auto;
  position: relative; isolation: isolate;
  padding: 2.2rem 2.4rem 3rem 2.4rem !important;
  margin-top: 1.2rem; margin-bottom: 2rem;
}}
/* the glass lives on a layer behind the content (a blur on the panel itself would trap the fixed nav inside it) */
.block-container::before {{
  content: ""; position: absolute; inset: 0; z-index: -1; border-radius: 22px;
  background: rgba(14,17,26,.55);
  backdrop-filter: blur(12px) saturate(130%);
  -webkit-backdrop-filter: blur(12px) saturate(130%);
  border: 1px solid rgba(255,255,255,.10);
  box-shadow: 0 18px 60px rgba(0,0,0,.45);
}}
section[data-testid="stSidebar"] > div {{
  background: rgba(12,15,24,.42);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255,255,255,.08);
}}

/* ---- no sidebar: navigation is the pill at the top ---- */
[data-testid="stSidebar"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {{
  display: none !important;
}}

/* ---- navigation: fully transparent, just words, fixed at the top left ---- */
.st-key-pillnav {{
  position: fixed; top: 16px; left: 26px; z-index: 999991;
  width: auto !important; flex-direction: row !important; flex-wrap: nowrap; gap: .3rem !important;
  background: none; border: none; box-shadow: none; padding: 0;
}}
/* soft dark fade across the very top (behind the words) so they read over any photo */
.st-key-pillnav::before {{
  content: ""; position: fixed; top: 0; left: 0; right: 0; height: 76px; z-index: -1; pointer-events: none;
  background: linear-gradient(180deg, rgba(6,8,12,.78) 0%, rgba(6,8,12,0) 100%);
}}
.st-key-pillnav a {{
  padding: .3rem .8rem; white-space: nowrap; background: transparent; text-decoration: none;
  color: rgba(255,255,255,.72) !important; border-bottom: 2px solid transparent;
  text-shadow: 0 1px 10px rgba(0,0,0,.85);
}}
.st-key-pillnav a p {{
  font-size: .74rem !important; letter-spacing: .16em; text-transform: uppercase; font-weight: 700;
}}
.st-key-pillnav a:hover {{ color: #fff !important; }}
.st-key-pillnav a[aria-current="page"] {{ color: #fff !important; border-bottom-color: {ACCENT}; }}
.block-container {{ margin-top: 4.6rem !important; }}

/* ---- type ---- */
h1 {{ letter-spacing: .02em; font-weight: 700; }}
h2, h3 {{ letter-spacing: .01em; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: rgba(238,240,244,.62) !important; }}
label p, [data-testid="stWidgetLabel"] p {{
  text-transform: uppercase; letter-spacing: .1em; font-size: .7rem !important; color: rgba(238,240,244,.7);
}}

/* ---- cards, inputs, tables ---- */
[data-testid="stMetric"] {{
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10);
  border-radius: 16px; padding: .8rem 1rem;
}}
[data-testid="stMetricValue"] {{ color: {ACCENT}; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{
  background: rgba(255,255,255,.08) !important; border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,.14) !important;
}}
[data-testid="stDataFrame"] {{
  border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,.10);
}}
.stButton > button {{
  border-radius: 999px; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.22);
  letter-spacing: .08em; text-transform: uppercase; font-size: .75rem;
}}
.stButton > button:hover {{ background: {ACCENT}; border-color: {ACCENT}; color: #111; }}
[data-testid="stTabs"] button[role="tab"] {{ letter-spacing: .1em; text-transform: uppercase; font-size: .75rem; }}
[data-testid="stExpander"] {{
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10); border-radius: 14px;
}}
hr {{ border-color: rgba(255,255,255,.12) !important; }}

/* ---- hero banner: the photo, unveiled, with the page title ---- */
.hero {{
  position: relative; height: 230px; margin: -.4rem 0 1.4rem 0; border-radius: 18px; overflow: hidden;
  background: {"url('app/static/background.webp') center 14% / cover no-repeat," if BACKGROUND.exists() else ""}
              linear-gradient(135deg, #2a1f3d, #0b0d12);
  border: 1px solid rgba(255,255,255,.14);
}}
.hero::after {{
  content: ""; position: absolute; inset: 0;
  background: linear-gradient(90deg, rgba(8,10,16,.82) 0%, rgba(8,10,16,.25) 55%, rgba(8,10,16,0) 100%);
}}
.hero .t {{ position: absolute; left: 28px; bottom: 22px; z-index: 2; }}
.hero .t small {{ display: block; letter-spacing: .22em; text-transform: uppercase; font-size: .7rem;
  color: {ACCENT}; font-weight: 700; margin-bottom: .2rem; }}
.hero .t span {{ font-size: 2.3rem; font-weight: 700; color: #fff; text-shadow: 0 2px 18px rgba(0,0,0,.6); }}
</style>
"""


def apply() -> None:
    st.markdown(_css(), unsafe_allow_html=True)


_PAGES: list = []


def pages() -> list:
    """The st.Page objects, built once so main.py (routing) and nav() (links) share the same ones."""
    if not _PAGES:
        for label, path in NAV:
            _PAGES.append(st.Page(path, title=label, default=(path == "Home.py"),
                                  url_path=None if path == "Home.py" else path.split("_", 1)[1][:-3]))
    return _PAGES


def nav() -> None:
    """The pill navigation (only when the app runs through app/main.py, which registers the pages)."""
    try:
        with st.container(horizontal=True, key="pillnav"):
            for page in pages():
                st.page_link(page, label=page.title)
    except Exception:          # a page run on its own (tests) has no registered pages to link to
        pass


def hero(title: str) -> None:
    """Photo banner with the page title (replaces the plain page title)."""
    st.markdown(f'<div class="hero"><div class="t"><small>Basketball analytics</small><span>{title}</span></div></div>',
                unsafe_allow_html=True)

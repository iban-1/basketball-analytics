"""Entry point: routes the pages; the pill navigation is drawn by theme.nav().   Run:  streamlit run app/main.py"""
import streamlit as st

import theme

st.set_page_config(page_title="Basketball analytics", layout="wide", initial_sidebar_state="collapsed")
pages = [st.Page(path, title=label, default=(path == "Home.py"), url_path=None if path == "Home.py" else path.split("_", 1)[1][:-3])
         for label, path in theme.NAV]
st.navigation(pages, position="hidden").run()

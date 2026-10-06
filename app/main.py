"""Entry point: routes the pages; the navigation words are drawn by theme.nav().   Run:  streamlit run app/main.py"""
import streamlit as st

import theme

st.set_page_config(page_title="Basketball analytics", layout="wide", initial_sidebar_state="collapsed")
st.navigation(theme.pages(), position="hidden").run()

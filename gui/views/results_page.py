import streamlit as st

from core.results_browser import render_results_page


def results_page() -> None:
    ws = st.session_state.workspace
    if ws is None:
        st.info("Open a workspace first.")
        return
    st.header("Results")
    render_results_page(ws.root)

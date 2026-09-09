import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from core.workspace import Workspace
from views.run_page import run_page
from views.results_page import results_page

st.set_page_config(page_title="Workspace GUI", layout="wide")

if "workspace" not in st.session_state:
    st.session_state.workspace = None
if "selected_model_idx" not in st.session_state:
    st.session_state.selected_model_idx = 0
if "running" not in st.session_state:
    st.session_state.running = False
if "final_output" not in st.session_state:
    st.session_state.final_output = []
if "_output_buf" not in st.session_state:
    st.session_state._output_buf = []
if "_proc" not in st.session_state:
    st.session_state._proc = None
if "_run_started" not in st.session_state:
    st.session_state._run_started = False

run_pg = st.Page(run_page, title="New Run", url_path="run")
view_pg = st.Page(results_page, title="View Results", url_path="results")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    if st.session_state.get("_close_ws_popover", False):
        st.session_state["ws_popover"] = False
        st.session_state._close_ws_popover = False

    with st.popover(
        "Open Workspace",
        width="stretch",
        disabled=st.session_state.running,
        key="ws_popover",
        on_change="rerun",
    ):
        ws_input = st.text_input(
            "Enter workspace path",
            value="",
            placeholder=r"C:\path\to\workspace",
        )

        if st.button("Open", type="primary", width="stretch", disabled=st.session_state.running):
            st.session_state.final_output = []
            st.session_state._output_buf = []
            st.session_state._proc = None
            st.session_state._run_started = False
            ws = Workspace(Path(ws_input))
            errors = ws.validate()
            if errors:
                for e in errors:
                    st.error(e)
                st.session_state.workspace = None
            else:
                ws.load_manifest()
                st.session_state.workspace = ws
                st.session_state.selected_model_idx = 0
                st.session_state._close_ws_popover = True
                st.rerun()

    ws: Workspace | None = st.session_state.workspace

    if ws:
        st.markdown(f":orange[**{ws.root.name}**]")
        st.caption(str(ws.root))

        with st.expander("Virtual Environment"):
            if ws.venv_exists:
                st.caption(":green[Ready]")
            else:
                st.caption(":orange[Not found] — create it to run models.")
                if st.button("Create Venv", disabled=st.session_state.running):
                    venv_placeholder = st.empty()
                    output_lines = []

                    def venv_callback(msg):
                        output_lines.append(msg)
                        venv_placeholder.text("\n".join(output_lines))

                    with st.spinner("Setting up virtual environment..."):
                        success = ws.create_venv(on_output=venv_callback)
                    if success:
                        st.success("Venv created successfully")
                    else:
                        st.error("Venv creation failed. Check output above.")
                    st.rerun()

        st.divider()
        col_nav_l, col_nav_r = st.columns(2)
        with col_nav_l:
            st.page_link(run_pg, label="New Run", width="stretch")
        with col_nav_r:
            st.page_link(view_pg, label="View Results", width="stretch")
        st.divider()

# ── Pages ────────────────────────────────────────────────────────
pg = st.navigation([run_pg, view_pg], position="hidden")
pg.run()

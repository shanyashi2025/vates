import time
from datetime import datetime

import streamlit as st

from core.form_builder import render_form
from core.runner import generate_run_json, run_model

_RUNNING_BTN_HTML = """
<style>
@keyframes runGradient {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes runPulse {
  0%   { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.5); }
  70%  { box-shadow: 0 0 0 12px rgba(255, 75, 75, 0); }
  100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
}
div.running-btn {
  width: 100%;
  text-align: center;
  padding: 0.55rem 1rem;
  border-radius: 0.5rem;
  color: #ffffff;
  font-weight: 700;
  letter-spacing: 0.05em;
  border: 1px solid #ff4b4b;
  background: linear-gradient(270deg, #ff4b4b, #ff8a34, #ff4b4b);
  background-size: 200% 200%;
  animation: runGradient 1.4s ease infinite, runPulse 1.6s ease infinite;
}
</style>
<div class="running-btn">&#9881; Running&#8230;</div>
"""

_BORDER_RUNNING_CSS = """
<style>
@keyframes runBorderPulse {
  0%   { border-color: rgba(255, 75, 75, 0.8); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.3); }
  70%  { border-color: rgba(255, 138, 52, 0.9); box-shadow: 0 0 0 8px rgba(255, 75, 75, 0); }
  100% { border-color: rgba(255, 75, 75, 0.8); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
}
.st-key-exec_output {
  border: 1px solid #ff4b4b !important;
  animation: runBorderPulse 1.6s ease infinite;
}
</style>
"""


def run_page() -> None:
    ws = st.session_state.workspace
    if ws is None:
        st.info("Click **Open Workspace** and enter a workspace path to begin.")
        return

    model_names = [m.get("name", f"Model {i}") for i, m in enumerate(ws.models)]
    if model_names:
        if len(model_names) <= 8:
            selected = st.radio(
                "Select a model",
                options=model_names,
                index=st.session_state.selected_model_idx,
                key="model_radio",
                horizontal=True,
            )
        else:
            selected = st.selectbox(
                "Select a model",
                options=model_names,
                index=st.session_state.selected_model_idx,
                key="model_radio",
            )
        st.session_state.selected_model_idx = model_names.index(selected)

    model = ws.models[st.session_state.selected_model_idx]
    model_id = model.get("name", f"model_{st.session_state.selected_model_idx}").lower().replace(" ", "_")

    st.header(model.get("name", "Model"))
    st.caption(str(ws.resolve_script_path(model["script_path"])))
    if model.get("description"):
        st.write(model["description"])
    if model.get("notes"):
        st.caption(model["notes"])

    st.divider()

    schema = model.get("run_config_json_schema", {})
    input_dirs = ws.list_input_dirs()

    values = render_form(schema, input_dirs, ws.root, key_prefix=f"model_{st.session_state.selected_model_idx}")

    st.divider()

    # ── Run Button ───────────────────────────────────────────────
    col_run, col_spacer = st.columns([1, 3])
    with col_run:
        if st.session_state.running:
            st.markdown(_RUNNING_BTN_HTML, unsafe_allow_html=True)
            run_tooltip = None
        else:
            run_disabled = not ws.venv_exists
            run_tooltip = None
            if not ws.venv_exists:
                run_tooltip = "Create the workspace venv first"
            if st.button(
                "Run Model",
                type="primary",
                disabled=run_disabled,
                width="stretch",
                help=run_tooltip,
            ):
                st.session_state.running = True
                st.session_state.final_output = []
                st.session_state._output_buf = []
                st.session_state._proc = None
                st.session_state._run_started = False
                st.session_state.last_run_end_time = None
                st.session_state.last_run_exit_code = None
                st.session_state.run_start_time = datetime.now()
                st.rerun()

    # ── Running State ────────────────────────────────────────────
    if st.session_state.running:
        proc = st.session_state._proc

        if proc is not None and proc.poll() is not None:
            st.session_state.running = False
            rc = proc.returncode
            if rc == 0:
                st.session_state._output_buf.append("")
                st.session_state._output_buf.append("Model completed successfully (exit code 0)")
            else:
                st.session_state._output_buf.append("")
                st.session_state._output_buf.append(f"Model failed (exit code {rc})")
            st.session_state.final_output = list(st.session_state._output_buf)
            st.session_state.last_run_end_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.last_run_exit_code = rc
            st.session_state._proc = None
            st.session_state._run_started = False
            st.rerun()

        st.markdown(_BORDER_RUNNING_CSS, unsafe_allow_html=True)
        with st.container(border=True, key="exec_output"):
            with st.status(
                "Running",
                state="running",
                expanded=True,
            ):
                st.caption(f"Running… started {st.session_state.run_start_time:%H:%M:%S}")
                output_display = st.empty()

                current_lines = list(st.session_state._output_buf)
                output_display.code(
                    "\n".join(current_lines) if current_lines else "(starting...)",
                    language=None,
                )

        if not st.session_state._run_started:
            st.session_state._run_started = True

            run_json_path = generate_run_json(ws.root, model_id, values)
            st.session_state._output_buf.append(
                f"Run config saved: {run_json_path.relative_to(ws.root)}"
            )
            st.session_state._output_buf.append("")

            script_path = ws.resolve_script_path(model["script_path"])
            buf = st.session_state._output_buf

            def on_output(msg):
                buf.append(msg)

            proc = run_model(
                venv_python=ws.venv_python,
                script_path=script_path,
                run_json_path=run_json_path,
                cwd=ws.root,
                on_output=on_output,
            )
            st.session_state._proc = proc

            if proc is None:
                # Failed to launch (error already written to output buffer): finalize now.
                st.session_state._output_buf.append("")
                st.session_state._output_buf.append("Model failed to start. Check output above.")
                st.session_state.running = False
                st.session_state.final_output = list(st.session_state._output_buf)
                st.session_state.last_run_end_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.session_state.last_run_exit_code = None
                st.session_state._proc = None
                st.session_state._run_started = False
                st.rerun()

        time.sleep(0.3)
        st.rerun()

    # ── Completed State (Last Run) ───────────────────────────────
    if not st.session_state.running and st.session_state.final_output:
        with st.container(border=True, key="exec_output"):
            rc = st.session_state.get("last_run_exit_code")
            with st.status(
                "Last run",
                state="complete" if rc == 0 else "error",
                expanded=True,
            ):
                ts = st.session_state.get("last_run_end_time", "")
                if ts:
                    note = f"Finished {ts}"
                else:
                    note = "Finished"
                if rc is None:
                    note += " · failed to start"
                else:
                    note += f" · exit code {rc}"
                st.caption(note)
                st.code("\n".join(st.session_state.final_output), language=None)

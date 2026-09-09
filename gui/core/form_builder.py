import streamlit as st
from pathlib import Path
from typing import Any


def _normalize_input_path(line: str, workspace_root: Path) -> str:
    """Convert a user-entered input directory to an absolute path.

    Accepts a workspace-relative form ('inputs/foo') or an absolute path.
    """
    line = line.strip().strip('"').strip("'")
    if not line:
        return ""
    p = Path(line)
    if p.is_absolute():
        return str(p)
    if line.startswith("inputs/") or line.startswith("inputs\\"):
        return str(workspace_root / line)
    # Treat as workspace-relative (e.g. a typo or 'inputsfoo'): resolve under workspace
    return str(workspace_root / line)


def render_input_directories(
    field_index: int,
    field_name: str,
    field_schema: dict,
    input_dirs: list[str],
    workspace_root: Path,
    key_prefix: str,
    required: bool,
    description: str | None,
) -> list[str]:
    title = field_schema.get("title", field_name.replace("_", " ").title())
    default = field_schema.get("default")
    default_list = default if isinstance(default, list) else []

    # A single multi-line text area is the source of truth: each line is one directory.
    state_key = f"{key_prefix}_{field_name}_text"
    if state_key not in st.session_state:
        init_lines = []
        for line in default_list:
            if isinstance(line, str):
                p = _normalize_input_path(line, workspace_root)
                # display friendly relative form when under workspace inputs/
                try:
                    rel = Path(p).relative_to(workspace_root / "inputs")
                    init_lines.append(f"inputs/{rel.as_posix()}")
                except ValueError:
                    init_lines.append(p)
        st.session_state[state_key] = "\n".join(init_lines)

    st.markdown(f"{field_index}. {title}")

    entries = [l for l in st.session_state[state_key].splitlines() if l.strip()]

    known_prefix = f"{key_prefix}_{field_name}_cb"

    def render_grid(groups: list, ncols: int):
        cols = st.columns(ncols)
        for ci, group in enumerate(groups):
            with cols[ci]:
                for d in group:
                    display = f"inputs/{d}"
                    checked = display in entries
                    cb_key = f"{known_prefix}_{d}"
                    changed = st.checkbox(d, value=checked, key=cb_key)
                    if changed != checked:
                        current = [l for l in st.session_state[state_key].splitlines() if l.strip()]
                        if changed and display not in current:
                            current.append(display)
                        elif not changed:
                            current = [l for l in current if l != display]
                        st.session_state[state_key] = "\n".join(current)

    n = len(input_dirs)
    if n > 0:
        ncols = 3 if n > 16 else (2 if n > 8 else 1)
        groups = [input_dirs[: min(n, 24)][i::ncols] for i in range(ncols)]
        with st.expander("Quick select"):
            render_grid(groups, ncols)

    if description:
        st.caption(description)
    st.caption("Both relative path (to workspace, e.g. `inputs/some_dir`) "
               "and absolute path (e.g. `C:/path/to/dir`) are supported; " \
               "one per line")

    st.text_area(
        "Enter directories",
        placeholder="C:/path/to/dir\ninputs/some_dir\n... (one per line)",
        key=state_key,
        height=120,
        label_visibility="collapsed",
    )

    lines = [l for l in st.session_state[state_key].splitlines() if l.strip()]
    result = [_normalize_input_path(l, workspace_root) for l in lines]

    seen = set()
    out = []
    for p in result:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def render_form(
    schema: dict,
    input_dirs: list[str],
    workspace_root: Path,
    key_prefix: str = "",
) -> dict[str, Any]:
    """Render Streamlit form widgets from a JSON Schema and return the user's values.

    Only fields that declare a `default` are pre-filled; otherwise widgets start
    at their neutral/empty state.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    values = {}

    for i, (field_name, field_schema) in enumerate(properties.items(), 1):
        field_type = field_schema.get("type", "string")
        title = field_schema.get("title", field_name.replace("_", " ").title())
        description = field_schema.get("description", "")
        default = field_schema.get("default")
        key = f"{key_prefix}_{field_name}" if key_prefix else field_name

        if field_name == "input_directories":
            values[field_name] = render_input_directories(
                i, field_name, field_schema, input_dirs, workspace_root,
                key_prefix, field_name in required, description,
            )
        elif field_name == "results_directory":
            st.markdown(f"{i}. {title}")
            if description:
                st.caption(description)
            st.caption("Enter subfolder name under `results/`, e.g. 'my_output' represents `results/my_output/`")
            values[field_name] = st.text_input(
                "Folder name under results/",
                value=str(default) if default is not None else "",
                key=key,
                label_visibility="collapsed",
                placeholder="folder_name",
            )
        elif "enum" in field_schema:
            st.markdown(f"{i}. {title}")
            if description:
                st.caption(description)
            options = field_schema["enum"]
            idx = options.index(default) if default in options else 0
            values[field_name] = st.selectbox(
                f"{title}",
                options=options,
                index=idx,
                key=key,
                label_visibility="collapsed",
            )
        elif field_type == "boolean":
            st.markdown(f"{i}. {title}")
            if description:
                st.caption(description)
            values[field_name] = st.checkbox(
                f"{title}",
                value=default if default is not None else False,
                key=key,
                label_visibility="collapsed",
            )
        elif field_type == "integer":
            st.markdown(f"{i}. {title}")
            if description:
                st.caption(description)
            min_val = field_schema.get("minimum", -999999)
            max_val = field_schema.get("maximum",  999999)
            values[field_name] = st.number_input(
                f"{title}",
                min_value=min_val,
                max_value=max_val,
                value=default if default is not None else min_val,
                step=1,
                key=key,
                label_visibility="collapsed",
            )
        elif field_type == "number":
            st.markdown(f"{i}. {title}")
            if description:
                st.caption(description)
            min_val = field_schema.get("minimum", -1e10)
            max_val = field_schema.get("maximum",  1e10)
            values[field_name] = st.number_input(
                f"{title}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default) if default is not None else float(min_val),
                key=key,
                label_visibility="collapsed",
            )
        else:
            st.markdown(f"{i}. {title}")
            if description:
                st.caption(description)
            values[field_name] = st.text_input(
                f"{title}",
                value=str(default) if default is not None else "",
                key=key,
                label_visibility="collapsed",
            )

    return values

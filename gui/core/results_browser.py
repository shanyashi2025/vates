import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

_RECENT_LIMIT = 5
_PREVIEW_MAX_ROWS = 1000
_PREVIEW_MAX_COLS = 250
_JSON_PREVIEW_MAX = 10 * 1024 * 1024
_RUNLOG_MSG_LIMIT = 10


def _glob_runlogs(results_root: Path) -> list[Path]:
    """Recursively find every *.runlog.json under results_root."""
    return sorted(results_root.rglob("*.runlog.json"))


def _read_runlog(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _parse_timestamp(dt_str: str) -> datetime | None:
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return None


def scan_runs(workspace_root: Path) -> list[dict]:
    """Scan workspace results and build a list of run summaries.

    One entry per runlog: run folder, model_name, scenario, start, end,
    duration, success, and the result file paths it declared.
    """
    results_root = workspace_root / "results"
    if not results_root.is_dir():
        return []

    rows = []
    for rl in _glob_runlogs(results_root):
        data = _read_runlog(rl)
        if not data:
            continue
        exec_info = data.get("execution", {})
        cfg = data.get("configuration", {})
        results = data.get("results", [])
        files = []
        for f in results:
            fp = Path(f)
            if not fp.is_absolute():
                fp = results_root / fp
            files.append(fp.resolve())
        rows.append({
            "run_dir": rl.parent.name,
            "run_path": rl.parent,
            "runlog_path": rl.resolve(),
            "model": data.get("model_name", ""),
            "scenario": cfg.get("scenario", ""),
            "success": bool(exec_info.get("success")),
            "start": _parse_timestamp(exec_info.get("start", "")),
            "end": _parse_timestamp(exec_info.get("end", "")),
            "duration": exec_info.get("duration", ""),
            "n_files": len(results),
            "files": files,
        })

    rows.sort(key=lambda r: (r["end"] is not None, r["end"] or datetime.min), reverse=True)
    return rows


def _stat_key(path: Path) -> tuple[int, int]:
    """(size, mtime_ns) for cache invalidation; cheap stat call."""
    try:
        s = path.stat()
        return s.st_size, s.st_mtime_ns
    except OSError:
        return 0, 0


def _count_lines(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _pin_cols_for(path: Path) -> list[str]:
    name = path.name.lower()
    if name.endswith(".proj.csv"):
        return ["group", "owner", "variable"]
    if name.endswith(".stoch.csv"):
        return ["simulation", "group", "owner", "variable"]
    return []


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_preview(
    path: Path,
    size: int,
    mtime_ns: int,
    max_preview_rows: int = _PREVIEW_MAX_ROWS,
    max_preview_cols: int = _PREVIEW_MAX_COLS,
):
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, nrows=max_preview_rows)
            total_cols = len(df.columns)
            col_cut = total_cols > max_preview_cols
            if col_cut:
                df = df.iloc[:, :max_preview_cols]
            row_cut = _count_lines(path) - 1 > max_preview_rows
            pinned = [c for c in _pin_cols_for(path) if c in df.columns]
            return "csv", df, row_cut, col_cut, pinned
        except Exception as e:
            return "error", f"Could not parse CSV: {e}", False, False, []
    if path.suffix.lower() == ".json":
        if size > _JSON_PREVIEW_MAX:
            return "too_large", None, False, False, []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (
                path.name.lower().endswith(".runlog.json")
                and isinstance(data, dict)
                and isinstance(data.get("messages"), list)
                and len(data["messages"]) > _RUNLOG_MSG_LIMIT
            ):
                total = len(data["messages"])
                data = dict(data)
                data["messages"] = data["messages"][:_RUNLOG_MSG_LIMIT]
                data["messages_total"] = total
            return "json", data, False, False, []
        except Exception as e:
            return "error", f"Could not parse JSON: {e}", False, False, []
    return "text", None, False, False, []


def _open_in_explorer(path: Path) -> None:
    try:
        subprocess.Popen(["explorer.exe", f"/select,{path}"])
    except Exception:
        try:
            os.startfile(str(path.parent))
        except Exception as e:
            st.error(f"Could not open Explorer: {e}")


def _download_btn(path: Path, namespace: str = "") -> None:
    data = None
    mime = "application/octet-stream"
    if path.suffix.lower() == ".csv":
        try:
            data = path.read_bytes()
            mime = "text/csv"
        except OSError:
            data = None
    elif path.suffix.lower() == ".json":
        try:
            data = path.read_bytes()
            mime = "application/json"
        except OSError:
            data = None
    if data is None:
        return
    st.download_button(
        label="Download",
        data=data,
        file_name=path.name,
        mime=mime,
        key=f"dl_{namespace}_{path}",
        width="stretch",
    )


def _run_short_label(r: dict) -> str:
    when = r["end"] if r["end"] is not None else r["start"]
    ts = when.strftime("%Y-%m-%d %H:%M") if when is not None else "unknown time"
    return f"**{ts}** · {r['run_dir']}\\ · {r['model']} · {r['scenario'] or '-'}"


def render_results_page(workspace_root: Path) -> None:
    if workspace_root is None:
        st.info("Open a workspace first.")
        return
    results_root = workspace_root / "results"
    if not results_root.is_dir():
        st.info("No `results/` directory found in this workspace.")
        return

    runs = scan_runs(workspace_root)

    # ── Files scope ──────────────────────────────────────────────
    subdirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    all_files = sorted(p for p in results_root.rglob("*") if p.is_file() and not p.name.startswith("."))

    show_all = st.checkbox("All runs", value=True, key="results_show_all")
    picked_runs = []
    if runs:
        for r in runs[:_RECENT_LIMIT]:
            if st.checkbox(
                _run_short_label(r),
                value=True,
                key=f"results_run_{r['runlog_path']}",
                disabled=show_all,
            ) and not show_all:
                picked_runs.append(r)
    if picked_runs:
        run_files = set()
        for r in picked_runs:
            run_files.add(r["runlog_path"])
            run_files.update(r["files"])
        all_files = [p for p in all_files if p.resolve() in run_files]

    col_sub, col_search = st.columns([1, 2])
    with col_sub:
        chosen = st.selectbox(
            "Subfolder",
            options=["All"] + [d.name for d in subdirs],
            index=0,
        )
    with col_search:
        search = st.text_input("Search", value="", key="results_search", placeholder="search...")

    scope_files = all_files
    if chosen != "All":
        scope_files = sorted(p for p in all_files if p.parent == results_root / chosen)
    if search.strip():
        needle = search.strip()
        scope_files = [p for p in scope_files if needle.lower() in p.name.lower() or needle.lower() in str(p).lower()]

    if not scope_files:
        st.info("No files match the current filters.")
        return

    st.caption(f"{len(scope_files)} file(s)")
    for p in scope_files:
        rel = p.relative_to(results_root)

        if p.suffix.lower() in (".csv", ".json"):
            with st.expander(str(rel), on_change="rerun") as ex:
                if ex.open:
                    st.code(str(p))
                    col_pv, col_dl, col_ex = st.columns(3)
                    with col_pv:
                        if st.button("Preview", key=f"preview_{p}", width="stretch"):
                            st.session_state[f"show_{p}"] = True
                    with col_dl:
                        _download_btn(p, namespace=f"browse_{chosen}")
                    with col_ex:
                        if st.button("Open in Explorer", key=f"explorer_{p}", width="stretch"):
                            _open_in_explorer(p)
                    if st.session_state.get(f"show_{p}", False):
                        kind, payload, row_cut, col_cut, pinned = _cached_preview(p, *_stat_key(p))
                        if kind == "csv":
                            st.dataframe(
                                payload,
                                width="stretch",
                                column_config=(
                                    {c: st.column_config.Column(pinned=True) for c in pinned}
                                    if pinned
                                    else None
                                ),
                            )
                            if row_cut or col_cut:
                                st.caption(
                                    ":orange[Preview limited to the first "
                                    f"{_PREVIEW_MAX_ROWS} rows / {_PREVIEW_MAX_COLS} columns "
                                    "due to file size. Open the original file in Explorer "
                                    "for the full content.]"
                                )
                        elif kind == "json":
                            if isinstance(payload, dict) and "messages_total" in payload:
                                st.caption(
                                    f":orange[`messages` truncated to the first "
                                    f"{_RUNLOG_MSG_LIMIT} of {payload['messages_total']} entries.]"
                                )
                            st.json(payload)
                        elif kind == "too_large":
                            st.caption(
                                f":orange[Preview skipped — JSON exceeds "
                                f"{_JSON_PREVIEW_MAX // 1048576} MB. Use Download or "
                                "Open in Explorer for the full content.]"
                            )
                        else:
                            st.warning(payload)
        else:
            st.markdown(f"`{rel}`  ({p.stat().st_size:,} bytes)")

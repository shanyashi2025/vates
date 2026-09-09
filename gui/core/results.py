import json
from pathlib import Path

import pandas as pd
import streamlit as st


def show_results(results_dir: Path) -> None:
    """Display results from a directory: runlog + CSVs."""
    if not results_dir or not results_dir.is_dir():
        st.info("No results directory found.")
        return

    st.subheader(f"Results: {results_dir.name}")

    runlog_files = sorted(results_dir.glob("*.runlog.json"))
    if runlog_files:
        with open(runlog_files[0], "r", encoding="utf-8") as f:
            runlog = json.load(f)
        with st.expander("Run Log", expanded=False):
            st.json(runlog)

    csv_files = sorted(results_dir.glob("*.csv"))
    if csv_files:
        selected = st.selectbox(
            "Select a result file to view",
            options=[f.name for f in csv_files],
            key="result_csv_selector",
        )
        if selected:
            df = pd.read_csv(results_dir / selected)
            st.dataframe(df, width="stretch")
            st.download_button(
                label=f"Download {selected}",
                data=df.to_csv(index=False),
                file_name=selected,
                mime="text/csv",
            )
    else:
        st.info("No CSV result files found in this directory.")

    other_files = [
        f for f in results_dir.iterdir()
        if f.is_file() and f.suffix not in (".csv", ".json") and not f.name.startswith(".")
    ]
    if other_files:
        with st.expander("Other output files"):
            for f in other_files:
                st.text(f"{f.name}  ({f.stat().st_size:,} bytes)")

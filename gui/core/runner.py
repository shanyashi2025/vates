import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable


def generate_run_json(workspace_root: Path, model_id: str, values: dict) -> Path:
    """Write run config JSON to runs/ and return its path."""
    runs_dir = workspace_root / "runs"
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"_gui_{model_id}_{timestamp}.json"
    run_path = runs_dir / filename

    cleaned = {k: v for k, v in values.items() if v is not None and v != ""}

    # results_directory is entered as a bare folder name -> build <workspace>/results/<foo>
    rd = cleaned.get("results_directory")
    if isinstance(rd, str) and rd.strip():
        p = Path(rd.strip())
        if not p.is_absolute():
            cleaned["results_directory"] = str(workspace_root / "results" / p)
        else:
            cleaned["results_directory"] = str(p)
    else:
        cleaned.pop("results_directory", None)

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=4)

    return run_path


def run_model(
    venv_python: Path,
    script_path: Path,
    run_json_path: Path,
    cwd: Path,
    on_output: Callable[[str], None],
) -> subprocess.Popen | None:
    """Execute a model script in a subprocess. Returns the Popen object.

    on_output(str) is called for each line of stdout/stderr from a background thread.
    The caller should poll proc.poll() to detect completion.

    Returns None if the process could not be started.
    """
    if not venv_python.exists():
        on_output(f"Error: Python executable not found: {venv_python}")
        return None
    if not script_path.exists():
        on_output(f"Error: Model script not found: {script_path}")
        return None

    cmd = [str(venv_python), str(script_path), str(run_json_path)]
    on_output(f"Running: {' '.join(cmd)}")
    on_output(f"Working directory: {cwd}")
    on_output("---")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd),
            bufsize=1,
        )
    except Exception as e:
        on_output(f"Error starting process: {e}")
        return None

    def reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            on_output(line.rstrip("\n"))
        proc.stdout.close()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return proc

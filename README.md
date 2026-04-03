# Vates Project

Open-source Python framework and sample implementations for actuarial modeling.

### How this repo is organized

1. **`vates/`** — The **standard** library / installable package: reusable framework code (`pip install -e .`).
2. **`examples/model/`** — **Example / tutorial for model developers** (company builders): reference models (`fund_model.py`, `port_monte_carlo.py`) and local package(s) (`local_package/`).
3. **Other folders under `examples/`** (`scripts/`, `cli/`, `input/`, …) — **Example / tutorial for most colleagues**: how to run models for analysis and reporting (commands, sample data), without editing framework or model code.
4. The **`gui/`** app is another way for analysts to run tasks.

## Install the framework (`vates`)

Work from the **repository root**.

### 1. Create a virtual environment (recommended)

**PowerShell** (Windows):

```powershell
cd path\to\your\project
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy, run once in that window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then run `Activate.ps1` again.

**Command Prompt** (`cmd.exe`):

```bat
cd path\to\your\project
python -m venv .venv
.venv\Scripts\activate.bat
```

You should see `(.venv)` in the prompt when the environment is active.

### 2. Install `vates` in editable mode

```powershell
pip install -e .
```

This installs the distribution **`vates`** in editable mode; import the library as **`import vates`**.

## Project layout

| Path                | Role                                             |
|---------------------|--------------------------------------------------|
| `vates/`            | Core framework and library (`pip install -e .`). |
| `examples/model/`   | Reference models and company-style package(s).   |
| `examples/scripts/` | Python scripts that call the reference models.   |
| `examples/cli/`     | CLI commands and model args json files.          |
| `examples/input/`   | Sample CSV input data.                           |
| `gui/`              | End-user launcher.                               |
| `docs/`             | Structure notes and discussions.                 |


## Quick start

### 1) Example Models

| No | Example Model         | Python Script                            | CLI + json                  |
|----|-----------------------|------------------------------------------|-----------------------------|
| 1  | 01_asset_proj         | `python .\examples\scripts\script_01.py` | `.\examples\cli\cli_01.bat` |
| 2  | 02_fund_proj          | `python .\examples\scripts\script_02.py` | `.\examples\cli\cli_02.bat` |
| 3  | 03_cross_proj         | `python .\examples\scripts\script_11.py` | `.\examples\cli\cli_11.bat` |
| 5  | 21_smith_wilson       | `python .\examples\scripts\script_21.py` | `.\examples\cli\cli_21.bat` |


- Model output can be found in `.\examples\results\script_*` and/or `.\examples\results\cli_*`
- CLI is configured via the corresponding json file `.\examples\cli\model_args_*.json`.


### 2) GUI

Start GUI:
```powershell
.\gui\start.bat
```

## For model developers

- Treat **`examples/model/`** as a **reference** layout: copy `local_package/` (rename to your company package) and the model files into your own repository.
- Keep company-specific code **outside** `vates/`; depend on **`vates`** via `pip install` or your dependency mechanism.

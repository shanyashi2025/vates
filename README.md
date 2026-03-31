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

### 1) Script demos

| No | Example Model         | Python                                   |
|----|-----------------------|------------------------------------------|
| 1  | 01_asset_proj         | `python .\examples\scripts\script_01.py` |
| 2  | 02_fund_proj          | `python .\examples\scripts\script_02.py` |
| 3  | 11_stoch_monte_carlo  | `python .\examples\scripts\script_11.py` |
| 4  | 12_stoch_ec_mvl       | `python .\examples\scripts\script_12.py` |
| 5  | 21_smith_wilson       | `python .\examples\scripts\script_21.py` |
| 6  | 22_efficient_frontier | `python .\examples\scripts\script_22.py` |

Model output can be found in `.\examples\results\script_*`

### 2) CLI command (Windows) + json demos

| No | Example Model         | Batch                       | Python                                                                                 |
|----|-----------------------|-----------------------------|----------------------------------------------------------------------------------------|
| 1  | 01_asset_proj         | `.\examples\cli\cli_01.bat` | `python .\examples\model\em01_asset_proj.py         .\examples\cli\model_args_01.json` |
| 2  | 02_fund_proj          | `.\examples\cli\cli_02.bat` | `python .\examples\model\em02_fund_proj.py          .\examples\cli\model_args_02.json` |
| 3  | 11_stoch_monte_carlo  | `.\examples\cli\cli_11.bat` | `python .\examples\model\em11_stoch_monte_carlo.py  .\examples\cli\model_args_11.json` |
| 4  | 12_stoch_ec_mvl       | `.\examples\cli\cli_12.bat` | `python .\examples\model\em12_stoch_ec_mvl.py       .\examples\cli\model_args_12.json` |
| 5  | 21_smith_wilson       | `.\examples\cli\cli_21.bat` | `python .\examples\model\em21_smith_wilson.py       .\examples\cli\model_args_21.json` |
| 6  | 22_efficient_frontier | `.\examples\cli\cli_22.bat` | `python .\examples\model\em22_efficient_frontier.py .\examples\cli\model_args_22.json` |

Model output can be found in `.\examples\results\cli_*`

### 3) GUI

Start GUI:
```powershell
.\gui\start.bat
```

## For model developers

- Treat **`examples/model/`** as a **reference** layout: copy `local_package/` (rename to your company package) and the model files into your own repository.
- Keep company-specific code **outside** `vates/`; depend on **`vates`** via `pip install` or your dependency mechanism.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Vates** — an open-source Python library (requires Python ≥ 3.12) for actuarial modeling:
monthly-step projection of assets, liabilities, funds, and economic/ESG scenarios, plus
stochastic (Monte Carlo) solvency capital measurement. Depends only on `numpy` and `pandas`;
`scipy` is optional and gates the equity-option asset class.

## Development Commands

- **Install (editable):** `pip install -e .` from the repo root (imports as `import vates`).
  `.venv/` and `uv.lock` are present; the project uses setuptools + `pyproject.toml` (version `0.1.5`),
  so `pip`/`uv` can also read it. There is no separate test suite or linter config.

- **Run example models:** from the repo root, pass a JSON run-config as the first CLI arg:
  ```
  python .\examples\models\em01_asset_proj_inner.py .\examples\runs\em01_asset_proj_inner.json
  ```
  Each `examples/runs/*.json` is a flat key-value config (`model_name`, `start_year`/`start_month`,
  `end_year`, `scenario`, `workspace_directory`, `input_directories`, `results_directory`)
  consumed by `main()` via `json.load(sys.argv[1])`. Other example models: `em01_asset_proj_top`,
  `em02_fund_proj`, `em03_cross_proj`, `em11_monte_carlo`, `em12_stoch_ec_mvl`.
  `.run_all.bat` scripts in `examples/runs/` and `examples/simple_use_cases/` run every variant in
  sequence (Windows only, they `cd` back to the repo root).

- **Simple use cases:** standalone scripts under `examples/simple_use_cases/`
  (`Black_Scholes_Option_Pricing_Model`, `efficient_frontier`, `interest_rate_term_structure`),
  each with its own `main.py`.

- **GUI:** `python .\gui\main.py` (Tkinter; also invoked via `gui/start.bat`). Wraps the
  `em01_asset_proj_*` asset models behind a form that writes a run config and shells out to the
  model script.

## Architecture

### Core projection engine (`vates/_core/`)

The heart of the library. The public surface (`vates/__init__.py`) re-exports the core engine,
variables, and result reader, plus subpackages `alm`, `finmath`, `solvency`, `utils`.

- **`ProjModelEngine`** — the deterministic, monthly-step engine. Usage contract is always:
  1. construct with `model_name`/`description`; 2. `configure_run(...)`; 3. `@model.bind_projection`
  a function whose first parameter is the model (or a zero-arg function); 4. `model.run()`.
  During `run()` it iterates `t = 0 .. MAX_T`, advancing a `ProjectionTimeSynchronizer` that keeps
  `model.time` and `model.period` (a `pd.Period` with monthly freq) in lockstep. The bound function
  reads `model.time` / `model.period` (and read-only constants `START_DATE`, `END_DATE`, `MAX_T`,
  `SCENARIO`, `SIMULATION`, ...) and mutates output variables.

- **`variables`** — bound functions write results into `ConstVariable` (scalar/array, constant) or
  `TDimVariable` (indexed by `t`/`pd.Period`) instances. Variables self-register with the engine via
  `include_proj_variable`. On run, the engine serializes them to `{model_name}.proj.csv`. Each
  variable carries `group`/`owner`/`name` and up to 3 labeled `dims` (lists or Enums), which become
  row labels like `name[a:b:c]`. Use `proj_result(...)` (`vates/_core/_utils.py`) to read `.proj.csv`
  back into a DataFrame or a single cell `(group, owner, variable [, date])` value.

- **`StochExecutor`** — multiprocessing wrapper over `ProjModelEngine` for stochastic runs.
  `configure_run` takes `simulations` (a string like `"1-10,13"`) and `max_workers`. It splits
  simulations into batches across a `ProcessPoolExecutor`; each batch runs simulations through a
  fresh engine instance, appending rows to per-batch `.stoch.csv` files. A `.stoch.stat.csv` is
  produced only when inputs include a `__stoch_setting__.json` with a `statistic` dict
  (mean/std/median/max/min and `perc%`). Two caveats: the bound function is executed in worker
  processes, so it must be top-level/picklable; and stochastic projection typically needs a shared
  per-period sync — see `add_projection_time_synchronizer` below.

- **`RunConfig`** (`vates/_core/_utils.py`) — frozen dataclass with exhaustive validation
  (`validate_number` / `validate_string` / `validate_period` / ...); created via `RunConfig.create`.
  `max_t` is constrained to 0–2400, months capped 0–2400 steps, period range 1900-01 to 5999-12.

### Time synchronization pattern

`ProjectionTimeSynchronizer` broadcasts `time`/`period` changes to registered observers via
`attach_time_observer`, and `add_projection_time_synchronizer` is a class decorator that injects a
`_time_synchronizer` into a class and (re)defines `time`/`period` properties bound to it. When a
class receives a `model_engine` keyword argument, the decorator wires it to the engine's
synchronizer so the object advances with the model. This is how economic/asset class objects stay in
sync with the running projection. `vates/alm/assets/asset_base.py` is the canonical example
(`@add_projection_time_synchronizer class Asset`).

### ALM library (`vates/alm/`)

High-level Asset–Liability Management building blocks that plug into the projection engine:

- **`econs/`** — economic/ESG items: `YieldCurve` (spot-rate curve with a `last_update` time tag),
  `CreditBand`, `EquityIndex`, `Currency`, `MarketInfo`. These are updated each time-step by an ESG
  master before assets roll forward.
- **`assets/`** — asset classes (`Cash`, `Equity`, `BondFixed`, derivatives `EquityOption`) built on
  the `Asset` abstract base. Each asset implements `mv`/`fav`/`bsv` (market / fund-accounting /
  balance-sheet value) plus a `roll_forward()` / `close_dealing()` lifecycle. Distinguishes *profile*
  (to-be-purchased) vs *existing* assets. `create_asset(cls_or_name, build_pipeline, ...)` is the
  factory: plain classes construct directly, while `BondFixed`/`EquityOption` route through a
  **builder** (`vates/alm/assets/builders/`). A builder performs a named `build_pipeline` of
  calibration steps (e.g. `derive_coupon_rate`, `calculate_amort_rate`, `calibrate_market_spread`,
  `calculate_market_price`, `risk_neutralization`) then validates required prices/spreads before
  constructing the asset. This builder layer is the most recently active area of the codebase.
- **`funds/`** — fund accounting (`Fund`) with an asset allocator and fund calculator (fund value,
  target weight / `RebalancePolicyParams`, `FundSizeType`, `TargetWeight`).
- **`liabs/`** — liability base class (`Liab`) and `ExtProjLiab` for liabilities projected by an
  external engine.
- **`enums.py`** — shared Enums (`AssetRepBasis`, `AssetClassification`, ...).

### Financial math (`vates/finmath/`)

Standalone functions/classes, independent of the projection engine: interest-rate conversion
(`InterestRateConvertor`, `convert_interest_rates`, `solve_ytm`, `solve_z_spread`), rate
spreading/interpolation/extrapolation, Smith–Wilson extrapolation, Black–Scholes
(`BlackScholesCalculator`), and quantitative-finance helpers (`multivariate_standard_normal`,
`validate_corr_matrix`, `geometric_brownian_motion`, `search_efficient_frontier`).

### Solvency (`vates/solvency/`)

Currently a China Cross-II module (`cn_cross2/`: regulatory `params.py` and
`quant_risk_min_cap.py`). Compute quantitative-risk minimum capital from stochastic projections.

### `_experiment/` and `utils/`

- `vates/_experiment/` — experimental code parked out of the main public API, currently an
  `autograd.py` (there is a matching `docs/tutorials/tut_autograd.py`).
- `vates/utils/` — helpers: `data_classes`, `json_share_code_tool/` (read/write JSON), `risk_module`,
  `uncategorized`.

### Example workflows

`examples/models/` are the reference implementations. The asset models
(`em01_*`) use a bespoke `company_package/` module (in `examples/models/`) with classes like
`EsgMaster`, `AssetMaster`, and file-loading helpers; input tables live in `examples/inputs/<name>/`
and are resolved by name through `_file_names.csv` + `_file_read_config.json` configs. The stochastic
models (`em11_monte_carlo`, `em12_stoch_ec_mvl`) demonstrate `StochExecutor` and solvency
calculation. Generated outputs (`examples/results/`, `examples/intermediate/`) are git-ignored.

## Conventions

- Private/internal modules are prefixed with `_` (e.g. `_core`, `_utils.py`, `_bond_fixed_component.py`);
  the public API is curated in each package's `__init__.py` `__all__`.
- Model/engine configuration follows a defensive-typing style: methods take keyword-only args, warn
  (not raise) on double-binding/double-configuration, and reject invalid types via explicit checks.
- Dates are represented as monthly `pd.Period` objects; time steps are integers `t` from `0`.
- Results use a long, 4-column key of `(group, owner, variable, date_or_constant)`.
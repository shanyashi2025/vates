# About This Repository

Open-source Python packages and sample implementations for actuarial models.

## Project layout

```text
.
├── vates/          # The core package
├── docs/           # Project documentation
├── examples/       # Example implementations
│   ├── model/      # Source code of example models and local package(s)
│   ├── input/      # Example input data (tables) associated with the models
│   ├── json/       # Examples of model set up by json file
│   └── scripts/    # Examples of model set up by Python script
└── gui/            # A graphical user interface for end-users to run models
```

## Quick Start - Example Implementations

Work from the **repository root**.

### 1. Asset projection
   - Model: `.\examples\model\01_asset_proj.py`
   - Input: `.\examples\input\01_asset_proj\`
   - Run:
     - set up by Python script
        ```powershell
        python .\examples\scripts\script_01.py
        ```
     - set up by json file
        ```powershell
        python .\examples\model\em01_asset_proj.py .\examples\json\model_args_01.json
        ```
   - Standard output (`.proj.csv`): `.\examples\results\script_01_asset_proj\`, `.\examples\results\json_01_asset_proj\`

### 2. Fund projection
   - Model: `.\examples\model\02_fund_proj.py`
   - Input: `.\examples\input\02_fund_proj\`
   - Run:
      - set up by Python script
         ```powershell
         python .\examples\scripts\script_02.py
         ```
      - set up by json file
         ```powershell
         python .\examples\model\em02_fund_proj.py .\examples\json\model_args_02.json
         ```
   - Standard output (`.proj.csv`): `.\examples\results\script_02_fund_proj\`, `.\examples\results\json_02_fund_proj\`
   - Other output: `.\examples\.output\02\aging_assets\`

### 3. C-ROSS projection
   - Model: `.\examples\model\03_cross_proj.py`
   - Input: `.\examples\input\03_cross_proj\`, `.output\02\aging_assets\`
   - Run:
     - set up by Python script
        ```powershell
        python .\examples\scripts\script_03.py
        ```
     - set up by json file
        ```powershell
        python .\examples\model\em03_cross_proj.py .\examples\json\model_args_03.json
        ```
   - Standard output (`.proj.csv`): `.\examples\results\script_03_cross_proj\`, `.\examples\results\json_03_cross_proj\`

### 4. Monte Carlo simulation (stochastic)
   - Model: `.\examples\model\11_monte_carlo.py`
   - Input: `.\examples\input\11_monte_carlo\`
   - Run:
     - set up by Python script
        ```powershell
        python .\examples\scripts\script_11.py
        ```
     - set up by json file
        ```powershell
        python .\examples\model\em11_monte_carlo.py .\examples\json\model_args_11.json
        ```
   - Standard output (`.proj.csv`, `.stoch.csv`, `.stoch.statistic.csv`): `.\examples\results\script_11_monte_carlo\`, `.\examples\results\json_11_monte_carlo\`

### 5. Smith-Wilson curve
   - Model: `.\examples\model\21_smith_wilson.py`
   - Input: `.\examples\input\21_smith_wilson\`
   - Run:
     - set up by Python script
        ```powershell
        python .\examples\scripts\script_21.py
        ```
     - set up by json file
        ```powershell
        python .\examples\model\em21_smith_wilson.py .\examples\json\model_args_21.json
        ```
   - Standard output (`.proj.csv`): None
   - Other output: `.\examples\.output\21\`

## Tutorial

### 1. KeyedArray
   - Script: `.docs\tutorials\tut_KeyedArray.py`
   - Description: introduction to basics of the `KeyedArray` class, and benchmarks performance of extensively using `.loc` to access (lookup) single elements

### 2. AutogradCell
   - Script: `.docs\tutorials\tut_AutogradCell.py`
   - Description: illustrates how to employ the `AutogradCell` class to implement the backpropagation algorithm to perform sensitivity testing in a fast way


## References

### Install the `vates` package

Work from the **repository root**.

#### 1. Create a virtual environment (recommended)

   **PowerShell** (Windows):
   
   ```powershell
   cd path\to\your\project
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   
   > If activation is blocked by execution policy, run once in that window:
   > ```powershell
   > Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   > ```
   > Then run `Activate.ps1` again.
   
   You should see `(.venv)` in the prompt when the environment is active.

#### 2. Install `vates` in editable mode

   ```powershell
   pip install -e .
   ```
   
   This installs the distribution **`vates`** in editable mode; import the library as **`import vates`**.

### About

`vates` is an open-source Python package for actuarial models.

### Installation

```powershell
pip install vates
```

### Quick Start

#### 1. Projection Model

The `ProjModelEngine` class is the projection model engine.

- step 1: `ProjModelEngine`: initialize a model instance
- step 2: `model.configure_run`: configure a run
- step 3: `def {my_projection}`: define a function to perform projection calculations
- step 4: `model.bind_projection`: bind it to the model
- step 5: `model.run`: run the model

```python
import vates

model = vates.ProjModelEngine(model_name='my_model', description='example model')

model.configure_run(start_year=2025, end_year=2026)

@model.bind_projection
def my_projection(model: vates.ProjModelEngine):
    t = model.time
    p = model.period
    if t == 0:
        print(f"Projection started, START_DATE = {model.START_DATE}")
    else:
        print(f"time: {t}, period: {p}")
    if t == model.MAX_T:
        print(f"Projection ended, END_DATE = {model.END_DATE}")

model.run()
```

Following information will display in the terminal:
```text
Projection started, START_DATE = 2025-12
time: 1, period: 2026-01
time: 2, period: 2026-02
...
time: 12, period: 2026-12
Projection ended, END_DATE = 2026-12
```

The runlog can be found in `.\results\{my_model}.runlog.json` file.


#### 2. Add Variables for Output

You can set up instances of `TDimVariable` and/or `ConstVariable`, the projected results will be automatically written 
to the `.\results\{my_model}.proj.csv` file.

```python
import vates

model = vates.ProjModelEngine(model_name='my_model', description='example model')

model.configure_run(start_year=2025, end_year=2026)

model.const_var = vates.ConstVariable('const_var', model_engine=model)
model.tdim_var = vates.TDimVariable('tdim_var', model_engine=model)

@model.bind_projection
def my_projection(model: vates.ProjModelEngine):
    t = model.time
    p = model.period
    if t == 0:
        model.const_var[...] = model.START_YEAR * 100 + model.START_MONTH
    else:
        model.tdim_var[t] = p.year * 100 + p.month

model.run()
```

You can use function `proj_result` to read the result from a `.proj.csv` file.

```python
from vates import proj_result

# Get the entire results 
df = proj_result(
    results_directory=r".\examples\results\em01_asset_proj_inner",
    model_name="asset_proj",
)
print(df)

# Get value of a specific cell (group + owner + variable + date)
val = proj_result(
    results_directory=r".\examples\results\em01_asset_proj_inner",
    model_name="asset_proj",
    group="equity_option",
    owner="option_1",
    variable="price",
    date="202502",
)
print(f"{val:.4f}")  # 5.2973
```

#### 3. Stochastic Model

The `StochExecutor` class is for stochastic model, multiprocessing is supported.

Similarly,
- step 1: `StochExecutor`: initialize a stochastic model instance
- step 2: `model.configure_run`: configure a run
- step 3: `def {my_projection}`: define a function to perform projection calculations
- step 4: `model.bind_projection`: bind it to the model
- step 5: `model.run`: run the model

```python
import vates

def my_projection(model: vates.ProjModelEngine):
    t = model.time
    if t == 0:
        print(f"simulation: {model.SIMULATION}")

def stoch_model():
    model = vates.StochExecutor(model_name='my_model', description='example model')
    model.configure_run(start_year=2025, end_year=2026, simulations="1-10", max_workers=2)
    model.bind_projection(my_projection)
    model.run()

if __name__ == '__main__':
    stoch_model()
```

Following information will display in the terminal:
```text
simulation: 1
simulation: 2
...
simulation: 10
```

#### 4. ALM Package

The `vates.alm` is the subpackage for asset-liability model.

There are many classes provided for modelling convenience:

- assets: `Cash`, `Equity`, `BondFixed`, `EquityOption`
- econs: `YieldCurve`, `CreditBand`, `EquityIndex`
- funds: `Fund`
- liabs: `ExtProjLiab`


### See

GitHub repository: https://github.com/shanyashi2025/vates

Documentation and tutorials: https://github.com/shanyashi2025/vates/tree/main/docs

Example implementations: https://github.com/shanyashi2025/vates/tree/main/examples
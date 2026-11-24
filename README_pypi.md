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
- step 3: `def < my_projection >`: define a function to perform projection calculations
- step 4: `model.bind_projection`: bind it to the model
- step 5: `model.run`: run the model

```python
from vates import ProjModelEngine

model = ProjModelEngine(slug='my_model', description='example model')
model.configure_run(start_year=2025, end_year=2026)

@model.bind_projection
def my_projection(m: ProjModelEngine):
    t = m.time
    p = m.period
    if t == 0:
        print(f"Projection started, START_DATE={m.START_DATE}")
    else:
        print(f"time: {t}, period: {p}")
    if t == m.MAX_T:
        print(f"Projection ended, END_DATE={m.END_DATE}")

model.run()
```

Following information will display in the terminal:
```text
Projection started, START_DATE=2025-12
time: 1, period: 2026-01
time: 2, period: 2026-02
...
time: 12, period: 2026-12
Projection ended, END_DATE=2026-12
```

The runlog can be found in `my_model.runlog.json` file.


#### 2. Add Variables for Model Output

Use `ScalarVariable` and `TDimVariable` for values without labeled dimensions.
Use `ConstantArrayVariable` and `TDimArrayVariable` when a value has labeled
dimensions. The `@` operator attaches each variable to the model for output,
which can be read by `model.proj_result`. Time-dimensioned variables are indexed
by integer (`model.time`).

```python
from vates import ProjModelEngine, ScalarVariable, TDimVariable

model = ProjModelEngine(slug='my_model', description='example model')
model.configure_run(start_year=2025, end_year=2026, results_directory=r"results")

const_var = ScalarVariable('const_var', owner="owner", group="group") @ model
tdim_var = TDimVariable('tdim_var', owner="owner", group="group", max_t=model.MAX_T) @ model

@model.bind_projection
def my_projection(m: ProjModelEngine):
    t = m.time
    p = m.period
    if t == 0:
        const_var.result = m.START_YEAR * 100 + m.START_MONTH
    tdim_var[t] = p.year * 100 + p.month + t / 100

model.run()

# 1. read the entire results 
df = model.proj_result()
print(df)

# 2. read value of a specific cell (group + owner + variable + date)
val = model.proj_result(
    group="group",
    owner="owner",
    variable="tdim_var",
    date="202602",
)
print(f"{val:.4f}")  # 202602.0200
```

The projected results `results/my_model.proj.csv` are saved on disk, which can be read by 
the function `proj_result`.

```python
from vates import proj_result

# 1. Get the entire results 
df = proj_result(
    results_directory=r"results",
    slug="my_model",
)
print(df)

# 2. Get value of a specific cell (group + owner + variable + date)
val = proj_result(
    results_directory=r"results",
    slug="my_model",
    group="group",
    owner="owner",
    variable="tdim_var",
    date="202602",
)
print(f"{val:.4f}")  # 202602.0200
```

#### 3. Time Synchronization

If a class needs the model's `time` and/or `period`, use `@time_synchronized` to decorate it. The decorator
wires the instance to the model so its cached values are automatically kept in sync.

```python
from vates import ProjModelEngine, time_synchronized

@time_synchronized("m")
class MyClass:

    def __init__(self, m, name):
        self.name = name

    def check_time(self):
        print(f"{self.name}: time={self.time} and period={self.period} are up to date.")

model = ProjModelEngine(slug='my_model', description='example model')
model.configure_run(start_year=2025, end_year=2026)

obj = MyClass(model, "my_obj")

@model.bind_projection
def my_projection(m: ProjModelEngine):
    obj.check_time()

model.run()
```

Following will be printed:
```text
my_obj: time=0 and period=2025-12 are up to date.
my_obj: time=1 and period=2026-01 are up to date.
...
my_obj: time=12 and period=2026-12 are up to date.
```

**Put them together: a simple monthly account-value projection**

```python
from vates import ProjModelEngine, TDimVariable, time_synchronized

@time_synchronized("m")
class Portfolio:

    time: int

    def __init__(self, m: ProjModelEngine, name: str, initial_value: float) -> None:
        self.name = name
        self.account_value = TDimVariable("account_value", owner=name, group="demo", max_t=m.MAX_T) @ m
        t = self.time
        self.account_value[t] = initial_value

    def roll_forward(self, monthly_rate: float, monthly_contribution: float) -> None:
        t = self.time
        self.account_value[t] = self.account_value[t - 1] * (1 + monthly_rate) + monthly_contribution


def projection(m: ProjModelEngine, portfolio_name: str, initial_account_value: float, monthly_contribution: float,
               annual_effective_rate: float) -> None:
    if m.time == 0:
        m.portfolio = Portfolio(m, name=portfolio_name, initial_value=initial_account_value)
    else:
        monthly_rate = (1 + annual_effective_rate) ** (1 / 12) - 1
        m.portfolio.roll_forward(monthly_rate=monthly_rate, monthly_contribution=monthly_contribution)


def main() -> None:
    model = ProjModelEngine(slug="first_projection", description="A simple monthly account-value projection")
    model.configure_run(start_year=2025, start_month=12, end_year=2026, end_month=12)
    model.bind_projection(projection)
    projection_kwargs = dict(
        portfolio_name = "portfolio",
        initial_account_value = 1_000.0,
        annual_effective_rate = 0.03,
        monthly_contribution = 100.0,
    )
    model.run(projection_kwargs=projection_kwargs)

    if not model.runlog["execution"]["success"]:
        raise RuntimeError("The projection run failed; see the run log for details.")

    ending_value = model.proj_result(group="demo", owner="portfolio", variable="account_value", date="202612")
    print(f"Ending account value at 2026-12: {ending_value:.2f}")  # 2246.41

    
if __name__ == "__main__":
    main()
```

#### 4. Stochastic Model

The `StochExecutor` class is for stochastic model, multiprocessing is supported.

Similarly,
- step 1: `StochExecutor`: initialize a stochastic model instance
- step 2: `model.configure_run`: configure a run
- step 3: `def < my_projection >`: define a function to perform projection calculations (the bound function is executed 
  in worker processes, so it must be top-level/picklable)
- step 4: `model.bind_projection`: bind it to the model
- step 5: `model.run`: run the model

```python
from vates import ProjModelEngine, StochExecutor

def my_projection(m: ProjModelEngine):
    t = m.time
    if t == 0:
        print(f"simulation: {m.SIMULATION}")

def stoch_model():
    model = StochExecutor(slug='my_stoch_model', description='example stochastic model')
    model.configure_run(start_year=2025, end_year=2026, simulations=list(range(1, 11)), max_workers=2)
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

#### 5. ALM Package

The `vates.alm` is the subpackage for asset-liability model.

There are building blocks for model developments:

- assets: `Cash`, `Equity`, `BondFixed`, `EquityOption`
- econs: `EquityIndex`, `YieldCurve`, `Spread`, `DefaultRisk`
- funds: `Fund`
- liabs: `ExternalLiab`


### See

GitHub repository: https://github.com/shanyashi2025/vates

Example workspace: https://github.com/shanyashi2025/vates/tree/main/example_ws
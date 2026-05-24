### About

`vates` is an open-source Python package for actuarial models.

### Installation

```powershell
pip install vates
```

### Synopsis

#### 1. `ProjModelEngine`

The `ProjModelEngine` class is the projection model engine.

- set up a model instance:

```python
import vates as vt
model_simple_params = vt.ProjModelEngine(model_name='your_model_name', start_year=2025, start_month=12)

model_full_params = vt.ProjModelEngine(
    model_name='your_model_name', 
    start_year=2025, 
    start_month=12,
    end_year=2026,
    model_desc='description of your model',
    scenario='scenario_to_run',
    simulation=1,
    workspace_directory='path/to/workspace',
    input_directories=['path/to/input/folder1', 'path/to/input/folder2'],
    results_directory='path/to/results/folder'    
)
```

- create your model class: inherit from `ProjModelEngine` and implement following concrete methods
    - `time_zero_calculations()`, 
    - `in_time_calculations()`, and 
    - `post_time_calculations()` 

- set up a model instance and call `.run()` to perform the projection

```python
import vates as vt

class YourModel(vt.ProjModelEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def time_zero_calculations(self):
        print(f"model name: {self.MODEL_NAME} | scenario: {self.SCENARIO} | simulation: {self.SIMULATION} | start date: {self.START_DATE} | end date: {self.END_DATE}")
    
    def in_time_calculations(self):
        print(f"time: {self.time} | period: {self.period}")
    
    def post_time_calculations(self):
        print(f"end of projection")

your_model_instance = YourModel(model_name='your_model_name', start_year=2025, start_month=12, end_year=2026)

your_model_instance.run()
```

#### 2. `TDepVariable` and `ConstVariable`

You can set up instances of `TDepVariable` and/or `ConstVariable`, the projected results will be automatically output to the `your_model_name.proj.csv` file.

```python
import vates as vt

class YourModel(vt.ProjModelEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_var1 = vt.ConstVariable(self, 'const_var1_name', 'owner1_name', 'group1_name')
        self.tdep_var1 = vt.TDepVariable(self, 'tdep_var1_name', 'owner1_name', 'group2_name')
        self.tdep_var2 = vt.TDepVariable(self, 'tdep_var2_name', 'owner2_name', 'group1_name')
        
    def time_zero_calculations(self):
        self.const_var1[0] = self.START_YEAR * 100 + self.START_MONTH
        
    def in_time_calculations(self):
        t, p = self.time, self.period
        self.tdep_var1[t] = p.year * 100 + p.month
        self.tdep_var2[t] = (t / 2) ** 2
        
    def post_time_calculations(self):
        pass

your_model_instance = YourModel(model_name='your_model_name', start_year=2025, start_month=12, end_year=2026)

your_model_instance.run()
```

#### 3. `StochExecutor`

The `StochExecutor` class is the executor for stochastic model, multiprocessing is supported.

- create your stoch executor class: inherit from `StochExecutor` and implement following concrete methods
  - `pre_stoch_calculations()`, and 
  - `post_stoch_calculations()`

```python
import vates as vt

class YourStochExecutor(vt.StochExecutor):
    def pre_stoch_calculations(self):
        print(f"model name: {self.MODEL_NAME} | scenario: {self.SCENARIO} | simulations: {self.SIMULATIONS}")

    def post_stoch_calculations(self):
        print(f'post stochastic calculations ...')

class YourModel(vt.ProjModelEngine):
    def time_zero_calculations(self):
        print(f"model name: {self.MODEL_NAME} | scenario: {self.SCENARIO} | simulation: {self.SIMULATION} | start date: {self.START_DATE} | end date: {self.END_DATE}")
    def in_time_calculations(self): pass
    def post_time_calculations(self): pass

if __name__ == '__main__': # must create the '__main__' block for multiprocessing
    your_stoch_model_instance = YourStochExecutor(
        model_cls=YourModel,
        model_name='your_stoch_model_name',
        start_year=2025,
        start_month=12,
        end_year=2026,
        simulations='1-3, 5',
        max_workers=4,
    )
    
    your_stoch_model_instance.run()
```

#### 4. `KeyedArray`

The `KeyedArray` class can be used as the alternative to `DataFrame` if `.at` or `.loc` is massively called for scalar access (lookup).

The `kr_from_df()` function is to create `KeyedArray` object from `DataFrame`.

```python
import pandas as pd
import random

random.seed(42)
import vates as vt

# --- set up the DataFrame ---
n_idx1, n_idx2, n_cols = 5, 3, 10

index1, index2 = [], []
for i in range(n_idx1):
  for j in range(n_idx2):
    index1.append(f"a{i}")  # a1, a2, ..
    index2.append(f"b{j}")  # b1, b2, ..

multi_index = pd.MultiIndex.from_arrays([index1, index2], names=['index1', 'index2'])
columns = [f"col{i}" for i in range(n_cols)]  # col1, col2, ..

data = [[random.uniform(1, 100) for i in range(n_cols)] for j in range(n_idx1 * n_idx2)]

df = pd.DataFrame(data, index=multi_index, columns=columns)

# --- KeyedArray ---
# 1. create KeyedArray object from DataFrame
kr = vt.kr_from_df(df)

# 2. get attributes `ndim`, `size`, `shape`, `dtype` just like numpy ndarray
print(f">>> {kr.ndim=}, {kr.size=}, {kr.shape=}, {kr.dtype=}")

# 3. use `[]` for scalar access by its integer-position index like numpy ndarray
print(f">>> {kr[1, 2]=}, {kr[11, 8]=}")

# 4. use `.at[]` for scalar access by its lable-based index like pandas DataFrame
print(f">>> {kr.at[('a0', 'b1'), 'col2']=}, {kr.at[('a3', 'b2'), 'col8']=}")
# - specially for 2D array, where the first index/key is a tuple, parentheses can be omitted
print(f">>> {kr.at['a0', 'b1', 'col2']=}, {kr.at['a3', 'b2', 'col8']=}")
# - display `df.at` for reference
print(f">>> {df.at[('a0', 'b1'), 'col2']=}, {df.at[('a3', 'b2'), 'col8']=}")

# 5. use `.get()` for scalar access by its lable-based index
# - positional arguments (*args)
print(f">>> {kr.get(('a0', 'b1'), 'col2')=}, {kr.get(('a3', 'b2'), 'col8')=}")
# - if the key is not found, it returns None or a specified default value
print(f">>> {kr.get(('a999', 'b1'), 'col2')=}, {kr.get(('a999', 'b1'), 'col2', default=-9999)=}")
# - keyword arguments (**kwargs)"
print(f">>> {kr.get(row_index=('a0', 'b1'), col_name='col2')=}")
print(f">>> {kr.get(col_name='col2', row_index=('a0', 'b1'))=}")
```

#### 5. `AutogradCell`

The `AutogradCell` class automates the backpropagation process to compute the gradient (partial derivative)

- `.value` holds the scalar value
- `.grad` holds the gradient (partial derivative)
- `.backward()` traverses the graph in reverse, applys the chain rule to compute the gradients 

For actuarial practice, you can employ `AutogradCell` to implement sensitivity test in a fast way.

```python
import vates as vt

def simple_cashflow_model(mort_rates, lapse_rates, expense_fixed, discount_rates):
    no_pols_if = 1
    bel, sumcf = 0.0, 0.0
    discount_factors = [1] + [0] * len(discount_rates)
    for t in range(2):
        prem_income = no_pols_if * 100
        expn_outgo = no_pols_if * expense_fixed
        no_deaths = no_pols_if * mort_rates[t]
        no_lapses = no_pols_if * (1 - mort_rates[t]) * lapse_rates[t]
        death_outgo = no_deaths * 10000
        surr_outgo = no_lapses * 100
        no_pols_if -= no_deaths + no_lapses
        sumcf += -prem_income + expn_outgo + death_outgo + surr_outgo
        discount_factors[t + 1] = discount_factors[t] / (1 + discount_rates[t])
        bel += (-prem_income + expn_outgo) * discount_factors[t] + (death_outgo + surr_outgo) * discount_factors[t + 1]
    return {"bel": bel, "sumcf": sumcf}

mort_rates = [0.001, 0.002]
lapse_rates = [0.10, 0.05]
expense_fixed = 50
discount_rates = [0.025, 0.035]

mort_rate_mul_sens, mort_rate_add_sens = vt.AutogradCell(1), vt.AutogradCell(0)
lapse_rate_mul_sens, lapse_rate_add_sens = vt.AutogradCell(1), vt.AutogradCell(0)
expense_fixed_mul_sens, expense_fixed_add_sens = vt.AutogradCell(1), vt.AutogradCell(0)
discount_rate_add_sens = vt.AutogradCell(0)

mort_rates = [x * mort_rate_mul_sens + mort_rate_add_sens for x in mort_rates]
lapse_rates = [x * lapse_rate_mul_sens + lapse_rate_add_sens for x in lapse_rates]
expense_fixed = expense_fixed * expense_fixed_mul_sens + expense_fixed_add_sens
discount_rates = [x + discount_rate_add_sens for x in discount_rates]

result = simple_cashflow_model(mort_rates, lapse_rates, expense_fixed, discount_rates)
print(f'{result["bel"].value=:.4f}')    # prints -53.1769, the outcome of this forward pass
print(f'{result["sumcf"].value=:.4f}')  # prints -52.4965, the outcome of this forward pass

print("\nGet selected variables sensitivity w.r.t. each assumption in one go:")
result["bel"].backward("bel")
result["sumcf"].backward("sumcf")
for i, name in enumerate(("bel", "sumcf"), 1):
    print(f"\n{i}. {name}'s sensitivity w.r.t. each assumption")
    print('a. mortality rates')
    print(f'   - all year percent : {mort_rate_mul_sens.grad.get(name, 0):.4f}')
    print(f'   - all year absolute: {mort_rate_add_sens.grad.get(name, 0):.4f}')
    print(f'   - by year percent  : [{mort_rates[0].grad.get(name, 0) * mort_rates[0].value:.4f}, {mort_rates[1].grad.get(name, 0) * mort_rates[1].value:.4f}]')
    print(f'   - by year absolute : [{mort_rates[0].grad.get(name, 0):.4f}, {mort_rates[1].grad.get(name, 0):.4f}]')
    print('b. lapse rates')
    print(f'   - all year percent : {lapse_rate_mul_sens.grad.get(name, 0):.4f}')
    print(f'   - all year absolute: {lapse_rate_add_sens.grad.get(name, 0):.4f}')
    print(f'   - by year percent  : [{lapse_rates[0].grad.get(name, 0) * lapse_rates[0].value:.4f}, {lapse_rates[1].grad.get(name, 0) * lapse_rates[1].value:.4f}]')
    print(f'   - by year absolute : [{lapse_rates[0].grad.get(name, 0):.4f}, {lapse_rates[1].grad.get(name, 0):.4f}]')
    print('c. expenses')
    print(f'   - percent : {expense_fixed_mul_sens.grad.get(name, 0):.4f}')
    print(f'   - absolute: {expense_fixed_add_sens.grad.get(name, 0):.4f}')
    print('d. discount rates')
    print(f'   - all year absolute: {discount_rate_add_sens.grad.get(name, 0):.4f}')
    print(f'   - by year absolute : [{discount_rates[0].grad.get(name, 0):.4f}, {discount_rates[1].grad.get(name, 0):.4f}]')
```

#### 6. Asset-Liability Model (ALM)

The `vates.alm` is the subpackage for asset-liability model.

It includes but are not limited to the following classes:

- assets: `Asset`, `Cash`, `Equity`, `BondFixed`, `EquityOption`
- econs: `YieldCurve`, `CreditBand`, `EquityIndex`
- funds: `Fund`, `AssetAllocator`
- liabs: `Liab`, `ExtProjLiab`


### See

GitHub repository: https://github.com/shanyashi2025/vates

Documentation and tutorials: https://github.com/shanyashi2025/vates/tree/main/docs

Example implementations: https://github.com/shanyashi2025/vates/tree/main/examples
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

With respect to actuarial practice, you can employ `AutogradCell` to implement sensitivity testing in a fast way.

```python
import vates as vt

a = vt.AutogradCell(-4.0)
b = vt.AutogradCell(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (a + b).apply_floor(0)
d += 3 * d - (a - b).apply_cap(0)
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.value:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

#### 6. Asset-Liability Model (ALM)

The `vates.alm` is the subpackage for asset-liability model.

It includes but are not limited to the following classes:

- assets: `Asset`, `Cash`, `Equity`, `BondFixed`, `EquityOption`, `BondFixedBuilder`, `EquityOptionBuilder`
- econs: `YieldCurve`, `CreditBand`, `EquityIndex`
- funds: `Fund`, `AssetAllocator`
- liabs: `Liab`, `ExtProjLiab`


### See

GitHub repository: https://github.com/shanyashi2025/vates

Documentation and tutorials: https://github.com/shanyashi2025/vates/tree/main/docs

Example implementations: https://github.com/shanyashi2025/vates/tree/main/examples
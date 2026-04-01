import pandas as pd
import numpy as np
import random
import time
from vates.utils import df_to_karray

def create_test_df(n_idx1: int, n_idx2: int, n_cols: int) -> pd.DataFrame:
    index1 = np.array([f"a{i}" for i in range(n_idx1)]).repeat(n_idx2)
    index2 = np.tile(np.array([f"b{i}" for i in range(n_idx2)]), n_idx1)
    index_array = [index1, index2]
    multi_index = pd.MultiIndex.from_arrays(index_array, names=['index1', 'index2'])
    columns = np.array([f"col{i}" for i in range(n_cols)])
    data = np.random.rand(n_idx1 * n_idx2, n_cols)
    return pd.DataFrame(data, index=multi_index, columns=columns)

def random_overhead(n_lookups: int, n_idx1: int, n_idx2: int, n_cols: int):
    s = time.time()
    for _ in range(n_lookups):
        idx1 = f"a{random.randint(0, n_idx1 - 1)}"
        idx2 = f"b{random.randint(0, n_idx2 - 1)}"
        col = f"col{random.randint(0, n_cols - 1)}"
    time_taken = time.time() - s
    print(f"`random_overhead` time taken: {time_taken:.2f} seconds / {n_lookups:,} times")
    return time_taken

def df_loc(df: pd.DataFrame, n_lookups: int, n_idx1: int, n_idx2: int, n_cols: int):
    s = time.time()
    for _ in range(n_lookups):
        idx1 = f"a{random.randint(0, n_idx1 - 1)}"
        idx2 = f"b{random.randint(0, n_idx2 - 1)}"
        col = f"col{random.randint(0, n_cols - 1)}"
        data_lookup = df.loc[(idx1, idx2), col]
    time_taken = time.time() - s
    print(f"`df_loc` time taken: {time_taken:.2f} seconds / {n_lookups:,} times")
    return time_taken

def karr_loc(df: pd.DataFrame, n_lookups: int, n_idx1: int, n_idx2: int, n_cols: int):
    s = time.time()
    karr = df_to_karray(df)
    for _ in range(n_lookups):
        idx1 = f"a{random.randint(0, n_idx1 - 1)}"
        idx2 = f"b{random.randint(0, n_idx2 - 1)}"
        col = f"col{random.randint(0, n_cols - 1)}"
        data_lookup = karr.loc[(idx1, idx2), col]
    time_taken = time.time() - s
    print(f"`karr_loc` time taken: {time_taken:.2f} seconds / {n_lookups:,} times")
    return time_taken


def main():
    n_lookups = 1_000_000
    n_idx1, n_idx2, n_cols = 200, 20, 200
    df = create_test_df(n_idx1, n_idx2, n_cols)
    print(f"shape: {(n_idx1, n_idx2, n_cols)}, size: {n_idx1 * n_idx2 * n_cols:,} ")
    t0 = random_overhead(n_lookups, n_idx1, n_idx2, n_cols)
    t1 = df_loc(df, n_lookups, n_idx1, n_idx2, n_cols)
    t2 = karr_loc(df, n_lookups, n_idx1, n_idx2, n_cols)
    print(f"`karr_loc` / `df_loc` = {(t2 - t0) / (t1 - t0):.2%}")

    """
    # Benchmark
    
    ## CPU: Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz, RAM: 16GB
    
    ## parameters:
    - n_idx1, n_idx2, n_cols = 200, 20, 200
    - size = 800,000
    
    ## summary of performance benchmark results (metric = `karr_loc` time taken / `df_loc` time taken)
    
                pandas version
    n_lookups   2.3.3   3.0.1
        1_000: 30.87%, 10.25%
       10_000:  7.17%,  1.34%
      100_000:  3.22%,  0.50%
    1_000_000:  3.20%,  0.47%
    
    """


if __name__ == '__main__':
    main()

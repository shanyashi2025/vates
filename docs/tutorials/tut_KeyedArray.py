import pandas as pd
import numpy as np
import time
import random
random.seed(42)

from vates import KeyedArray

def create_test_df(n_idx1: int, n_idx2: int, n_cols: int) -> pd.DataFrame:
    np.random.seed(42)
    index1 = np.array([f"a{i}" for i in range(n_idx1)]).repeat(n_idx2)
    index2 = np.tile(np.array([f"b{i}" for i in range(n_idx2)]), n_idx1)
    index_array = [index1, index2]
    multi_index = pd.MultiIndex.from_arrays(index_array, names=['index1', 'index2'])
    columns = np.array([f"col{i}" for i in range(n_cols)])
    data = np.random.rand(n_idx1 * n_idx2, n_cols)
    return pd.DataFrame(data, index=multi_index, columns=columns)

def rand_lookup_keys(n_lookups: int, n_idx1: int, n_idx2: int, n_cols: int):
    return [((f"a{random.randint(0, n_idx1 - 1)}", f"b{random.randint(0, n_idx2 - 1)}"), f"col{random.randint(0, n_cols - 1)}")
            for _ in range(n_lookups)]

def df_loc(df: pd.DataFrame, lookup_keys: list):
    s = time.time()
    for key in lookup_keys:
        _ = df.loc[key]
    time_taken = time.time() - s
    return time_taken

def df_at(df: pd.DataFrame, lookup_keys: list):
    s = time.time()
    for key in lookup_keys:
        _ = df.at[key]
    time_taken = time.time() - s
    return time_taken

def kr_at(kr: KeyedArray, lookup_keys: list):
    s = time.time()
    for key in lookup_keys:
        _ = kr.at[key]
    time_taken = time.time() - s
    return time_taken

def main():
    print(f"This python script gives the introduction to basics of the `KeyedArray` class, and benchmarks performance of"
          f" scalar access.")

    n_idx1, n_idx2, n_cols = 200, 20, 200
    df = create_test_df(n_idx1, n_idx2, n_cols)

    print(f"\n--- DataFrame created, filled with random numbers ---")
    print(f'>>> df')
    print(df)

    # Basics of KeyedArray
    print(f"\n--- Basics of KeyedArray ---")
    print(f"1. crate the `KeyedArray` object from DataFrame")
    print(f">>> KeyedArray.from_df(df)")
    kr = KeyedArray.from_df(df)
    print(f">>> {type(kr)=}")

    print(f"\n2. get values of the array")
    print(f">>> {kr.values=}")

    print(f"\n3. get attributes `ndim`, `size`, `shape`, `dtype` just like numpy ndarray")
    print(f">>> {kr.ndim=}, {kr.size=}, {kr.shape=}, {kr.dtype=}")

    print(f"\n4. use `[]` for scalar access by its integer-position index like numpy ndarray")
    print(f">>> {kr[1, 2]=}, {kr[305, 50]=}")

    print(f"\n5. use `.at[]` for scalar access by its lable-based index like pandas DataFrame")
    print(f">>> {kr.at[('a0', 'b1'), 'col2']=}, {kr.at[('a15', 'b5'), 'col50']=}")
    print(f"-   specially for 2D array, where the first index/key is a tuple, parentheses can be omitted:")
    print(f">>> {kr.at['a0', 'b1', 'col2']=}, {kr.at['a15', 'b5', 'col50']=}")
    print(f"-   display `df.at` for reference:")
    print(f">>> {df.at[('a0', 'b1'), 'col2']=}, {df.at[('a15', 'b5'), 'col50']=}")

    print(f"\n6. use `.get()` for scalar access by its lable-based index")
    print(f"6.1. positional arguments (*args)")
    print(f">>> {kr.get(('a0', 'b1'), 'col2')=}, {kr.get(('a15', 'b5'), 'col50')=}")
    print(f"-   this is similar to `.at`, the difference is that it returns None or a specified default value if the key is not found:")
    print(f">>> {kr.get(('a999', 'b1'), 'col2')=}, {kr.get(('a999', 'b1'), 'col2', default=-9999)=}")
    print(f"6.2. keyword arguments (**kwargs)")
    print(f">>> {kr.get(row_index=('a0', 'b1'), col_name='col2')=}")
    print(f">>> {kr.get(col_name='col2', row_index=('a0', 'b1'))=} # sequance does not matter")
    print(f"-   notes with respect to `row_index` and `col_name`:")
    print(f"(1) for SingleIndex Dataframe:")
    print(f"    default dimension names are 'row_index' and 'col_name' created by `KeyedArray.from_df(df)`")
    print(f"    you can specify dimension names: `KeyedArray.from_df(df, multi_index_name=< your_row_index_name >, col_index_name=< your_col_index_name >)`")
    print(f"(2) for MultiIndex Dataframe without unpacking MultiIndex - like this case:")
    print(f"    default dimension names are 'row_index' and 'col_name' created by `KeyedArray.from_df(df)`")
    print(f"    you can specify dimension names: `KeyedArray.from_df(df, multi_index_name=< your_row_index_name >, col_index_name=< your_col_index_name >)`")
    print(f"(3) for MultiIndex Dataframe with unpacking MultiIndex:")
    print(f"    default dimension names are original df index names and 'col_name' created by `KeyedArray.from_df(df, unpack_multi_index=True)`")
    print(f"    you can specify col index name: `KeyedArray.from_df(df, unpack_multi_index=True, col_index_name=< your_col_index_name >)`")

    print(f"\n7. use `.key_to_pos()` to map the key to integer-position index of the dimension")
    print(f"-   dimension can be specified by either int or str")
    print(f">>> {kr.key_to_pos(0, ('a15', 'b5'))=}, {kr.key_to_pos(1, 'col50')=}")
    print(f">>> {kr.key_to_pos('row_index', ('a15', 'b5'))=}, {kr.key_to_pos('col_name', 'col50')=}")
    print(f"-   let's view lable-based indexing (pandas style) vs integer-position indexing (numpy style):")
    print(f">>> {kr.at[('a15', 'b5'), 'col50']=}")
    i, j = kr.key_to_pos(0, ('a15', 'b5')), kr.key_to_pos(1, 'col50')
    print(f">>> kr[{i}, {j}]={kr[i, j]}")

    print(f"\n--- scalar access performance benchmark: `KeyedArray` vs `DataFrame` ---")
    print(f">>> display benchmarking results (unit: seconds, %):")
    print(f"{'n_lookups':^11}| {'kr.at':^6} | {'df.at':^6} | {'kr.at / df.at %':^16} | {'df.loc':^6} | {'kr.at / df.loc %':^16} ")
    for n_lookups in (10_000, 100_000, 1_000_000):
        lookup_keys = rand_lookup_keys(n_lookups, n_idx1, n_idx2, n_cols)
        t1, t2, t3 = kr_at(kr, lookup_keys), df_at(df, lookup_keys), df_loc(df, lookup_keys)
        print(f"{n_lookups:>10,} | {t1:>6.2f} | {t2:>6.2f} | {t1 / t2:>16.2%} | {t3:>6.2f} | {t1 / t3:>16.2%} ")
    print(f"- testing data size: size={kr.size} | shape={kr.shape} | ndim={kr.ndim}")
    print(f"- pandas version: {pd.__version__} (df.at and df.loc are much slower in version 3.0.x compared with version 2.3.x)")

if __name__ == '__main__':
    main()

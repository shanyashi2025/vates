import pandas as pd
import random
import time
from vates.utils import df_to_karray


def random_overhead(n: int):
    s = time.time()
    for _ in range(n):
        liab_id = f"id{random.randint(1, 200)}"
        var_name = f"var{random.randint(1, 12)}"
        col_name = f"col{random.randint(1, 1200)}"
    time_taken = time.time() - s
    print(f"`random_overhead` time taken: {time_taken:.2f} seconds / {n} times")
    return time_taken

def df_loc(n: int):
    df = pd.read_csv("karray_test_performance.csv", index_col=['liab_id', 'var_name'])

    s = time.time()
    for _ in range(n):
        liab_id = f"id{random.randint(1, 200)}"
        var_name = f"var{random.randint(1, 12)}"
        col_name = f"col{random.randint(1, 1200)}"
        data_read = df.loc[(liab_id, var_name), col_name]
    time_taken = time.time() - s
    print(f"`df_loc` time taken: {time_taken:.2f} seconds / {n} times")
    return time_taken

def karr_loc(n: int):
    df = pd.read_csv("karray_test_performance.csv", index_col=['liab_id', 'var_name'])

    s = time.time()
    karr = df_to_karray(df)
    for _ in range(n):
        liab_id = f"id{random.randint(1, 200)}"
        var_name = f"var{random.randint(1, 12)}"
        col_name = f"col{random.randint(1, 1200)}"
        data_read = karr.loc[(liab_id, var_name), col_name]
    time_taken = time.time() - s
    print(f"`karr_loc` time taken: {time_taken:.2f} seconds / {n} times")
    return time_taken


def test_performance():
    n = 100_000
    t0 = random_overhead(n)
    t1 = df_loc(n)
    t2 = karr_loc(n)
    print(f"`karr_loc` / `df_loc` = {(t2 - t0) / (t1 - t0):.2%}")

    """
    # pandas == 2.3.3
    # n =     1_000: 94.63%, 86.47%, 87.78%
    # n =    10_000: 14.05%, 15.14%, 14.84%
    # n =   100_000:  5.30%,  4.09%,  4.33%
    # n = 1_000_000:  3.26%,  3.06%,  2.82%

    # pandas == 3.0.1
    # n =     1_000: 32.35%, 33.60%, 27.88%
    # n =    10_000:  3.59%,  3.26%,  3.36%
    # n =   100_000:  1.04%,  0.93%,  0.77%
    # n = 1_000_000:  0.63%,  0.54%,  0.59%
    """

def main():
    test_performance()


if __name__ == '__main__':
    main()

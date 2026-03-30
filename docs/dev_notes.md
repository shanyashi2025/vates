## Developer Notes

### [2026-MAR-25] About pandas 3.0.1 performance deterioration
pandas 2.3.3 -> 3.0.1, significant performance deterioration observed.

- [2026-MAR-26] Develop the new class `IndexedNumArray` (in `vates.utils`) to replace dataframe for numeric arrays.

Below exhibites testing results on `em12_stoch_ec_mvl.py` (simulations = 10, max_workers = 3):

| pandas | time (seconds) | remark              |
|--------|----------------|---------------------|
| 2.3.0  | 21.9           | 2026-MAR-25         |
| 2.3.3  | 22.1           | 2026-MAR-25         |
| 3.0.1  | 33.0           | 2026-MAR-25         |
| 3.0.1  | 27.9           | 2026-MAR-25 updated |
| 2.3.3  | 17.0           | 2026-MAR-26 updated |
| 3.0.1  | 21.9           | 2026-MAR-26 updated |


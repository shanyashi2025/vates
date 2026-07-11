import warnings
from typing import Any

import numpy as np
import pandas as pd


class KeyedArray:
    """NumPy array with label-based indexing support.

    This class wraps a NumPy ndarray and provides an additional `.at`
    interface for scalar access using user-defined keys for each dimension.

    Integer-position-based indexing (`[]`) follows standard NumPy semantics, while
    label-based (key) indexing is handled via `.at`.

    """

    __slots__ = ('_arr', '_at',)

    def __init__(self, nparray: np.ndarray, /, *, frozen: bool, key_pos_pairs: list[dict[Any, int]],
                 dim_names: list[str] | None = None):
        """Initialize a KeyedArray.

        Args:
            nparray (np.ndarray): Underlying NumPy array.
            frozen (bool): True if frozen.
            key_pos_pairs (list): A list of dictionaries mapping keys to integer-position indices
                for each dimension. Length must match `nparray.ndim`.
            dim_names (list, optional): A list of string storing dimension names.
        """
        self._arr: np.ndarray = nparray
        self._arr.flags.writeable = not frozen
        self._at: _AtIndexer = _AtIndexer(nparray, key_pos_pairs, dim_names)

    @property
    def values(self) -> np.ndarray:
        return self._arr

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return self._arr.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._arr.size

    @property
    def shape(self) -> tuple[int]:
        """Shape of the array."""
        return self._arr.shape

    @property
    def dtype(self):
        """Data type of the array."""
        return self._arr.dtype

    def __getitem__(self, index):
        """Return element(s) using NumPy-style integer-position-based indexing."""
        return self._arr[index]

    def __setitem__(self, index, value):
        """Set element(s) using NumPy-style integer-position-based indexing."""
        self._arr[index] = value

    @property
    def at(self) -> '_AtIndexer':
        """Label-based indexer for scalar access via keys."""
        return self._at

    @property
    def dim_names(self) -> list[str] | None:
        return self._at.dim_names

    @property
    def key_pos_pairs(self) -> list[dict[Any, int]]:
        """Mapping from keys to integer-position indices for each dimension."""
        return self._at.key_pos_pairs

    def key_to_pos(self, dim: int | str, key: Any, if_not_found = None) -> int | None:
        """Return the integer-position index mapped from the key of the dimension.

        Args:
            dim (int | str): Dimension, either dimension (int) or dimension name (str) is supported.
            key: Key to lookup.
            if_not_found (Optional): Value to return if no match is found, defaults to None.

        Returns:
            The integer-position index mapped from the key of the dimension.
        """
        if isinstance(dim, int):
            axis = dim
        elif isinstance(dim, str):
            axis = self._at.dim_name_to_axis.get(dim, None)
            if axis is None:
                return if_not_found
        else:
            return if_not_found
        pos = self._at.key_pos_pairs[axis].get(key, None)
        if pos is None:
            return if_not_found
        return pos

    def get(self, *args, default=None, **kwargs):
        """Return element by keys, or a default value if not found.

        Args:
            *args: Positional keys corresponding to each dimension.
            default: Value to return if any key is invalid.
            **kwargs: Keyword keys.

        Returns:
            The element at the specified keys, or `default` if lookup fails.

        Raises:
            TypeError: If both positional keys (`*args`) and keyword keys (`**kwargs`) are provided.
        """
        if args and kwargs:
            raise TypeError("Cannot mix positional and keyword keys.")

        if args:
            return self._at.get(args, default_value=default)

        keys = [None] * self._arr.ndim
        for dim, val in kwargs.items():
            axis = self._at.dim_name_to_axis.get(dim, None)
            if axis is None:
                return default
            keys[axis] = val

        return self._at.get(tuple(keys), default_value=default)


class _AtIndexer:
    """Internal label-based indexer for KeyedArray."""

    __slots__ = ('_nparray', '_key_pos_pairs', '_dim_names', '_dim_name_to_axis', '_cached_key_pos')

    def __init__(self, nparray: np.ndarray, key_pos_pairs: list[dict[Any, int]], dim_names: list[str] | None = None):
        """Initialize the indexer.

        Args:
            nparray: Underlying NumPy array.
            key_pos_pairs: List of key-to-pos mappings per dimension.
            dim_names: List of dimension names.

        Raises:
            TypeError: If `key_pos_pairs` is not a list.
            ValueError: If `key_pos_pairs` length does not match array dimensions.
        """
        if not isinstance(key_pos_pairs, list):
            raise TypeError(
                f"key_pos_pairs: type {type(key_pos_pairs)} is not allowed, expected 'list'."
            )
        if len(key_pos_pairs) != nparray.ndim:
            raise ValueError(
                f"key_pos_pairs length {len(key_pos_pairs)} does not match nparray ndim {nparray.ndim}."
            )

        self._nparray: np.ndarray = nparray
        self._dim_names: list[str] = self._validate_dim_names(dim_names)
        self._dim_name_to_axis: dict[str, int] = {name: i for i, name in enumerate(dim_names)}
        self._key_pos_pairs: list[dict[Any, int]] = key_pos_pairs
        self._cached_key_pos: dict[tuple[int, Any], tuple[Any, bool, str]] = {}

    @property
    def dim_names(self) -> list[str]:
        return self._dim_names

    @property
    def dim_name_to_axis(self) -> dict[str, int]:
        return self._dim_name_to_axis

    @property
    def key_pos_pairs(self) -> list[dict[Any, int]]:
        """Mapping from keys to integer-position indices for each dimension."""
        return self._key_pos_pairs

    def _validate_dim_names(self, dim_names: list[str] | None) -> list[str]:
        ndim = self._nparray.ndim
        default = [f"dimension{i}" for i in range(1, ndim + 1)]
        if dim_names is None:
            return default
        if len(dim_names) != ndim:
            warnings.warn(f"dim_names length {len(dim_names)} does not match nparray ndim {ndim}, default names "
                          f"(dimension1/2/...) will be used")
            return default
        if len(set(dim_names)) != len(dim_names):
            warnings.warn(f"dim_names are not unique, default names (dimension1/2/...) will be used")
            return default
        return dim_names

    def parse_pos_tuple(self, keys, if_not_found = None) -> tuple[int, ...] | None:
        """Convert a tuple of keys into a tuple of integer-position indices.

        Args:
            keys: Keys corresponding to each dimension.
            if_not_found (Optional): Value to return if no match is found, defaults to None.

        Returns:
            Tuple of integer indices. None (or specified `if_not_found` value) if not found.

        """
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) != self._nparray.ndim:
            if self._nparray.ndim == 2 and len(keys) > 2:  # attempt to infer tuple of keys for 2D array
                keys = keys[:-1], keys[-1]
            else:
                warnings.warn(f"length of keys {len(keys)} does not match ndim {self._nparray.ndim}")
                return if_not_found

        pos_list: list[int] = []
        for axis, key in enumerate(keys):
            pos, valid, _ = self.resolve_key(axis, key)
            if not valid:
                return if_not_found
            pos_list.append(pos)

        return tuple(pos_list)

    def resolve_key(self, axis, key) -> tuple[Any, bool, str]:
        resolved = self._cached_key_pos.get((axis, key), None)
        if resolved is not None:
            return resolved

        pos = self._key_pos_pairs[axis].get(key, None)

        if pos is None:
            return pos, False, f"key {key} does not exist in dimension '{self.dim_names[axis]}'"

        if not isinstance(pos, int):
            return pos, False, f"position type {type(pos)} is not valid, must be 'int'"

        if not 0 <= pos <= self._nparray.shape[axis] - 1:
            return pos, False, f"position {pos} is out of bounds range(0, {self._nparray.shape[axis]})"

        resolved = pos, True, ""
        self._cached_key_pos[(axis, key)] = resolved
        return resolved

    def get(self, args, /, *, default_value):
        """Return element by keys, or a default value if not found.

        Args:
            args: Tuple of positional keys corresponding to each dimension.
            default_value: Value to return if lookup fails.

        Returns:
            The element at the specified keys, or `default_value` if lookup fails.
        """
        pos_tuple = self.parse_pos_tuple(args)
        if pos_tuple is None:
            return default_value
        return self._nparray[pos_tuple]

    def __getitem__(self, keys):
        """Return element(s) using label-based indexing.

        Args:
            keys: Keys corresponding to each dimension.

        Returns:
            Element(s) from the underlying array.

        Raises:
            IndexError: If keys are incorrect.
        """
        pos_tuple = self.parse_pos_tuple(keys)
        if pos_tuple is None:
            raise IndexError(f"index {keys} is not valid")
        return self._nparray[pos_tuple]

    def __setitem__(self, keys, value):
        """Set element(s) using label-based indexing.

        Args:
            keys: Keys corresponding to each dimension.
            value: Value to assign.

        Raises:
            IndexError: If keys are incorrect.
        """
        pos_tuple = self.parse_pos_tuple(keys)
        if pos_tuple is None:
            raise IndexError(f"index {keys} is not valid")
        self._nparray[pos_tuple] = value


def kr_from_df(df: pd.DataFrame, *, frozen: bool = True, unpack_multi_index: bool = False,
               multi_index_name: str = 'row_index', col_index_name: str = 'col_name') -> KeyedArray | None:
    """Creat KeyedArray object from pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to be processed.
        frozen (bool): True if frozen. Defaults to True.
        unpack_multi_index (bool): True if unpacking multi-index, applicable to MultiIndex only.
        multi_index_name (str): Single name for multi-index, applicable only when `unpack_multi_index` is False and
            `df.index.nlevels` > 1, defaults to 'row_index'.
        col_index_name (str): Name of column index, e.g. 'col_name', 'var_name', 'field_name', defaults to 'col_name'.

    Returns:
        KeyedArray object: return None for empty DataFrame (`n_rows` is 0).

    """
    n_rows, n_cols = df.shape
    if n_rows == 0:
        return None

    col_key_pos_pair = {key: pos for pos, key in enumerate(df.columns)}
    values = df.to_numpy().ravel()

    df_index = df.index
    n_levels = df_index.nlevels
    if n_levels > 1 and unpack_multi_index:
        row_codes = df_index.codes
        dim_sizes = [len(level) for level in df_index.levels] + [n_cols]
        row_codes_repeated = [np.repeat(code, n_cols) for code in row_codes]
        col_codes = np.tile(np.arange(n_cols), n_rows)
        indices = tuple(row_codes_repeated + [col_codes])
        row_key_pos_pair = [{key: pos for pos, key in enumerate(df_index.levels[i])} for i in range(n_levels)]
        key_pos_pairs = row_key_pos_pair + [col_key_pos_pair]
        dim_names = df_index.names + [col_index_name]
    else:
        row_codes, uniques = pd.factorize(df_index)
        row_codes_repeated = np.repeat(row_codes, n_cols)
        col_codes = np.tile(np.arange(n_cols), n_rows)
        indices = (row_codes_repeated, col_codes)
        dim_sizes = [len(uniques), n_cols]
        row_key_pos_pair = {key: pos for pos, key in enumerate(df_index.unique())}
        key_pos_pairs = [row_key_pos_pair, col_key_pos_pair]
        dim_names = (df_index.names if n_levels == 1 else [multi_index_name]) + [col_index_name]

    dtype = values.dtype
    if dtype in ('i1', 'i2', 'i4', 'i8', 'f2', 'f4', 'f8'):
        arr = np.full(dim_sizes, np.nan, dtype=np.float64)
    else:
        warnings.warn(f"note: {dtype=}!")
        arr = np.full(dim_sizes, '', dtype=dtype)

    arr[indices] = values

    return KeyedArray(arr, frozen=frozen, key_pos_pairs=key_pos_pairs, dim_names=dim_names)

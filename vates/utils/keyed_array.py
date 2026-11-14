import numpy as np
import pandas as pd
import warnings
from collections.abc import Hashable
from typing import Any, Literal, overload


_MISSING: object = object()

class KeyedArray:
    """NumPy array with label-based indexing support.

    This class wraps a NumPy ndarray and provides an additional `.at`
    interface for scalar access using user-defined keys for each dimension.

    Integer-position-based indexing (`[]`) follows standard NumPy semantics, while
    label-based (key) indexing is handled via `.at`.

    """

    __slots__ = ('_arr', '_at',)

    def __init__(
        self,
        nparray: np.ndarray,
        /,
        *,
        frozen: bool,
        key_pos_pairs: list[dict[Hashable, int]],
        dim_names: list[str] | None = None
    ):
        """Initialize a KeyedArray.

        Args:
            nparray (np.ndarray): Underlying NumPy array.
            frozen (bool): True if frozen (`nparray.flags.writeable = False`).
            key_pos_pairs (list): A list of dictionaries mapping keys to integer-position indices
                for each dimension. Length must match `nparray.ndim`.
            dim_names (list, optional): A list of string storing dimension names; if not provided, "dimension1/2/..." will be used.
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
    def dim_names(self) -> list[str]:
        return self._at.dim_names

    @property
    def key_pos_pairs(self) -> list[dict[Hashable, int]]:
        """Mapping from keys to integer-position indices for each dimension."""
        return self._at.key_pos_pairs

    @overload
    def key_to_pos(self, dim: int | str, key: Hashable) -> int: ...

    @overload
    def key_to_pos[T](self, dim: int | str, key: Hashable, *, default: T) -> int | T: ...

    def key_to_pos(self, dim: int | str, key: Hashable, *, default: Any = _MISSING) -> int | Any:
        """Return the integer-position index mapped from the key of the dimension.

        Args:
            dim (int | str): Dimension, either dimension (int) or dimension name (str) is supported.
            key (Hashable): Key to lookup.
            default (Optional): Value to return if no match is found; raise if not provided.

        Returns:
            The integer-position index mapped from the key of the dimension.
        """
        if isinstance(dim, int):
            axis = dim
        elif isinstance(dim, str):
            axis = self._at.dim_name_to_axis.get(dim, None)
            if axis is None:
                if default is _MISSING:
                    raise KeyError(f"'{dim}' is not a valid dimension name.")
                return default
        else:
            if default is _MISSING:
                raise TypeError(f"Invalid type of 'dim': {type(dim)}, expected ('int', 'str').")
            return default
        pos = self._at.key_pos_pairs[axis].get(key, None)
        if pos is None:
            if default is _MISSING:
                raise KeyError(f"'{key}' is not a valid key in dimension '{dim}'.")
            return default
        return pos

    def get(self, *args, default=None, **kwargs):
        """Return element by keys, or a default value if not found.

        Args:
            *args (Hashable): Positional keys corresponding to each dimension.
            default: Value to return if any key is invalid.
            **kwargs: Keyword keys.

        Returns:
            The element at the specified keys, or `default` if lookup fails.

        Raises:
            TypeError: If both positional keys (`*args`) and keyword keys (`**kwargs`) are provided.
        """
        return self._at.get(*args, default=default, **kwargs)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        *,
        frozen: bool = True,
        unpack_multi_index: bool = False,
        multi_index_name: str = 'row_index',
        col_index_name: str = 'col_name'
    ) -> 'KeyedArray':
        """Creat KeyedArray object from pandas DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to be processed.
            frozen (bool): True if frozen. Defaults to True.
            unpack_multi_index (bool): True if unpacking multi-index, applicable to MultiIndex only.
            multi_index_name (str): Single name for multi-index, applicable only when `unpack_multi_index` is False and
                `df.index.nlevels` > 1, defaults to 'row_index'.
            col_index_name (str): Name of column index, e.g. 'col_name', 'var_name', 'field_name', defaults to 'col_name'.

        Returns:
            KeyedArray object.

        """
        n_rows, n_cols = df.shape
        if n_rows == 0:
            return KeyedArray(np.array([]), frozen=frozen, key_pos_pairs=[{}])

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


class _AtIndexer:
    """Internal label-based indexer for KeyedArray."""

    __slots__ = ('_arr', 'key_pos_pairs', 'dim_names', 'dim_name_to_axis')

    def __init__(self, arr: np.ndarray, key_pos_pairs: list[dict[Hashable, int]], dim_names: list[str] | None = None):
        """Initialize the indexer.

        Args:
            arr: Underlying NumPy array.
            key_pos_pairs: List of key-to-pos mappings per dimension.
            dim_names: List of dimension names; if not provided, "dimension1/2/..." will be used.

        Raises:
            TypeError: If `key_pos_pairs` is not a list.
            ValueError: If `key_pos_pairs` length does not match array dimensions.
        """
        if not isinstance(key_pos_pairs, list):
            raise TypeError(f"key_pos_pairs: type {type(key_pos_pairs)} is not allowed, expected 'list'.")
        if len(key_pos_pairs) != arr.ndim:
            raise ValueError(f"key_pos_pairs length {len(key_pos_pairs)} does not match nparray ndim {arr.ndim}.")

        self._arr: np.ndarray = arr
        self.dim_names: list[str] = self._resolve_dim_names(dim_names)
        self.dim_name_to_axis: dict[str, int] = {name: i for i, name in enumerate(self.dim_names)}
        self.key_pos_pairs: list[dict[Hashable, int]] = key_pos_pairs

    def _resolve_dim_names(self, dim_names: list[str] | None) -> list[str]:
        ndim = self._arr.ndim
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

    @overload
    def _resolve_indices(self, *keys: Hashable, on_fail_raise: Literal[True] = True) -> tuple[int, ...]: ...

    @overload
    def _resolve_indices(self, *keys: Hashable, on_fail_raise: Literal[False]) -> tuple[int, ...] | None: ...

    def _resolve_indices(self, *keys: Hashable, on_fail_raise: bool = True) -> tuple[int, ...] | None:
        """Convert a tuple of keys into a tuple of integer-position indices.

        Args:
            *keys (Hashable): Keys corresponding to each dimension.
            on_fail_raise (bool): True if raise on fail to match; defaults to True.

        Returns:
            Tuple of integer indices.

        """
        if len(keys) != self._arr.ndim:
            if self._arr.ndim == 2 and len(keys) > 2:  # attempt to infer tuple of keys for 2D array
                keys = keys[:-1], keys[-1]
            else:
                msg = f"length of keys {len(keys)} does not match ndim {self._arr.ndim}"
                if on_fail_raise:
                    raise ValueError(msg)
                warnings.warn(msg)
                return None

        pos_indices: list[int] = []
        for axis, key in enumerate(keys):
            pos = self.key_pos_pairs[axis].get(key, None)
            if pos is None:
                if on_fail_raise:
                    raise KeyError(f"'{key}' is not a valid key in dimension '{self.dim_names[axis]}'")
                return None
            pos_indices.append(pos)

        return tuple(pos_indices)

    def get(self, *args: Hashable, default, **kwargs):
        """Return element by keys, or a default value if not found.

        Args:
            *args (Hashable): Positional keys corresponding to each dimension.
            default: Value to return if lookup fails.
            **kwargs: Keyword keys.

        Returns:
            The element at the specified keys, or `default` if lookup fails.

        Raises:
            TypeError: If both positional keys (`*args`) and keyword keys (`**kwargs`) are provided.
        """
        if args and kwargs:
            raise TypeError("Cannot mix positional and keyword keys.")
        if args:
            keys = args
        else:
            keys: list[Hashable] = [None] * self._arr.ndim
            for dim, val in kwargs.items():
                axis = self.dim_name_to_axis.get(dim, None)
                if axis is None:
                    return default
                keys[axis] = val
        indices = self._resolve_indices(*keys, on_fail_raise=False)
        if indices is None:
            return default
        return self._arr[indices]

    def __getitem__(self, keys: Hashable | tuple[Hashable, ...]):
        """Return element(s) using label-based indexing.

        Args:
            keys (keys: Hashable | tuple[Hashable, ...]): Keys corresponding to each dimension.

        Returns:
            Element(s) from the underlying array.

        Raises:
            KeyError: If keys are incorrect.
        """
        keys = keys if isinstance(keys, tuple) else (keys,)
        indices = self._resolve_indices(*keys)  # will raise if keys are invalid
        return self._arr[indices]

    def __setitem__(self, keys: Hashable | tuple[Hashable, ...], value):
        """Set element(s) using label-based indexing.

        Args:
            keys (Hashable | tuple[Hashable, ...]): Keys corresponding to each dimension.
            value: Value to assign.

        Raises:
            KeyError: If keys are incorrect.
        """
        keys = keys if isinstance(keys, tuple) else (keys,)
        indices = self._resolve_indices(*keys)  # will raise if keys are invalid
        self._arr[indices] = value

import numpy as np
import pandas as pd
import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from itertools import product
from typing import Self


type DimElemItem = str | int
type DimItem = list[DimElemItem] | tuple[DimElemItem, ...] | Enum


def _maybe_convert_to_str(x) -> str:
    if isinstance(x, str):
        return x
    elif isinstance(x, int):
        return str(x)
    else:
        raise TypeError(f"Invalid type of {x}: {type(x)}, expected ('str', 'int').")

def _maybe_hit_dim_cache(x: tuple, cache: set[tuple]) -> tuple:
    hit = next((item for item in cache if item == x), None)
    if hit is not None:
        return hit
    cache.add(x)
    return x

def _resolve_dims(dims: list[DimItem] | tuple[DimItem, ...] | None,
                  cache: set[tuple[tuple[str, ...], ...]] | None = None) -> tuple[tuple[str, ...], ...]:
    """Normalize dims to a tuple of label lists.

    Args:
        dims (list[DimItem] | tuple[DimItem]): List or tuple of `DimItem`; `DimItem` can be list (of str or int),
            tuple (of str or int) or Enum class.
        cache (set[tuple[tuple[str, ...], ...]]): Set of cache.

    Returns:
        tuple[tuple[str, ...], ...]: Normalized labels.

    Raises:
        TypeError: If `dims` is not list or tuple; if `dim` (item in `dims`) is not list, tuple or Enum subclass;
            if an item in a `dim` is not str or int.
    """
    if dims is None:
        return () if cache is None else _maybe_hit_dim_cache((), cache)

    if not isinstance(dims, (list, tuple)):
        raise TypeError(f"Invalid type of 'dims': {type(dims)}, expected ('list', 'tuple').")

    resolved = []
    for dim in dims:
        if isinstance(dim, list):
            resolved.append(tuple([_maybe_convert_to_str(x) for x in dim]))
        elif isinstance(dim, tuple):
            resolved.append(dim)
        elif isinstance(dim, type) and issubclass(dim, Enum):
            resolved.append(tuple([x.name for x in dim]))
        else:
            raise TypeError(f"Invalid type of 'dims' item: {type(dim)}, expected ('list', 'tuple', 'Enum' subclass).")

    resolved = tuple(resolved)
    return resolved if cache is None else _maybe_hit_dim_cache(resolved, cache)


class ProjVariable(ABC):

    __slots__ = ()

    name: str
    owner: str
    group: str
    dims: tuple[tuple[str, ...], ...]
    ndim: int
    result: float | int | np.ndarray | None
    is_constant: bool

    _dims_cache: set[tuple[tuple[str, ...], ...]] = set()

    @abstractmethod
    def __getitem__(self, index):
        ...

    @abstractmethod
    def __setitem__(self, index, value):
        ...

    def __matmul__(self, other) -> Self:
        func = getattr(other, "attach_proj_variable", lambda _: None)
        func(self)
        return self


class ConstVariable(ProjVariable):
    """Container for constant (non-time-dimensioned) variables.

    Holds a scalar, string, or an array that does not vary over time.
    Optional dimension labels can be provided as lists, tuples or Enums; they are used in CSV output.

    Attributes:
        _name (str): Variable name used in outputs.
        _owner (str): Variable owner used in outputs.
        _group (str): Variable group used in outputs.
        _dims (tuple[tuple[str, ...], ...]): Dimension labels.
        _ndim (int): Number of dimensions.
        _result: Stored value (copied if array-like).
    """

    __slots__ = ('__weakref__', '_name', '_owner', '_group', '_dims', '_ndim', '_result',)

    def __init__(
        self,
        name: str,
        /,
        *,
        owner: str,
        group: str,
        dims: list[DimItem] | tuple[DimItem, ...] | None = None
    ):
        """
        Initialize the Constant Variable.

        Args:
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list[DimItem] | tuple[DimItem, ...] | None): Dimensions.
        """
        self._name: str = name
        self._owner: str = owner
        self._group: str = group
        self._dims: tuple[tuple[str, ...], ...] = _resolve_dims(dims, self._dims_cache)
        self._ndim: int = len(self._dims)
        self._result = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def group(self) -> str:
        return self._group

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    @property
    def is_constant(self) -> bool:
        """bool: Always True for constant variables."""
        return True

    @property
    def dims(self) -> tuple[tuple[str, ...], ...]:
        """tuple[tuple[str, ...], ...]: Dimension labels."""
        return self._dims

    @property
    def ndim(self) -> int:
        """int: Number of dimensions."""
        return self._ndim

    def __getitem__(self, index):
        """Return the stored value.

        Returns:
            Any: The stored scalar/array/string value.
        """
        return self._result

    def __setitem__(self, index, value):
        """Set the stored value.

        Args:
            value: A float/int/str or numpy array matching the dimensions.
        """
        self._result = value


class TDimVariable(ProjVariable):
    """Container for time-dimensioned variables (indexed by the time or period).

    Values are stored for each `t` from 0 to `max_t` (inclusive). Labeled dimensions (lists, tuples or Enums)
    are supported and preserved for CSV output.

    Attributes:
        _owner (str): Variable owner used in outputs.
        _dims (tuple[tuple[str, ...], ...]): Dimension labels.
        _ndim (int): Number of dimensions.
        _result (np.ndarray): Values across time and optional dimensions.
    """

    __slots__ = ('__weakref__', '_meta', '_owner', '_dims', '_ndim', '_result',)

    _meta_cache: set[tuple[str, str, int, pd.Period | None]] = set()  # tuple of (name, group, max_t, start_date)

    def __init__(
        self,
        name: str,
        /,
        *,
        owner: str,
        group: str,
        dims: list[DimItem] | tuple[DimItem, ...] | None = None,
        max_t: int,
        start_date: pd.Period | None = None,
    ):
        """
        Initialize the Time-dimensioned Variable.

        Args:
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list[DimItem] | tuple[DimItem, ...] | None): Dimensions.
            max_t (int): Max time.
            start_date (pd.Period): Start date (corresponds to time 0).
        """
        self._meta: tuple[str, str, int, pd.Period | None] = self._maybe_hit_meta_cache(name, group, max_t, start_date)
        self._owner: str = owner
        self._dims: tuple[tuple[str, ...], ...] = _resolve_dims(dims, self._dims_cache)
        self._ndim: int = len(self._dims)
        shape = tuple([max_t + 1] + ([len(dim) for dim in self._dims] if self._ndim > 0 else []))
        self._result = np.zeros(shape=shape)

    @property
    def name(self) -> str:
        return self._meta[0]

    @property
    def group(self) -> str:
        return self._meta[1]

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def max_t(self) -> int:
        return self._meta[2]

    @property
    def start_date(self) -> pd.Period | None:
        return self._meta[3]

    @classmethod
    def _maybe_hit_meta_cache(cls, name: str, group: str, max_t: int, start_date: pd.Period | None
                              ) -> tuple[str, str, int, pd.Period | None]:
        meta = (name, group, max_t, start_date)
        hit = next((x for x in cls._meta_cache if x == meta), None)
        if hit is not None:
            return hit
        if not isinstance(max_t, int):
            raise TypeError(f"Invalid type of 'max_t': {type(max_t)}, expected 'int'.")
        if max_t < 0:
            raise ValueError(f"Invalid value of 'max_t': {max_t}, expected positive.")
        if start_date is not None and not isinstance(start_date, pd.Period):
            raise TypeError(f"Invalid type of 'start_date': {type(start_date)}, expected 'pd.Period'.")
        cls._meta_cache.add(meta)  # noqa
        return meta

    @property
    def result(self) -> np.ndarray:
        return self._result

    @property
    def is_constant(self) -> bool:
        """bool: Always False for time-dimensioned variables."""
        return False

    @property
    def dims(self) -> tuple[tuple[str, ...], ...]:
        """tuple[tuple[str, ...], ...]: Dimension labels."""
        return self._dims

    @property
    def ndim(self) -> int:
        """int: Number of dimensions."""
        return self._ndim

    def _resolve_time_index(self, index: int | pd.Period) -> int:
        if isinstance(index, int):
            t = index
        elif isinstance(index, pd.Period):
            start_date = self.start_date
            if start_date is None:
                raise ValueError(f"'start_date' is None, cannot index by 'pd.Period'.")
            t = (index - start_date).n
        else:
            raise TypeError(f"Invalid {type(index)=}, expected 'int' or 'pd.Period'.")
        if not (0 <= t <= self.max_t):
            raise IndexError(f"Invalid {index=}, expected: '0 to {self.max_t}', or "
                             f"'{self.start_date} to {self.start_date + self.max_t}.")
        return t

    def __getitem__(self, index: int | pd.Period):
        """Return the result at a certain time or period.

        Args:
            index (int | pd.Period): Return the value at a certain time or period.

        Returns:
            Any: Result at `t` (copied for ndim>0).

        """
        t = self._resolve_time_index(index)
        return self._result[t] if self._ndim == 0 else self._result[t,].copy()

    def __setitem__(self, index: int | pd.Period, value):
        """Set the value at the current time index.

        Args:
            value: Scalar or array whose shape matches the variable's dimensions.
        """
        t = self._resolve_time_index(index)
        if self._ndim == 0:
            self._result[t] = value
        else:
            self._result[t,] = value.copy()


class ProjVariableManager:

    __slots = ("_proj_variables",)

    def __init__(self):
        self._proj_variables: list[weakref.ref[ProjVariable]] = []

    def attach_proj_variable(self, obj: ProjVariable, /) -> None:
        """Attach a projection variable into `_proj_variables`

        Args:
            obj (ProjVariable): Projection variable to be attached.

        Raises:
            TypeError: Invalid type.
        """
        if not isinstance(obj, ProjVariable):
            raise TypeError(f"Invalid type: {type(obj)}, expected 'ProjVariable'.")
        ref = weakref.ref(obj)
        if ref not in self._proj_variables:
            self._proj_variables.append(ref)

    def __rmatmul__(self, other: ProjVariable) -> ProjVariable:
        self.attach_proj_variable(other)
        return other

    def clear_dead(self) -> None:
        self._proj_variables[:] = [ref for ref in self._proj_variables if ref() is not None]

    def filter(self, rule: dict[str, dict] | None = None) -> list[ProjVariable]:
        alive = (v for ref in self._proj_variables if (v := ref()) is not None)
        if rule is None:
            return list(alive)
        return [v for v in alive if self._is_kept(v, rule)]

    @classmethod
    def _is_kept(cls, v: ProjVariable, rule: dict[str, dict]) -> bool:
        group: dict[str, str | list] | None = rule.get(v.group, None)
        if group is None:
            return False

        include = group.get("include", None)
        if include is not None:
            if isinstance(include, str):
                return (include.lower() == "__all__") or (v.name == include)
            elif isinstance(include, list):
                return v.name in include
            else:
                raise TypeError(f"Invalid type of 'include': {type(include)}, expected ('str', 'list[str]').")

        exclude = group.get("exclude", None)
        if exclude is not None:
            if isinstance(exclude, list):
                return v.name not in exclude
            elif isinstance(exclude, str):
                return v.name != exclude
            else:
                raise TypeError(f"Invalid type of 'exclude': {type(exclude)}, expected ('str', 'list[str]').")

        return True  # if neither 'include' nor 'exclude' provided, all variables in the group are included

    @classmethod
    def iter_label_result(cls, v: ProjVariable, dim_joiner: str = ":", dim_bracket: tuple[str, str] = ("[", "]")
                          ) -> Iterator[tuple[str, float | np.ndarray | None]]:
        if v.ndim == 0:
            yield v.name, v.result
        else:
            _left, right_ = dim_bracket
            indexed_dims = (list(enumerate(dim)) for dim in v.dims)
            for item in product(*indexed_dims):
                dim_idx, dim_str = map(tuple, zip(*item))
                label = f"{v.name}{_left}{dim_joiner.join(dim_str)}{right_}"
                result = v.result[(slice(0, None),) + dim_idx]  # noqa
                yield label, result

    def __contains__(self, item) -> bool:
        ref = item if isinstance(item, weakref.ref) else weakref.ref(item)
        return ref in self._proj_variables

    def __iter__(self):
        for ref in self._proj_variables:
            obj = ref()
            if obj is not None:
                yield obj

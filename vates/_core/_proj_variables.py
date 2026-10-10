import numpy as np
import pandas as pd
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal


class ProjVariable(ABC):

    __slots__ = ()

    name: str
    owner: str
    group: str
    dims: list | None
    ndim: int
    result: float | np.ndarray | None
    is_constant: bool

    _cached_dims: set[tuple[tuple, ...]] = set()

    @classmethod
    def _resolve_dims(cls, dims) -> tuple[tuple, ...] | None:
        """Normalize dims to a tuple of label lists.

        Args:
            dims: None, a list of lists (of str or int), or Enum classes.

        Returns:
            tuple[tuple[str]] | None: Normalized labels or None.

        Raises:
            ValueError: If dims are malformed or exceed 3 dimensions.
        """

        if dims is None:
            return None

        if not isinstance(dims, (list, tuple)):
            raise ValueError("dims must be a list.")
        if len(dims) > 3:
            raise ValueError("Number of dimensions exceeds maximum (3).")

        def _maybe_convert_to_str(x) -> str:
            if isinstance(x, str):
                return x
            elif isinstance(x, int):
                return str(x)
            else:
                raise ValueError(f"{x}: variable dimension list should contain str or int.")

        resolved = []
        for dim in dims:
            if isinstance(dim, list):
                resolved.append(tuple([_maybe_convert_to_str(x) for x in dim]))
            elif isinstance(dim, tuple):
                resolved.append(dim)
            elif issubclass(dim, Enum):
                resolved.append(tuple([x.name for x in dim]))
            else:
                raise ValueError(f"{dim}: variable dimension must be either list or enumeration.")

        resolved = tuple(resolved)
        cached = next((x for x in cls._cached_dims if x == resolved), None)
        if cached is not None:
            return cached
        cls._cached_dims.add(resolved)
        return resolved

    @abstractmethod
    def __getitem__(self, index):
        ...

    @abstractmethod
    def __setitem__(self, index, value):
        ...

    def __matmul__(self, other):
        return other.__rmatmul__(self)


class ConstVariable(ProjVariable):
    """Container for constant (non-time-dimensioned) variables.

    Holds a scalar, string, or an array (up to 3 dimensions) that does not vary over time.
    Optional dimension labels can be provided as lists or Enums; they are used in CSV output.

    Attributes:
        _name (str): Variable name used in outputs.
        _owner (str): Variable owner used in outputs.
        _group (str): Variable group used in outputs.
        _dims (list[list[str]] | None): Dimension labels (expanded from lists or Enums), or None if scalar.
        _ndim (int): Number of dimensions (0-3).
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
        dims: list | None = None
    ):
        """
        Initialize the Constant Variable.

        Args:
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list|None): Dimensions.
        """
        self._name: str = name
        self._owner: str = owner
        self._group: str = group
        self._dims: tuple[tuple[str]] | None = self._resolve_dims(dims)
        self._ndim: int = len(dims) if dims is not None else 0
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

    @property
    def is_constant(self) -> bool:
        """bool: Always True for constant variables."""
        return True

    @property
    def dims(self) -> list | None:
        """list | None: Dimension labels or None if scalar."""
        return self._dims

    @property
    def ndim(self) -> int:
        """int: Number of dimensions (0-3)."""
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

    Values are stored for each `t` from 0 to `max_t` (inclusive). Optional up to 3 labeled
    dimensions (lists or Enums) are supported and preserved for CSV output.

    Attributes:
        _owner (str): Variable owner used in outputs.
        _dims (list[list[str]] | None): Dimension labels (expanded from lists or Enums), or None if scalar.
        _ndim (int): Number of dimensions (0-3).
        _result (np.ndarray): Values across time and optional dimensions.
    """

    __slots__ = ('__weakref__', '_meta', '_owner', '_dims', '_ndim', '_result',)

    _cached_metas: set[tuple[str, str, int, pd.Period | None]] = set()

    def __init__(
        self,
        name: str,
        /,
        *,
        owner: str,
        group: str,
        dims: list | None = None,
        max_t: int,
        start_date: pd.Period | None = None,
    ):
        """
        Initialize the Time-dimensioned Variable.

        Args:
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list|None): Dimensions.
            max_t (int): Max time.
            start_date (pd.Period): Start date (coresponding to time 0).
        """
        self._meta: tuple[str, str, int, pd.Period | None] = self._make_meta(name, group, max_t, start_date)
        self._owner: str = owner
        self._dims: tuple[tuple[str]] | None = self._resolve_dims(dims)
        self._ndim: int = len(dims) if dims is not None else 0
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
    def _make_meta(cls, name: str, group: str, max_t: int, start_date: pd.Period
                   ) -> tuple[str, str, int, pd.Period | None]:
        meta = (name, group, max_t, start_date)
        cached = next((x for x in cls._cached_metas if x == meta), None)
        if cached is not None:
            return cached
        if not isinstance(max_t, int):
            raise TypeError(f"Invalid type of 'max_t': {type(max_t)}, expected 'int'.")
        if max_t <= 0:
            raise ValueError(f"Invalid value of 'max_t': {max_t}, expected positive.")
        if start_date is not None and not isinstance(start_date, pd.Period):
            raise TypeError(f"Invalid type of 'start_date': {type(max_t)}, expected 'pd.Period'.")
        cls._cached_metas.add(meta)
        return meta

    @property
    def result(self) -> np.ndarray:
        return self._result

    @property
    def is_constant(self) -> bool:
        """bool: Always False for time-dimensioned variables."""
        return False

    @property
    def dims(self) -> list | None:
        """list | None: Dimension labels or None if scalar."""
        return self._dims

    @property
    def ndim(self) -> int:
        """int: Number of dimensions (0-3)."""
        return self._ndim

    def __getitem__(self, index: int | pd.Period):
        """Return the result at a certain time or period.

        Args:
            index (int | pd.Period): Return the value at a certain time or period.

        Returns:
            Any: Result at `t` (copied for ndim>0).

        """
        if type(index) == int:
            t = index
        elif type(index) == pd.Period:
            if self.start_date is None:
                raise ValueError(f"'start_date' is None, cannot index by 'pd.Period'.")
            t = (index - self.start_date).n
        else:
            raise TypeError(f"Invalid {type(index)=}, expected 'int' or 'pd.Period'.")

        if not (0 <= t <= self.max_t):
            raise ValueError(f"Invalid {index=}, expected t: 0 to {self.max_t}.")

        return self._result[t] if self._ndim == 0 else self._result[t,].copy()

    def __setitem__(self, index: int | pd.Period, value):
        """Set the value at the current time index.

        Args:
            value: Scalar or array whose shape matches the variable's dimensions.
        """
        if type(index) == int:
            t = index
        elif type(index) == pd.Period:
            if self.start_date is None:
                raise ValueError(f"'start_date' is None, cannot index by 'pd.Period'.")
            t = (index - self.start_date).n
        else:
            raise TypeError(f"Invalid {type(index)=}, expected 'int' or 'pd.Period'.")

        if not (0 <= t <= self.max_t):
            raise ValueError(f"Invalid {index=}, expected t: 0 to {self.max_t}.")

        if self._ndim == 0:
            self._result[t] = value
        else:
            self._result[t,] = value.copy()


def make_proj_variable(
    name: str,
    /,
    *,
    model_engine = None,
    variable_type: Literal["const", "c", "tdim", "t", "time_dimensioned"] = "t",
    owner: str,
    group: str,
    dims: list | None = None,
) -> ConstVariable | TDimVariable:
    if not isinstance(variable_type, str):
        raise TypeError(f"Invalid type of 'variable_type': '{type(variable_type)}', expected 'str'.")

    if variable_type.lower() in ("const", "c"):
        var = ConstVariable(name, owner=owner, group=group, dims=dims)
    elif variable_type.lower() in ("tdim", "t", "time_dimensioned"):
        if model_engine is not None:
            var = TDimVariable(name, owner=owner, group=group, dims=dims,
                               max_t=getattr(model_engine, "MAX_T"), start_date=getattr(model_engine, "START_DATE", None))
        else:
            var = TDimVariable(name, owner=owner, group=group, dims=dims, max_t=1200)
            warnings.warn(f"Create 'TDimVariable': 'max_t=1200' and 'start_date=None'; 'model_engine' is 'None'.")
    else:
        raise ValueError(f"Invalid value of 'variable_type': '{variable_type}', expected ('const', 'tdim').")

    if model_engine is not None and (func := getattr(model_engine, "attach_proj_variable", None)) is not None:
        func(var)

    return var

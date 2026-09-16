from __future__ import annotations
import numpy as np
import pandas as pd
import typing
import weakref
from abc import ABC, abstractmethod
from enum import Enum

if typing.TYPE_CHECKING:
    from vates._core.proj_model_engine import ProjModelEngine
    from vates._core._utils import RunConfiguration


class ProjVariable(ABC):
    """Container for projection variables used in model output.

    Optional dimension labels can be provided as lists or Enums; they are used in CSV output.

    Attributes:
        name (str): Variable name used in outputs.
        owner (str): Variable owner used in outputs.
        group (str): Variable group used in outputs.
        _dims (list[list[str]] | None): Dimension labels (expanded from lists or Enums), or None if scalar.
        _ndim (int): Number of dimensions (0-3).
    """

    __slots__ = ('__weakref__', 'name', 'owner', 'group', '_dims', '_ndim',)

    _unique_dims = []

    def __init__(
        self, name: str,
        /,
        *,
        model_engine: ProjModelEngine | None,
        owner: str,
        group: str,
        dims: list | None = None
    ):
        """
        Initialize the Projection Variable.

        Args:
            model_engine (ProjModelEngine): Model engine object.
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (tuple[tuple[str]]|None): Dimensions.
        """
        self.name: str = name
        self.owner: str = owner
        self.group: str = group
        self._dims: tuple[tuple[str]] | None = self._resolve_dims(dims)
        self._ndim: int = len(dims) if dims is not None else 0
        if model_engine is not None:
            model_engine.include_proj_variable(weakref.ref(self))

    @property
    @abstractmethod
    def result(self):
        pass

    @property
    @abstractmethod
    def is_constant(self) -> bool:
        """bool: True if constant, False if time-dimensioned."""
        pass

    @property
    def dims(self) -> list | None:
        """list | None: Dimension labels or None if scalar."""
        return self._dims

    @property
    def ndim(self) -> int:
        """int: Number of dimensions (0-3)."""
        return self._ndim

    @classmethod
    def _resolve_dims(cls, dims) -> tuple[tuple] | None:
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
            elif issubclass(dim, Enum):
                resolved.append(tuple([x.name for x in dim]))
            else:
                raise ValueError(f"{dim}: variable dimension must be either list or enumeration.")

        resolved = tuple(resolved)
        for dim in cls._unique_dims:
            if resolved == dim:
                return dim
        cls._unique_dims.append(resolved)
        return resolved

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __setitem__(self, index, value):
        pass


class ConstVariable(ProjVariable):
    """Container for constant (non-time-dimensioned) variables.

    Holds a scalar, string, or an array (up to 3 dimensions) that does not vary over time.
    Optional dimension labels can be provided as lists or Enums; they are used in CSV output.

    Attributes:
        name (str): Variable name used in outputs.
        owner (str): Variable owner used in outputs.
        group (str): Variable group used in outputs.
        _dims (list[list[str]] | None): Dimension labels (expanded from lists or Enums), or None if scalar.
        _ndim (int): Number of dimensions (0-3).
        _result: Stored value (copied if array-like).
    """
    __slots__ = ('_result',)

    def __init__(
        self, name: str,
        /,
        *,
        model_engine: ProjModelEngine | None = None,
        owner: str = 'unowned',
        group: str = 'ungrouped',
        dims: list | None = None
    ):
        """
        Initialize the Constant Variable.

        Args:
            name (str): Variable name.
            model_engine (ProjModelEngine): Model engine object.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list|None): Dimensions.
        """
        super().__init__(name, model_engine=model_engine, owner=owner, group=group, dims=dims)
        self._result = None

    @property
    def result(self):
        return self._result

    @property
    def is_constant(self) -> bool:
        """bool: Always True for constant variables."""
        return True

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
        name (str): Variable name used in outputs.
        owner (str): Variable owner used in outputs.
        group (str): Variable group used in outputs.
        _dims (list[list[str]] | None): Dimension labels (expanded from lists or Enums), or None if scalar.
        _ndim (int): Number of dimensions (0-3).
        _result (np.ndarray): Values across time and optional dimensions.
        _assigned (np.ndarray): True if value has been assigned otherwise False.
    """

    __slots__ = ('_cfg', '_result', '_assigned',)

    fallback_cfg = None


    def __init__(
        self,
        name: str,
        /,
        *,
        model_engine: ProjModelEngine | None,
        owner: str = 'unowned',
        group: str = 'ungrouped',
        dims: list | None = None
    ):
        """
        Initialize the Time-dimensioned Variable.

        Args:
            model_engine (ProjModelEngine): Model engine object.
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list|None): Dimensions.
        """
        super().__init__(name, model_engine=model_engine, owner=owner, group=group, dims=dims)
        if model_engine:
            self._cfg: RunConfiguration = model_engine._run_config
        elif self.fallback_cfg:
            self._cfg = self.fallback_cfg  # for advanced users who deliberately want to use 'fallback_cfg'
        else:
            raise ValueError("'model_engine' is None.")
        self._result = np.zeros(self._shape)
        self._assigned = np.array([False] * (self._cfg.max_t + 1))


    @property
    def result(self) -> np.ndarray:
        return self._result

    @property
    def is_constant(self) -> bool:
        """bool: Always False for time-dimensioned variables."""
        return False

    @property
    def _shape(self) -> tuple[int]:
        """Compute the internal numpy array shape.

        Returns:
            tuple[int]: (max_t+1, [dim1, dim2, dim3]).
        """
        shape = (self._cfg.max_t + 1,)
        if self._ndim > 0:
            for dim in self._dims:
                shape += (len(dim),)
        return shape

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
            t = (index - self._cfg.start_date).n
        else:
            raise TypeError(f"Invalid {type(index)=}, expected 'int' or 'pd.Period'.")

        if not (0 <= t <= self._cfg.max_t):
            raise ValueError(f"Invalid {index=}, expected t: 0 to {self._cfg.max_t} (period: {self._cfg.start_date} to {self._cfg.end_date}).")

        if self._assigned[t]:
            return self._result[t] if self._ndim == 0 else self._result[t,].copy()
        else:
            return None

    def __setitem__(self, index: int | pd.Period, value):
        """Set the value at the current time index.

        Args:
            value: Scalar or array whose shape matches the variable's dimensions.
        """
        if type(index) == int:
            t = index
        elif type(index) == pd.Period:
            t = (index - self._cfg.start_date).n
        else:
            raise TypeError(f"Invalid {type(index)=}, expected 'int' or 'pd.Period'.")

        if not (0 <= t <= self._cfg.max_t):
            raise ValueError(f"Invalid {index=}, expected t: 0 to {self._cfg.max_t} (period: {self._cfg.start_date} to {self._cfg.end_date}).")

        if self._ndim == 0:
            self._result[t] = value
        else:
            self._result[t,] = value.copy()

        self._assigned[t] = True

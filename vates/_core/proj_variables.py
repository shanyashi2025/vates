from __future__ import annotations
import pandas as pd
import typing
import numpy as np
import weakref
import warnings
from enum import Enum
from abc import ABC, abstractmethod

if typing.TYPE_CHECKING:
    from vates._core.proj_model_engine import ProjModelEngine


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

    def __init__(self, model_engine: ProjModelEngine, name: str, owner: str, group: str, dims: list | None = None):
        """
        Initialize the Projection Variable.

        Args:
            model_engine (ProjModelEngine): Model engine object.
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list|None): Dimensions.
        """
        self.name: str = name
        self.owner: str = owner
        self.group: str = group
        self._dims: list[list[str]] | None = self._parse_dims(dims)
        self._ndim: int = len(dims) if dims is not None else 0
        model_engine.include_proj_variable(weakref.ref(self))

    @property
    @abstractmethod
    def result(self):
        pass

    @property
    @abstractmethod
    def is_constant(self) -> bool:
        """bool: True if constant, False if time-dependent."""
        pass

    @property
    def dims(self) -> list | None:
        """list | None: Dimension labels or None if scalar."""
        return self._dims

    @property
    def ndim(self) -> int:
        """int: Number of dimensions (0-3)."""
        return self._ndim

    @staticmethod
    def _parse_dims(dims) -> list[list[str]] | None:
        """Normalize dims to a list of label lists.

        Args:
            dims: None, a list of lists, or a list of Enum classes.

        Returns:
            list[list[str]] | None: Normalized labels or None.

        Raises:
            ValueError: If dims are malformed or exceed 3 dimensions.
        """
        if dims is None:
            return None

        import copy
        dims = copy.deepcopy(dims)

        if not isinstance(dims, list):
            raise ValueError("dims must be a list.")
        if len(dims) > 3:
            raise ValueError("Number of dimensions exceeds maximum (3).")

        result = []
        for dim in dims:
            if isinstance(dim, list):
                for index, value in enumerate(dim):
                    if type(value) == str:
                        pass
                    elif type(value) == int:
                        dim[index] = str(value)
                    else:
                        raise ValueError("Variable dimension list should contain str or int.")
                result.append(dim)
            elif issubclass(dim, Enum):
                result.append([x.name for x in dim])
            else:
                raise ValueError("Variable dimension must be either list or enumeration.")

        return result

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __setitem__(self, index, value):
        pass


class ConstVariable(ProjVariable):
    """Container for constant (non-time-dependent) variables.

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

    def __init__(self, model_engine: ProjModelEngine, name: str, owner: str, group: str, dims: list | None = None):
        """
        Initialize the Constant Variable.

        Args:
            model_engine (ProjModelEngine): Model engine object.
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list|None): Dimensions.
        """
        super().__init__(model_engine, name, owner, group, dims)
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


class TDepVariable(ProjVariable):
    """Container for time-dependent variables (indexed by the time or period).

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

    __slots__ = ('_max_t', '_start_date_ordinal', '_result', '_assigned',)

    def __init__(self, model_engine: ProjModelEngine, name: str, owner: str, group: str, dims: list | None = None):
        """
        Initialize the Time-memory Variable.

        Args:
            model_engine (ProjModelEngine): Model engine object.
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list|None): Dimensions.
        """
        super().__init__(model_engine, name, owner, group, dims)
        self._max_t: int = model_engine.MAX_T
        self._start_date_ordinal: int = model_engine.START_YEAR * 12 + model_engine.START_MONTH  # ordinal encoding
        self._result = np.zeros(self._shape)
        self._assigned = np.array([False] * (self._max_t + 1))

    @property
    def result(self) -> np.ndarray:
        return self._result

    @property
    def is_constant(self) -> bool:
        """bool: Always False for time-dependent variables."""
        return False

    @property
    def _shape(self) -> tuple[int]:
        """Compute the internal numpy array shape.

        Returns:
            tuple[int]: (max_t+1, [dim1, dim2, dim3]).
        """
        shape = (self._max_t + 1,)
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
            if not (0 <= index <= self._max_t):
                warnings.warn(f"Invalid {index=}, expected 0 to {self._max_t}.")
                return None
            t = index
        elif type(index) == pd.Period:
            t = index.year * 12 + index.month - self._start_date_ordinal
            if t is None:
                warnings.warn(f"Invalid {index=}.")
                return None
        else:
            warnings.warn(f"Invalid {type(index)=}, expected 'int' or 'pd.Period'.")
            return None

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
            if not (0 <= index <= self._max_t):
                warnings.warn(f"Assignment failed, invalid {index=}, expected 0 to {self._max_t}.")
                return
            t = index
        elif type(index) == pd.Period:
            t = index.year * 12 + index.month - self._start_date_ordinal
            if t is None:
                warnings.warn(f"Assignment failed, invalid {index=}.")
                return
        else:
            warnings.warn(f"Assignment failed, invalid {type(index)=}, expected int or pd.Period.")
            return

        if self._ndim == 0:
            self._result[t] = value
        else:
            self._result[t,] = value.copy()

        self._assigned[t] = True

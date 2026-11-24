import numpy as np
import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from itertools import product
from typing import Any, Self, overload


type DimElemItem = str | int
type DimItem = list[DimElemItem] | tuple[DimElemItem, ...] | type(Enum)


def _maybe_convert_to_str(x: str | int | Enum) -> str:
    if isinstance(x, str):
        return x
    elif isinstance(x, int):
        return str(x)
    elif isinstance(x, Enum):
        return x.name
    else:
        raise TypeError(f"Invalid type of {x}: {type(x)}, expected ('str', 'int', 'Enum').")

def _maybe_hit_dim_cache(x: tuple, cache: set[tuple]) -> tuple:
    hit = next((item for item in cache if item == x), None)
    if hit is not None:
        return hit
    cache.add(x)
    return x

def _resolve_dims(dims: list[DimItem] | tuple[DimItem, ...],
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
    if not isinstance(dims, (list, tuple)):
        raise TypeError(f"Invalid type of 'dims': {type(dims)}, expected ('list', 'tuple').")

    resolved = []
    for dim in dims:
        if isinstance(dim, (list, tuple)):
            resolved.append(tuple([_maybe_convert_to_str(x) for x in dim]))
        elif isinstance(dim, type) and issubclass(dim, Enum):
            resolved.append(tuple([x.name for x in dim]))
        else:
            raise TypeError(f"Invalid type of 'dims' item: {type(dim)}, expected ('list', 'tuple', 'Enum' subclass).")

    resolved = tuple(resolved)
    return resolved if cache is None else _maybe_hit_dim_cache(resolved, cache)


class BaseProjVariable(ABC):

    __slots__ = ('__weakref__', )

    name: str
    owner: str
    group: str
    is_tdim: bool
    dims: tuple[tuple[str, ...], ...]
    result: float | int | np.ndarray | None

    def resolve_indices(self, *args: DimElemItem | Enum) -> tuple[int, ...]:
        dims = self.dims
        if len(args) != len(dims):
            raise ValueError(f"'{self.name}': {len(args)} key(s) provided, expected {len(dims)}.")
        indices: list[int] = []
        for axis, arg in enumerate(args):
            try:
                indices.append(dims[axis].index(_maybe_convert_to_str(arg)))
            except ValueError:
                raise KeyError(f"'{self.name}': '{arg}' not in dims[{axis}].") from None
        return tuple(indices)

    @abstractmethod
    def __getitem__(self, index: int | slice):
        ...

    @abstractmethod
    def __setitem__(self, index: int | slice, value):
        ...

    def __matmul__(self, other) -> Self:
        func = getattr(other, "attach_proj_variable", lambda _: None)
        func(self)
        return self


class ScalarVariable(BaseProjVariable):
    """Container for scalar variables.

    Holds a scalar that does not vary over time.

    Attributes:
        _ng (tuple[str, str]): Variable name and group used in outputs.
        _owner (str): Variable owner used in outputs.
        _result: Stored scalar value.
    """

    __slots__ = ('_ng', '_owner', '_result',)
    is_tdim = False
    _ng_cache: set[tuple[str, str]] = set()
    dims = ()

    def __init__(
        self,
        name: str,
        /,
        *,
        owner: str,
        group: str,
    ):
        """
        Initialize the Scalar Variable.

        Args:
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
        """
        self._ng: tuple[str, str] = _maybe_hit_dim_cache((name, group), self._ng_cache)
        self._owner: str = owner
        self._result = None

    @property
    def name(self) -> str:
        return self._ng[0]

    @property
    def group(self) -> str:
        return self._ng[1]

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def result(self) -> Any:
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    def __getitem__(self, index):
        """Return the stored result value.

        Returns:
            Any: The result value.
        """
        return self._result

    def __setitem__(self, index, value):
        """Set the stored value.

        Args:
            value: A scalar value.
        """
        self._result = value


class ConstantArrayVariable(BaseProjVariable):
    """Container for constant array variables.

    Holds an array that does not vary over time.
    Dimension labels can be provided as lists, tuples or Enums; they are used in CSV output.

    Attributes:
        _ng (tuple[str, str]): Variable name and group used in outputs.
        _owner (str): Variable owner used in outputs.
        _dims (tuple[tuple[str, ...], ...]): Dimension labels.
        _ndim (int): Number of dimensions.
        _result: Stored value.
    """

    __slots__ = ('_ng', '_owner', '_dims', '_ndim', '_result',)
    is_tdim = False
    _ng_cache: set[tuple[str, str]] = set()
    _dims_cache: set[tuple[tuple[str, ...], ...]] = set()

    def __init__(
        self,
        name: str,
        /,
        *,
        owner: str,
        group: str,
        dims: list[DimItem] | tuple[DimItem, ...]
    ):
        """
        Initialize the Constant Variable.

        Args:
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list[DimItem] | tuple[DimItem, ...]): Dimensions.
        """
        self._ng: tuple[str, str] = _maybe_hit_dim_cache((name, group), self._ng_cache)
        self._owner: str = owner
        self._dims: tuple[tuple[str, ...], ...] = _resolve_dims(dims, self._dims_cache)
        self._ndim: int = len(self._dims)
        self._result: np.ndarray = np.zeros([len(dim) for dim in self._dims])

    @property
    def name(self) -> str:
        return self._ng[0]

    @property
    def group(self) -> str:
        return self._ng[1]

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def result(self) -> np.ndarray:
        return self._result

    @result.setter
    def result(self, value):
        self._result[:] = value

    @property
    def dims(self) -> tuple[tuple[str, ...], ...]:
        """tuple[tuple[str, ...], ...]: Dimension labels."""
        return self._dims

    @property
    def ndim(self) -> int:
        """int: Number of dimensions."""
        return self._ndim

    def __getitem__(self, index: int | slice):
        """Return the stored value.

        Returns:
            Any: The stored array value.
        """
        value = self._result[index]
        return value.item() if np.ndim(value) == 0 else value

    def __setitem__(self, index: int | slice, value):
        """Set the stored value.

        Args:
            value: A numpy array matching the dimensions.
        """
        self._result[index] = value


class TDimVariable(BaseProjVariable):
    """Container for time-dimensioned variables (indexed by the time).

    Values are stored for each `t` from 0 to `max_t` (inclusive).

    Attributes:
        _ng (tuple[str, str]): Variable name and group used in outputs.
        _owner (str): Variable owner used in outputs.
        _result (np.ndarray): Values across time and optional dimensions.
    """

    __slots__ = ('_ng', '_owner', '_result',)
    is_tdim = True
    _ng_cache: set[tuple[str, str]] = set()
    dims = ()

    def __init__(
        self,
        name: str,
        /,
        *,
        owner: str,
        group: str,
        max_t: int,
    ):
        """
        Initialize the Time-dimensioned Variable.

        Args:
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            max_t (int): Max time.
        """
        self._ng: tuple[str, str] = _maybe_hit_dim_cache((name, group), self._ng_cache)
        self._owner: str = owner
        self._result = np.zeros(max_t + 1)

    @property
    def name(self) -> str:
        return self._ng[0]

    @property
    def group(self) -> str:
        return self._ng[1]

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def result(self) -> np.ndarray:
        return self._result

    def __getitem__(self, index: int | slice):
        """Return the result at a time index or slice.

        Args:
            index: Integer time index or slice of time indices.

        Returns:
            Any: The selected value.

        """
        return self._result[index]

    def __setitem__(self, index: int | slice, value):
        """Set the value at the current time index.

        Args:
            value: Scalar or array whose shape matches the variable's dimensions.
        """
        self._result[index] = value


class TDimArrayVariable(BaseProjVariable):
    """Container for time-dimensioned array variables (indexed by the (time, dims)).

    Values are stored for each `t` from 0 to `max_t` (inclusive). Labeled dimensions (lists, tuples or Enums)
    are supported and preserved for CSV output.

    Attributes:
        _ng (tuple[str, str]): Variable name and group used in outputs.
        _owner (str): Variable owner used in outputs.
        _dims (tuple[tuple[str, ...], ...]): Dimension labels.
        _ndim (int): Number of dimensions.
        _result (np.ndarray): Values across time and optional dimensions.
    """

    __slots__ = ('_ng', '_owner', '_dims', '_ndim', '_result',)
    is_tdim = True
    _ng_cache: set[tuple[str, str]] = set()
    _dims_cache: set[tuple[tuple[str, ...], ...]] = set()

    def __init__(
        self,
        name: str,
        /,
        *,
        owner: str,
        group: str,
        dims: list[DimItem] | tuple[DimItem, ...],
        max_t: int,
    ):
        """
        Initialize the Time-dimensioned Variable.

        Args:
            name (str): Variable name.
            owner (str): Variable owner.
            group (str): Variable group.
            dims (list[DimItem] | tuple[DimItem, ...]): Dimensions.
            max_t (int): Max time.
        """
        self._ng: tuple[str, str] = _maybe_hit_dim_cache((name, group), self._ng_cache)
        self._owner: str = owner
        self._dims: tuple[tuple[str, ...], ...] = _resolve_dims(dims, self._dims_cache)
        self._ndim: int = len(self._dims)
        self._result = np.zeros([max_t + 1] + [len(dim) for dim in self._dims])

    @property
    def name(self) -> str:
        return self._ng[0]

    @property
    def group(self) -> str:
        return self._ng[1]

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def result(self) -> np.ndarray:
        return self._result

    @property
    def dims(self) -> tuple[tuple[str, ...], ...]:
        """tuple[tuple[str, ...], ...]: Dimension labels."""
        return self._dims

    @property
    def ndim(self) -> int:
        """int: Number of dimensions."""
        return self._ndim

    def __getitem__(self, index):
        """Return results at a time index or slice.

        Args:
            index: Integer time index or slice of time indices.

        Returns:
            Any: The selected value.

        """
        return self._result[index]

    def __setitem__(self, index, value):
        """Set the value at the current time index.

        Args:
            value: Scalar or array whose shape matches the variable's dimensions.
        """
        self._result[index] = value


@overload
def make_variable(name: str, /, *, owner: str, group: str) -> ScalarVariable: ...

@overload
def make_variable(name: str, /, *, owner: str, group: str, dims: list | tuple) -> ConstantArrayVariable: ...

@overload
def make_variable(name: str, /, *, owner: str, group: str, max_t: int) -> TDimVariable: ...

@overload
def make_variable(name: str, /, *, owner: str, group: str, max_t: int, dims: list | tuple) -> TDimArrayVariable: ...

@overload
def make_variable(name: str, /, *, owner: str, group: str, max_t: int | None = None, dims: list | tuple | None = None,
                  ) -> ScalarVariable | ConstantArrayVariable | TDimVariable | TDimArrayVariable: ...

def make_variable(name: str, /, *, owner: str, group: str, max_t: int | None = None, dims: list | tuple | None = None,
                  ) -> ScalarVariable | ConstantArrayVariable | TDimVariable | TDimArrayVariable:
    if max_t is None and dims is None:
        return ScalarVariable(name, owner=owner, group=group)
    if max_t is None and dims is not None:
        return ConstantArrayVariable(name, owner=owner, group=group, dims=dims)
    if max_t is not None and dims is None:
        return TDimVariable(name, owner=owner, group=group, max_t=max_t)
    if max_t is not None and dims is not None:
        return TDimArrayVariable(name, owner=owner, group=group, max_t=max_t, dims=dims)
    raise ValueError("Should never reach here.")


class ProjVariableManager:

    __slots__ = ("_proj_variables",)

    def __init__(self):
        self._proj_variables: list[weakref.ref[BaseProjVariable]] = []

    def attach_proj_variable(self, obj: BaseProjVariable, /) -> None:
        """Attach a projection variable into `_proj_variables`

        Args:
            obj (BaseProjVariable): Projection variable to be attached.

        Raises:
            TypeError: Invalid type.
        """
        if not isinstance(obj, BaseProjVariable):
            raise TypeError(f"Invalid type: {type(obj)}, expected 'ProjVariable'.")
        ref = weakref.ref(obj)
        if ref not in self._proj_variables:
            self._proj_variables.append(ref)

    def __rmatmul__(self, other: BaseProjVariable) -> BaseProjVariable:
        self.attach_proj_variable(other)
        return other

    def clear_dead(self) -> None:
        self._proj_variables[:] = [ref for ref in self._proj_variables if ref() is not None]

    def filter(self, rule: dict[str, dict] | None = None) -> list[BaseProjVariable]:
        alive = (v for ref in self._proj_variables if (v := ref()) is not None)
        if rule is None:
            return list(alive)
        return [v for v in alive if self._is_kept(v, rule)]

    @classmethod
    def _is_kept(cls, v: BaseProjVariable, rule: dict[str, dict]) -> bool:
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
    def iter_label_result(cls, v: BaseProjVariable, *, dim_joiner: str = ":", dim_bracket: tuple[str, str] = ("[", "]")
                          ) -> Iterator[tuple[str, float | np.ndarray | None]]:
        if len(v.dims) == 0:
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

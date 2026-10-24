import numpy as np
from collections import defaultdict
from typing import overload


class ObjList[T](list[T]):
    """A subclass of list, implement convenience methods for sum attr.
    """

    @overload
    def sum_attr(self, attr_name: str, /, *, allow_not_exist: bool = False) -> float:
        ...

    @overload
    def sum_attr(self, attr_name: list[str], /, *, allow_not_exist: bool = False) -> np.ndarray:
        ...

    @overload
    def sum_attr(self, attr_name: str, /, *, group_by: str, group_keys: str | list[str] | None = None,
                 allow_not_exist: bool = False) -> dict[str, float]:
        ...

    @overload
    def sum_attr(self, attr_name: list[str], /, *, group_by: str, group_keys: str | list[str] | None = None,
                 allow_not_exist: bool = False) -> dict[str, np.ndarray]:
        ...

    def sum_attr(
        self,
        attr_name: str | list[str],
        /,
        *,
        group_by: str | None = None,
        group_keys: str | list[str] | None = None,
        allow_not_exist: bool = False,
    ) -> float | np.ndarray | dict[str, float | np.ndarray]:
        if group_by is None:
            return _sum_attr(list(self), attr_name=attr_name, allow_not_exist=allow_not_exist)
        else:
            return _sum_attr_by_group(list(self), attr_name=attr_name, group_by=group_by, group_keys=group_keys,
                                      allow_not_exist=allow_not_exist)


def _require_attr(obj_list, attr_name: str | list[str]) -> None:
    attr_name = attr_name if isinstance(attr_name, list) else [attr_name]
    for name in attr_name:
        for obj in obj_list:
            if not hasattr(obj, name):
                raise AttributeError(f"'{type(obj).__name__}' has no attribute '{name}'.")

@overload
def _sum_attr(obj_list: list, *, attr_name: str, allow_not_exist: bool = False) -> float:
    ...

@overload
def _sum_attr(obj_list: list, *, attr_name: list[str], allow_not_exist: bool = False) -> np.ndarray:
    ...

def _sum_attr(obj_list: list, *, attr_name: str | list[str], allow_not_exist: bool = False) -> float | np.ndarray:
    if not isinstance(attr_name, (str, list)):
        raise TypeError(f"Invalid type of name {type(attr_name)}, expected 'str' or 'list[str]'.")
    if not allow_not_exist:
        _require_attr(obj_list, attr_name)
    if isinstance(attr_name, str):
        return sum(getattr(obj, attr_name, 0.0) for obj in obj_list)
    else: # isinstance(attr_name, list)
        n_name: int = len(attr_name)
        total: list[float] = [0.0] * n_name
        for i in range(n_name):
            total[i] = sum(getattr(obj, attr_name[i], 0.0) for obj in obj_list)
        return np.array(total, dtype=float)


@overload
def _sum_attr_by_group(obj_list: list, *, attr_name: str, group_by: str, group_keys: str | list[str] | None = None,
                       allow_not_exist: bool = False) -> dict[str, float]:
    ...

@overload
def _sum_attr_by_group(obj_list: list, *, attr_name: list[str], group_by: str, group_keys: str | list[str] | None = None,
                       allow_not_exist: bool = False) -> dict[str, np.ndarray]:
    ...

def _sum_attr_by_group(obj_list: list, *, attr_name: str | list[str], group_by: str,
                       group_keys: str | list[str] | None = None, allow_not_exist: bool = False,
                       ) -> dict[str, float | np.ndarray]:
    """
    Returns:
        dict[str, float]: if type of `attr_name` is str
        dict[str, np.ndarray]: if type of `attr_name` is list[str]
    """
    if not isinstance(group_by, str):
        raise TypeError(f"Invalid type of 'group_by': {type(group_by)}, expected 'str'.")
    if not isinstance(attr_name, (str, list)):
        raise TypeError(f"Invalid type of name {type(attr_name)}, expected 'str' or 'list[str]'.")
    if group_keys is None:
        filtered = obj_list
    elif isinstance(group_keys, str):
        filtered = [x for x in obj_list if getattr(x, group_by) == group_keys]
    elif isinstance(group_keys, list):
        filtered = [x for x in obj_list if getattr(x, group_by) in group_keys]
    else:
        raise TypeError(f"Invalid type of 'group_keys': {type(group_keys)}, expected 'str' or 'list[str]'.")

    if not allow_not_exist:
        _require_attr(filtered, attr_name)

    if isinstance(attr_name, str):
        totals: defaultdict[str, float] = defaultdict(float)
        for obj in filtered:
            key = getattr(obj, group_by)
            val = getattr(obj, attr_name, 0.0)
            totals[key] += val
        return dict(totals)  # dict[str, float]
    else:  # isinstance(attr_name, list)
        n_name = len(attr_name)
        totals: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_name))
        for obj in filtered:
            key = getattr(obj, group_by)
            for i, name in enumerate(attr_name):
                val = getattr(obj, name, 0.0)
                totals[key][i] += val
        return dict(totals)  # dict[str, np.ndarray]

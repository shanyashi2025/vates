import numpy as np
from collections import defaultdict
from typing import overload


class ObjList[T](list[T]):
    """A subclass of list, implement convenience methods for sum attr.
    """

    @overload
    def sum_attr(self, name: str, /, *, allow_not_exist: bool = False) -> float:
        ...

    @overload
    def sum_attr(self, name: list[str], /, *, allow_not_exist: bool = False) -> np.ndarray:
        ...

    @overload
    def sum_attr(self, name: str, /, *, group_by: str, filter_in: str | list[str] | None = None,
                 allow_not_exist: bool = False) -> dict[str, float]:
        ...

    @overload
    def sum_attr(self, name: list[str], /, *, group_by: str, filter_in: str | list[str] | None = None,
                 allow_not_exist: bool = False) -> dict[str, np.ndarray]:
        ...

    def sum_attr(
        self,
        name: str | list[str],
        /,
        *,
        group_by: str | None = None,
        filter_in: str | list[str] | None = None,
        allow_not_exist: bool = False,
    ) -> float | np.ndarray | dict[str, float | np.ndarray]:
        if group_by is None:
            return sum_attr(list(self), attr_name=name, allow_not_exist=allow_not_exist)
        else:
            return sum_attr_group_by(list(self), attr_name=name, group_by=group_by, filter_in=filter_in,
                                     allow_not_exist=allow_not_exist)


@overload
def sum_attr(obj_list: list, *, attr_name: str, allow_not_exist: bool = False) -> float:
    ...

@overload
def sum_attr(obj_list: list, *, attr_name: list[str], allow_not_exist: bool = False) -> np.ndarray:
    ...

def sum_attr(obj_list: list, *, attr_name: str | list[str], allow_not_exist: bool = False) -> float | np.ndarray:
    default = 0.0 if allow_not_exist else None
    if isinstance(attr_name, str):
        return sum(getattr(obj, attr_name, default) for obj in obj_list)
    elif isinstance(attr_name, list):
        n_name: int = len(attr_name)
        total: list[float] = [0.0] * n_name
        for i in range(n_name):
            total[i] = sum(getattr(obj, attr_name[i], default) for obj in obj_list)
        return np.array(total, dtype=float)
    else:
        raise TypeError(f"Invalid {type(attr_name)=}, expected 'str' or 'list[str]'.")


@overload
def sum_attr_group_by(obj_list: list, *, attr_name: str, group_by: str, filter_in: str | list[str] | None = None,
                      allow_not_exist: bool = False) -> dict[str, float]:
    ...

@overload
def sum_attr_group_by(obj_list: list, *, attr_name: list[str], group_by: str, filter_in: str | list[str] | None = None,
                      allow_not_exist: bool = False) -> dict[str, np.ndarray]:
    ...

def sum_attr_group_by(obj_list: list, *, attr_name: str | list[str], group_by: str,
                      filter_in: str | list[str] | None = None, allow_not_exist: bool = False,
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
    default = 0.0 if allow_not_exist else None
    if filter_in is None:
        filtered = obj_list
    elif isinstance(filter_in, str):
        filtered = filter(lambda x: getattr(x, group_by) == filter_in, obj_list)
    elif isinstance(filter_in, list):
        filtered = filter(lambda x: getattr(x, group_by) in filter_in, obj_list)
    else:
        raise TypeError(f"Invalid type of 'slice_key': {type(filter_in)}, expected 'str' or 'list[str]'.")

    if isinstance(attr_name, str):
        totals: defaultdict[str, float] = defaultdict(float)
        for obj in filtered:
            key = getattr(obj, group_by)
            val = getattr(obj, attr_name, default)
            totals[key] += val
        return dict(totals)  # dict[str, float]
    else:  # isinstance(attr_name, list)
        n_name = len(attr_name)
        totals: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_name))
        for obj in filtered:
            key = getattr(obj, group_by)
            for i in range(n_name):
                val = getattr(obj, attr_name[i], default)
                totals[key][i] += val
        return dict(totals)  # dict[str, np.ndarray]

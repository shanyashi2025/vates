import numpy as np
from typing import Self, overload


class Connector[T]:
    """A wrapper of a list, convenience for get attr.
    """
    __slots__ = ('_objs', )

    def __init__(self, objs: list[T] | None = None):
        self._objs: list[T] = objs or []

    @property
    def objs(self) -> list[T]:
        return self._objs

    def append(self, item: T, /) -> None:
        self._objs.append(item)

    def extend(self, item: list[T] | Self, /) -> None:
        if isinstance(item, list):
            self._objs.extend(item)
        elif isinstance(item, Connector):
            self._objs.extend(item._objs)

    def count(self, value) -> int:
        return self._objs.count(value)

    def __len__(self) -> int:
        return len(self._objs)

    def __getattr__(self, item):
        return getattr(self._objs, item)

    @overload
    def sum_attr(self, name: str, /, *, allow_not_exist: bool = False) -> float:
        ...

    @overload
    def sum_attr(self, name: list[str], /, *, allow_not_exist: bool = False) -> np.ndarray:
        ...

    @overload
    def sum_attr(self, name: str, /, *, group_by: str, slice_key: str | list[str] | None = None,
                 allow_not_exist: bool = False) -> dict[str, float]:
        ...

    @overload
    def sum_attr(self, name: list[str], /, *, group_by: str, slice_key: str | list[str] | None = None,
                 allow_not_exist: bool = False) -> dict[str, np.ndarray]:
        ...

    def sum_attr(
        self,
        name: str | list[str],
        /,
        *,
        group_by: str | None = None,
        slice_key: str | list[str] | None = None,
        allow_not_exist: bool = False,
    ) -> float | np.ndarray | dict[str, float | np.ndarray]:
        if group_by is None:
            return sum_attr(self.objs, attr_name=name, allow_not_exist=allow_not_exist)
        else:
            return sum_attr_group_by(self.objs, attr_name=name, group_by=group_by, slice_key=slice_key,
                                     allow_not_exist=allow_not_exist)

    def __contains__(self, item) -> bool:
        return item in self.objs

    def __iter__(self):
        return iter(self.objs)


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
def sum_attr_group_by(obj_list: list, *, attr_name: str, group_by: str, slice_key: str | list[str] | None = None,
                      allow_not_exist: bool = False) -> dict[str, float]:
    ...

@overload
def sum_attr_group_by(obj_list: list, *, attr_name: list[str], group_by: str, slice_key: str | list[str] | None = None,
                      allow_not_exist: bool = False) -> dict[str, np.ndarray]:
    ...

def sum_attr_group_by(obj_list: list, *, attr_name: str | list[str], group_by: str,
                      slice_key: str | list[str] | None = None, allow_not_exist: bool = False,
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
    if slice_key is not None and not isinstance(slice_key, (str, list)):
        raise TypeError(f"Invalid type of 'slice_key': {type(slice_key)}, expected 'str' or 'list[str]'.")
    default = 0.0 if allow_not_exist else None

    if isinstance(attr_name, str):
        # --- path 1.1 ---
        if isinstance(slice_key, str):
            return {group_by: sum(getattr(obj, attr_name, default) for obj in obj_list if
                                  getattr(obj, group_by) == slice_key)}  # dict[str, float]
        # --- path 1.2 ---
        if isinstance(slice_key, list):
            total: dict[str, float] = {key: 0.0 for key in slice_key}
            for obj in obj_list:
                key = getattr(obj, group_by)
                if key in total:
                    val = getattr(obj, attr_name, default)
                    total[key] += val
            return total  # dict[str, float]
        # --- path 1.3 ---
        if slice_key is None:
            total: dict[str, float] = {}
            for obj in obj_list:
                key = getattr(obj, group_by)
                val = getattr(obj, attr_name, default)
                if key in total:
                    total[key] += val
                else:
                    total[key] = val
            return total  # dict[str, float]
        raise ValueError(f"Invalid type of 'slice_key' {type(slice_key)}, expected ('str', 'list[str]', None)")
    else:  # isinstance(attr_name, list)
        n_name = len(attr_name)
        # --- path 2.1 ---
        if isinstance(slice_key, str):
            arr: list[float] = [0.0] * n_name
            for obj in obj_list:
                key = getattr(obj, group_by)
                if key == slice_key:
                    for i in range(n_name):
                        val = getattr(obj, attr_name[i], default)
                        arr[i] += val
            return {group_by: np.array(arr, dtype=float)}  # dict[str, np.ndarray]
        # --- path 2.2 ---
        if isinstance(slice_key, list):
            total: dict[str, list[float]] = {key: [0.0] * n_name for key in slice_key}
            for obj in obj_list:
                key = getattr(obj, group_by)
                if key in total:
                    for i in range(n_name):
                        val = getattr(obj, attr_name[i], default)
                        total[key][i] += val
            return {key: np.array(arr, dtype=float) for key, arr in total.items()}  # dict[str, np.ndarray]
        # --- path 2.3 ---
        if slice_key is None:
            total: dict[str, list[float]] = {}
            for obj in obj_list:
                key = getattr(obj, group_by)
                if key in total:
                    for i in range(n_name):
                        val = getattr(obj, attr_name[i], default)
                        total[key][i] += val
                else:
                    total[key] = [getattr(obj, bas) for bas in attr_name]
            return {key: np.array(val, dtype=float) for key, val in total.items()}  # dict[str, np.ndarray]

        raise ValueError(f"Calculation not defined: {group_by=}, {slice_key=}")

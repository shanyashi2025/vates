import pandas as pd
import weakref
import warnings
from typing import Literal


class ValidatedBool:
    """Descriptor for bool"""
    def __init__(self, max_sets=None):
        self.max_sets = max_sets if isinstance(max_sets, (int, float)) else None
        self._set_counts = weakref.WeakKeyDictionary()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        set_count = self._set_counts.get(instance, 0)
        if self.max_sets is not None and set_count >= self.max_sets:
            warnings.warn(f"{self.name} can only be modified up to {self.max_sets} times.")
            return
        if not isinstance(value, bool):
            raise TypeError(f"{self.name}: type '{type(value)}' is not allowed, expected 'bool'.")
        instance.__dict__[self.name] = value
        self._set_counts[instance] = set_count + 1

class ValidatedNumber:
    """Descriptor for number (int and float)"""
    def __init__(self, value_type=(float, int), value_min: float | int | None=None, value_max: float | int | None=None,
                 value_lst: list | tuple | range | None=None, allow_none: bool=False, max_sets=None):
        if isinstance(value_type, tuple):
            self.value_type = value_type
        elif isinstance(value_type, list):
            self.value_type = tuple(value_type)
        else:
            self.value_type = (value_type,)
        self.value_min: float | int | None = value_min
        self.value_max: float | int | None = value_max
        self.value_lst: list | tuple | range | None = value_lst
        self.allow_none: bool = allow_none
        self.max_sets = max_sets if isinstance(max_sets, (int, float)) else None
        self._set_counts = weakref.WeakKeyDictionary()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        set_count = self._set_counts.get(instance, 0)
        if self.max_sets is not None and set_count >= self.max_sets:
            warnings.warn(f"{self.name} can only be modified up to {self.max_sets} times.")
            return
        if value is None:
            if not self.allow_none:
                raise TypeError(f"{self.name}: None is not an allowd value.")
        else:
            if not isinstance(value, self.value_type):
                raise TypeError(f"{self.name}: type '{type(value)}' is not allowed, expected in {self.value_type}.")
            if self.value_min is not None and value < self.value_min:
                raise ValueError(f"{self.name}: {value=}, expected >={self.value_min}.")
            if self.value_max is not None and value > self.value_max:
                raise ValueError(f"{self.name}: {value=}, expected <={self.value_max}.")
            if self.value_lst is not None and value not in self.value_lst:
                raise ValueError(f"{self.name}: {value=}, expected in {self.value_lst}.")
        instance.__dict__[self.name] = value
        self._set_counts[instance] = set_count + 1

class ValidatedString:
    """Descriptor for string"""
    def __init__(self, len_min: int | None=None, len_max: int | None=None, str_literal: list | tuple | None=None,
                 allow_none: bool=False, max_sets=None):
        self.len_min: int | None = len_min
        self.len_max: int | None = len_max
        self.str_literal: list | tuple | None = str_literal
        self.allow_none: bool = allow_none
        self.max_sets = max_sets if isinstance(max_sets, (int, float)) else None
        self._set_counts = weakref.WeakKeyDictionary()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        set_count = self._set_counts.get(instance, 0)
        if self.max_sets is not None and set_count >= self.max_sets:
            warnings.warn(f"{self.name} can only be modified up to {self.max_sets} times.")
            return
        if value is None:
            if not self.allow_none:
                raise TypeError(f"{self.name}: None is not an allowd value.")
        else:
            if not isinstance(value, str):
                raise TypeError(f"{self.name}: type '{type(value)}' is not allowed, expected 'str'.")
            if self.len_min is not None and len(value) < self.len_min:
                raise ValueError(f"{self.name}: {len(value)=}, expected >={self.len_min}.")
            if self.len_max is not None and len(value) > self.len_max:
                raise ValueError(f"{self.name}: {len(value)=}, expected <={self.len_max}.")
            if self.str_literal is not None and value not in self.str_literal:
                raise ValueError(f"{self.name}: {value=}, expected in {self.str_literal}.")
        instance.__dict__[self.name] = value
        self._set_counts[instance] = set_count + 1

class ValidatedPeriod:
    """Descriptor for (pandas) period """
    def __init__(self, value_min: pd.Period | None=None, value_max: pd.Period | None=None, allow_none: bool=False,
                 max_sets=None):
        self.value_min: pd.Period | None = value_min
        self.value_max: pd.Period | None = value_max
        self.allow_none: bool = allow_none
        self.max_sets = max_sets if isinstance(max_sets, (int, float)) else None
        self._set_counts = weakref.WeakKeyDictionary()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        set_count = self._set_counts.get(instance, 0)
        if self.max_sets is not None and set_count >= self.max_sets:
            warnings.warn(f"{self.name} can only be modified up to {self.max_sets} times.")
            return
        if value is None:
            if not self.allow_none:
                raise TypeError(f"{self.name}: None is not an allowd value.")
        else:
            if not isinstance(value, pd.Period):
                raise TypeError(f"{self.name}: type '{type(value)}' is not allowed, expected pandas.Period.")
            if self.value_min is not None and value < self.value_min:
                raise ValueError(f"{self.name}: {value=}, expected >={self.value_min}.")
            if self.value_max is not None and value > self.value_max:
                raise ValueError(f"{self.name}: {value=}, expected <={self.value_max}.")
        instance.__dict__[self.name] = value
        self._set_counts[instance] = set_count + 1

class ValidatedList:
    """Descriptor for list"""
    def __init__(self, len_min: int | None=None, len_max: int | None=None, item_type=None, allow_none: bool=False,
                 max_sets=None):
        self.len_min: int | None = len_min
        self.len_max: int | None = len_max
        if item_type is None:
            self.item_type = None
        elif isinstance(item_type, (list, tuple)):
            self.item_type = item_type
        else:
            self.item_type = [item_type]
        self.allow_none: bool = allow_none
        self.max_sets = max_sets if isinstance(max_sets, (int, float)) else None
        self._set_counts = weakref.WeakKeyDictionary()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        set_count = self._set_counts.get(instance, 0)
        if self.max_sets is not None and set_count >= self.max_sets:
            warnings.warn(f"{self.name} can only be modified up to {self.max_sets} times.")
            return
        if value is None:
            if not self.allow_none:
                raise TypeError(f"{self.name}: None is not an allowd value.")
        else:
            if not isinstance(value, list):
                raise TypeError(f"{self.name}: type '{type(value)}' is not allowed, expected 'list'.")
            if self.len_min is not None and len(value) < self.len_min:
                raise ValueError(f"{self.name}: {len(value)=}, expected >={self.len_min}.")
            if self.len_max is not None and len(value) > self.len_max:
                raise ValueError(f"{self.name}: {len(value)=}, expected <={self.len_max}.")
            if self.item_type is not None:
                for item in value:
                    if type(item) not in self.item_type:
                        raise ValueError(f"{self.name} <item: {item}>: type {type(item)} is not allowed, expected in "
                                         f"{self.item_type}.")
        instance.__dict__[self.name] = value
        self._set_counts[instance] = set_count + 1

def parse_str_to_int_list(str_in: str, separator: str=',', joinner: str= '-',
                          sort_list: Literal[None, 'ascending', 'asc', 'descending', 'desc']=None,
                          handler: Literal['keep', 'remove', 'error']= 'error') -> list[int]:
    """Parse string to a list of non-negative integers"""
    if not isinstance(str_in, str):
        raise TypeError(f"{str_in}: type {type(str_in)} is not allowed, expected 'str'.")

    import re
    int_lst = []
    parts = str_in.replace(' ', '').split(separator)

    for part in parts:
        if not part:
            pass
        elif re.fullmatch(rf'\d+\s*{joinner}\s*\d+', part):  # range like "1-5"
            s1, s2 = re.split(rf'\s*{joinner}\s*', part)
            n1, n2 = int(s1), int(s2)
            n1, n2 = min(n1, n2), max(n1, n2)
            int_lst.extend(range(n1, n2 + 1))
        elif re.fullmatch(r'\d+', part):  # single integer
            n = int(part)
            int_lst.append(n)
        else:
            raise ValueError(f"{part} is not allowed, expected non-negative integer, separated by '{separator}' "
                             f"and/or a range joined by '{joinner}'.")

    if sort_list is None:
        pass
    elif sort_list.lower() in ('ascending', 'asc'):
        int_lst.sort()
    elif sort_list.lower() in ('descending', 'desc'):
        int_lst.sort(reverse=True)

    int_set = set(int_lst)
    if len(int_lst) != len(int_set):
        handler = handler.lower()
        if handler == 'keep':
            pass
        elif handler == 'remove':
            int_lst = list(int_set)
        elif handler == 'error':
            dup_lst = [x for x in int_set if int_lst.count(x) > 1]
            raise ValueError(f"Duplicate entries in '{str_in}': e.g. {dup_lst[:min(5, len(dup_lst))]}. "
                             f"Revise the input, or specify duplicate_handler='keep' or 'remove'.")

    return int_lst

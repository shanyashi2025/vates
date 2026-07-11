from dataclasses import dataclass
from typing import Literal, Any


@dataclass(frozen=True)
class RunConfig:
    start_year: int | None
    start_month: int | None
    end_year: int | None
    end_month: int | None
    scenario: str | None
    simulations: list[int] | None
    simulation: int | None
    wsdir: str
    input_directories: list[str] | None
    results_directory: str
    is_delete_existing_results: bool
    enable_write_proj_result: bool
    stoch_result_file_mode: Literal['w', 'a', None]
    stoch_result_file_id: str | None
    enable_write_runlog: bool
    max_workers: int | None = None

    def __post_init__(self):
        if self.start_year is None and self.start_month is None and self.end_year is None and self.end_month is None:
            pass  # bypass if all dates are `None`
        else:
            self.validate_number("start_year", self.start_year, value_type=int, value_min=1900, value_max=5999)
            self.validate_number("start_month", self.start_month, value_type=int, value_lst=range(1, 13))
            self.validate_number("end_year", self.end_year, value_type=int, value_min=1900, value_max=5999)
            self.validate_number("end_month", self.end_month, value_type=int, value_lst=range(1, 13))
            if self.end_year * 12 + self.end_month < self.start_year * 12 + self.start_month:
                raise ValueError(f"end_date ('{self.end_year}-{self.end_month}') < start_date ('{self.start_year}-{self.start_month}')")
        self.validate_string("scenario", self.scenario, allow_none=True)
        self.validate_string("workspace_directory", self.wsdir)
        self.validate_list("input_directories", self.input_directories, item_type=str, allow_none=True)
        self.validate_string("results_directory", self.results_directory)
        self.validate_bool("is_delete_existing_results", self.is_delete_existing_results)
        self.validate_bool("enable_write_proj_result", self.enable_write_proj_result)
        self.validate_string("stoch_result_file_mode", self.stoch_result_file_mode, str_literal=['w', 'a'], allow_none=True)
        self.validate_string("stoch_result_file_id", self.stoch_result_file_id, allow_none=True)
        self.validate_bool("enable_write_runlog", self.enable_write_runlog)
        self.validate_number("simulation", self.simulation, value_type=int, value_min=1, value_max=100_000, allow_none=True)
        self.validate_list("simulations", self.simulations, item_type=int, len_min=0, len_max=100_000, allow_none=True)
        self.validate_number("max_workers", self.max_workers, value_type=int, value_min=1, value_max=999, allow_none=True)

    @staticmethod
    def validate_bool(name: str, value, /) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected 'bool'.")

    @staticmethod
    def validate_number(name: str, value, /, *,
                        value_type=(float, int), value_min: float | int | None = None,
                        value_max: float | int | None = None, value_lst: list | tuple | range | None = None,
                        allow_none: bool = False) -> None:
        if isinstance(value_type, tuple):
            pass
        elif isinstance(value_type, list):
            value_type = tuple(value_type)
        else:
            value_type = (value_type,)

        if value is None:
            if not allow_none:
                raise TypeError(f"{name}: value 'None' is not allowd.")
            return

        if not isinstance(value, value_type):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected in {value_type}.")
        if value_min is not None and value < value_min:
            raise ValueError(f"{name}: {value=}, expected >={value_min}.")
        if value_max is not None and value > value_max:
            raise ValueError(f"{name}: {value=}, expected <={value_max}.")
        if value_lst is not None and value not in value_lst:
            raise ValueError(f"{name}: {value=}, expected in {value_lst}.")

    @staticmethod
    def validate_string(name: str, value, /, *,
                        len_min: int = 0, len_max: int | None = None, str_literal: list | tuple | None = None,
                        allow_none: bool = False) -> None:
        if value is None:
            if not allow_none:
                raise TypeError(f"{name}: value 'None' is not allowd.")
            return

        if not isinstance(value, str):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected 'str'.")
        if len(value) < len_min:
            raise ValueError(f"{name}: {len(value)=}, expected >={len_min}.")
        if len_max is not None and len(value) > len_max:
            raise ValueError(f"{name}: {len(value)=}, expected <={len_max}.")
        if str_literal is not None and value not in str_literal:
            raise ValueError(f"{name}: {value=}, expected in {str_literal}.")

    @staticmethod
    def validate_list(name: str, value, /, *,
                      len_min: int | None = None, len_max: int | None = None, item_type = Any, allow_none: bool = False,
                      ) -> None:
        if item_type is Any:
            pass
        elif isinstance(item_type, tuple):
            pass
        elif isinstance(item_type, list):
            item_type = tuple(item_type)
        else:
            item_type = (item_type,)

        if value is None:
            if not allow_none:
                raise TypeError(f"{name}: value 'None' is not allowd.")
            return

        if not isinstance(value, list):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected 'list'.")
        if len_min is not None and len(value) < len_min:
            raise ValueError(f"{name}: {len(value)=}, expected >={len_min}.")
        if len_max is not None and len(value) > len_max:
            raise ValueError(f"{name}: {len(value)=}, expected <={len_max}.")
        if item_type is not Any:
            for item in value:
                if not isinstance(item, item_type):
                    raise ValueError(f"{name}: item '{item}' type '{type(item)}' is not allowed, expected in {item_type}.")


def parse_str_to_int_list(str_in: str, separator: str = ',', joiner: str = '-',
                          sort_list: Literal[None, 'ascending', 'asc', 'descending', 'desc'] = None,
                          handler: Literal['keep', 'remove', 'error'] = 'error') -> list[int]:
    """Parse string to a list of non-negative integers"""
    if not isinstance(str_in, str):
        raise TypeError(f"{str_in}: type {type(str_in)} is not allowed, expected 'str'.")

    import re
    int_lst = []
    parts = str_in.replace(' ', '').split(separator)

    for part in parts:
        if not part:
            pass
        elif re.fullmatch(rf'\d+\s*{joiner}\s*\d+', part):  # range like "1-5"
            s1, s2 = re.split(rf'\s*{joiner}\s*', part)
            n1, n2 = int(s1), int(s2)
            n1, n2 = min(n1, n2), max(n1, n2)
            int_lst.extend(range(n1, n2 + 1))
        elif re.fullmatch(r'\d+', part):  # single integer
            n = int(part)
            int_lst.append(n)
        else:
            raise ValueError(f"{part} is not allowed, expected non-negative integer, separated by '{separator}' "
                             f"and/or a range joined by '{joiner}'.")

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

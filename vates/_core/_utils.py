import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any, Self


@dataclass(frozen=True, slots=True)
class RunConfig:
    start_year: int | None
    start_month: int | None
    start_date: pd.Period | None
    end_year: int | None
    end_month: int | None
    end_date: pd.Period | None
    max_t: int
    scenario: str | None
    simulations: list[int] | None
    simulation: int | None
    workspace_directory: str
    workspace_directory_path: Path
    input_directories: list[str] | None
    results_directory: str
    results_directory_path: Path
    is_delete_existing_results: bool
    enable_write_proj_result: bool
    stoch_result_file_mode: Literal['w', 'a', None]
    stoch_result_file_id: str | None
    enable_write_runlog: bool
    max_workers: int | None = None

    def __post_init__(self):
        if self.end_date is None and self.start_date is None and self.start_year is None and self.start_month is None \
                and self.end_year is None and self.end_month is None:
            pass  # bypass validations of dates
        else:
            self.validate_number("start_year", self.start_year, value_type=int, value_min=1900, value_max=5999)
            self.validate_number("start_month", self.start_month, value_type=int, value_lst=range(1, 13))
            self.validate_number("end_year", self.end_year, value_type=int, value_min=1900, value_max=5999)
            self.validate_number("end_month", self.end_month, value_type=int, value_lst=range(1, 13))
            self.validate_number("max_t", self.max_t, value_type=int, value_min=0, value_max=2400)
            self.validate_period("start_date", self.start_date, value_min=pd.Period("1900-1", freq="M"),
                                 value_max=self.end_date)
            self.validate_period("end_date", self.end_date, value_min=self.start_date,
                                 value_max=pd.Period("5999-12", freq="M"))
            if (self.end_date - self.start_date).n != self.max_t:
                raise ValueError(f"Number of months from start_date ({self.start_date}) to end_date {self.end_date} is "
                                 f"inconsistent with max_t {self.max_t}.")

        self.validate_string("scenario", self.scenario, allow_none=True)
        self.validate_list("simulations", self.simulations, item_type=int, len_min=0, len_max=100_000, allow_none=True)
        self.validate_number("simulation", self.simulation, value_type=int, value_min=1, value_max=100_000, allow_none=True)
        self.validate_string("workspace_directory", self.workspace_directory)
        self.validate_path("workspace_directory_path", self.workspace_directory_path)
        self.validate_list("input_directories", self.input_directories, item_type=str, allow_none=True)
        self.validate_string("results_directory", self.results_directory)
        self.validate_path("results_directory_path", self.results_directory_path)
        self.validate_bool("is_delete_existing_results", self.is_delete_existing_results)
        self.validate_bool("enable_write_proj_result", self.enable_write_proj_result)
        self.validate_string("stoch_result_file_mode", self.stoch_result_file_mode, str_literal=['w', 'a'], allow_none=True)
        self.validate_string("stoch_result_file_id", self.stoch_result_file_id, allow_none=True)
        self.validate_bool("enable_write_runlog", self.enable_write_runlog)
        self.validate_number("max_workers", self.max_workers, value_type=int, value_min=1, value_max=999, allow_none=True)

    @classmethod
    def create(
        cls,
        *,
        start_year: int | None,
        start_month: int | None,
        end_year: int | None,
        end_month: int | None,
        scenario: str | None,
        simulations: str | list[int] | None = None,
        simulation: int | None = None,
        workspace_directory: str,
        input_directories: list[str] | None = None,
        results_directory: str,
        is_delete_existing_results: bool,
        enable_write_proj_result: bool,
        stoch_result_file_mode: Literal['w', 'a', None] = None,
        stoch_result_file_id: str | None = None,
        enable_write_runlog: bool,
        max_workers: int | None = None
    ) -> Self:

        if start_year is None and start_month is None and end_year is None and end_month is None:
            start_date = None
            end_date = None
            max_t = 0
        else:
            start_date = pd.Period(f'{start_year}-{start_month}', freq='M')
            end_date = pd.Period(f'{end_year}-{end_month}', freq='M')
            max_t = (end_date - start_date).n

        if isinstance(simulations, str):
            simulations = parse_str_to_int_list(simulations)

        workspace_directory_path = Path(workspace_directory).resolve()
        results_directory_path = (workspace_directory_path / results_directory).resolve()

        return cls(
            start_year=start_year,
            start_month=start_month,
            start_date=start_date,
            end_year=end_year,
            end_month=end_month,
            end_date=end_date,
            max_t=max_t,
            scenario=scenario,
            simulations=simulations,
            simulation=simulation,
            workspace_directory=workspace_directory,
            workspace_directory_path=workspace_directory_path,
            input_directories=input_directories,
            results_directory=results_directory,
            results_directory_path=results_directory_path,
            is_delete_existing_results=is_delete_existing_results,
            enable_write_proj_result=enable_write_proj_result,
            stoch_result_file_mode=stoch_result_file_mode,
            stoch_result_file_id=stoch_result_file_id,
            enable_write_runlog=enable_write_runlog,
            max_workers=max_workers,
        )

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
    def validate_period(name: str, value, /, *,
                        value_min: pd.Period | None = None, value_max: pd.Period | None = None, allow_none: bool = False
                        ) -> None:
        if value is None:
            if not allow_none:
                raise TypeError(f"{name}: value 'None' is not allowd.")
            return

        if not isinstance(value, pd.Period):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected 'Period'.")
        if value_min is not None and value < value_min:
            raise ValueError(f"{name}: {value=}, expected >={value_min}.")
        if value_max is not None and value > value_max:
            raise ValueError(f"{name}: {value=}, expected <={value_max}.")

    @staticmethod
    def validate_path(name: str, value, /, *,
                      dir_or_file: str | None = None, must_exist: bool = False, allow_none: bool = False) -> None:
        if value is None:
            if not allow_none:
                raise TypeError(f"{name}: value 'None' is not allowd.")
            return

        if not isinstance(value, Path):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected 'Path'.")

        if must_exist:
            if not value.exists():
                raise ValueError(f"Path {name} does not exist.")
            if dir_or_file == "dir" and not value.is_file():
                raise ValueError(f"Path {name} is not a directory.")
            elif dir_or_file == "file":
                raise ValueError(f"Path {name} is not a file.")

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


def parse_str_to_int_list(str_in: str, /, *, separator: str = ',', joiner: str = '-',
                          sort_list: Literal[None, 'ascending', 'asc', 'descending', 'desc'] = None,
                          on_duplicate: Literal['keep', 'remove', 'error'] = 'error') -> list[int]:
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
        on_duplicate = on_duplicate.lower()
        if on_duplicate == 'keep':
            pass
        elif on_duplicate == 'remove':
            int_lst = list(int_set)
        elif on_duplicate == 'error':
            dup_lst = [x for x in int_set if int_lst.count(x) > 1]
            raise ValueError(f"Duplicate entries in '{str_in}': e.g. {dup_lst[:min(5, len(dup_lst))]}. "
                             f"Revise the input, or specify duplicate_handler='keep' or 'remove'.")

    return int_lst

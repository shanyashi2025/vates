import pandas as pd
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True)
class RunConfiguration:
    start_date: pd.Period
    end_date: pd.Period
    scenario: str
    simulations: list[int] | tuple[int, ...]
    simulation: int
    workspace_directory_path: Path
    input_directory_paths: list[Path] | tuple[Path, ...]
    results_directory_path: Path
    is_delete_existing_results: bool
    is_write_proj_result: bool
    is_write_stoch_result: bool
    stoch_result_file_id: str
    is_write_runlog: bool
    max_workers: int

    def __post_init__(self):
        self.validate_period("start_date", self.start_date, value_min=pd.Period("1900-1", freq="M"),
                             value_max=self.end_date)
        self.validate_period("end_date", self.end_date, value_min=self.start_date,
                             value_max=pd.Period("5999-12", freq="M"))
        self.validate_str("scenario", self.scenario)
        self.validate_list_tuple("simulations", self.simulations, item_type=int, len_min=0, len_max=100_000)
        self.validate_int("simulation", self.simulation, value_min=0, value_max=100_000)
        self.validate_path("workspace_directory_path", self.workspace_directory_path, must_absolute=True)
        self.validate_list_tuple("input_directory_paths", self.input_directory_paths, item_type=Path)
        for d in self.input_directory_paths:
            self.validate_path("input_directory", d, must_absolute=True)
        self.validate_path("results_directory_path", self.results_directory_path, must_absolute=True)
        self.validate_bool("is_delete_existing_results", self.is_delete_existing_results)
        self.validate_bool("enable_write_proj_result", self.is_write_proj_result)
        self.validate_bool("enable_write_stoch_result", self.is_write_stoch_result)
        self.validate_str("stoch_result_file_id", self.stoch_result_file_id)
        self.validate_bool("enable_write_runlog", self.is_write_runlog)
        self.validate_int("max_workers", self.max_workers, value_min=1, value_max=999)

    @cached_property
    def max_t(self) -> int:
        return (self.end_date - self.start_date).n

    @classmethod
    def make_for_proj(cls, **kwargs) -> Self:
        unused = dict(simulations = (), max_workers = 1)
        return cls(**(unused | kwargs))

    @classmethod
    def make_for_stoch(cls, **kwargs) -> Self:
        unused = dict(
            is_delete_existing_results=True,
            is_write_proj_result=False,
            is_write_stoch_result=False,
            stoch_result_file_id="",
            simulation=0,
        )
        return cls(**(unused | kwargs))

    @staticmethod
    def validate_bool(name: str, value, /) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected 'bool'.")

    @staticmethod
    def validate_int(name: str, value, /, *,
                     value_min: int | None = None, value_max: int | None = None,
                     value_lst: list | tuple | range | None = None) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected 'int'.")
        if value_min is not None and value < value_min:
            raise ValueError(f"{name}: {value=}, expected >={value_min}.")
        if value_max is not None and value > value_max:
            raise ValueError(f"{name}: {value=}, expected <={value_max}.")
        if value_lst is not None and value not in value_lst:
            raise ValueError(f"{name}: {value=}, expected in {value_lst}.")

    @staticmethod
    def validate_number(name: str, value, /, *,
                        value_min: float | int | None = None, value_max: float | int | None = None,
                        value_lst: list | tuple | range | None = None) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected ('float', 'int').")
        if value_min is not None and value < value_min:
            raise ValueError(f"{name}: {value=}, expected >={value_min}.")
        if value_max is not None and value > value_max:
            raise ValueError(f"{name}: {value=}, expected <={value_max}.")
        if value_lst is not None and value not in value_lst:
            raise ValueError(f"{name}: {value=}, expected in {value_lst}.")

    @staticmethod
    def validate_str(name: str, value, /, *, len_max: int | None = None,
                     str_literal: list | tuple | None = None) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected 'str'.")
        if len_max is not None and len(value) > len_max:
            raise ValueError(f"{name}: {len(value)=}, expected <={len_max}.")
        if str_literal is not None and value not in str_literal:
            raise ValueError(f"{name}: {value=}, expected in {str_literal}.")

    @staticmethod
    def validate_period(name: str, value: Any, /, *,
                        value_min: pd.Period | None = None, value_max: pd.Period | None = None) -> None:
        if not isinstance(value, pd.Period):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected 'Period'.")
        if value_min is not None and value < value_min:
            raise ValueError(f"{name}: {value=}, expected >={value_min}.")
        if value_max is not None and value > value_max:
            raise ValueError(f"{name}: {value=}, expected <={value_max}.")

    @staticmethod
    def validate_path(name: str, value, /, *, must_absolute: bool = False,
                      dir_or_file: str | None = None, must_exist: bool = False) -> None:
        if not isinstance(value, Path):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected 'Path'.")
        if must_absolute and not value.is_absolute():
            raise ValueError(f"Path {name} must be absolute.")
        if must_exist and not value.exists():
            raise ValueError(f"Path {name} does not exist.")
        # The directory/file type can only be asserted once the path actually exists.
        if value.exists():
            if dir_or_file == "dir" and not value.is_dir():
                raise ValueError(f"Path {name} is not a directory.")
            elif dir_or_file == "file" and not value.is_file():
                raise ValueError(f"Path {name} is not a file.")

    @staticmethod
    def validate_list_tuple(name: str, value, /, *, len_min: int | None = None, len_max: int | None = None,
                            item_type = Any) -> None:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{name}: type '{type(value)}' is not allowed, expected ('list', 'tuple').")
        if len_min is not None and len(value) < len_min:
            raise ValueError(f"{name}: {len(value)=}, expected >={len_min}.")
        if len_max is not None and len(value) > len_max:
            raise ValueError(f"{name}: {len(value)=}, expected <={len_max}.")
        if item_type is not Any:
            for item in value:
                if not isinstance(item, item_type):
                    raise ValueError(f"{name}: item '{item}' type '{type(item)}' is not allowed, expected in {item_type}.")

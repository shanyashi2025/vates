import csv
import gc
import glob
import inspect
import json
import os
import pandas as pd
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Literal, Self, get_type_hints, overload

from vates._core._helper import get_environ, get_f_filename_lineno, search_file_in_dirs
from vates._core._proj_result_reader import proj_result
from vates._core._proj_variables import BaseProjVariable, ProjVariableManager
from vates._core._run_configuration import RunConfiguration
from vates._core._time_synchronizer import ProjectionTimeSynchronizer

_MISSING: object = object()

class ProjModelEngine:
    """Actuarial projection model engine.
    """

    _projection: Callable
    _run_config: RunConfiguration

    def __init__(
        self,
        *,
        slug: str,
        description: str = '...'
    ) -> None:
        """Initialize a projection model engine.

        Args:
            slug (str): The slug of the model.
            description (str, optional): The description of the model.
        """
        self._slug: str = str(slug)
        self._description: str = str(description)

        # runtime stuffs
        self._time_synchronizer: ProjectionTimeSynchronizer = ProjectionTimeSynchronizer()
        self._proj_variable_mgr: ProjVariableManager = ProjVariableManager()
        self._filepath_cache: dict[str, Path | None] = {}
        self._result_files: set = set()
        self._messages: list[str] = []
        self._runlog: dict = {}

        self._initialized: bool = True

    def bind_projection(
        self,
        func: Callable,
        /
    ) -> Self:
        """Bind the projection model to the engine.

        Args:
            func (callable): The projection function.

        Raises:
            ValueError: If `func` is already bound, not callable, or a class.
            ValueError: If the type hints of `func` cannot be resolved (e.g.
                string/forward annotations referencing names that are not
                importable at module scope).
            ValueError: If `func`'s first argument is missing an annotation,
                is annotated with a non-class, or is annotated with a type
                that `self` is not a subclass of.
        """
        if hasattr(self, '_projection'):
            raise ValueError(f"{self._projection} is already bound.")
        if not callable(func):
            raise ValueError(f"Cannot bind un-callable object: {func}.")
        if inspect.isclass(func):
            raise ValueError(f"Cannot bind a class: {func}.")

        sig_params = inspect.signature(func).parameters
        if len(sig_params) == 0:
            f, l = get_f_filename_lineno()
            self._messages.append(f"{f}:{l}: [INFO] Function '{func.__name__}' has no argument, it will be bound as a "
                                  f"function instead of a method.")
            super().__setattr__('_projection', func)
        else:
            try:
                if inspect.isfunction(func) or inspect.ismethod(func):
                    globalns = getattr(func, "__globals__", None)
                else:
                    globalns = getattr(func.__call__, "__globals__", None)
                hints = get_type_hints(func, globalns=globalns)
            except NameError as e:
                raise ValueError(
                    f"Function '{func.__name__}': type hints cannot be resolved. If you use string/forward "
                    f"annotations (e.g. via 'from __future__ import annotations'), referenced names must be "
                    f"importable at module scope. Cause: {e}"
                ) from e

            first_arg_name = list(sig_params.keys())[0]
            if first_arg_name not in hints:
                raise ValueError(
                    f"Function '{func.__name__}' first argument '{first_arg_name}': must annotate with the "
                    f"model engine type, e.g.: '{type(self).__name__}'. String/forward annotations (e.g. via "
                    f"'from __future__ import annotations') are only resolved when the referenced name is "
                    f"importable at module scope."
                )
            annotated = hints[first_arg_name]
            if not isinstance(annotated, type):
                raise ValueError(
                    f"Function '{func.__name__}' first argument '{first_arg_name}': annotation must be a "
                    f"class (the model engine type), got '{annotated}'."
                )
            if not issubclass(type(self), annotated):
                raise ValueError(
                    f"Function '{func.__name__}' first argument '{first_arg_name}': model engine type "
                    f"'{type(self)}' is not a subclass of the annotated type '{annotated}'."
                )
            super().__setattr__('_projection',  MethodType(func, self))

        f, l = get_f_filename_lineno()
        self._messages.append(f"{f}:{l}: [INFO] Function {func} has been bound to {self}.")

        return self

    def configure_run(
        self,
        *,
        start_year: int,
        start_month: int = 12,
        end_year: int | None = None,
        end_month: int = 12,
        scenario: str = "",
        simulation: int = 0,
        workspace_directory: str | Path = Path.cwd(),
        input_directories: list[str | Path] | tuple[str | Path, ...] = (),
        results_directory: str | Path = "",
        is_delete_existing_results: bool = True,
        enable_write_proj_result: bool = True,
        stoch_result_file_mode: Literal['w', 'a', 'n'] = 'n',
        stoch_result_file_id: str = "",
        enable_write_runlog: bool = True,
    ) -> Self:
        """Set the configuration for a run.

        Args:
            start_year (int): Projection start year.
            start_month (int, optional): Projection start month. Defaults to 12.
            end_year (int, optional): Projection end year. Defaults to `start_year`.
            end_month (int, optional): Projection end month. Defaults to 12.
            scenario (str, optional): Scenario. Defaults to empty str `""`.
            simulation (int, optional): Simulation (stochastic). Defaults to 0.
            workspace_directory (str | Path, optional): Workspace directory, must be absolute path. Defaults to `{Path.cwd()}`.
            input_directories (list[str | Path] | tuple[str | Path, ...], optional): List or tuple of input directory,
                relative paths are resolved against `workspace_directory`. Defaults to empty tuple `()`.
            results_directory (str | Path, optional): Results directory, relative paths are resolved against
                `workspace_directory`. Defaults to `{Path.cwd()}`.
            is_delete_existing_results (bool, optional): Delete existing results if any. Defaults to True.
            enable_write_proj_result (bool, optional): Enable writing projection result. Defaults to True.
            stoch_result_file_mode (Literal['w', 'a', 'n'], optional): Stochastic result writer mode. Defaults to "n".
            stoch_result_file_id (str, optional): Stochastic result id. Defaults to empty `""`.
            enable_write_runlog (bool, optional): Enable writing run log. Defaults to True.
        """
        if hasattr(self, '_run_config'):
            raise ValueError(f"Run configuration is already set.")
        if start_year is None:
            raise ValueError(f"start_year: value 'None' is not allowed.")
        start_date = pd.Period(f"{start_year}-{start_month}", freq='M')
        end_date = pd.Period(f"{end_year or start_year}-{end_month}", freq='M')
        workspace_directory_path = Path(workspace_directory)
        super().__setattr__('_run_config', RunConfiguration(
            start_date=start_date,
            end_date=end_date,
            scenario=scenario,
            simulation=simulation,
            workspace_directory_path=workspace_directory_path,
            input_directory_paths=[workspace_directory_path / d for d in input_directories],  # resolve against workspace_directory
            results_directory_path=workspace_directory_path / Path(results_directory),  # resolve against workspace_directory
            is_delete_existing_results=is_delete_existing_results,
            enable_write_proj_result=enable_write_proj_result,
            stoch_result_file_mode=stoch_result_file_mode,
            stoch_result_file_id=stoch_result_file_id,
            enable_write_runlog=enable_write_runlog,
            simulations=(),
            max_workers=1,
        ))
        return self

    def run(
        self,
        *,
        projection_args: dict[str, Any] | None = None,
    ) -> Self:
        if not hasattr(self, '_projection'):
            raise ValueError(f"Projection function has not been bound.")
        if not hasattr(self, '_run_config'):
            raise ValueError("Run configuration has not been set.")
        projection_args = projection_args or {}

        if self.time is not None:
            msg = f"'time={self.time} will be reset to iterate from 0 to {self.MAX_T}."
            warnings.warn(msg)
            f, l = get_f_filename_lineno()
            self._messages.append(f"{f}:{l}: [WARNING] {msg}")

        exec_start_time = datetime.now()
        try:
            time_step = 1  # might allow different time step in the future
            for t in range(self.MAX_T + 1):
                if t == 0:
                    self._time_synchronizer.set(time=0, period=self.START_DATE)
                else:
                    self._time_synchronizer.elapse(time_step)
                self._projection(**projection_args)
            gc.collect()
            self._proj_variable_mgr.clear_dead()
            self._write_results()
            exec_success = True
        except Exception:
            traceback.print_exc()
            self._messages.append(traceback.format_exc())
            exec_success = False
        self._dump_runlog(exec_success, exec_start_time, datetime.now())
        return self

    def __call__(
        self,
        *,
        projection_args: dict[str, Any] | None = None,
    ) -> Self:
        return self.run(projection_args=projection_args)

    def _write_results(self) -> None:
        if self.RESULTS_DIRECTORY_PATH.is_dir():
            if self._run_config.is_delete_existing_results:
                remove_pattern = ('.proj.csv', '.stoch.csv', 'stoch.stat.csv', '.runlog.json')
                for f in glob.glob(str(self.RESULTS_DIRECTORY_PATH / f'{self._slug}*')):
                    if f.endswith(remove_pattern):
                        os.remove(f)
                    else:
                        msg = f"Existing file NOT deleted: '{f}'."
                        warnings.warn(msg)
                        f, l = get_f_filename_lineno()
                        self._messages.append(f"{f}:{l}: [INFO] {msg}")
        else:
            os.makedirs(self.RESULTS_DIRECTORY_PATH, exist_ok=True)
        self._write_projection_result()
        self._write_stochastic_result()

    def _write_projection_result(self) -> None:
        """Write the projection result."""
        if not self._run_config.enable_write_proj_result:
            return
        variables = self._proj_variable_mgr.filter(rule=self.load_json('__proj_variables__.json', default=None))
        if not variables:
            return  # empty list
        variables.sort(key=lambda x: (x.group, x.owner, x.name))

        periods = pd.period_range(start=self.START_DATE, end=self.END_DATE, freq='M')
        period_col_list = list(map(lambda p: p.year * 100 + p.month, periods))

        output_file = self.RESULTS_DIRECTORY_PATH / f'{self._slug}.proj.csv'
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['group', 'owner', 'variable', 'constant'] + period_col_list)
            for v in variables:
                meta_col = [v.group, v.owner]
                for label, result in self._proj_variable_mgr.iter_label_result(v):
                    result_ls = ([''] + result.tolist()) if v.is_tdim else [result]
                    writer.writerow(meta_col + [label] + result_ls)
        self._result_files.add(output_file)

    def _write_stochastic_result(self) -> None:
        """Write the stochastic result."""
        if not self._run_config.stoch_result_file_mode:
            return
        rule = self.load_json('__stoch_variables__.json', default=None)
        if rule is None:
            return
        variables = self._proj_variable_mgr.filter(rule)
        if not variables:
            return  # empty list
        variables.sort(key=lambda x: (x.group, x.owner, x.name))

        stoch_setting = self.load_json('__stoch_setting__.json', default={})
        noy_mres = stoch_setting.get('number_of_years_monthly_results_retained', 0)

        periods = pd.period_range(start=self.START_DATE, end=self.END_DATE, freq='M')
        periods_m = pd.period_range(
            start=self.START_DATE, end=f'{min(self.START_YEAR + noy_mres, self.END_YEAR)}-{12}', freq='M')
        periods_y = pd.period_range(start=pd.Period(str(self.START_YEAR), freq='Y'),
                                    end=pd.Period(str(self.END_YEAR), freq='Y'), freq='Y')
        dates = list(map(lambda p: p.year * 100 + p.month, periods))
        dates_m = list(map(lambda p: p.year * 100 + p.month, periods_m))
        dates_y = list(map(lambda p: p.year, periods_y))
        indices = [dates.index(x) for x in dates_m] + [dates.index(x * 100 + 12) for x in dates_y]

        file_id = self._run_config.stoch_result_file_id
        file_id += '.' if file_id else ''
        output_file = self.RESULTS_DIRECTORY_PATH / f'{self._slug}.{file_id}stoch.csv'

        output_mode = self._run_config.stoch_result_file_mode
        is_file_exist = output_file.is_file()
        if output_mode == 'w' and is_file_exist:
            msg = f"stoch_result_file_mode='w': existing '{output_file}' will be overwritten."
            warnings.warn(msg)
            f, l = get_f_filename_lineno()
            self._messages.append(f"{f}:{l}: {msg}")
        elif output_mode == 'a' and not is_file_exist:
            output_mode = 'w'
            msg = f"stoch_result_file_mode='a': but 'w' mode will be used because '{output_file}' does not exist."
            warnings.warn(msg)
            f, l = get_f_filename_lineno()
            self._messages.append(f"{f}:{l}: {msg}")

        with open(output_file, output_mode, newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            sim = self._run_config.simulation
            if output_mode == 'w':  # write header
                writer.writerow(['simulation', 'group', 'owner', 'variable', 'constant'] + dates_m + dates_y)
            for v in variables:
                meta_col = ([sim] if sim else []) + [v.group, v.owner]
                for label, result in self._proj_variable_mgr.iter_label_result(v):
                    result_ls = ([''] + result[indices].tolist()) if v.is_tdim else [result]
                    writer.writerow(meta_col + [label] + result_ls)

        self._result_files.add(output_file)

    @property
    def runlog(self) -> dict[str, Any] | None:
        return self._runlog

    def _dump_runlog(self, exec_success: bool, exec_start_time: datetime, exec_end_time: datetime) -> None:
        exec_total_seconds = int((exec_end_time - exec_start_time).total_seconds())
        exec_hours = exec_total_seconds // 3600
        exec_minutes = (exec_total_seconds % 3600) // 60
        exec_seconds = exec_total_seconds % 60

        self._runlog.update({
            "model": {
                "slug": self._slug,
                "description": self._description,
                "projection_function": f"{inspect.getfile(self._projection)}::{self._projection.__name__}",
                "projection_engine": f"{inspect.getfile(type(self))}::{type(self).__name__}",
            },
            "execution": {
                "success": exec_success,
                "start": exec_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end": exec_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                "duration": f"{exec_hours:02}:{exec_minutes:02}:{exec_seconds:02}",
            },
            "configuration": {
                "start_year": self.START_YEAR,
                "start_month": self.START_MONTH,
                "end_year": self.END_YEAR,
                "end_month": self.END_MONTH,
                "scenario": self.SCENARIO,
                "simulation": self.SIMULATION,
                "workspace_directory": str(self.WORKSPACE_DIRECTORY_PATH),
                "input_directories": [str(d) for d in self._run_config.input_directory_paths],
                "results_directory": str(self.RESULTS_DIRECTORY_PATH),
            },
            "environment": get_environ(),
            "results": list(map(str, self._result_files)),
            "messages": self._messages,
        })

        if self._run_config.enable_write_runlog:
            os.makedirs(self.RESULTS_DIRECTORY_PATH, exist_ok=True)
            output_file = self.RESULTS_DIRECTORY_PATH / f'{self._slug}.runlog.json'
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(self._runlog, jsonfile, indent=4)

    @overload
    def load_json[T](self, filename: str, /, *, default: T, **kwargs: Any) -> dict | T: ...

    @overload
    def load_json(self, filename: str, /, **kwargs: Any) -> dict: ...

    def load_json(self, filename: str, /, *, default: Any = _MISSING, **kwargs: Any) -> dict | Any:
        filepath = self.get_filepath(filename)
        if filepath is not None:
            with open(filepath, 'r', **kwargs) as jsonfile:
                return json.load(jsonfile)
        if default is not _MISSING:
            return default
        extra_msg = "" if filename.lower().endswith('.json') else "Perhaps forget to include '.json' in `filename`."
        raise FileNotFoundError(f"JSON file '{filename}' not exists in input directories. {extra_msg}")

    @overload
    def read_csv[T](self, filename: str, /, *, default: T, **kwargs: Any) -> pd.DataFrame | T: ...

    @overload
    def read_csv(self, filename: str, /, **kwargs: Any) -> pd.DataFrame: ...

    def read_csv(self, filename: str, /, *, default: Any = _MISSING, **kwargs: Any) -> pd.DataFrame | Any:
        filepath = self.get_filepath(filename)
        if filepath is not None:
            return pd.read_csv(filepath, **kwargs)
        if default is not _MISSING:
            return default
        extra_msg = "" if filename.lower().endswith('.csv') else "Perhaps forget to include '.csv' in `filename`."
        raise FileNotFoundError(f"CSV file '{filename}' does not exist in input directories. {extra_msg}")

    @overload
    def read_excel[T](self, filename: str, /, *, default: T, **kwargs: Any) -> pd.DataFrame | T: ...

    @overload
    def read_excel(self, filename: str, /, **kwargs: Any) -> pd.DataFrame: ...

    def read_excel(self, filename: str, /, *, default: Any = _MISSING, **kwargs: Any) -> pd.DataFrame | Any:
        filepath = self.get_filepath(filename)
        if filepath is not None:
            return pd.read_excel(filepath, **kwargs)
        if default is not _MISSING:
            return default
        extra_msg = "" if filename.lower().endswith('.xlsx') else "Perhaps forget to include '.xlsx' in `filename`."
        raise FileNotFoundError(f"Excel file '{filename}' does not exist in input directories. {extra_msg}")

    def get_filepath(self, filename: str, /) -> Path | None:
        """Return the path of `filename` in the input directories, first match wins.

        When the same filename exists in more than one input directory, the first match
        (per `input_directories` order) is returned.
        """
        cache = self._filepath_cache
        filepath = cache.get(filename, _MISSING)
        if filepath is _MISSING:
            matches = search_file_in_dirs(filename, dirs=self._run_config.input_directory_paths)
            filepath = matches[0]
            if len(matches) > 1:
                ignored = ", ".join(f"'{p}'" for p in matches[1:])
                f, l = get_f_filename_lineno()
                self._messages.append(f"{f}:{l}: [INFO] '{filename}': duplicate found in multiple input directories; "
                                      f"using '{filepath}' (first match); ignored: {ignored}.")
            cache[filename] = filepath
        return filepath

    @property
    def WORKSPACE_DIRECTORY_PATH(self) -> Path:
        return self._run_config.workspace_directory_path

    @property
    def RESULTS_DIRECTORY_PATH(self) -> Path:
        return self._run_config.results_directory_path

    @property
    def SLUG(self) -> str:
        """str: Model slug."""
        return self._slug

    @property
    def SCENARIO(self) -> str:
        """str: Scenario code used for a run."""
        return self._run_config.scenario

    @property
    def SIMULATION(self) -> int:
        """int: Simulation."""
        return self._run_config.simulation

    @property
    def START_YEAR(self) -> int:
        """int: Year of the start date of the projection."""
        return self._run_config.start_date.year

    @property
    def START_MONTH(self) -> int:
        """int: Month (1-12) of the start date of the projection."""
        return self._run_config.start_date.month

    @property
    def START_DATE(self) -> pd.Period:
        """pd.Period: Start date of the projection."""
        return self._run_config.start_date

    @property
    def END_YEAR(self) -> int:
        """int: Year of the end date of the projection."""
        return self._run_config.end_date.year

    @property
    def END_MONTH(self) -> int:
        """int: Month (1-12) of the end date of the projection."""
        return self._run_config.end_date.month

    @property
    def END_DATE(self) -> pd.Period:
        """pd.Period: End date of the projection."""
        return self._run_config.end_date

    @property
    def MAX_T(self) -> int:
        """int: Number of projection months."""
        return self._run_config.max_t

    def attach_time_listener(self, listener, /) -> None:
        self._time_synchronizer.attach_time_listener(listener)

    @property
    def time(self) -> int:
        """int: Current projection time index."""
        return self._time_synchronizer.time

    @time.setter
    def time(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"time: type {type(value)} is not allowed, expected 'int'.")
        if not 0 <= value <= self.MAX_T:
            raise ValueError(f"time: value {value} is not allowed, expected 0 to {self.MAX_T}.")
        self._time_synchronizer.set(time=value, period=self.START_DATE + value)

    @property
    def period(self) -> pd.Period:
        """pd.Period: Current projection period."""
        return self._time_synchronizer.period

    @period.setter
    def period(self, value: pd.Period | str) -> None:
        if isinstance(value, str):
            value = pd.Period(value, freq="M")
        if not isinstance(value, pd.Period):
            raise TypeError(f"period: type {type(value)} is not allowed, expected 'pd.Period'.")
        if not self.START_DATE <= value <= self.END_DATE :
            raise ValueError(f"period: value {value} is not allowed, expected {self.START_DATE}) to {self.END_DATE}.")
        self._time_synchronizer.set(period=value, time=(value - self.START_DATE).n)

    @property
    def proj_variables(self) -> list[BaseProjVariable]:
        return list(self._proj_variable_mgr)

    def attach_proj_variable(self, proj_variable: BaseProjVariable, /) -> None:
        self._proj_variable_mgr.attach_proj_variable(proj_variable)

    @overload
    def proj_result(self) -> pd.DataFrame:
        ...

    @overload
    def proj_result(self, *, group: str, owner: str, variable: str, date: str | int | None = None) -> float:
        ...

    def proj_result(self, *, group: str | None = None, owner: str | None = None, variable: str | None = None,
                    date: str | int | None = None) -> pd.DataFrame | Any:
        return proj_result(results_directory=self.RESULTS_DIRECTORY_PATH, slug=self.SLUG,
                           group=group, owner=owner, variable=variable, date=date)

    def __setattr__(self, name, value):
        if hasattr(self, '_initialized'):
            if name in type(self).__dict__:  # check if the attribute name already exists in the class definition
                obj = type(self).__dict__.get(name)
                if isinstance(obj, property) and obj.fset is not None:  # check if it is a property and has a setter
                    pass
                else:
                    raise AttributeError(f"Cannot overwrite protected member '{name}'")
            elif name.startswith('_'):
                raise AttributeError(f"Cannot add a private member (underscore-prefixed) '{name}'.")
            if not hasattr(self, name) and hasattr(self, "_messages"):
                f, l = get_f_filename_lineno()
                self._messages.append(f"{f}:{l}: [INFO] Add member: '{name}' {type(value)}")
        super().__setattr__(name, value)

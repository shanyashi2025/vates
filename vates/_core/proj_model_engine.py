import csv
import glob
import inspect
import json
import os
import pandas as pd
import traceback
import warnings
import weakref
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Callable, Literal, Self, get_type_hints

from vates._core.proj_variables import ProjVariable
from vates._core._utils import RunConfig

class ProjModelEngine:
    """Actuarial projection model engine.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = '...'
    ) -> None:
        """Initialize a projection model engine.

        Args:
            name (str): The name of the model.
            description (str, optional): The description of the model.
        """
        self._name: str = str(name)
        self._description: str = str(description)

        self._proj_func: Callable | None = None
        self._run_config: RunConfig | None = None

        # runtime stuffs
        self._cached_filepath: dict[str, tuple] = {}
        self._time_observers: list[weakref.ref] = []
        self._proj_variables: list[weakref.ref[ProjVariable]] = []
        self._output_files: set = set()
        self._time: int | None = None
        self._period: pd.Period | None = None
        self._messages: list[str] = []
        self._runlog: dict | None = None

        super().__setattr__('_initialized', True)

    def bind_proj_func(
        self,
        func: Callable,
        /
    ) -> Self:
        """Bind the projection model to the engine.

        Args:
            func (callable): The projection function.

        Raises:
            ValueError: If `func` is not callable.
        """
        if self._proj_func is not None:
            msg = (f"{self._proj_func} is already bound. If you are sure you want to reset it, "
                   f"use 'foo._proj_func = None', then call 'foo.bind_proj_func(...)' method.")
            warnings.warn(msg); self.include_traced_message(f"WARNING: {msg}")
            return self
        if not callable(func):
            raise ValueError(f"Cannot bind un-callable object: {func}.")

        sig = inspect.signature(func)
        hints = get_type_hints(func)

        sig_params = inspect.signature(func).parameters
        if len(sig.parameters) == 0:
            self.include_traced_message(f"INFO: Function '{func.__name__}' has no argument, it will be bound as a "
                                        f"function instead of a method.")
            self._proj_func = func
        else:
            first_arg_name = list(sig_params.keys())[0]
            if first_arg_name not in hints:
                raise ValueError(f"Function '{func.__name__}' first argument '{first_arg_name}': must have annotation.")
            if hints[first_arg_name] is not type(self):
                raise ValueError(f"Function '{func.__name__}' first argument '{first_arg_name}': model engine type "
                                 f"'{type(self)}' inconsistent with type hint '{str(hints[first_arg_name])}'.")
            self._proj_func = MethodType(func, self)

        self.include_traced_message(f"INFO: Function {func} has been bound to {self}.")

        return self

    def set_run_config(
        self,
        *,
        start_year: int,
        start_month: int | None = None,
        end_year: int | None = None,
        end_month: int | None = None,
        scenario: str | None = None,
        simulation: int | None = None,
        workspace_directory: str | None = None,
        input_directories: list[str] | None = None,
        results_directory: str | None = None,
        is_delete_existing_results: bool = True,
        enable_write_proj_result: bool = True,
        stoch_result_file_mode: Literal['w', 'a', None] = None,
        stoch_result_file_id: str | None = None,
        enable_write_runlog: bool = True,
    ) -> Self:
        """Set the configuration for a run.

        Args:
            start_year (int, optional): Projection start year. Defaults to None.
            start_month (int, optional): Projection start month. Defaults to 12.
            end_year (int, optional): Projection end year. Defaults to `start_year`.
            end_month (int, optional): Projection end month. Defaults to 12.
            scenario (str, optional): Scenario. Defaults to None.
            simulation (int, optional): Simulation (stochastic). Defaults to None.
            workspace_directory (str, optional): Workspace directory. Defaults to `{os.getcwd()}`.
            input_directories (list[str], optional): List of input directory. Defaults to None.
            results_directory (str, optional): Results directory. Defaults to 'results/`scenario`'.
            is_delete_existing_results (bool, optional): Delete existing results if any. Defaults to True.
            enable_write_proj_result (bool, optional): Enable writing projection result. Defaults to True.
            stoch_result_file_mode (Literal['w', 'a'], optional): Stochastic result writer mode. Defaults to None.
            stoch_result_file_id (str, optional): Stochastic result id. Defaults to None.
            enable_write_runlog (bool, optional): Enable writing run log. Defaults to True.
        """
        if self._run_config is not None:
            msg = (f"Run configuration is already set. If you are sure you want to reset it, "
                   f"use 'foo._run_config = None', then call 'foo.set_run_config(...)' method.")
            warnings.warn(msg); self.include_traced_message(f"WARNING: {msg}")
            return self

        none_items = []

        if start_year is None:
            raise ValueError(f"start_year: value 'None' is not allowed.")
        if start_month is None:
            start_month = 12
            none_items.append(f"start_month={start_month}")
        if end_year is None:
            end_year = start_year
            none_items.append(f"end_year={end_year}")
        if end_month is None:
            end_month = 12
            none_items.append(f"end_month={end_month}")
        if workspace_directory is None:
            workspace_directory = os.getcwd()
            none_items.append(f"workspace_directory='{workspace_directory}'")
        if results_directory is None:
            results_directory = f"./results/{scenario or ''}"
            none_items.append(f"results_directory='{results_directory}'")
        self._run_config = RunConfig.create(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            scenario=scenario,
            simulation=simulation,
            workspace_directory=workspace_directory,
            input_directories=input_directories,
            results_directory=results_directory,
            is_delete_existing_results=is_delete_existing_results,
            enable_write_proj_result=enable_write_proj_result,
            stoch_result_file_mode=stoch_result_file_mode,
            stoch_result_file_id=stoch_result_file_id,
            enable_write_runlog=enable_write_runlog,
        )

        if len(none_items) > 0:
            msg = f"Following items are set by default: {', '.join(none_items)}."
            warnings.warn(msg); self.include_traced_message(f"INFO: {msg}")

        return self

    def run(
        self,
        *,
        proj_func_args: dict[str, ...] | None = None,
    ) -> dict:
        if self._proj_func is None:
            raise ValueError(f"Projection function has not been bound.")
        if self._run_config is None:
            raise ValueError("Run configuration has not been set.")
        proj_func_args = proj_func_args or {}

        if self._time is not None:
            msg = f"'time={self._time} will be reset to iterate from 0 to {self.MAX_T}."
            warnings.warn(msg); self.include_traced_message(f"WARNING: {msg}")

        exec_start_time = datetime.now()
        try:
            for self.time in range(self.MAX_T + 1):
                self._proj_func(**proj_func_args)
            self._proj_variables[:] = [ref for ref in self._proj_variables if ref()]  # remove dead
            self._write_results()
            exec_success = True
        except Exception as e:
            traceback.print_exc()
            self._messages.append(traceback.format_exc())
            exec_success = False
        self._dump_runlog(exec_success, exec_start_time, datetime.now())
        return self._runlog

    def __call__(
        self,
        *,
        proj_func_params: dict[str, ...] | None = None,
    ) -> dict:
        return self.run(proj_func_args=proj_func_params)

    def _write_results(self) -> None:
        if self.results_directory_path.is_dir():
            if self._run_config.is_delete_existing_results:
                remove_pattern = ('.proj.csv', '.stoch.csv', 'stoch.stat.csv', '.runlog.json')
                for f in glob.glob(str(self.results_directory_path / f'{self._name}*')):
                    if f.endswith(remove_pattern):
                        os.remove(f)
                    else:
                        msg = f"Exsiting file NOT deleted: '{f}'."
                        warnings.warn(msg); self.include_traced_message(f"INFO: {msg}")
        else:
            os.makedirs(self.results_directory_path, exist_ok=True)
        self._write_projection_result()
        self._write_stochastic_result()

    def _write_projection_result(self) -> None:
        """Write the projection result."""
        if not self._run_config.enable_write_proj_result: return
        variables = self._select_variables(self._proj_variables,'__proj_variables__.json', 'full')
        if not variables: return  # empty list

        periods = pd.period_range(start=self.START_DATE, end=self.END_DATE, freq='M')
        period_col_list = list(map(lambda p: p.year * 100 + p.month, periods))

        output_file = self._concat_output_file_path('.proj.csv')
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['group', 'owner', 'variable', 'constant'] + period_col_list)
            for v in variables:
                self._write_variable(v, writer)
        self._output_files.add(output_file)

    def _write_stochastic_result(self) -> None:
        """Write the stochastic result."""
        if not self._run_config.stoch_result_file_mode:
            return
        variables = self._select_variables(self._proj_variables,'__stoch_variables__.json', 'none')
        if not variables: return  # empty list

        stoch_setting = self.load_json('__stoch_setting__.json', allow_not_found=True)
        noy_mres = stoch_setting.get('number_of_years_monthly_results_retained', 0) if stoch_setting else 0

        periods = pd.period_range(start=self.START_DATE, end=self.END_DATE, freq='M')
        periods_m = pd.period_range(
            start=self.START_DATE, end=f'{min(self.START_YEAR + noy_mres, self.END_YEAR)}-{12}', freq='M')
        periods_y = pd.period_range(start=self.START_YEAR, end=self.END_YEAR, freq='Y')
        period_lst = list(map(lambda p: p.year * 100 + p.month, periods))
        period_lst_m = list(map(lambda p: p.year * 100 + p.month, periods_m))
        period_lst_y = list(map(lambda p: p.year, periods_y))
        pos_lst_m = [period_lst.index(x) for x in period_lst_m]
        pos_lst_y = [period_lst.index(x * 100 + 12) for x in period_lst_y]
        period_col_list = period_lst_m + period_lst_y

        file_id = self._run_config.stoch_result_file_id
        file_id += '.' if file_id else ''
        output_file = self._concat_output_file_path(f'.{file_id}stoch.csv')

        output_mode = self._run_config.stoch_result_file_mode
        is_file_exist = output_file.is_file()
        if output_mode == 'w' and is_file_exist:
            msg = f"stoch_result_file_mode='w': existing '{output_file}' will be overwritten."
            warnings.warn(msg); self.include_traced_message(msg)
        elif output_mode == 'a' and not is_file_exist:
            output_mode = 'w'
            msg = f"stoch_result_file_mode='a': but 'w' mode will be used because '{output_file}' does not exist."
            warnings.warn(msg); self.include_traced_message(msg)

        with open(output_file, output_mode, newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            if output_mode == 'w':  # write header
                writer.writerow(['simulation', 'group', 'owner', 'variable', 'constant'] + period_col_list)
            for v in variables:
                self._write_stoch_variable(v, writer, self._run_config.simulation, pos_lst_m, pos_lst_y)
        self._output_files.add(output_file)

    def attach_time_observer(self, observer, /) -> None:
        if isinstance(observer, weakref.ref):
            self._time_observers.append(observer)
        else:
            self._time_observers.append(weakref.ref(observer))

    def _notify_time_sync(self) -> None:
        alive_observers = []
        for ref in self._time_observers:
            obs = ref()
            if obs is not None:
                obs.sync_time(self)
                alive_observers.append(ref)

        total_count = len(self._time_observers)
        dead_count = total_count - len(alive_observers)
        if dead_count > 0 and (dead_count > 50 or dead_count / total_count > 0.25):
            self._time_observers[:] = alive_observers

    def include_proj_variable(self, proj_variable: ProjVariable | weakref.ref[ProjVariable]) -> None:
        """Include a projection variable into `_proj_variables`

        Args:
            proj_variable (ProjVariable | weakref.ref[ProjVariable]): Projection variable to be included.

        Raises:
            TypeError: Invalid type.
            ValueError: Projection variable already included.
        """
        if isinstance(proj_variable, weakref.ref) and isinstance(proj_variable(), ProjVariable):
            proj_var_ref = proj_variable
        elif isinstance(proj_variable, ProjVariable):
            proj_var_ref = weakref.ref(proj_variable)
        else:
            raise TypeError(f"Invalid type {type(proj_variable)}, expected 'ProjVariable'.")
        if proj_var_ref in self._proj_variables:
            raise ValueError(f"Variable is already included: name '{proj_var_ref().name}', "
                             f"owner '{proj_var_ref().owner}', group '{proj_var_ref().group}'")
        self._proj_variables.append(proj_var_ref)

    def _select_variables(self, variable_list: list[weakref.ref[ProjVariable]], user_select: str,
                          default_full_or_none: Literal['full', 'none']) -> list | None:
        sel_spc_dict = self.load_json(user_select, allow_not_found=True)
        if sel_spc_dict is None:
            if default_full_or_none == 'full':
                return [r() for r in variable_list if r() is not None]
            else:
                return None

        def _is_select(v: ProjVariable) -> bool:
            if v is None:
                return False
            group_dict = sel_spc_dict.get(v.group)
            if group_dict is None:
                return False
            if not group_dict:  # group key is included and var name is empty: regarded as all var names included
                return True
            include, exclude = group_dict.get("include"), group_dict.get("exclude")
            if include:
                if isinstance(include, str) and include.upper() == "__ALL__":
                    return True
                return isinstance(include, list) and v.name in include
            if exclude:
                return isinstance(exclude, list) and v.name not in exclude
            return True  # dict has key other than 'include' or 'exclude': regarded as all var names included

        return [r() for r in variable_list if _is_select(r())]

    def _write_variable(self, variable: ProjVariable, writer: csv.writer) -> None:
        if variable.ndim == 0:
            self._write_0d_variable(variable, writer)
        else:
            self._write_nd_variable(variable, writer)

    @staticmethod
    def _write_0d_variable(variable: ProjVariable, writer: csv.writer) -> None:
        result = variable.result
        result = [result] if variable.is_constant else [''] + result.tolist()
        writer.writerow([variable.group, variable.owner, variable.name] + result)

    @staticmethod
    def _write_nd_variable(variable: ProjVariable, writer: csv.writer) -> None:
        from itertools import product
        fixcol = [variable.group, variable.owner]
        dim_ranges = [range(len(dim)) for dim in variable.dims]
        for dim_index in product(*dim_ranges):
            dimstr = ":".join([variable.dims[i][dim_index[i]] for i in range(variable.ndim)])
            result = variable.result[(slice(0, None),) + dim_index]
            result = [result] if variable.is_constant else [''] + result.tolist()
            writer.writerow(fixcol + [f'{variable.name}[{dimstr}]'] + result)

    def _write_stoch_variable(self, variable: ProjVariable, writer: csv.writer, sim: int,
                              pos_lst_m: list[int], pos_lst_y: list[int]) -> None:
        if variable.ndim == 0:
            self._write_0d_stoch_variable(variable, writer, sim, pos_lst_m, pos_lst_y)
        else:
            self._write_nd_stoch_variable(variable, writer, sim, pos_lst_m, pos_lst_y)

    @staticmethod
    def _write_0d_stoch_variable(variable: ProjVariable, writer: csv.writer, sim: int,
                                 pos_lst_m: list[int], pos_lst_y: list[int]) -> None:
        result = variable.result
        if variable.is_constant:
            result_lst = [result]
        else:
            result_lst_m = result[pos_lst_m].tolist()
            result_lst_y = result[pos_lst_y].tolist()  # to be enhanced, currently only correct for BS variable
            result_lst = [''] + result_lst_m + result_lst_y
        writer.writerow([sim, variable.group, variable.owner, variable.name] + result_lst)

    @staticmethod
    def _write_nd_stoch_variable(variable: ProjVariable, writer: csv.writer, sim: int,
                                 pos_lst_m: list[int], pos_lst_y: list[int]) -> None:
        from itertools import product
        fixcol = ([sim] if sim else []) + [variable.group, variable.owner]
        dim_ranges = [range(len(dim)) for dim in variable.dims]
        for dim_index in product(*dim_ranges):
            dimstr = ":".join([variable.dims[i][dim_index[i]] for i in range(variable.ndim)])
            result = variable.result[(slice(0, None),) + dim_index]
            if variable.is_constant:
                result_lst = [result]
            else:
                result_lst_m = result[pos_lst_m].tolist()
                result_lst_y = result[pos_lst_y].tolist()  # to be enhanced, currently only correct for BS variable
                result_lst = [''] + result_lst_m + result_lst_y
            writer.writerow(fixcol + [f'{variable.name}[{dimstr}]'] + result_lst)

    @property
    def runlog(self) -> dict[str, ...] | None:
        return self._runlog

    def _dump_runlog(self, exec_success: bool, exec_start_time: datetime, exec_end_time: datetime) -> None:
        exec_total_seconds = int((exec_end_time - exec_start_time).total_seconds())
        exec_hours = exec_total_seconds // 3600
        exec_minutes = (exec_total_seconds % 3600) // 60
        exec_seconds = exec_total_seconds % 60

        self._runlog = {
            "model": {
                "name": self._name,
                "description": self._description,
                "projection_function": f"{inspect.getfile(self._proj_func)}:{self._proj_func.__name__}",
                "projection_engine": f"{inspect.getfile(type(self))}:{type(self).__name__}",
            },
            "execution": {
                "success": exec_success,
                "start": exec_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end": exec_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                "duration": f"{exec_hours:02}:{exec_minutes:02}:{exec_seconds:02}",
            },
            "run_config": {
                "start_year": self.START_YEAR,
                "start_month": self.START_MONTH,
                "end_year": self.END_YEAR,
                "end_month": self.END_MONTH,
                "scenario": self.SCENARIO,
                "simulation": self.SIMULATION,
                "workspace_directory": self._run_config.workspace_directory,
                "input_directories": self._run_config.input_directories,
                "results_directory": self._run_config.results_directory,
            },
            "output_files": list(map(str, self._output_files)),
            "messages": self._messages,
        }

        if self._run_config.enable_write_runlog:
            with open(self._concat_output_file_path(".runlog.json"), 'w', encoding='utf-8') as jsonfile:
                json.dump(self._runlog, jsonfile, indent=4)

    def load_json(self, filename: str, /, *, first_or_last_seen: str = 'first_seen',
                  allow_not_found: bool = False, **kwargs) -> dict | None:
        filepath = self.get_filepath(filename, first_or_last_seen=first_or_last_seen)
        if filepath:
            with open(filepath, 'r', **kwargs) as jsonfile:
                return json.load(jsonfile)
        elif allow_not_found:
            self.include_traced_message(f"INFO: JSON file '{filename}' not found, 'None' is return.")
            return None
        else:
            ext_warn = "" if filename.lower().endswith('.json') else "You might forget to include '.json' in filename."
            raise FileNotFoundError(f"JSON file '{filename}' not exists in input directories. {ext_warn}")

    def read_csv(self, filename: str, /, *, first_or_last_seen: str = 'first_seen',
                 allow_not_found: bool = False, **kwargs) -> pd.DataFrame | None:
        filepath = self.get_filepath(filename, first_or_last_seen=first_or_last_seen)
        if filepath:
            return pd.read_csv(filepath, **kwargs)
        elif allow_not_found:
            self.include_traced_message(f"INFO: CSV file '{filename}' not found, 'None' is return.")
            return None
        else:
            ext_warn = "" if filename.lower().endswith('.csv') else "You might forget to include '.csv' in filename."
            raise FileNotFoundError(f"CSV file '{filename}' does not exist in input directories. {ext_warn}")

    def read_excel(self, filename: str, /, *, first_or_last_seen: str = 'first_seen',
                   allow_not_found: bool = False, **kwargs) -> pd.DataFrame | None:
        filepath = self.get_filepath(filename, first_or_last_seen=first_or_last_seen)
        if filepath:
            return pd.read_excel(filepath, **kwargs)
        elif allow_not_found:
            self.include_traced_message(f"INFO: Excel file '{filename}' not found, 'None' is return.")
            return None
        else:
            ext_warn = "" if filename.lower().endswith('.xlsx') else "You might forget to include '.xlsx' in filename."
            raise FileNotFoundError(f"Excel file '{filename}' does not exist in input directories. {ext_warn}")

    def read_parquet(self, filename: str, /, *, first_or_last_seen: str = 'first_seen',
                     allow_not_found: bool = True, **kwargs) -> pd.DataFrame | None:
        filepath = self.get_filepath(filename, first_or_last_seen=first_or_last_seen)
        if filepath is not None:
            import pyarrow.dataset as ds
            dataset = ds.dataset(filepath, format="parquet")
            table = dataset.to_table(**kwargs)
            return table.to_pandas()
        elif allow_not_found:
            self.include_traced_message(f"INFO: parquet file '{filename}' not found, 'None' is return.")
            return None
        else:
            ext_warn = "" if filename.lower().endswith('.parquet') else "You might forget to include '.parquet' in filename."
            raise FileNotFoundError(f"Parquet file '{filename}' not exists in input directories. {ext_warn}")

    def get_filepath(self, filename: str, /, *, first_or_last_seen: str = "first_seen") -> Path | None:
        if filename in self._cached_filepath:  # get from cache
            filepath_tuple = self._cached_filepath[filename]
        else:  # search
            filepath_tuple = self._search_filepath(filename)
            self._cached_filepath[filename] = filepath_tuple

        if first_or_last_seen.lower() in ("first", "first_seen", "first-seen"):
            return filepath_tuple[0]
        elif first_or_last_seen.lower() in ("last", "last_seen", "last-seen"):
            return filepath_tuple[1]
        else:
            msg = f"'first_or_last_seen': value '{first_or_last_seen}' is unkown, use 'first_seen' as fallback."
            warnings.warn(msg); self.include_traced_message(f"WARNING: {msg}")
            return filepath_tuple[0]

    def _search_filepath(self, filename: str) -> tuple[Path | None, Path | None]:
        if (filepath := Path(filename)).is_absolute() and filepath.is_file():  # absolute path
            return filepath, filepath
        if self._run_config.input_directories is None:  # return None if input_directories not specified
            return None, None
        first_seen, last_seen = None, None
        for directory in self._run_config.input_directories:  # search input_directories sequentially
            filepath = (self.workspace_directory_path / directory / filename).resolve()
            if filepath.is_file():
                if first_seen is None:
                    first_seen = filepath
                last_seen = filepath
        return first_seen, last_seen

    def _concat_output_file_path(self, name: str) -> Path:
        return self.results_directory_path / f'{self._name}{name}'

    @property
    def workspace_directory_path(self) -> Path:
        return self._run_config.workspace_directory_path

    @property
    def results_directory_path(self) -> Path:
        return self._run_config.results_directory_path

    @property
    def MODEL_NAME(self) -> str:
        """str: Model alias used to prefix output files."""
        return self._name

    @property
    def SCENARIO(self) -> str | None:
        """str: Scenario code that identifies the set of input used for a run."""
        return self._run_config.scenario

    @property
    def SIMULATION(self) -> int | None:
        """int: Simulation."""
        return self._run_config.simulation

    @property
    def START_YEAR(self) -> int:
        """int: Year of the start date of the projection."""
        return self._run_config.start_year

    @property
    def START_MONTH(self) -> int:
        """int: Month (1-12) of the start date of the projection."""
        return self._run_config.start_month

    @property
    def START_DATE(self) -> pd.Period:
        """pd.Period: Start date of the projection."""
        return self._run_config.start_date

    @property
    def END_YEAR(self) -> int:
        """int: Year of the end date of the projection."""
        return self._run_config.end_year

    @property
    def END_MONTH(self) -> int:
        """int: Month (1-12) of the end date of the projection."""
        return self._run_config.end_month

    @property
    def END_DATE(self) -> pd.Period:
        """pd.Period: End date of the projection."""
        return self._run_config.end_date

    @property
    def MAX_T(self) -> int:
        """int: Number of projection months."""
        return self._run_config.max_t

    @property
    def time(self) -> int | None:
        """pd.Period: Current projection time index."""
        return self._time

    @time.setter
    def time(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"time: type {type(value)} is not allowed, expected 'int'.")
        if not 0 <= value <= self.MAX_T:
            raise ValueError(f"time: value {value} is not allowed, execpted 0 to {self.MAX_T}.")
        self._time = value
        self._period = self.START_DATE + value
        self._notify_time_sync()

    @property
    def period(self) -> pd.Period | None:
        """pd.Period: Current projection period."""
        return self._period

    @period.setter
    def period(self, value: pd.Period) -> None :
        if not isinstance(value, pd.Period):
            raise TypeError(f"period: type {type(value)} is not allowed, expected 'pd.Period'.")
        if self.START_DATE <= value <= self.END_DATE :
            raise ValueError(f"period: value {value} is not allowed, expected {self.START_DATE}) to {self.END_DATE}.")
        self._period = value
        self._time = (value - self.START_DATE).n
        self._notify_time_sync()

    def include_traced_message(self, /, msg: str):
        frame = inspect.currentframe().f_back
        filename = os.path.abspath(frame.f_code.co_filename)
        lineno = frame.f_lineno
        self._messages.append(f"{filename}:{lineno}: {msg}")

    def __setattr__(self, name, value):
        if hasattr(self, '_initialized'):
            if name in type(self).__dict__:  # check if the attribute name already exists in the class definition
                obj = type(self).__dict__.get(name)
                if isinstance(obj, property) and obj.fset is not None:  # check if it is a property and has a setter
                    pass
                else:
                    raise AttributeError(f"Cannot overwrite protected member '{name}'")
            if not hasattr(self, name) and hasattr(self, "_messages"):
                self.include_traced_message(f"INFO: Add member: '{name}' {type(value)}")
        super().__setattr__(name, value)

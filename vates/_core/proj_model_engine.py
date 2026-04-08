from functools import reduce, cached_property
from typing import Literal
import pandas as pd
import os
import glob
import csv
from datetime import datetime
from abc import abstractmethod
import operator

from vates._core.proj_variables import ProjVariable
from vates._core.utils import ValidatedBool, ValidatedNumber, ValidatedString, ValidatedPeriod, ValidatedList
from vates._core._html_generator import generate_runlog_html

class ProjModelEngine:
    """Actuarial projection model engine.

    Lifecycle:
        - time_zero_calculations(): Initialize state, load inputs, etc.
        - in_time_calculations(): Perform calculations for each subsequent projection month.
        - post_time_calculations(): Finalize aggregations and produce any end-of-horizon measures.

    I/O:
        - Validates settings on initialization.
        - Writes CSV outputs with standardized headers and an HTML run log for traceability.

    Attributes:

        _exe_start_time (str): Model execution start time.
        _model_name (str): Model name.
        _model_desc (str): Model description.
        _start_year (int): Projection start year.
        _start_month (int): Projection start month.
        _end_year (int): Projection end year.
        _scenario (str): Projection scenario.
        _simulation (int | None): Simulation for stochastic runs (None means deterministic).
        _wsdir (str): Workspace directory.
        _input_directories (list[str]): Directories to locate model input files.
        _results_directory (str): Output directory where CSVs and the run log are written.
        _enable_proj_result (bool): True/False means enable/dienable writing projection result.
        _stoch_result_mode (Literal[None, 'w', 'a']): Stochastic result file write mode.
        _stoch_result_file_id (None | str): Stochastic result file id.
        _enable_run_log (bool): True/False means enable/dienable writing run log.
        _wsdir (str): Current working directory `os.getcwd().
        _cached_filepath (dict): Dictionary of cached file path.
    """

    _model_name = ValidatedString(len_min=1, max_sets=1)
    _model_desc = ValidatedString(max_sets=1)
    _start_year = ValidatedNumber(value_type=int, value_min=1900, value_max=5999, max_sets=1)
    _start_month = ValidatedNumber(value_type=int, value_lst=range(1, 13), max_sets=1)
    _end_year = ValidatedNumber(value_type=int, value_min=1900, value_max=5999, allow_none=True, max_sets=1)
    _scenario = ValidatedString(allow_none=True, max_sets=1)
    _simulation = ValidatedNumber(value_type=int, value_min=1, value_max=100_000, allow_none=True, max_sets=1)
    _input_directories = ValidatedList(item_type=str, allow_none=True, len_min=0, max_sets=1)
    _results_directory = ValidatedString(max_sets=1)
    _wsdir = ValidatedString(max_sets=1)
    _enable_proj_result = ValidatedBool(max_sets=1)
    _stoch_result_mode = ValidatedString(str_literal=['w', 'a'], allow_none=True, max_sets=1)
    _stoch_result_file_id = ValidatedString(allow_none=True, max_sets=1)
    _enable_run_log = ValidatedBool(max_sets=1)
    _time = ValidatedNumber(value_type=int, value_min=0, value_max=6000, allow_none=True)
    _period = ValidatedPeriod(allow_none=True)

    def __init__(
            self,
            model_name: str,
            start_year: int,
            start_month: int,
            end_year: int | None = None,
            model_desc: str | None = None,
            scenario: str | None = None,
            simulation: int | None = None,
            workspace_directory: str | None = None,
            input_directories: list[str] | None = None,
            results_directory: str = '',
            is_delete_existing_results: bool = True,
            enable_proj_result: bool = True,
            stoch_result_mode: Literal['w', 'a', None] = None,
            stoch_result_file_id: str | None = None,
            enable_run_log: bool = True,
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the projection model engine.

        Args:
            model_name (str): Model name.
            start_year (int): Projection start year.
            start_month (int): Projection start month.
            end_year (int): Projection end year.
            model_desc (str, optional): Model description.
            scenario (str, optional): Scenario. Defaults to None.
            simulation (int, optional): Simulation (stochastic). Defaults to None.
            workspace_directory (str, optional): Workspace directory.
            input_directories (list[str], optional): List of input directory. Defaults to None.
            results_directory (str, optional): Results directory. Defaults to ''.
            enable_proj_result (bool, optional): Enable writing projection result. Defaults to True.
            stoch_result_mode (Literal['w', 'a'], optional): Stochastic result writer mode. Defaults to None.
            stoch_result_file_id (str, optional): Stochastic result id. Defaults to None.
            enable_run_log (bool, optional): Enable writing run log. Defaults to True.
        """
        self._exe_start_time: str = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self._model_name: str = model_name
        self._start_year: int = start_year
        self._start_month: int = start_month
        self._end_year: int | None = end_year
        self._model_desc: str = model_desc if model_desc else (f'{self.__doc__}' if type(self) != ProjModelEngine
                                           else 'Default projection model engine.')
        self._scenario: str | None = scenario
        self._simulation: int | None = simulation
        self._wsdir: str = workspace_directory if workspace_directory else os.getcwd()
        self._input_directories: list[str] | None = input_directories
        self._results_directory: str = results_directory
        self._enable_proj_result: bool = enable_proj_result
        self._stoch_result_mode: Literal['w', 'a', None] = stoch_result_mode
        self._stoch_result_file_id: str | None = stoch_result_file_id
        self._enable_run_log: bool = enable_run_log
        if self._results_directory:
            os.makedirs(os.path.join(self._wsdir, self._results_directory), exist_ok=True)
            if is_delete_existing_results:
                existing_files = glob.glob(os.path.join(self._wsdir, self._results_directory, f'{self._model_name}*'))
                for f in existing_files:
                    os.remove(f)
        self._cached_filepath: dict = {}
        self._proj_variables: list[ProjVariable] = []
        self._time: int | None = None
        self._period: pd.Period | None = None

    def run(self):
        """Execute a full projection run.

        Steps:
            1) Call `time_zero_calculations()` at start date.
            2) For each subsequent month: call `in_time_calculations()`.
            3) After the last month: call `post_time_calculations()`.
            4) Call `_write_projection_results()` to CSV and generate the run log HTML.
        """
        self._time, self._period = 0, self.START_DATE
        self.time_zero_calculations()
        for self._time in range(1, self.MAX_T + 1):
            self._period += 1
            self.in_time_calculations()
        self._time, self._period = None, None
        self.post_time_calculations()
        self._write_projection_result()
        self._write_stochastic_result()
        self._write_runlog()

    @abstractmethod
    def time_zero_calculations(self):
        """Initialize state and compute for the start of the projection."""
        pass

    @abstractmethod
    def in_time_calculations(self):
        """Compute for each month of the projection."""
        pass

    @abstractmethod
    def post_time_calculations(self):
        """Finalize results after the last projection month."""
        pass

    def _write_projection_result(self) -> None:
        """Write the projection result."""
        if not self._enable_proj_result: return
        variables = self._select_variables(self._proj_variables,'__proj_variables__', 'full')
        if not variables: return  # empty list

        periods = pd.period_range(start=self.START_DATE, end=self.END_DATE, freq='M')
        period_col_list = list(map(lambda p: p.year * 100 + p.month, periods))

        output_file = self._concat_output_file_path('.proj.csv')
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['group', 'owner', 'variable', 'constant'] + period_col_list)
            for v in variables:
                self._write_variable(v, writer)

    def _write_stochastic_result(self) -> None:
        """Write the stochastic result."""
        if not self._stoch_result_mode: return
        variables = self._select_variables(self._proj_variables,'__stoch_variables__', 'none')
        if not variables: return  # empty list

        stoch_setting = self.load_json('__stoch_setting__', allow_not_found=True)
        if stoch_setting is None:
            noy_mres = 0
        else:
            noy_mres = stoch_setting.get('number_of_years_monthly_results_retained', 0)

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

        file_id = f'.{self._stoch_result_file_id}' if self._stoch_result_file_id else ''
        output_file = self._concat_output_file_path(file_id + '.stoch.csv')

        with open(output_file, self._stoch_result_mode, newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            if self._stoch_result_mode == 'w':
                writer.writerow(['simulation', 'group', 'owner', 'variable', 'constant'] + period_col_list)
            for v in variables:
                self._write_stoch_variable(v, writer, self._simulation, pos_lst_m, pos_lst_y)

    def _select_variables(self, full_list: list, user_select: str,
                          default_full_or_none: Literal['full', 'none']) -> list | None:
        df = self.read_csv(user_select, extension='.txt', allow_not_found=True)
        if df is None:
            if default_full_or_none == 'full':
                return [r() for r in full_list if r() is not None]
            else:
                return None

        select_spec_list = df[['var_group', 'var_name', 'included']].values.tolist()
        select_spec_dict = {(var_grp, var_name): True if incl.lower() in ['y', 'yes'] else False
                            for (var_grp, var_name, incl) in select_spec_list}

        def _is_select(v) -> bool:
            if v is None:
                return False
            elif (v.group, v.name) in select_spec_dict:
                return select_spec_dict[(v.group, v.name)]
            elif (v.group, "*") in select_spec_dict:
                return select_spec_dict[(v.group, "*")]
            else:
                return False

        return [r() for r in full_list if _is_select(r())]

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

    def _write_runlog(self) -> None:
        if not self._enable_run_log: return
        result_files = self._scan_results_directory()
        runlog: dict = {
            "model_name": self._model_name,
            "model_desc": self._model_desc,
            "exe_start_time": self._exe_start_time,
            "exe_end_time": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "setting": {
                "start_year": self._start_year,
                "start_month": self._start_month,
                "end_year": self._end_year,
                "scenario": self._scenario,
                "simulation": self._simulation,
                "workspace_directory": self._wsdir,
                "input_directories": self._input_directories,
                "results_directory": self._results_directory
            },
            "input_files": self.input_files,
            "proj_result_files": result_files['proj'],
            "stoch_result_files": result_files['stoch'] | result_files['stoch_stat'],
            "other_result_files": result_files['other'],
        }

        html_content = generate_runlog_html(runlog)
        runlog_file = self._concat_output_file_path(".runlog.html")

        with open(runlog_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def load_json(self, filename: str, encoding='utf-8', allow_not_found: bool = False) -> dict | None:
        import json
        filepath = self._get_filepath(filename, 'json')
        if filepath is not None:
            with open(filepath, 'r', encoding=encoding) as jsonfile:
                parsed_data = json.load(jsonfile)
            return parsed_data
        elif allow_not_found:
            return None
        else:
            raise FileNotFoundError(f"File '{filename}.json' not exists in input directories.")

    def read_csv(self, filename: str, extension: str | None = None, encoding: str = 'utf-8',
                 allow_not_found: bool = False, **kwargs) -> pd.DataFrame | None:
        filepath = self._get_filepath(filename, extension)  # in case filename is absolute path and extension is None
        if filepath is None and extension is None:
            filepath = self._get_filepath(filename, '.csv')
        if filepath is not None:
            return pd.read_csv(filepath_or_buffer=filepath, encoding=encoding, **kwargs)
        elif allow_not_found:
            return None
        else:
            raise FileNotFoundError(f"File '{filename}' not exists in input directories.")

    def read_parquet(self, filename: str, columns: list[str] | None = None, filter_dict: dict | None = None,
                     raise_not_found_error: bool = True, **kwargs) -> pd.DataFrame | None:
        extension = None if filename.endswith('.parquet') else '.parquet'
        filepath = self._get_filepath(filename, extension)
        if filepath is not None:
            import pyarrow.dataset as ds
            dataset = ds.dataset(filepath, format="parquet")
            filter_expr = None if filter_dict is None else reduce(operator.and_, [ds.field(k) == v
                                                                                  for k, v in filter_dict.items()])
            table = dataset.to_table(columns=columns, filter=filter_expr, **kwargs)
            return table.to_pandas()
        elif raise_not_found_error:
            file_lookup = filename if extension is None else f'{filename}.parquet'
            raise FileNotFoundError(f"File '{file_lookup}' not exists in input directories.")
        else:
            return None

    def _get_filepath(self, filename: str, extension: str | None) -> str | None:
        if (filename, extension) in self._cached_filepath:
            return self._cached_filepath[(filename, extension)]
        if extension is None:
            filepath = self._scan_filepath(filename)
        else:
            filepath = self._scan_filepath(f'{filename}{'' if extension.startswith('.') else '.'}{extension}')
        self._cached_filepath[(filename, extension)] = filepath
        return filepath

    def _scan_filepath(self, filename: str) -> str | None:
        if os.path.exists(filename):  # absolute path
            return filename
        if self._input_directories is None:
            return None
        for directory in self._input_directories:
            filepath = os.path.join(self._wsdir, directory, filename)
            if os.path.exists(filepath):
                return filepath
        return None  # return None if file not exists

    def _scan_results_directory(self) -> dict[str, dict]:
        files = glob.glob(os.path.join(self._wsdir, self._results_directory, f'{self._model_name}*'))
        proj, stoch, stoch_stat, log, other = {}, {}, {}, {}, {}
        for file in files:
            filename = os.path.basename(file)
            if filename.endswith('.proj.csv'):
                proj[filename] = file
            elif filename.endswith('.stoch.csv'):
                stoch[filename] = file
            elif filename.endswith('.stoch.statistic.csv'):
                stoch_stat[filename] = file
            elif filename.endswith('.runlog.html'):
                log[filename] = file
            else:
                other[filename] = file
        return {'proj': proj, 'stoch': stoch, 'stoch_stat': stoch_stat, 'log': log, 'other': other}

    def _concat_output_file_path(self, name: str) -> str:
        return os.path.join(self._wsdir, self._results_directory, f'{self._model_name}{name}')

    @property
    def input_files(self) -> dict[str, str]:
        input_files = {}
        for (filename, extension), filepath in self._cached_filepath.items():
            if filepath is None or filename in ('__proj_variables__', '__stoch_variables__', '__stoch_setting__'):
                continue
            if extension is not None:
                filename += f'{'' if extension.startswith('.') else '.'}{extension}'
            input_files[filename] = filepath
        return input_files

    @property
    def workspace_directory(self) -> str:
        return self._wsdir

    @property
    def results_directory(self) -> str | None:
        return os.path.join(self._wsdir, self._results_directory) if self._results_directory else None

    @property
    def MODEL_NAME(self) -> str:
        """str: Model alias used to prefix output files."""
        return self._model_name

    @property
    def SCENARIO(self) -> str | None:
        """str: Scenario code that identifies the set of input used for a run."""
        return self._scenario

    @property
    def SIMULATION(self) -> int | None:
        """int: Simulation."""
        return self._simulation

    @property
    def START_YEAR(self) -> int:
        """int: Year of the start date of the projection."""
        return self._start_year

    @property
    def START_MONTH(self) -> int:
        """int: Month (1-12) of the start date of the projection."""
        return self._start_month

    @property
    def END_YEAR(self) -> int | None:
        """int: Year of the end date of the projection (inclusive, months run to December)."""
        return self._end_year

    @cached_property
    def START_DATE(self) -> pd.Period:
        """pd.Period: Start date of the projection."""
        return pd.Period(f'{self.START_YEAR}-{self.START_MONTH}', freq='M')

    @cached_property
    def END_DATE(self) -> pd.Period | None:
        """pd.Period: End date of the projection."""
        if self.END_YEAR is None:
            return None
        return pd.Period(f'{self.END_YEAR}-12', freq='M')

    @cached_property
    def MAX_T(self) -> int:
        """int: Number of projection months."""
        if self.END_DATE is None:
            return 0
        max_t = (self.END_DATE - self.START_DATE).n
        if max_t < 0:
            raise ValueError(f"END_DATE ({self.END_DATE}) is earlier than START_DATE ({self.START_DATE}).")
        if max_t > 6000:
            raise ValueError(f"Projection period ({self.START_DATE} to {self.END_DATE}) exceeds 500 years.")
        return max_t

    @cached_property
    def period_time_pairs(self) -> dict[pd.Period, int]:
        return {self.START_DATE + t: t for t in range(0, self.MAX_T + 1)}

    @property
    def time(self) -> int | None:
        """pd.Period: Current projection time index."""
        return self._time

    @time.setter
    def time(self, value: int) -> None:
        """pd.Period: Current projection time index."""
        if not isinstance(value, int):
            raise TypeError(f"time: type {type(value)} is not allowed, expected 'int'.")
        if value < 0:
            raise ValueError(f"time: value {value} is not allowed, execpted non-nagative.")
        self._time = value
        self._period = self.START_DATE + value

    @property
    def period(self) -> pd.Period | None:
        """pd.Period: Current projection period."""
        return self._period

    @period.setter
    def period(self, value: pd.Period) -> None :
        """pd.Period: Current projection period."""
        if not isinstance(value, pd.Period):
            raise TypeError(f"period: type {type(value)} is not allowed, expected 'pd.Period'.")
        self._period = value
        self._time = (value - self.START_DATE).n

    @property
    def proj_variables(self) -> list[ProjVariable]:
        return self._proj_variables

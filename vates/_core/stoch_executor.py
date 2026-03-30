import glob
import os
import warnings
from abc import abstractmethod
from datetime import datetime
import pandas as pd
from functools import cached_property

from .proj_model_engine import ProjModelEngine
from ._html_generator import generate_runlog_html
from .utils import ValidatedNumber, ValidatedString, ValidatedList, parse_str_to_int_list


class StochExecutor:
    """Stochastic model executor."""

    _model_name = ValidatedString(len_min=1, max_sets=1)
    _model_desc = ValidatedString(max_sets=1)
    _start_year = ValidatedNumber(value_type=int, value_min=1900, value_max=5999, max_sets=1)
    _start_month = ValidatedNumber(value_type=int, value_lst=range(1, 13), max_sets=1)
    _end_year = ValidatedNumber(value_type=int, value_min=1900, value_max=5999, allow_none=True, max_sets=1)
    _scenario = ValidatedString(allow_none=True, max_sets=1)
    _simulations = ValidatedList(item_type=int, allow_none=True, len_min=0, len_max=100_000, max_sets=1)
    _wsdir = ValidatedString(max_sets=1)
    _input_directories = ValidatedList(item_type=str, allow_none=True, len_min=0, max_sets=1)
    _results_directory = ValidatedString(max_sets=1)
    _max_workers = ValidatedNumber(value_type=int, value_min=1, max_sets=1)

    def __init__(
            self,
            model_cls,
            model_name: str,
            start_year: int,
            start_month: int,
            end_year: int,
            model_desc: str | None = None,
            scenario: str | None = None,
            simulations: str | None = None,
            workspace_directory: str | None = None,
            input_directories: list[str] | None = None,
            results_directory: str = '',
            max_workers: int | None = None,
            *args,
            **kwargs
    ) -> None:
        self._exe_start_time: str = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self._model_cls = model_cls
        self._model_name: str = model_name
        self._model_desc: str = model_desc if model_desc else f'**Stochastic** {self._model_cls.__doc__}'
        self._start_year: int = start_year
        self._start_month: int = start_month
        self._end_year: int | None = end_year
        self._scenario: str | None = scenario
        self._sims_str: str | None = simulations
        self._simulations: list[int] = [] if simulations is None else parse_str_to_int_list(simulations)
        self._wsdir: str = workspace_directory if workspace_directory else os.getcwd()
        self._input_directories: list[str] | None = input_directories
        self._results_directory: str = results_directory
        if self._results_directory:
            os.makedirs(os.path.join(self._wsdir, self._results_directory), exist_ok=True)
            existing_files = glob.glob(os.path.join(self._wsdir, self._results_directory, f'{self._model_name}*'))
            for f in existing_files:
                os.remove(f)
        self._max_workers: int = self._parse_max_workers(max_workers)
        self._cached_filepath: dict = {}
        self._input_files: dict = {}
        self._sim_exe_results = {'success': [], 'fail': []}
        self._init_kwargs: dict = kwargs

    @staticmethod
    def _parse_max_workers(requested_workers: int | None) -> int:
        if requested_workers is None:
            warnings.warn(f"max_workers is set to 1, reason: max_workers not specified.")
            return 1
        if not isinstance(requested_workers, int):
            warnings.warn(f"max_workers is set to 1, reason: type <{type(requested_workers)}> is not allowed, expect int.")
            return 1
        if requested_workers <= 0:
            warnings.warn(f"max_workers is set to 1, reason: input={requested_workers}, exptect positive.")
            return 1

        from multiprocessing import cpu_count
        _cpu_count = cpu_count()
        if requested_workers <= _cpu_count:
            return requested_workers
        else:
            warnings.warn(f"max_workers is set to {_cpu_count}, reason: requested {requested_workers} > cpu count.")
            return _cpu_count

    def run(self):
        self.pre_stoch_calculations()
        self._run_all_simulations()
        self.post_stoch_calculations()
        self._write_runlog()

    @abstractmethod
    def pre_stoch_calculations(self):
        pass

    @abstractmethod
    def post_stoch_calculations(self):
        pass

    def _run_all_simulations(self):
        self._run_simulations_multiprocess()
        self._write_stochastic_statistic()

    load_json = ProjModelEngine.load_json
    read_csv = ProjModelEngine.read_csv
    workspace_directory = ProjModelEngine.workspace_directory
    _concat_output_file_path = ProjModelEngine._concat_output_file_path
    _get_filepath = ProjModelEngine._get_filepath
    _scan_filepath = ProjModelEngine._scan_filepath
    _scan_results_directory = ProjModelEngine._scan_results_directory

    @staticmethod
    def _create_batches(simulations: list[int], n_batches: int) -> list[tuple[int, ...]]:
        """Split simulations into batches for workers"""
        from itertools import batched
        quotient, remainder = divmod(len(simulations), n_batches)
        batched_simulations = batched(simulations, n=quotient + (1 if remainder > 0 else 0))
        return [batch for batch in batched_simulations]

    def _run_simulation_batch(self, simulation_batch: tuple[int, ...], batch_id: str):
        result: dict = {'input_files': {}, 'success': [], 'fail': []}

        for simulation in simulation_batch:
            try:
                model_instance = self._model_cls(
                    model_name=self._model_name,
                    model_desc=self._model_desc,
                    start_year=self._start_year,
                    start_month=self._start_month,
                    end_year=self._end_year,
                    scenario=self._scenario,
                    simulation=simulation,
                    workspace_directory=self._wsdir,
                    input_directories=self._input_directories,
                    results_directory=self._results_directory,
                    is_delete_existing_results=False,
                    enable_proj_result=simulation == self._simulations[0],
                    stoch_result_mode='w' if simulation == simulation_batch[0] else 'a',
                    stoch_result_file_id=batch_id,
                    enable_run_log=False,
                    **self._init_kwargs,
                )
                res = model_instance.run()
                result['success'].append((simulation, res))
                result['input_files'] |= model_instance.input_files
            except Exception as e:
                result['fail'].append((simulation, str(e)))
                warnings.warn(f'! simulation {simulation} failed: {str(e)}')

            del model_instance

        return result

    def _run_simulations_multiprocess(self):
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_workers = min(self._max_workers, len(self._simulations))
        simulation_batches = self._create_batches(self._simulations, n_workers)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self._run_simulation_batch, batch, str(i))
                       for i, batch in enumerate(simulation_batches, 1)]

            for future in as_completed(futures):
                res = future.result()
                self._sim_exe_results['success'].extend(res['success'])
                self._sim_exe_results['fail'].extend(res['fail'])
                self._input_files |= res['input_files']

    # def _run_simulations_singleprocess(self):
    #     res = self._run_simulation_batch(self._simulations, '1')
    #     self._sim_exe_results['success'].extend(res['success'])
    #     self._sim_exe_results['fail'].extend(res['fail'])
    #     self._input_files |= res['input_files']

    def _write_stochastic_statistic(self):
        stoch_setting = self.load_json('__stoch_setting__', allow_not_found=True)
        if stoch_setting is None or stoch_setting.get('statistic', None) is None:
            return

        stoch_file_paths = self.list_stoch_file_paths()
        if len(stoch_file_paths) == 0: return

        df = pd.concat((pd.read_csv(f) for f in stoch_file_paths), ignore_index=True)
        meta_cols = ['group', 'owner', 'variable']
        time_cols = [col for col in df.columns if col not in meta_cols + ['simulation']]

        stat_dict = stoch_setting['statistic']
        stat_funcs: dict[str, ...] = {stat: stat for stat in ['mean', 'std', 'median', 'max', 'min'] if stat in stat_dict}
        if 'perc%' in stat_dict:
            for q in stat_dict['perc%']:
                if not 0 < q < 100: raise ValueError(f"Invalidd perc%={q}, expected 0 to 100.")
                stat_funcs[f"perc_{q}"] = lambda x, p=q: x.quantile(p / 100)
        agg_dict = {f"{col}:{stat}": (col, func) for col in time_cols for stat, func in stat_funcs.items()}

        df = df.copy()
        with warnings.catch_warnings():  # latest version of pandas might raise PerformanceWarning
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            statistic = df.groupby(meta_cols).agg(**agg_dict).reset_index()

        output_file = self._concat_output_file_path('.stoch.statistic.csv')
        statistic.to_csv(output_file, index=False)

    def _write_runlog(self):
        result_files = self._scan_results_directory()
        runlog: dict = {
            "model_name": self._model_name,
            "model_desc": self._model_desc,
            "exe_start_time": self._exe_start_time,
            "exe_end_time": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "failed_simulations": self._sim_exe_results['fail'],
            "setting": {
                "start_year": self._start_year,
                "start_month": self._start_month,
                "end_year": self._end_year,
                "scenario": self._scenario,
                "simulations": self._sims_str,
                "workspace_directory": self._wsdir,
                "input_directories": self._input_directories,
                "results_directory": self._results_directory,
                "max_workers": self._max_workers,
            },
            "input_files": self._input_files,
            "proj_result_files": result_files['proj'],
            "stoch_result_files": result_files['stoch'] | result_files['stoch_stat'],
            "other_result_files": result_files['other'],
        }

        html_content = generate_runlog_html(runlog)
        runlog_file = self._concat_output_file_path(".runlog.html")

        with open(runlog_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def list_stoch_file_paths(self) -> list:
        filepath_lst = glob.glob(os.path.join(self._wsdir, self._results_directory, f'{self._model_name}*.stoch.csv'))
        if filepath_lst:
            return filepath_lst
        else:
            return []

    def get_stoch_stat_file_path(self) -> str | None:
        filepath = self._concat_output_file_path('.stoch.statistic.csv')
        if os.path.exists(filepath):
            return filepath
        else:
            return None

    @property
    def MODEL_NAME(self) -> str:
        """str: Model alias used to prefix output files."""
        return self._model_name

    @property
    def SCENARIO(self) -> str | None:
        """str: Scenario code that identifies the set of input used for a run."""
        return self._scenario

    @property
    def SIMULATIONS(self) -> list[int] | None:
        """int: Simulation."""
        return self._simulations

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

    @property
    def ALL_SIM_KWARGS(self) -> dict:
        return self._init_kwargs

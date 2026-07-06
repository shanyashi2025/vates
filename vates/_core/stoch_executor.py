import os
import glob
import warnings
from pathlib import Path
from abc import abstractmethod
from datetime import datetime
import pandas as pd
from functools import cached_property

from vates._core.proj_model_engine import ProjModelEngine
from vates._core.utils import ValidatedNumber, ValidatedString, ValidatedList, parse_str_to_int_list


class StochExecutor:
    """Stochastic model executor."""

    _name = ValidatedString(len_min=1, max_sets=1)
    _description = ValidatedString(max_sets=1)
    _start_year = ValidatedNumber(value_type=int, value_min=1900, value_max=5999, max_sets=1)
    _start_month = ValidatedNumber(value_type=int, value_lst=range(1, 13), max_sets=1)
    _end_year = ValidatedNumber(value_type=int, value_min=1900, value_max=5999, max_sets=1)
    _end_month = ValidatedNumber(value_type=int, value_lst=range(1, 13), max_sets=1)
    _scenario = ValidatedString(allow_none=True, max_sets=1)
    _simulations = ValidatedList(item_type=int, len_min=0, len_max=100_000, max_sets=1)
    _wsdir = ValidatedString(max_sets=1)
    _input_directories = ValidatedList(item_type=str, allow_none=True, len_min=0, max_sets=1)
    _results_directory = ValidatedString(max_sets=1)
    _max_workers = ValidatedNumber(value_type=int, value_min=1, max_sets=1)

    def __init__(
            self,
            name: str,
            *,
            description: str = '...',
            model_cls,
            simulations: str,
            start_year: int,
            start_month: int = 12,
            end_year: int | None = None,
            end_month: int = 12,
            scenario: str | None = None,
            workspace_directory: str | None = None,
            input_directories: list[str] | None = None,
            results_directory: str = '',
            max_workers: int | None = None,
    ) -> None:
        self._exec_start_time: datetime = datetime.now()
        self._name: str = name
        self._description: str = description
        self._model_cls = model_cls
        self._sims_str: str = simulations
        self._simulations: list[int] = parse_str_to_int_list(simulations)
        self._start_year: int = start_year
        self._start_month: int = start_month
        self._end_year: int = end_year or start_year
        self._end_month: int = end_month
        self._scenario: str | None = scenario
        self._wsdir: str = workspace_directory or os.getcwd()
        self._input_directories: list[str] | None = input_directories
        self._results_directory: str = results_directory
        self._max_workers: int = self._parse_max_workers(max_workers)
        self._cached_filepath: dict = {}
        self._output_files: list = []
        self._sim_input_files: dict = {}
        self._sim_output_files: dict = {}
        self._enable_write_runlog: bool = True
        self._all_sim_params: dict = {}

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

    def run(self) -> dict:
        self.results_directory.mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(self.results_directory / f'{self._name}*')):
            if f.endswith(('.proj.csv', '.stoch.csv', 'stoch.stat.csv', '.runlog.json')):
                os.remove(f)
        self.pre_stoch_calculations()
        self._run_all_simulations()
        self.post_stoch_calculations()
        return self._generate_runlog()

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
    read_excel = ProjModelEngine.read_excel
    read_parquet = ProjModelEngine.read_parquet
    _concat_output_file_path = ProjModelEngine._concat_output_file_path
    _get_filepath = ProjModelEngine._get_filepath
    _scan_filepath = ProjModelEngine._scan_filepath
    _generate_runlog = ProjModelEngine._generate_runlog

    @staticmethod
    def _create_batches(simulations: list[int], n_batches: int) -> list[tuple[int, ...]]:
        """Split simulations into batches for workers"""
        from itertools import batched
        quotient, remainder = divmod(len(simulations), n_batches)
        batched_simulations = batched(simulations, n=quotient + (1 if remainder > 0 else 0))
        return [batch for batch in batched_simulations]

    def _run_simulation_batch(self, simulation_batch: tuple[int, ...], batch_id: str) -> dict:
        result: dict = {'input_files': {}, 'output_files': {}, 'err_msg': []}

        for simulation in simulation_batch:
            try:
                model_instance = self._model_cls(
                    name=self._name,
                    description=self._description,
                    start_year=self._start_year,
                    start_month=self._start_month,
                    end_year=self._end_year,
                    end_month=self._end_month,
                    scenario=self._scenario,
                    simulation=simulation,
                    workspace_directory=self._wsdir,
                    input_directories=self._input_directories,
                    results_directory=self._results_directory,
                    is_delete_existing_results=False,
                    enable_write_proj_result=simulation == self._simulations[0],
                    stoch_result_mode='w' if simulation == simulation_batch[0] else 'a',
                    stoch_result_file_id=batch_id,
                    enable_write_runlog=False,
                    **self._all_sim_params,
                )
            except Exception as e:
                sim_err = f'simulation #{simulation} init error: {str(e)}'
                result['err_msg'].append(sim_err)
                warnings.warn(f'! {sim_err}')
                continue # `model_instance` is not successfully initialized, skip `.run()`

            try:
                res = model_instance.run()
                result['input_files'] |= res.get('input_files', {})
                result['output_files'] |= res.get('output_files', {})
            except Exception as e:
                sim_err = f'simulation #{simulation} run error: {str(e)}'
                result['err_msg'].append(sim_err)
                warnings.warn(f'! {sim_err}')

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
                self._sim_input_files |= res['input_files']
                self._sim_output_files |= res['output_files']

    def _write_stochastic_statistic(self):
        stoch_setting = self.load_json('__stoch_setting__', allow_not_found=True)
        if stoch_setting is None or stoch_setting.get('statistic', None) is None:
            return

        stoch_file_paths = sorted([f for f in self._sim_output_files if f and f.endswith('.stoch.csv')])
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

        output_file = self._concat_output_file_path('.stoch.stat.csv')
        statistic.to_csv(output_file, index=False)
        self._output_files.append(output_file)

    @property
    def _runlog(self) -> dict[str, ...]:
        def _file_stat(file_path: Path) -> dict[str, str]:
            """Get file modification time and size"""
            stat_info = os.stat(file_path)
            file_mtime = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size_bytes = stat_info.st_size
            if size_bytes < 1024 * 1024: file_size = f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024: file_size = f"{size_bytes / (1024 * 1024):.1f} MB"
            else: file_size = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
            return {"mtime": file_mtime, "size": file_size}

        return {
            "model": {
                "name": self._name,
                "description": self._description,
                "projection_engine": self._model_cls.__name__,
                "stochastic_executor": self.__class__.__name__,
            },
            "execution": {
                "start": self._exec_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            "run_setting": {
                "start_year": self._start_year,
                "start_month": self._start_month,
                "end_year": self._end_year,
                "end_month": self._end_month,
                "scenario": self._scenario,
                "simulations": self._sims_str,
                "workspace_directory": str(self.workspace_directory),
                "input_directories": self._input_directories,
                "results_directory": self._results_directory,
                "max_workers": self._max_workers,
            },
            "output_files": self._sim_output_files | {str(f): _file_stat(f) for f in self._output_files},
        }

    @cached_property
    def workspace_directory(self) -> Path:
        return Path(self._wsdir).resolve()

    @cached_property
    def results_directory(self) -> Path | None:
        return (Path(self._wsdir) / self._results_directory).resolve()

    @property
    def MODEL_NAME(self) -> str:
        """str: Model alias used to prefix output files."""
        return self._name

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
    def END_YEAR(self) -> int:
        """int: Year of the end date of the projection."""
        return self._end_year

    @property
    def END_MONTH(self) -> int:
        """int: Month (1-12) of the end date of the projection."""
        return self._end_month

    @cached_property
    def START_DATE(self) -> pd.Period:
        """pd.Period: Start date of the projection."""
        return pd.Period(f'{self.START_YEAR}-{self.START_MONTH}', freq='M')

    @cached_property
    def END_DATE(self) -> pd.Period | None:
        """pd.Period: End date of the projection."""
        return pd.Period(f'{self.END_YEAR}-{self.END_MONTH}', freq='M')

    @cached_property
    def MAX_T(self) -> int:
        """int: Number of projection months."""
        if self.END_DATE is None:
            return 0
        max_t = (self.END_DATE - self.START_DATE).n
        if max_t < 0:
            raise ValueError(f"END_DATE ({self.END_DATE}) cannot be earlier than START_DATE ({self.START_DATE}).")
        if max_t > 6000:
            raise ValueError(f"Projection period ({self.START_DATE} to {self.END_DATE}) exceeds 500 years.")
        return max_t

    @property
    def ALL_SIM_PARAMS(self) -> dict:
        return self._all_sim_params

    @ALL_SIM_PARAMS.setter
    def ALL_SIM_PARAMS(self, value) -> None:
        self._all_sim_params = value

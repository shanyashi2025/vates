import glob
import inspect
import json
import os
import pandas as pd
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import batched
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Self, get_type_hints

from vates._core.proj_model_engine import ProjModelEngine
from vates._core._utils import RunConfig


class StochExecutor:
    """Stochastic model executor."""

    _proj_cls: type[ProjModelEngine]
    _projection: Callable
    _run_config: RunConfig
    _sims_str: str

    include_traced_message = ProjModelEngine.include_traced_message
    load_json = ProjModelEngine.load_json
    read_csv = ProjModelEngine.read_csv
    read_excel = ProjModelEngine.read_excel
    _concat_output_file_path = ProjModelEngine._concat_output_file_path
    get_filepath = ProjModelEngine.get_filepath
    _search_filepath = ProjModelEngine._search_filepath
    proj_result = ProjModelEngine.proj_result

    def __init__(
        self,
        *,
        model_name: str,
        description: str = '...',
    ) -> None:
        self._model_name: str = str(model_name)
        self._description: str = str(description)

        # runtime stuffs
        self._cached_filepath: dict[str, tuple] = {}
        self._result_files: set = set()
        self._messages: list[str] = []
        self._sim_messages: list = []
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
            ValueError: If `func` is not callable.
        """
        if hasattr(self, '_projection'):
            raise ValueError(f"{self._projection} is already bound.")
        if not callable(func):
            raise ValueError(f"Cannot bind un-callable object: {func}.")

        sig_params = inspect.signature(func).parameters
        if len(sig_params) == 0:
            raise ValueError(f"Function '{func.__name__}' has no argument, cannot be bound.")

        first_arg_name = list(sig_params.keys())[0]
        proj_cls = get_type_hints(func).get(first_arg_name)
        if proj_cls is not None:
            super().__setattr__('_proj_cls', proj_cls)
            self.include_traced_message(
                f"INFO: {proj_cls} is set as the projection model engine according to type hints of "
                f"first argument '{first_arg_name}'."
            )
        else:
            raise ValueError(f"First argument '{first_arg_name}': type hint is missing.")

        super().__setattr__('_projection', func)
        self.include_traced_message(f"INFO: Function {func} has been bound to {self}.")
        return self

    def configure_run(
        self,
        *,
        start_year: int | None = None,
        start_month: int | None = None,
        end_year: int | None = None,
        end_month: int | None = None,
        scenario: str | None = None,
        simulations: str,
        workspace_directory: str | None = None,
        input_directories: list[str] | None = None,
        results_directory: str | None = None,
        max_workers: int | None = None,
    ) -> Self:
        """Set the configuration for a run.

        Args:
            start_year (int): Projection start year.
            start_month (int, optional): Projection start month. Defaults to 12.
            end_year (int, optional): Projection end year. Defaults to `start_year`.
            end_month (int, optional): Projection end month. Defaults to 12.
            scenario (str, optional): Scenario. Defaults to None.
            simulations: (str, optional): Simulations.
            workspace_directory (str, optional): Workspace directory. Defaults to `{os.getcwd()}`.
            input_directories (list[str], optional): List of input directory. Defaults to None.
            results_directory (str, optional): Results directory. Defaults to 'results/`scenario`'.
            max_workers (int, optional): Max workers. Defaults to 1.
        """
        if hasattr(self, '_run_config'):
            raise ValueError(f"Run configuration is already set.")

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
        if max_workers is None:
            max_workers = 1
            none_items.append(f"max_workers='{max_workers}'")
        super().__setattr__('_sims_str', simulations)
        super().__setattr__('_run_config', RunConfig.create(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            scenario=scenario,
            simulations=simulations,
            workspace_directory=workspace_directory,
            input_directories=input_directories,
            results_directory=results_directory,
            is_delete_existing_results=True,
            enable_write_proj_result=False,
            stoch_result_file_mode=None,
            stoch_result_file_id=None,
            enable_write_runlog=True,
            max_workers=self._parse_max_workers(max_workers),
        ))

        if len(none_items) > 0:
            self.include_traced_message(f"INFO: Following items are set by default: {', '.join(none_items)}.")

        return self

    def _parse_max_workers(self, requested_workers: int | None) -> int:
        if not isinstance(requested_workers, int):
            self.include_traced_message(f"max_workers is set to 1: invalid type '{type(requested_workers)}', expect int.")
            return 1
        if requested_workers <= 0:
            self.include_traced_message(f"max_workers is set to 1: request={requested_workers}, exptect positive.")
            return 1

        _cpu_count = cpu_count()
        if requested_workers <= _cpu_count:
            return requested_workers
        else:
            self.include_traced_message(f"max_workers is set to {_cpu_count}: requested {requested_workers} > cpu_count.")
            return _cpu_count

    def run(
        self,
        *,
        projection_args: dict[str, ...] | None = None,
    ) -> dict:
        if not hasattr(self, '_projection'):
            raise ValueError(f"Projection function has not been bound.")
        if not hasattr(self, '_proj_cls'):
            raise ValueError(f"Projection model engine class is None.")
        if not hasattr(self, '_run_config'):
            raise ValueError("Run configuration has not been set.")
        projection_args = projection_args or {}

        if self.results_directory_path.is_dir():
            remove_pattern = ('.proj.csv', '.stoch.csv', 'stoch.stat.csv', '.runlog.json')
            for f in glob.glob(str(self.results_directory_path / f'{self._model_name}*')):
                if f.endswith(remove_pattern):
                    os.remove(f)
                else:
                    self.include_traced_message(f"INFO: Exsiting file NOT deleted: '{f}'.")
        else:
            os.makedirs(self.results_directory_path, exist_ok=True)

        exec_start_time = datetime.now()
        exec_success = self._run_simulations_multiprocess(projection_args=projection_args)
        self._write_stochastic_statistic()
        self._dump_runlog(exec_success, exec_start_time, datetime.now())
        return self._runlog

    @staticmethod
    def _create_batches(simulations: list[int], n_batches: int) -> list[tuple[int, ...]]:
        """Split simulations into batches for workers"""
        quotient, remainder = divmod(len(simulations), n_batches)
        batched_simulations = batched(simulations, n=quotient + (1 if remainder > 0 else 0))
        return [batch for batch in batched_simulations]

    def _run_simulation_batch(
        self,
        *,
        simulation_batch: tuple[int, ...],
        batch_id: str,
        projection_args: dict[str, ...]
    ) -> tuple[bool, list, list]:
        success: bool = True
        result: list = []
        output_files: list = []

        for simulation in simulation_batch:
            try:
                model_instance = self._proj_cls(
                    model_name=self._model_name,
                    description=self._description
                ).bind_projection(
                    self._projection
                ).configure_run(
                    simulation=simulation,
                    stoch_result_file_id=batch_id,
                    stoch_result_file_mode='w' if simulation == simulation_batch[0] else 'a',
                    enable_write_proj_result=(simulation == self._run_config.simulations[0]),
                    start_year=self.START_YEAR,
                    start_month=self.START_MONTH,
                    end_year=self.END_YEAR,
                    end_month=self.END_MONTH,
                    scenario=self.SCENARIO,
                    workspace_directory=self._run_config.workspace_directory,
                    input_directories=self._run_config.input_directories,
                    results_directory=self._run_config.results_directory,
                    is_delete_existing_results=False,
                    enable_write_runlog=False,
                )
            except Exception as e:
                traceback.print_exc()
                result.append(f"{simulation=}: init error:\n{traceback.format_exc()}")
                success = False
                continue # `model_instance` is not successfully initialized, skip `.run()`

            try:
                res = model_instance.run(projection_args=projection_args)
                result.append(res)
                output_files.extend(model_instance._result_files)

            except Exception as e:
                traceback.print_exc()
                result.append(f"{simulation=}: run error:\n{traceback.format_exc()}")
                success = False

            del model_instance

        return success, result, output_files

    def _run_simulations_multiprocess(
        self,
        *,
        projection_args: dict[str, ...]
    ) -> bool:
        n_workers = min(self._run_config.max_workers, len(self._run_config.simulations))
        simulation_batches = self._create_batches(self._run_config.simulations, n_workers)
        exec_success = True

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    self._run_simulation_batch,
                    simulation_batch=batch,
                    batch_id=str(i),
                    projection_args=projection_args
                )
                for i, batch in enumerate(simulation_batches, 1)
            ]

            for future in as_completed(futures):
                success, result, output_files = future.result()
                exec_success = exec_success and success
                self._sim_messages.extend(result)
                self._result_files.update(output_files)

        return exec_success

    def _write_stochastic_statistic(self):
        stoch_setting = self.load_json('__stoch_setting__.json', allow_not_found=True)
        if stoch_setting is None or stoch_setting.get('statistic', None) is None:
            return

        stoch_file_paths = sorted([f for f in self._result_files if f and str(f).endswith('.stoch.csv')])
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
        self._result_files.add(output_file)

    @property
    def runlog(self) -> dict[str, ...] | None:
        return self._runlog

    def _dump_runlog(self, exec_success: bool, exec_start_time: datetime, exec_end_time: datetime) -> None:
        exec_total_seconds = int((exec_end_time - exec_start_time).total_seconds())
        exec_hours = exec_total_seconds // 3600
        exec_minutes = (exec_total_seconds % 3600) // 60
        exec_seconds = exec_total_seconds % 60

        self._runlog.update({
            "model_name": self._model_name,
            "description": self._description,
            "srouce_code": {
                "projection_function": f"{inspect.getfile(self._projection)}: <function '{self._projection.__name__}'>",
                "projection_engine": f"{inspect.getfile(self._proj_cls)}: <class '{self._proj_cls.__name__}'>",
                "stochastic_executor": f"{inspect.getfile(type(self))}: <class '{type(self).__name__}'>",
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
                "simulations": self._sims_str,
                "workspace_directory": self._run_config.workspace_directory,
                "input_directories": self._run_config.input_directories,
                "results_directory": self._run_config.results_directory,
                "max_workers": self._run_config.max_workers,
            },
            "environment": self._environ,
            "results": list(map(str, self._result_files)),
            "messages": self._messages + self._sim_messages,
        })

        if self._run_config.enable_write_runlog:
            with open(self._concat_output_file_path(".runlog.json"), 'w', encoding='utf-8') as jsonfile:
                json.dump(self._runlog, jsonfile, indent=4)

    @property
    def _environ(self) -> dict[str, str]:
        return {
            "COMPUTERNAME": os.getenv("COMPUTERNAME"),
            "USERNAME": os.getenv("USERNAME"),
            "USERDOMAIN": os.getenv("USERDOMAIN"),
            "VIRTUAL_ENV": os.getenv("VIRTUAL_ENV"),
            "process_id": os.getpid(),
        }

    @property
    def workspace_directory_path(self) -> Path:
        return self._run_config.workspace_directory_path

    @property
    def results_directory_path(self) -> Path:
        return self._run_config.results_directory_path

    @property
    def MODEL_NAME(self) -> str:
        """str: Model name used for result files."""
        return self._model_name

    @property
    def SCENARIO(self) -> str:
        """str: Scenario code used for a run."""
        return self._run_config.scenario

    @property
    def SIMULATIONS(self) -> list[int]:
        """list[int]: Simulations for a stochastic run."""
        return self._run_config.simulations

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

    def __setattr__(self, name, value):
        if hasattr(self, '_initialized'):
            if name in type(self).__dict__:  # check if the attribute name already exists in the class definition
                raise AttributeError(f"Cannot overwrite protected member '{name}'")
            elif name.startswith('_'):
                raise AttributeError(f"Cannot add a private member (underscore-prefixed) '{name}'.")
            if not hasattr(self, name) and hasattr(self, "_messages"):
                self.include_traced_message(f"INFO: Add member: '{name}' {type(value)}")
        super().__setattr__(name, value)

import gc
import glob
import inspect
import json
import os
import pandas as pd
import traceback
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import batched
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Self, get_type_hints

from vates._core._helper import get_environ, get_f_filename_lineno
from vates._core.proj_model_engine import ProjModelEngine
from vates._core._run_configuration import RunConfiguration


class StochExecutor:
    """Stochastic model executor."""

    _proj_cls: type[ProjModelEngine]
    _projection: Callable
    _run_config: RunConfiguration
    _n_sims: int

    load_json = ProjModelEngine.load_json
    read_csv = ProjModelEngine.read_csv
    read_excel = ProjModelEngine.read_excel
    get_filepath = ProjModelEngine.get_filepath
    proj_result = ProjModelEngine.proj_result

    def __init__(
        self,
        *,
        slug: str,
        description: str = '...',
    ) -> None:
        self._slug: str = str(slug)
        self._description: str = str(description)

        # runtime stuffs
        self._filepath_cache: dict[str, Path | None] = {}
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
            ValueError: If `func` is already bound, not callable, or a class.
            ValueError: If `func` has no argument. A projection function bound
                through `StochExecutor` must take the model engine as its first
                argument, because its type annotation selects the worker engine
                class.
            ValueError: If the type hints of `func` cannot be resolved (e.g.
                string/forward annotations referencing names that are not
                importable at module scope).
            ValueError: If `func`'s first argument is missing an annotation,
                or is annotated with a type that is not a `ProjModelEngine`
                subclass.
        """
        if hasattr(self, '_projection'):
            raise ValueError(f"{self._projection} is already bound.")
        if not callable(func):
            raise ValueError(f"Cannot bind un-callable object: {func}.")
        if inspect.isclass(func):
            raise ValueError(f"Cannot bind a class: {func}.")

        sig_params = inspect.signature(func).parameters
        if len(sig_params) == 0:
            raise ValueError(f"Function '{func.__name__}' has no argument, cannot be bound.")
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
        proj_cls = hints.get(first_arg_name)
        if proj_cls is None:
            raise ValueError(
                f"Function '{func.__name__}' first argument '{first_arg_name}': must annotate with the "
                f"projection engine type (a 'ProjModelEngine' subclass), e.g. ': ProjModelEngine'."
            )
        if not isinstance(proj_cls, type) or not issubclass(proj_cls, ProjModelEngine):
            raise ValueError(
                f"Function '{func.__name__}' first argument '{first_arg_name}': annotated type '{proj_cls}' "
                f"must be a 'ProjModelEngine' subclass (it is used as the worker engine class)."
            )
        super().__setattr__('_proj_cls', proj_cls)
        super().__setattr__('_projection', func)
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
        simulations: list[int] | tuple[int],
        workspace_directory: str | Path = Path.cwd(),
        input_directories: list[str | Path] | tuple[str | Path, ...] = (),
        results_directory: str | Path ="",
        max_workers: int = 1,
    ) -> Self:
        """Set the configuration for a run.

        Args:
            start_year (int): Projection start year.
            start_month (int, optional): Projection start month. Defaults to 12.
            end_year (int, optional): Projection end year. Defaults to `start_year`.
            end_month (int, optional): Projection end month. Defaults to 12.
            scenario (str, optional): Scenario. Defaults to empty str `""`.
            simulations: (list[int] | tuple[int]): Simulations.
            workspace_directory (str | Path, optional): Workspace directory, must be absolute path. Defaults to `{Path.cwd()}`.
            input_directories (list[str | Path] | tuple[str | Path, ...], optional): List or tuple of input directory,
                relative paths are resolved against `workspace_directory`. Defaults to empty tuple `()`.
            results_directory (str | Path, optional): Results directory, relative paths are resolved against
                `workspace_directory`. Defaults to `{Path.cwd()}`.
            max_workers (int, optional): Max workers. Defaults to 1.
        """
        if hasattr(self, '_run_config'):
            raise ValueError(f"Run configuration is already set.")
        if start_year is None:
            raise ValueError(f"start_year: value 'None' is not allowed.")
        start_date = pd.Period(f"{start_year}-{start_month}", freq='M')
        end_date = pd.Period(f"{end_year or start_year}-{end_month}", freq='M')
        workspace_directory_path = Path(workspace_directory)
        super().__setattr__('_n_sims', len(simulations))
        super().__setattr__('_run_config', RunConfiguration(
            start_date=start_date,
            end_date=end_date,
            scenario=scenario,
            simulations=simulations,
            workspace_directory_path=workspace_directory_path,
            input_directory_paths=[workspace_directory_path / d for d in input_directories],  # resolve against workspace_directory
            results_directory_path=workspace_directory_path / Path(results_directory),  # resolve against workspace_directory
            is_delete_existing_results=True,
            enable_write_proj_result=False,
            stoch_result_file_mode="n",
            stoch_result_file_id="",
            enable_write_runlog=True,
            max_workers=self._apply_max_workers(max_workers),
            simulation=0,
        ))
        return self

    def _apply_max_workers(self, requested_workers: int) -> int:
        if not isinstance(requested_workers, int):
            f, l = get_f_filename_lineno()
            self._messages.append(f"{f}:{l}: max_workers is set to 1: invalid type '{type(requested_workers)}', expect int.")
            return 1
        if requested_workers <= 0:
            f, l = get_f_filename_lineno()
            self._messages.append(f"{f}:{l}: max_workers is set to 1: request={requested_workers}, expected positive.")
            return 1

        n_cpu = cpu_count()
        if requested_workers > n_cpu:
            f, l = get_f_filename_lineno()
            self._messages.append(f"{f}:{l}: max_workers is set to {n_cpu}: requested {requested_workers} > cpu_count.")
        return min(requested_workers, n_cpu)

    def run(
        self,
        *,
        projection_kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        if not hasattr(self, '_projection'):
            raise ValueError(f"Projection function has not been bound.")
        if not hasattr(self, '_proj_cls'):
            raise ValueError(f"Projection model engine class is None.")
        if not hasattr(self, '_run_config'):
            raise ValueError("Run configuration has not been set.")

        if self.RESULTS_DIRECTORY_PATH.is_dir():
            remove_pattern = ('.proj.csv', '.stoch.csv', 'stoch.stat.csv', '.runlog.json')
            for f in glob.glob(str(self.RESULTS_DIRECTORY_PATH / f'{self._slug}*')):
                if f.endswith(remove_pattern):
                    os.remove(f)
                else:
                    f, l = get_f_filename_lineno()
                    self._messages.append(f"{f}:{l}: [INFO] Existing file NOT deleted: '{f}'.")
        else:
            os.makedirs(self.RESULTS_DIRECTORY_PATH, exist_ok=True)

        exec_start_time = datetime.now()
        exec_success = self._run_simulations_multiprocess(projection_kwargs=projection_kwargs)
        self._write_stochastic_statistic()
        self._dump_runlog(exec_success, exec_start_time, datetime.now())
        return self

    def __call__(
        self,
        *,
        projection_kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        return self.run(projection_kwargs=projection_kwargs)

    @staticmethod
    def _create_batches(simulations: list[int] | tuple[int, ...], n_batches: int) -> list[tuple[int, ...]]:
        """Split simulations into batches for workers"""
        quotient, remainder = divmod(len(simulations), n_batches)
        batched_simulations = batched(simulations, n=quotient + (1 if remainder > 0 else 0))
        return [batch for batch in batched_simulations]

    def _run_simulation_batch(
        self,
        *,
        simulation_batch: tuple[int, ...],
        batch_id: str,
        projection_kwargs: Mapping[str, Any] | None
    ) -> tuple[bool, list, list]:
        success: bool = True
        messages: list = []
        output_files: list = []

        for simulation in simulation_batch:
            try:
                model_instance = self._proj_cls(
                    slug=self._slug,
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
                    workspace_directory=self.WORKSPACE_DIRECTORY_PATH,
                    input_directories=self._run_config.input_directory_paths,
                    results_directory=self.RESULTS_DIRECTORY_PATH,
                    is_delete_existing_results=False,
                    enable_write_runlog=False,
                )
            except Exception:
                traceback.print_exc()
                messages.append(f"{simulation=}: `__init__()` ERROR:\n{traceback.format_exc()}")
                success = False
                continue # `model_instance` is not successfully initialized, skip `.run()`

            try:
                model_instance.run(projection_kwargs=projection_kwargs)
                runlog = model_instance.runlog
                messages.append(runlog)
                output_files.extend(model_instance._result_files)
                if success and not ("execution" in runlog and runlog["execution"].get("success")):
                    success = False

            except Exception:
                traceback.print_exc()
                messages.append(f"{simulation=}: `run()` error:\n{traceback.format_exc()}")
                success = False

            del model_instance
            gc.collect()

        return success, messages, output_files

    def _run_simulations_multiprocess(
        self,
        *,
        projection_kwargs: Mapping[str, Any] | None
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
                    projection_kwargs=projection_kwargs
                )
                for i, batch in enumerate(simulation_batches, 1)
            ]

            for future in as_completed(futures):
                success, messages, output_files = future.result()
                exec_success = exec_success and success
                self._sim_messages.extend(messages)
                self._result_files.update(output_files)

        return exec_success

    def _write_stochastic_statistic(self):
        stoch_setting = self.load_json('__stoch_setting__.json', default=None)
        if stoch_setting is None or stoch_setting.get('statistic', None) is None:
            return

        stoch_file_paths = sorted([f for f in self._result_files if f and str(f).endswith('.stoch.csv')])
        if len(stoch_file_paths) == 0: return

        df = pd.concat((pd.read_csv(f) for f in stoch_file_paths), ignore_index=True)
        meta_cols = ['group', 'owner', 'variable']
        time_cols = [col for col in df.columns if col not in meta_cols + ['simulation']]

        stat_dict = stoch_setting['statistic']
        stat_funcs: dict[str, Any] = {stat: stat for stat in ['mean', 'std', 'median', 'max', 'min'] if stat in stat_dict}
        if 'perc%' in stat_dict:
            for q in stat_dict['perc%']:
                if not 0 < q < 100: raise ValueError(f"Invalid perc%={q}, expected 0 to 100.")
                stat_funcs[f"perc_{q}"] = lambda x, p=q: x.quantile(p / 100)
        agg_dict = {f"{col}:{stat}": (col, func) for col in time_cols for stat, func in stat_funcs.items()}

        statistic = df.groupby(meta_cols).agg(**agg_dict).reset_index()  # pandas 3.3.x might warn PerformanceWarning

        output_file = self.RESULTS_DIRECTORY_PATH / f'{self._slug}.stoch.stat.csv'
        statistic.to_csv(output_file, index=False)
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
                "projection_engine": f"{inspect.getfile(self._proj_cls)}::{self._proj_cls.__name__}",
                "stochastic_executor": f"{inspect.getfile(type(self))}::{type(self).__name__}",
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
                "n_simulations": self._n_sims,
                "workspace_directory": str(self.WORKSPACE_DIRECTORY_PATH),
                "input_directories": [str(d) for d in self._run_config.input_directory_paths],
                "results_directory": str(self.RESULTS_DIRECTORY_PATH),
                "max_workers": self._run_config.max_workers,
            },
            "environment": get_environ(),
            "results": list(map(str, self._result_files)),
            "messages": self._messages + self._sim_messages,
        })

        if self._run_config.enable_write_runlog:
            output_file = self.RESULTS_DIRECTORY_PATH / f'{self._slug}.runlog.json'
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(self._runlog, jsonfile, indent=4)

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
    def SIMULATIONS(self) -> list[int] | tuple[int, ...]:
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
                f, l = get_f_filename_lineno()
                self._messages.append(f"{f}:{l}: [INFO] Add member: '{name}' {type(value)}")
        super().__setattr__(name, value)

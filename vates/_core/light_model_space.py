import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import Self

from vates._core.proj_model_engine import ProjModelEngine
from vates._core._utils import RunConfig

class LightModelSpace:
    """Lightweight model space.
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

        self._run_config: RunConfig | None = None

        # runtime stuffs
        self._cached_filepath: dict[tuple, tuple] = {}
        self._messages = []

    def set_run_config(
        self,
        *,
        scenario: str | None = None,
        workspace_directory: str | None = None,
        input_directories: list[str] | None = None,
    ) -> Self:
        """Set the configuration for a run.

        Args:
            scenario (str, optional): Scenario. Defaults to None.
            workspace_directory (str, optional): Workspace directory. Defaults to `{os.getcwd()}`.
            input_directories (list[str], optional): List of input directory. Defaults to None.
        """
        if self._run_config is not None:
            msg = (f"Run configuration is already set. If you are sure you want to reset it, use 'foo._run_config ="
                        f" None', then call 'foo.set_run_config(...)' method.")
            warnings.warn(msg); self.add_traced_message(f"Warning: {msg}")
            return self

        none_items = []

        if workspace_directory is None:
            workspace_directory = os.getcwd()
            none_items.append(f"workspace_directory='{workspace_directory}'")
        self._run_config = RunConfig(
            scenario=scenario,
            wsdir=workspace_directory,
            input_directories=input_directories,
            start_year=None,
            start_month=None,
            end_year=None,
            end_month=None,
            results_directory='',
            is_delete_existing_results=False,
            enable_write_proj_result=False,
            stoch_result_file_mode=None,
            stoch_result_file_id=None,
            enable_write_runlog=False,
            simulation=None,
            simulations=None,
        )

        if len(none_items) > 0:
            msg = f"Following items are set by default: {', '.join(none_items)}."
            warnings.warn(msg); self.add_traced_message(f"Warning: {msg}")

        return self

    add_traced_message = ProjModelEngine.add_traced_message
    load_json = ProjModelEngine.load_json
    read_csv = ProjModelEngine.read_csv
    read_excel = ProjModelEngine.read_excel
    read_parquet = ProjModelEngine.read_parquet
    get_filepath = ProjModelEngine.get_filepath
    _search_filepath = ProjModelEngine._search_filepath

    @cached_property
    def workspace_directory(self) -> Path:
        return Path(self._run_config.wsdir or os.getcwd()).resolve()

    @property
    def MODEL_NAME(self) -> str:
        """str: Model alias used to prefix output files."""
        return self._name

    @property
    def SCENARIO(self) -> str | None:
        """str: Scenario code that identifies the set of input used for a run."""
        return self._run_config.scenario

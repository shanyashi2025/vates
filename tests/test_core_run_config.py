"""Tests for `vates/_core/_utils.py`: `RunConfiguration.create` and `parse_str_to_int_list`."""

import pandas as pd
import pytest
from pathlib import Path

from vates._core._run_configuration import RunConfiguration

WORKSPACE = "C:\\work"

def _p(year, month):
    return pd.Period(f"{year}-{month}", freq="M")


def _create(**overrides):
    params = dict(
        start_date=_p(2026, 12),
        end_date=_p(2028, 12),
        scenario="base",
        workspace_directory_path=Path(WORKSPACE),
        results_directory_path=Path(WORKSPACE) / "results" / "base",
        is_delete_existing_results=True,
        enable_write_proj_result=True,
        enable_write_runlog=True,
        simulations=(),
        simulation=0,
        input_directory_paths=(),
        stoch_result_file_mode="n",
        stoch_result_file_id="",
        max_workers=1,
    )
    params.update(overrides)
    return RunConfiguration(**params)


class TestRunConfigDates:
    def test_month_count_equals_max_t(self):
        cfg = _create()
        # 2026-12 to 2028-12 = 24 months (both endpoints inclusive)
        assert cfg.max_t == 24
        assert (cfg.end_date - cfg.start_date).n == cfg.max_t
        assert cfg.start_date.month == 12 and cfg.end_date.month == 12

    def test_boundary_start_month_ok(self):
        cfg = _create(start_date=_p(1900, 1), end_date=_p(1900,12))
        assert cfg.start_date.year == 1900 and cfg.start_date.month == 1

    def test_early_start_below_minimum_raises(self):
        # 1899-01 is before the RunConfiguration floor of 1900-01.
        with pytest.raises(ValueError):
            _create(start_date=_p(1899, 12), end_date=_p(1900, 12))

    def test_late_end_beyond_maximum_raises(self):
        with pytest.raises(ValueError):
            _create(start_date=_p(5999, 12), end_date=_p(6000, 1))


class TestRunConfigValidation:
    def test_max_workers_validated(self):
        with pytest.raises(ValueError):
            _create(max_workers=0)
        with pytest.raises(ValueError):
            _create(max_workers=1000)
        assert _create(max_workers=3).max_workers == 3

    def test_stoch_file_mode_literal(self):
        with pytest.raises(ValueError):
            _create(stoch_result_file_mode="x")
        assert _create(stoch_result_file_mode="w").stoch_result_file_mode == "w"


class TestPathFields:
    def test_relative_workspace_directory_raises(self):
        with pytest.raises(ValueError):
            _create(workspace_directory_path=Path("work"))

    def test_relative_results_directory_raises(self):
        with pytest.raises(ValueError):
            _create(results_directory_path=Path("./results/base"))

    def test_relative_input_directory_raises(self):
        with pytest.raises(ValueError):
            _create(input_directory_paths=[Path("inputs")])

    def test_absolute_results_directory_resolved(self):
        results = Path(WORKSPACE) / "results" / "base"
        cfg = _create(results_directory_path=results)
        assert cfg.results_directory_path == Path(results).resolve()

    def test_input_directory_paths_resolved(self):
        inputs = Path(WORKSPACE) / "inputs"
        cfg = _create(input_directory_paths=[inputs])
        assert cfg.input_directory_paths == [inputs.resolve()]

    def test_input_directory_paths_default_empty(self):
        assert _create().input_directory_paths == ()

    def test_path_fields_are_absolute(self):
        cfg = _create()
        assert cfg.workspace_directory_path.is_absolute()
        assert cfg.results_directory_path.is_absolute()


class TestValidatePath:
    def test_relative_path_raises_when_absolute_required(self):
        with pytest.raises(ValueError):
            RunConfiguration.validate_path("p", Path("relative/dir"), must_absolute=True)

    def test_missing_path_raises_when_must_exist(self, tmp_path):
        with pytest.raises(ValueError):
            RunConfiguration.validate_path("p", tmp_path / "missing", must_exist=True)

    def test_dir_check_skipped_for_missing_path(self, tmp_path):
        # The type cannot be asserted for a path that does not exist yet.
        RunConfiguration.validate_path("p", tmp_path / "missing", dir_or_file="dir")

    def test_file_check_on_directory_raises(self, tmp_path):
        with pytest.raises(ValueError):
            RunConfiguration.validate_path("p", tmp_path, dir_or_file="file")

    def test_dir_check_on_file_raises(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        with pytest.raises(ValueError):
            RunConfiguration.validate_path("p", f, dir_or_file="dir")

    def test_dir_and_file_checks_pass(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        RunConfiguration.validate_path("p", tmp_path, dir_or_file="dir", must_exist=True)
        RunConfiguration.validate_path("p", f, dir_or_file="file", must_exist=True)

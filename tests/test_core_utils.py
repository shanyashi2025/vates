"""Tests for `vates/_core/_utils.py`: `RunConfiguration.create` and `parse_str_to_int_list`."""

import pytest
from pathlib import Path

from vates._core._utils import RunConfiguration, parse_int_list_from_str

WORKSPACE = "C:\\work"


def _create(**overrides):
    params = dict(
        start_year=2026,
        start_month=12,
        end_year=2028,
        end_month=12,
        scenario="base",
        workspace_directory=WORKSPACE,
        results_directory=str(Path(WORKSPACE) / "results" / "base"),
        is_delete_existing_results=True,
        enable_write_proj_result=True,
        enable_write_runlog=True,
    )
    params.update(overrides)
    return RunConfiguration.create(**params)


class TestRunConfigDates:
    def test_month_count_equals_max_t(self):
        cfg = _create()
        # 2026-12 to 2028-12 = 24 months (both endpoints inclusive)
        assert cfg.max_t == 24
        assert (cfg.end_date - cfg.start_date).n == cfg.max_t
        assert cfg.start_date.month == 12 and cfg.end_date.month == 12

    def test_boundary_start_month_ok(self):
        cfg = _create(start_year=1900, start_month=1)
        assert cfg.start_date.year == 1900 and cfg.start_date.month == 1

    def test_early_start_below_minimum_raises(self):
        # 1899-01 is before the RunConfiguration floor of 1900-01.
        with pytest.raises(ValueError):
            _create(start_year=1899, start_month=1)

    def test_late_end_beyond_maximum_raises(self):
        with pytest.raises(ValueError):
            _create(end_year=6000, end_month=1)

    def test_inconsistent_month_count_raises(self):
        # max_t is derived from the two dates, so this cannot normally trigger
        # through `create`; build directly to exercise the guard.
        with pytest.raises(ValueError):
            RunConfiguration(
                start_date=_p(2026, 12),
                end_date=_p(2027, 12),
                max_t=10,  # 13 months expected -> inconsistent
                scenario="base",
                simulations=None,
                simulation=None,
                workspace_directory_path=Path(WORKSPACE),
                input_directory_paths=[],
                results_directory_path=Path(WORKSPACE) / "results/base",
                is_delete_existing_results=True,
                enable_write_proj_result=True,
                stoch_result_file_mode=None,
                stoch_result_file_id=None,
                enable_write_runlog=True,
            )


def _p(year, month):
    import pandas as pd
    return pd.Period(f"{year}-{month}", freq="M")


class TestRunConfigValidation:
    def test_max_t_exactly_2400_ok(self):
        # 2000-01 to 2200-01 = exactly 2400 months; the bound is inclusive.
        assert _create(start_year=2000, start_month=1, end_year=2200, end_month=1).max_t == 2400

    def test_max_t_exceeds_2400_raises(self):
        with pytest.raises(ValueError):
            _create(start_year=2000, start_month=1, end_year=2201, end_month=1)

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

    def test_simulations_parsed_from_string(self):
        cfg = _create(simulations="1-3,5")
        assert cfg.simulations == [1, 2, 3, 5]


class TestPathFields:
    def test_relative_workspace_directory_raises(self):
        with pytest.raises(ValueError):
            _create(workspace_directory="work")

    def test_relative_results_directory_raises(self):
        with pytest.raises(ValueError):
            _create(results_directory="./results/base")

    def test_relative_input_directory_raises(self):
        with pytest.raises(ValueError):
            _create(input_directories=["inputs"])

    def test_absolute_results_directory_resolved(self):
        results = str(Path(WORKSPACE) / "results" / "base")
        cfg = _create(results_directory=results)
        assert cfg.results_directory_path == Path(results).resolve()

    def test_input_directory_paths_resolved(self):
        inputs = Path(WORKSPACE) / "inputs"
        cfg = _create(input_directories=[str(inputs)])
        assert cfg.input_directory_paths == [inputs.resolve()]

    def test_input_directory_paths_default_empty(self):
        assert _create().input_directory_paths == []

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


class TestParseStrToIntList:
    def test_simple_list(self):
        assert parse_int_list_from_str("1,2,3") == [1, 2, 3]

    def test_range(self):
        assert parse_int_list_from_str("1-10") == list(range(1, 11))

    def test_mixed_list_and_range(self):
        assert parse_int_list_from_str("1-10,13") == list(range(1, 11)) + [13]

    def test_reversed_range_normalised(self):
        assert parse_int_list_from_str("5-1") == [1, 2, 3, 4, 5]

    def test_sort_ascending(self):
        assert parse_int_list_from_str("3,1-2", sort_list="asc") == [1, 2, 3]

    def test_sort_descending(self):
        assert parse_int_list_from_str("1-3", sort_list="desc") == [3, 2, 1]

    def test_duplicates_raise_by_default(self):
        with pytest.raises(ValueError):
            parse_int_list_from_str("1,1")

    def test_duplicates_keep(self):
        assert parse_int_list_from_str("1,1", on_duplicate="keep") == [1, 1]

    def test_duplicates_remove(self):
        assert parse_int_list_from_str("1,1,2", on_duplicate="remove") == [1, 2]

    def test_bad_token_raises(self):
        with pytest.raises(ValueError):
            parse_int_list_from_str("abc")

    def test_negative_disallowed(self):
        with pytest.raises(ValueError):
            parse_int_list_from_str("-1")

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            parse_int_list_from_str(5)
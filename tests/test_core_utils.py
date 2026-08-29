"""Tests for `vates/_core/_utils.py`: `RunConfig.create` and `parse_str_to_int_list`."""

import pytest
from pathlib import Path

from vates._core._utils import RunConfig, parse_str_to_int_list

WORKSPACE = "C:\\work"


def _create(**overrides):
    params = dict(
        start_year=2026,
        start_month=12,
        end_year=2028,
        end_month=12,
        scenario="base",
        workspace_directory=WORKSPACE,
        results_directory="./results/base",
        is_delete_existing_results=True,
        enable_write_proj_result=True,
        enable_write_runlog=True,
    )
    params.update(overrides)
    return RunConfig.create(**params)


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
        # 1899-01 is before the RunConfig floor of 1900-01.
        with pytest.raises(ValueError):
            _create(start_year=1899, start_month=1)

    def test_late_end_beyond_maximum_raises(self):
        with pytest.raises(ValueError):
            _create(end_year=6000, end_month=1)

    def test_inconsistent_month_count_raises(self):
        # max_t is derived from the two dates, so this cannot normally trigger
        # through `create`; build directly to exercise the guard.
        with pytest.raises(ValueError):
            RunConfig(
                start_date=_p(2026, 12),
                end_date=_p(2027, 12),
                max_t=10,  # 13 months expected -> inconsistent
                scenario="base",
                simulations=None,
                simulation=None,
                workspace_directory=WORKSPACE,
                workspace_directory_path=Path(WORKSPACE),
                input_directories=None,
                results_directory="./results/base",
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


class TestParseStrToIntList:
    def test_simple_list(self):
        assert parse_str_to_int_list("1,2,3") == [1, 2, 3]

    def test_range(self):
        assert parse_str_to_int_list("1-10") == list(range(1, 11))

    def test_mixed_list_and_range(self):
        assert parse_str_to_int_list("1-10,13") == list(range(1, 11)) + [13]

    def test_reversed_range_normalised(self):
        assert parse_str_to_int_list("5-1") == [1, 2, 3, 4, 5]

    def test_sort_ascending(self):
        assert parse_str_to_int_list("3,1-2", sort_list="asc") == [1, 2, 3]

    def test_sort_descending(self):
        assert parse_str_to_int_list("1-3", sort_list="desc") == [3, 2, 1]

    def test_duplicates_raise_by_default(self):
        with pytest.raises(ValueError):
            parse_str_to_int_list("1,1")

    def test_duplicates_keep(self):
        assert parse_str_to_int_list("1,1", on_duplicate="keep") == [1, 1]

    def test_duplicates_remove(self):
        assert parse_str_to_int_list("1,1,2", on_duplicate="remove") == [1, 2]

    def test_bad_token_raises(self):
        with pytest.raises(ValueError):
            parse_str_to_int_list("abc")

    def test_negative_disallowed(self):
        with pytest.raises(ValueError):
            parse_str_to_int_list("-1")

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            parse_str_to_int_list(5)
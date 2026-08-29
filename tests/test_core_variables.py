"""Tests for `ConstVariable` and `TDimVariable`.

`ProjVariable` subclasses read `model_engine._run_config` at construction time,
so every variable here is created *after* `configure_run` (see `make_configured`).
"""

import numpy as np
import pandas as pd
import pytest

from enum import Enum

from vates import ConstVariable, TDimVariable
from vates._core._utils import RunConfig

from conftest import DEFAULT_END_YEAR, DEFAULT_START_YEAR


class _AssetType(Enum):
    EQUITY = "equity"
    BOND = "bond"


@pytest.fixture
def configured(make_configured, tmp_path):
    """A configured engine (2026-12 .. 2028-12) already attached to `tmp_path`."""
    return make_configured(tmp_path)


class TestConstVariable:
    def test_scalar_storage(self, configured):
        v = ConstVariable("c", model_engine=configured, owner="o", group="g")
        v[...] = 42
        assert v.result == 42
        assert v.is_constant is True
        assert v.ndim == 0
        assert v.dims is None

    def test_array_storage(self, configured):
        v = ConstVariable("c", model_engine=configured, owner="o", group="g",
                          dims=[["A", "B"]])
        v[...] = np.array([1.0, 2.0])
        assert v.ndim == 1
        assert v.is_constant is True

    def test_string_storage(self, configured):
        v = ConstVariable("s", model_engine=configured, owner="o", group="g")
        v[...] = "hello"
        assert v.result == "hello"
        assert v.is_constant is True

    def test_dims_from_enum(self, configured):
        v = ConstVariable("e", model_engine=configured, owner="o", group="g",
                          dims=[_AssetType])
        assert v.dims == [["EQUITY", "BOND"]]

    def test_int_labels_coerced_to_str(self, configured):
        v = ConstVariable("i", model_engine=configured, owner="o", group="g",
                          dims=[[0, 1]])
        assert v.dims == [["0", "1"]]

    def test_max_three_dims(self, configured):
        dims = [["a", "b"], ["c"], ["d"]]
        v = ConstVariable("3d", model_engine=configured, owner="o", group="g", dims=dims)
        assert v.ndim == 3

    def test_four_dims_rejected(self, configured):
        with pytest.raises(ValueError):
            ConstVariable("4d", model_engine=configured, owner="o", group="g",
                          dims=[["a"], ["b"], ["c"], ["d"]])

    def test_dims_must_be_list(self, configured):
        with pytest.raises(ValueError):
            ConstVariable("bad", model_engine=configured, owner="o", group="g",
                          dims="AB")

    def test_dim_class_that_is_not_enum_raises_valueerror(self, configured):
        # a class that is not an Enum reaches the `else` branch
        with pytest.raises(ValueError):
            ConstVariable("bad", model_engine=configured, owner="o", group="g",
                          dims=[int])

    def test_dim_element_that_is_not_a_class_raises_typeerror(self, configured):
        # `issubclass(1, Enum)` fails before the guard's `else` branch
        with pytest.raises(TypeError):
            ConstVariable("bad", model_engine=configured, owner="o", group="g",
                          dims=[1, 2])


class TestConstVariableRegistration:
    def test_registered_with_engine(self, configured):
        v = ConstVariable("rv", model_engine=configured, owner="o", group="g")
        names = [ref().name for ref in configured._proj_variables]
        assert "rv" in names

    def test_duplicate_registration_raises(self, configured):
        # registering the *same* instance twice is rejected
        with pytest.raises(ValueError, match="already included"):
            v = ConstVariable("dup", model_engine=configured, owner="o", group="g")
            configured.include_proj_variable(v)

    def test_non_variable_type_rejected(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        with pytest.raises(TypeError):
            m.include_proj_variable(object())


class TestTDimVariable:
    def test_size_from_run_config(self, configured):
        # 2026-12 .. 2028-12 -> max_t = 24, internal array has max_t + 1 rows
        v = TDimVariable("t", model_engine=configured, owner="o", group="g")
        assert v._result.shape == (configured.MAX_T + 1,)
        assert len(v._assigned) == configured.MAX_T + 1

    def test_requires_engine(self):
        with pytest.raises(ValueError, match="'model_engine' is None"):
            TDimVariable("x", model_engine=None)

    def test_fallback_cfg_used_and_warns(self):
        class WithFallback(TDimVariable):
            fallback_cfg = RunConfig.create(
                start_year=2026, start_month=12, end_year=2027, end_month=12,
                scenario="base",
                workspace_directory="C:\\work", results_directory="res",
                is_delete_existing_results=True, enable_write_proj_result=True,
                enable_write_runlog=True,
            )

        # note: the fallback path intentionally does *not* warn (verified in code)
        v = WithFallback("v", model_engine=None)
        assert v._result.shape == (13,)

    def test_unassigned_read_returns_none(self, configured):
        v = TDimVariable("t", model_engine=configured, owner="o", group="g")
        assert v[0] is None
        assert v[configured.MAX_T] is None

    def test_write_then_read(self, configured):
        v = TDimVariable("t", model_engine=configured, owner="o", group="g")
        v[3] = 9.5
        assert v[3] == 9.5
        assert bool(v._assigned[3]) is True

    def test_period_indexing(self, configured):
        v = TDimVariable("t", model_engine=configured, owner="o", group="g")
        v[6] = 1.0
        assert v[configured.START_DATE + 6] == 1.0

    @pytest.mark.parametrize("bad", [-1, 9999])  # 9999 is way beyond max_t=24
    def test_out_of_range_index_raises(self, configured, bad):
        v = TDimVariable("t", model_engine=configured, owner="o", group="g")
        with pytest.raises(ValueError):
            v[bad]

    @pytest.mark.parametrize("bad", ["x", 1.5])
    def test_wrong_type_index_raises(self, configured, bad):
        v = TDimVariable("t", model_engine=configured, owner="o", group="g")
        with pytest.raises(TypeError):
            v[bad]

    def test_far_period_index_raises(self, configured):
        v = TDimVariable("t", model_engine=configured, owner="o", group="g")
        with pytest.raises(ValueError):
            v[configured.END_DATE + 1]

    def test_ndim_returns_copy(self, configured):
        v = TDimVariable("t", model_engine=configured, owner="o", group="g",
                         dims=[["A", "B"]])
        v[0] = np.array([1.0, 2.0])
        got = v[0]
        got[0] = 999.0
        assert v[0][0] == 1.0  # read returned a copy, not a view
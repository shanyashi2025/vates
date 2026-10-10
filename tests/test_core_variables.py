"""Tests for `ConstVariable` and `TDimVariable`.

`ProjVariable` subclasses read `model_engine._run_config` at construction time,
so every variable here is created *after* `configure_run` (see `make_configured`).
"""

import numpy as np
import pandas as pd
import pytest

from enum import Enum

from vates import ConstVariable, TDimVariable


class _AssetType(Enum):
    EQUITY = "equity"
    BOND = "bond"


@pytest.fixture
def configured(make_configured, tmp_path):
    """A configured engine (2026-12 .. 2028-12) already attached to `tmp_path`."""
    return make_configured(tmp_path)


class TestConstVariable:
    def test_scalar_storage(self):
        v = ConstVariable("c", owner="o", group="g")
        v[...] = 42
        assert v.result == 42
        assert v.is_constant is True
        assert v.ndim == 0
        assert v.dims is None

    def test_array_storage(self):
        v = ConstVariable("c", owner="o", group="g",
                          dims=[["A", "B"]])
        v[...] = np.array([1.0, 2.0])
        assert v.ndim == 1
        assert v.is_constant is True

    def test_string_storage(self):
        v = ConstVariable("s", owner="o", group="g")
        v[...] = "hello"
        assert v.result == "hello"
        assert v.is_constant is True

    def test_dims_from_enum(self):
        v = ConstVariable("e", owner="o", group="g",
                          dims=[_AssetType])
        assert v.dims == (("EQUITY", "BOND"), )

    def test_int_labels_coerced_to_str(self):
        v = ConstVariable("i", owner="o", group="g",
                          dims=[[0, 1]])
        assert v.dims == (("0", "1"), )

    def test_max_three_dims(self):
        dims = [["a", "b"], ["c"], ["d"]]
        v = ConstVariable("3d", owner="o", group="g", dims=dims)
        assert v.ndim == 3
        assert v.dims == (("a", "b"), ("c", ), ("d", ))

    def test_four_dims_rejected(self):
        with pytest.raises(ValueError):
            ConstVariable("4d", owner="o", group="g",
                          dims=[["a"], ["b"], ["c"], ["d"]])

    def test_dims_must_be_list(self):
        with pytest.raises(ValueError):
            ConstVariable("bad", owner="o", group="g", dims="AB")

    def test_dim_class_that_is_not_enum_raises_valueerror(self):
        # a class that is not an Enum reaches the `else` branch
        with pytest.raises(ValueError):
            ConstVariable("bad", owner="o", group="g", dims=[int])

    def test_dim_element_that_is_not_a_class_raises_typeerror(self):
        # `issubclass(1, Enum)` fails before the guard's `else` branch
        with pytest.raises(TypeError):
            ConstVariable("bad", owner="o", group="g",
                          dims=[1, 2])


class TestTDimVariable:
    def test_size_from_run_config(self, configured):
        # 2026-12 .. 2028-12 -> max_t = 24, internal array has max_t + 1 rows
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE)
        assert v._result.shape == (configured.MAX_T + 1,)
        assert v._meta[2] == configured.MAX_T
        assert v._meta[3] == configured.START_DATE

    def test_unassigned_read_returns_zero(self, configured):
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE)
        assert v[0] == 0.0
        assert v[configured.MAX_T] == 0.0

    def test_write_then_read(self, configured):
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE)
        v[3] = 9.5
        assert v[3] == 9.5

    def test_period_indexing(self, configured):
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE)
        v[6] = 1.0
        assert v[configured.START_DATE + 6] == 1.0

    @pytest.mark.parametrize("bad", [None, "foo", 3.14, pd.Period("1999-12", "M")])
    def test_wrong_type_max_t_raises(self, bad):
        with pytest.raises(TypeError):
            v = TDimVariable("t", owner="o", group="g", max_t=bad, start_date=pd.Period("1999-12", "M"))

    @pytest.mark.parametrize("bad", [0, -1, -24])
    def test_wrong_non_postive_max_t_raises(self, bad):
        with pytest.raises(ValueError):
            v = TDimVariable("t", owner="o", group="g", max_t=bad, start_date=pd.Period("1999-12", "M"))

    @pytest.mark.parametrize("bad", ["foo", 3.14, 123])
    def test_wrong_type_start_date_raises(self, bad):
        with pytest.raises(TypeError):
            v = TDimVariable("t", owner="o", group="g", max_t=24, start_date=bad)

    def test_period_index_raises_while_start_date_is_none(self):
        v = TDimVariable("t", owner="o", group="g", max_t=24, start_date=None)
        assert v._meta[3] is None
        v = TDimVariable("t", owner="o", group="g", max_t=24)
        assert v._meta[3] is None
        with pytest.raises(ValueError, match="cannot index by 'pd.Period'"):
            v[pd.Period("1999-12", "M")]

    @pytest.mark.parametrize("bad", [-1, 9999])  # 9999 is way beyond max_t=24
    def test_out_of_range_index_raises(self, configured, bad):
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE)
        with pytest.raises(ValueError):
            v[bad]

    @pytest.mark.parametrize("bad", ["x", 1.5])
    def test_wrong_type_index_raises(self, configured, bad):
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE)
        with pytest.raises(TypeError):
            v[bad]

    def test_far_period_index_raises(self, configured):
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE)
        with pytest.raises(ValueError):
            v[configured.END_DATE + 1]

    def test_ndim_returns_copy(self, configured):
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE,
                         dims=[["A", "B"]])
        v[0] = np.array([1.0, 2.0])
        got = v[0]
        got[0] = 999.0
        assert v[0][0] == 1.0  # read returned a copy, not a view

    def test_cached_meta(self, configured):
        assert len(TDimVariable._cached_metas) == 0
        v = TDimVariable("t", owner="o1", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE)
        assert len(TDimVariable._cached_metas) == 1
        v = TDimVariable("t", owner="o2", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE)
        assert len(TDimVariable._cached_metas) == 1
        v = TDimVariable("t1", owner="o1", group="g", max_t=configured.MAX_T, start_date=configured.START_DATE)
        assert len(TDimVariable._cached_metas) == 2
        v = TDimVariable("t", owner="o1", group="g", max_t=120, start_date=configured.START_DATE)
        assert len(TDimVariable._cached_metas) == 3
        v = TDimVariable("t", owner="o1", group="g", max_t=configured.MAX_T, start_date=pd.Period("1999-12", "M"))
        assert len(TDimVariable._cached_metas) == 4


class TestCachedDims:
    def test_cached_dims(self):
        assert len(ConstVariable._cached_dims) == 0
        assert len(TDimVariable._cached_dims) == 0
        assert not (ConstVariable._cached_dims is TDimVariable._cached_dims)
        v = ConstVariable("t", owner="o", group="g", dims=[["A", "B"]])
        assert len(ConstVariable._cached_dims) == 1
        v = ConstVariable("t", owner="o", group="g", dims=[["A", "B"], ("C", "D")])
        assert len(ConstVariable._cached_dims) == 2
        v = TDimVariable("t", owner="o", group="g", dims=[["A", "B"]], max_t=24, start_date = pd.Period("1999-12", "M"))
        assert len(ConstVariable._cached_dims) == 2
        assert len(TDimVariable._cached_dims) == 1

"""Tests for the four projection variable types and their factory."""

from enum import Enum

import numpy as np
import pytest

from vates import (
    ConstArrayVariable,
    ScalarVariable,
    TDimArrayVariable,
    TDimVariable,
    make_proj_variable,
)
from vates._core.tools import _FALLBACK


class _AssetType(Enum):
    EQUITY = "equity"
    BOND = "bond"


@pytest.fixture
def configured(make_configured, tmp_path):
    """A configured engine (2026-12 .. 2028-12) already attached to `tmp_path`."""
    return make_configured(tmp_path)


class TestScalarVariable:
    def test_scalar_storage(self):
        v = ScalarVariable("c", owner="o", group="g")
        v[...] = 42
        assert v.result == 42
        assert v.is_constant is True
        v.result = 21
        assert v.result == 21

    def test_string_storage(self):
        v = ScalarVariable("s", owner="o", group="g")
        v[...] = "hello"
        assert v.result == "hello"
        v.result = "world"
        assert v.result == "world"


class TestConstArrayVariable:
    def test_array_storage_and_slice(self):
        v = ConstArrayVariable("c", owner="o", group="g", dims=[["A", "B", "C"]])
        v[...] = np.array([1.0, 2.0, 3.0])

        assert v.ndim == 1
        assert v.is_constant is True
        assert v[1] == 2.0
        np.testing.assert_array_equal(v[1:], [2.0, 3.0])

        v[1:] = [12.0, 13.0]
        np.testing.assert_array_equal(v.result, [1.0, 12.0, 13.0])

    def test_result_setter(self):
        v = ConstArrayVariable("c", owner="o", group="g", dims=[["A", "B"]])
        v.result = np.array([11.0, 12.0])
        np.testing.assert_array_equal(v.result, [11.0, 12.0])

    def test_dims_from_enum(self):
        v = ConstArrayVariable("e", owner="o", group="g", dims=[_AssetType])
        assert v.dims == (("EQUITY", "BOND"),)

    def test_int_labels_coerced_to_str(self):
        v = ConstArrayVariable("i", owner="o", group="g", dims=[[0, 1]])
        assert v.dims == (("0", "1"),)

    def test_multiple_dimensions(self):
        v = ConstArrayVariable("3d", owner="o", group="g", dims=[["a", "b"], ["c"], ["d"]])
        assert v.ndim == 3
        assert v.dims == (("a", "b"), ("c",), ("d",))

    def test_label_based_access(self):
        v = ConstArrayVariable(
            "c",
            owner="o",
            group="g",
            dims=[["A", "B"], _AssetType],
        )
        v[...] = [[1.0, 2.0], [3.0, 4.0]]

        indices = v.resolve_dim_indices("B", _AssetType.BOND)
        assert indices == (1, 1)
        assert v[...][indices] == 4.0

        v[...][indices] = 9.0
        assert v[1, 1] == 9.0

    @pytest.mark.parametrize("bad", ["foo", 42, 3.14])
    def test_dims_must_be_list_or_tuple(self, bad):
        with pytest.raises(TypeError):
            ConstArrayVariable("bad", owner="o", group="g", dims=bad)

    @pytest.mark.parametrize("bad", [int, 42, "foo"])
    def test_wrong_dim_type_raises(self, bad):
        with pytest.raises(TypeError):
            ConstArrayVariable("bad", owner="o", group="g", dims=[bad])


class TestTDimVariable:
    def test_size_and_default_values(self, configured):
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T)
        assert v.result.shape == (configured.MAX_T + 1,)
        assert v[0] == 0.0
        assert v[configured.MAX_T] == 0.0

    def test_write_and_slice(self, configured):
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T)
        v[3] = 9.5
        v[4:6] = [10.5, 11.5]
        assert v[3] == 9.5
        np.testing.assert_array_equal(v[3:6], [9.5, 10.5, 11.5])

    def test_period_indexing_is_not_supported(self, configured):
        v = TDimVariable("t", owner="o", group="g", max_t=configured.MAX_T)
        with pytest.raises(IndexError):
            v[configured.START_DATE]


class TestTDimArrayVariable:
    def test_dimensions_and_time_slices(self, configured):
        v = TDimArrayVariable(
            "t",
            owner="o",
            group="g",
            dims=[["A", "B"]],
            max_t=configured.MAX_T,
        )
        v[0] = [1.0, 2.0]
        v[1:3] = [[3.0, 4.0], [5.0, 6.0]]

        assert v.dims == (("A", "B"),)
        assert v.ndim == 1
        np.testing.assert_array_equal(v[0], [1.0, 2.0])
        np.testing.assert_array_equal(v[1:3], [[3.0, 4.0], [5.0, 6.0]])

    def test_multidimensional_time_slice_and_assignment(self, configured):
        v = TDimArrayVariable(
            "t",
            owner="o",
            group="g",
            dims=[["A", "B", "C"], ["x", "y", "z"]],
            max_t=configured.MAX_T,
        )
        values = np.arange((configured.MAX_T + 1) * 9).reshape(configured.MAX_T + 1, 3, 3)
        v.result[:] = values

        np.testing.assert_array_equal(v[1:3, 2], values[1:3, 2])
        v[1:3, 2] = [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]]
        np.testing.assert_array_equal(
            v[1:3, 2],
            [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        )
        assert v[1, 2, 0] == 100.0
        assert v[2, 2, 2] == 202.0

    def test_label_based_access_after_time_index(self, configured):
        v = TDimArrayVariable(
            "t",
            owner="o",
            group="g",
            dims=[["A", "B"], _AssetType],
            max_t=configured.MAX_T,
        )
        v[3] = [[1.0, 2.0], [3.0, 4.0]]

        indices = v.resolve_dim_indices("B", _AssetType.BOND)
        assert indices == (1, 1)
        assert v[3][indices] == 4.0

        v[3][indices] = 9.0
        assert v[3, 1, 1] == 9.0


class TestResolveDimIndices:
    @pytest.mark.parametrize(
        ("keys", "expected"),
        [
            (("B", "Y"), (1, 2)),
            (("B", 2), (1, 1)),
        ],
    )
    def test_resolves_labels(self, keys, expected):
        v = ConstArrayVariable(
            "c",
            owner="o",
            group="g",
            dims=[["A", "B"], ["X", 2, "Y"]],
        )
        assert v.resolve_dim_indices(*keys) == expected

    def test_single_key_for_one_dimension(self):
        v = ConstArrayVariable("c", owner="o", group="g", dims=[["A", "B"]])
        assert v.resolve_dim_indices("A") == (0,)

    def test_resolves_enum_labels(self):
        v = ConstArrayVariable("c", owner="o", group="g", dims=[["A", "B"], _AssetType])
        assert v.resolve_dim_indices("B", _AssetType.BOND) == (1, 1)

    def test_empty_dimensions_resolve_to_empty_tuple(self):
        v = TDimVariable("t", owner="o", group="g", max_t=2)
        assert v.resolve_dim_indices() == ()

    def test_wrong_number_of_keys_raises(self):
        v = ConstArrayVariable("c", owner="o", group="g", dims=[["A"], ["X"]])
        with pytest.raises(ValueError, match="expected 2"):
            v.resolve_dim_indices("A")

    def test_unknown_label_raises_key_error(self):
        v = ConstArrayVariable("c", owner="o", group="g", dims=[["A", "B"]])
        with pytest.raises(KeyError, match="not in dims"):
            v.resolve_dim_indices("missing")


class TestVariableCaches:
    def test_dimension_caches_are_per_array_class(self):
        assert ConstArrayVariable._dims_cache is not TDimArrayVariable._dims_cache
        ConstArrayVariable("c1", owner="o", group="g", dims=[["A", "B"]])
        ConstArrayVariable("c2", owner="o", group="g", dims=[["A", "B"], ("C", "D")])
        TDimArrayVariable("t", owner="o", group="g", dims=[["A", "B"]], max_t=24)
        assert len(ConstArrayVariable._dims_cache) == 2
        assert len(TDimArrayVariable._dims_cache) == 1


class TestMakeProjVariable:
    @pytest.mark.parametrize("bad", [None, 1, 3.14, []])
    def test_wrong_variable_type_type(self, bad):
        with pytest.raises(TypeError, match="expected 'str'"):
            make_proj_variable("t", variable_type=bad, owner="o", group="g")

    @pytest.mark.parametrize("bad", ["Nope", "42"])
    def test_wrong_variable_type_value(self, bad):
        with pytest.raises(ValueError, match=r"expected \('const', 'tdim'\)"):
            make_proj_variable("t", variable_type=bad, owner="o", group="g")

    def test_const_scalar_factory(self, configured):
        v = make_proj_variable("c", variable_type="c", owner="o", group="g", model_engine=configured)
        assert isinstance(v, ScalarVariable)
        assert v.name == "c"
        assert v.owner == "o"
        assert v.group == "g"
        assert v in configured.proj_variable_manager

    def test_const_array_factory(self, configured):
        v = make_proj_variable(
            "c",
            variable_type="const",
            owner="o",
            group="g",
            dims=[["A", "B"]],
            model_engine=configured,
        )
        assert isinstance(v, ConstArrayVariable)
        assert v.dims == (("A", "B"),)

    @pytest.mark.parametrize("good", ["c", "const", "CoNsT"])
    def test_correct_const_var_type_value(self, configured, good):
        v = make_proj_variable("c", variable_type=good, owner="o", group="g", model_engine=configured)
        assert isinstance(v, ScalarVariable)

    def test_tdim_scalar_factory(self, configured):
        v = make_proj_variable("t", variable_type="t", owner="o", group="g", model_engine=configured)
        assert isinstance(v, TDimVariable)
        assert v.result.shape == (configured.MAX_T + 1,)
        assert v in configured.proj_variable_manager

    def test_tdim_array_factory(self, configured):
        v = make_proj_variable(
            "t",
            variable_type="t",
            owner="o",
            group="g",
            dims=[["A", "B"]],
            model_engine=configured,
        )
        assert isinstance(v, TDimArrayVariable)
        assert v.dims == (("A", "B"),)
        assert v.result.shape == (configured.MAX_T + 1, 2)

    @pytest.mark.parametrize("good", ["t", "tdim", "time_DiMeNSioned"])
    def test_correct_tdim_var_type_value(self, configured, good):
        v = make_proj_variable("t", variable_type=good, owner="o", group="g", model_engine=configured)
        assert isinstance(v, TDimVariable)

    def test_var_type_defaults_to_tdim(self, configured):
        v = make_proj_variable("t", owner="o", group="g", model_engine=configured)
        assert isinstance(v, TDimVariable)

    def test_no_engine_uses_fallback(self):
        v1 = make_proj_variable("c", variable_type="c", owner="o", group="g")
        fallback = _FALLBACK.engine
        v2 = make_proj_variable("c", variable_type="c", owner="o", group="g", model_engine=None)
        v3 = make_proj_variable("t", variable_type="t", owner="o", group="g")
        v4 = make_proj_variable("t", variable_type="t", owner="o", group="g", model_engine=None)
        assert isinstance(v1, ScalarVariable)
        assert isinstance(v2, ScalarVariable)
        assert isinstance(v3, TDimVariable)
        assert isinstance(v4, TDimVariable)
        assert all(v in fallback.proj_variable_manager for v in (v1, v2, v3, v4))

    def test_engine_has_no_max_t_raises(self):
        class NoMaxTEngine:
            pass

        with pytest.raises(AttributeError, match="MAX_T"):
            make_proj_variable("t", owner="o", group="g", model_engine=NoMaxTEngine())

    def test_engine_without_attach_method_leaves_variable_unattached(self):
        class NoAttachEngine:
            MAX_T = 24

        v = make_proj_variable("t", owner="o", group="g", model_engine=NoAttachEngine())
        assert isinstance(v, TDimVariable)

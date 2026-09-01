"""Tests for `vates/_core/keyed_array.py`: the `KeyedArray` wrapper and its
label-based `.at` / `.get` / `key_to_pos` interfaces, plus the `from_df`
constructor.

`KeyedArray` pairs a NumPy-style positional accessor (`[]`) with a pandas-style
label accessor (`.at`, `.get`). The real-world usage is documented in
`docs/tutorials/tut_KeyedArray.py` and exercised by `econ_master.py` /
`em12_stoch_ec_mvl.py`; these tests pin that contract.
"""

import numpy as np
import pandas as pd
import pytest

from vates import KeyedArray


@pytest.fixture
def make_kr():
    """Factory for a small 2D (2x3) writable `KeyedArray` with row/col labels."""

    def _make(frozen=False, key_pos_pairs=None, dim_names=None, arr=None):
        arr = arr if arr is not None else np.arange(6.0).reshape(2, 3)
        key_pos_pairs = key_pos_pairs or [{"r0": 0, "r1": 1}, {"c0": 0, "c1": 1, "c2": 2}]
        dim_names = dim_names or ["row", "col"]
        return KeyedArray(arr, frozen=frozen, key_pos_pairs=key_pos_pairs,
                          dim_names=dim_names)

    return _make


class TestConstructionAndProperties:
    def test_values_ndim_shape_dtype_size(self, make_kr):
        kr = make_kr()
        assert isinstance(kr.values, np.ndarray)
        assert kr.ndim == 2
        assert kr.shape == (2, 3)
        assert kr.size == 6
        assert kr.dtype == np.float64

    def test_frozen_flags_writeable_false(self, make_kr):
        kr = make_kr(frozen=True)
        assert kr.values.flags.writeable is False

    def test_not_frozen_flags_writeable_true(self, make_kr):
        kr = make_kr(frozen=False)
        assert kr.values.flags.writeable is True

    def test_dim_names_and_key_pos_pairs_exposed(self, make_kr):
        kr = make_kr()
        assert kr.dim_names == ["row", "col"]
        assert kr.key_pos_pairs == [{"r0": 0, "r1": 1}, {"c0": 0, "c1": 1, "c2": 2}]

    def test_key_pos_pairs_not_a_list_raises(self, make_kr):
        with pytest.raises(TypeError):
            make_kr(key_pos_pairs=({"a": 0}, {"b": 1}))

    def test_key_pos_pairs_length_mismatch_raises(self, make_kr):
        with pytest.raises(ValueError):
            make_kr(key_pos_pairs=[{"a": 0}])  # 1 pair for a 2-D array

    def test_dim_names_length_mismatch_warns_and_defaults(self, make_kr):
        with pytest.warns(UserWarning, match="does not match"):
            kr = make_kr(dim_names=["only"])
        assert kr.dim_names == ["dimension1", "dimension2"]

    def test_dim_names_duplicate_warns_and_defaults(self, make_kr):
        with pytest.warns(UserWarning, match="not unique"):
            kr = make_kr(dim_names=["row", "row"])
        assert kr.dim_names == ["dimension1", "dimension2"]


class TestPositionalIndexing:
    """`[]` follows NumPy integer-position semantics."""

    def test_getitem_by_position(self, make_kr):
        kr = make_kr()
        assert kr[1, 2] == 5.0  # row 1, col 2

    def test_setitem_by_position(self, make_kr):
        kr = make_kr()
        kr[0, 1] = 99.0
        assert kr[0, 1] == 99.0
        assert kr.values[0, 1] == 99.0

    def test_frozen_blocked_by_position(self, make_kr):
        kr = make_kr(frozen=True)
        with pytest.raises(ValueError):
            kr[0, 0] = 1.0


class TestLabelIndexingAt:
    """`.at` provides label-based access, raising IndexError on bad keys."""

    def test_getitem_by_label(self, make_kr):
        kr = make_kr()
        assert kr.at[("r1", "c2")] == 5.0
        assert kr.at["r1", "c2"] == 5.0

    def test_2d_parenthesized_row_key(self, make_kr):
        # row key is itself a tuple: at[(row_key_tuple, col_key)]
        row_kpp = [{"a": 0, "b": 1}, {"z": 0}]
        kr = KeyedArray(np.arange(2.0).reshape(2, 1), frozen=False,
                        key_pos_pairs=row_kpp, dim_names=["dim1", "dim2"])
        assert kr.at[("a", "z")] == 0.0
        assert kr.at["a", "z"] == 0.0

    def test_2d_tuple_omission(self, make_kr):
        # for 2-D arrays the row-key parentheses may be omitted:
        # at["a", "b1", "c0"] == at[("a", "b1"), "c0"]
        kpp = [{("a", "b1"): 0, ("a", "b2"): 1}, {"c0": 0, "c1": 1}]
        kr = KeyedArray(np.arange(4.0).reshape(2, 2), frozen=False,
                        key_pos_pairs=kpp, dim_names=["row", "col"])
        # 3 keys on a 2-D array: row key is a 2-tuple, plus the col key
        assert kr.at["a", "b1", "c1"] == 1.0
        assert kr.at[("a", "b1"), "c1"] == 1.0
        assert kr.at[("a", "b1", "c1")] == 1.0
        assert kr.at[(("a", "b1"), "c1")] == 1.0

    def test_at_invalid_key_raises(self, make_kr):
        kr = make_kr()
        with pytest.raises(IndexError):
            kr.at[("nope", "c0")]

    def test_at_wrong_key_length_warns_then_raises(self, make_kr):
        # key count != ndim: parse warns and IndexError is raised
        kr = make_kr()

        with pytest.warns(UserWarning, match="does not match ndim"):
            with pytest.raises(IndexError):
                kr.at[("r0",)]  # 1 key for a 2-D array

    def test_setitem_by_label(self, make_kr):
        kr = make_kr()
        kr.at[("r0", "c1")] = 42.0
        assert kr.at[("r0", "c1")] == 42.0

    def test_frozen_blocked_by_label(self, make_kr):
        kr = make_kr(frozen=True)
        with pytest.raises(ValueError):
            kr.at[("r0", "c0")] = 1.0


class TestGetMethod:
    """`.get` is `.at` that falls back to a default value instead of raising."""

    def test_get_by_positional(self, make_kr):
        kr = make_kr()
        # one positional key per dimension; a single tuple argument would be an
        # incomplete key set (len == 1 != ndim) and fall back to default
        assert kr.get("r1", "c2") == 5.0

    def test_get_by_keyword_order_independent(self, make_kr):
        kr = make_kr()
        assert kr.get(row="r1", col="c2") == 5.0
        assert kr.get(col="c2", row="r1") == 5.0

    def test_get_missing_returns_none_default(self, make_kr):
        kr = make_kr()
        assert kr.get(("nope", "c0")) is None

    def test_get_missing_returns_explicit_default(self, make_kr):
        kr = make_kr()
        assert kr.get(("nope", "c0"), default=-1) == -1

    def test_get_unknown_kwarg_dim_returns_default(self, make_kr):
        kr = make_kr()
        assert kr.get(bogus="x", default=42) == 42

    def test_get_incomplete_kwargs_returns_default(self, make_kr):
        kr = make_kr()
        assert kr.get(row="r0", default=7) == 7  # col missing

    def test_get_mixing_args_and_kwargs_raises(self, make_kr):
        kr = make_kr()
        with pytest.raises(TypeError):
            kr.get(("r0",), col="c0")


class TestKeyToPos:
    """`key_to_pos` maps a single dimension key to an integer position."""

    def test_int_dim(self, make_kr):
        kr = make_kr()
        assert kr.key_to_pos(1, "c2") == 2

    def test_str_dim(self, make_kr):
        kr = make_kr()
        assert kr.key_to_pos("col", "c2") == 2

    def test_unknown_key_returns_none(self, make_kr):
        kr = make_kr()
        assert kr.key_to_pos("col", "zzz") is None

    def test_unknown_dim_name_returns_none(self, make_kr):
        kr = make_kr()
        assert kr.key_to_pos("bogus", "c2") is None

    def test_invalid_dim_type_returns_none(self, make_kr):
        kr = make_kr()
        assert kr.key_to_pos(1.5, "c2") is None

    def test_unknown_key_with_if_not_found(self, make_kr):
        kr = make_kr()
        assert kr.key_to_pos("col", "zzz", if_not_found=-9) == -9


class TestFromDf:
    def test_empty_returns_none(self):
        df = pd.DataFrame({"A": []})
        assert KeyedArray.from_df(df) is None

    def test_single_index_labelled(self):
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]},
                          index=pd.Index(["x", "y"], name="idx"))
        kr = KeyedArray.from_df(df)
        assert kr.dim_names == ["idx", "col_name"]
        assert kr.shape == (2, 2)
        assert kr.at[("x", "B")] == 3.0
        assert kr.get("y", "A") == 2.0

    def test_multi_index_unpacked(self):
        df = pd.DataFrame(
            {"A": [1.0, 2.0], "B": [3.0, 4.0]},
            index=pd.MultiIndex.from_arrays([["g1", "g2"], ["t1", "t2"]],
                                            names=["gp", "tp"]),
        )
        kr = KeyedArray.from_df(df, unpack_multi_index=True)
        assert kr.dim_names == ["gp", "tp", "col_name"]
        assert kr.shape == (2, 2, 2)  # 2 groups x 2 types x 2 cols
        assert kr.at[("g1", "t1", "A")] == 1.0
        assert kr.at["g1", "t1", "A"] == 1.0

    def test_multi_index_collapsed_when_not_unpacked(self):
        # without unpack_multi_index the multi-index becomes a single row
        # dimension labelled 'row_index'.
        df = pd.DataFrame(
            {"A": [1.0, 2.0]},
            index=pd.MultiIndex.from_arrays([["g1", "g1"], ["t1", "t2"]]),
        )
        kr = KeyedArray.from_df(df)
        assert kr.dim_names == ["row_index", "col_name"]
        assert kr.shape == (2, 1)
        assert kr.at[("g1", "t1"), "A"] == 1.0
        assert kr.at[(("g1", "t1"), "A")] == 1.0

    def test_numeric_dtype_promoted_to_float(self):
        df = pd.DataFrame({"A": [1, 2]})  # int64
        kr = KeyedArray.from_df(df)
        assert kr.dtype == np.float64

    def test_object_dtype_warns(self):
        df = pd.DataFrame({"A": ["x", "y"]})
        with pytest.warns(UserWarning):
            kr = KeyedArray.from_df(df)
        assert kr.dtype == object
        assert kr.at[(0, "A")] == "x"

    def test_frozen_default_true(self):
        df = pd.DataFrame({"A": [1.0]}, index=pd.Index(["x"]))
        kr = KeyedArray.from_df(df)
        assert kr.values.flags.writeable is False

    def test_frozen_false_writeable(self):
        df = pd.DataFrame({"A": [1.0]}, index=pd.Index(["x"]))
        kr = KeyedArray.from_df(df, frozen=False)
        assert kr.values.flags.writeable is True


class TestDirectConstructionWithoutDimNames:
    """`dim_names` defaults to `None` in the constructor. The indexer routes it
    through `_validate_dim_names`, which synthesizes `dimension1/2/...`; the
    array must be usable with those default names."""

    def test_no_dim_names_synthesizes_defaults(self):
        kr = KeyedArray(np.zeros((2, 2)), frozen=False,
                        key_pos_pairs=[{"a": 0}, {"b": 1}])
        assert kr.dim_names == ["dimension1", "dimension2"]
        # default names are wired into the indexer, not a dead attribute
        assert kr.at[("a", "b")] == 0.0

    def test_no_dim_names_key_to_pos_by_default_name(self):
        kr = KeyedArray(np.arange(4.0).reshape(2, 2), frozen=False,
                        key_pos_pairs=[{"a": 0}, {"b": 1}])
        assert kr.key_to_pos("dimension1", "a") == 0
        assert kr.key_to_pos("dimension2", "b") == 1
        assert kr.get(dimension1="a", dimension2="b") == 1.0  # value at [0, 1]
"""Tests for `vates/solvency/risk_tree.py`: the `RiskNode` / `RiskTree` pair and
the `risk_aggregation` correlation formula it is built on.

The module is the backbone of the three solvency standard formulas shipped in this
package (`cn_cross2`, `eu_solvency2`, `hk_rbc`): each builder assembles a
risk-module hierarchy once, then leaf capitals are pushed through it.  These tests
pin that contract -- path resolution, the identifier -> keyword-argument mapping
used by `agg_func`, the capital cache and its invalidation, and the copy/view
semantics -- plus the integration with the three builders.
"""

import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from vates.solvency.risk_tree import RiskNode, RiskTree, preorder_traversal, risk_aggregation
from vates.solvency.cn_cross2 import make_cross2_mc_module
from vates.solvency.eu_solvency2 import make_solvency2_scr_module
from vates.solvency.eu_solvency2.rules import CORR_MATRIX_LIFE
from vates.solvency.hk_rbc import make_hkrbc_pcr_module

BUILDERS = [make_cross2_mc_module, make_solvency2_scr_module, make_hkrbc_pcr_module]
BUILDER_IDS = ["cn_cross2", "eu_solvency2", "hk_rbc"]


def _sum_agg(**capitals: float) -> float:
    """Aggregation rule that adds up whatever children it is given."""
    return sum(capitals.values())


class _RecordingAgg:
    """Callable object carrying mutable state, to observe sharing across copies."""

    def __init__(self):
        self.seen = []

    def __call__(self, **capitals: float) -> float:
        self.seen.append(capitals)
        return sum(capitals.values())


@pytest.fixture
def make_tree():
    """Factory for a two-module tree `T -> M(E, F), L(H)`, aggregated by sum.
    T
    ├── M
    │   ├── E (1.0)
    │   └── F (2.0)
    └── L
        └── H (4.0)
    """

    def _make(values=None):
        tree = RiskTree(root="T")
        tree.root.add_child("M", "L", agg_func=_sum_agg)
        tree.get_node("M").add_child("E", "F", agg_func=_sum_agg)
        tree.get_node("L").add_child("H", agg_func=_sum_agg)
        tree.batch_set_risk_capital(values or {"M/E": 1.0, "M/F": 2.0, "L/H": 4.0})
        return tree

    return _make


@pytest.fixture
def tree(make_tree):
    """A tree with `M/E = 1`, `M/F = 2`, `L/H = 4` (so `T = 7`, `M = 3`, `L = 4`)."""
    return make_tree()


class TestRiskAggregation:
    def test_uncorrelated_risks_are_combined_in_quadrature(self):
        assert risk_aggregation(3.0, 4.0, corr_matrix=np.eye(2)) == pytest.approx(5.0)

    def test_perfectly_correlated_risks_add_up(self):
        assert risk_aggregation(3.0, 4.0, corr_matrix=np.ones((2, 2))) == pytest.approx(7.0)

    def test_sequence_form_matches_variadic_form(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        assert risk_aggregation([3.0, 4.0], corr_matrix=corr) == pytest.approx(
            risk_aggregation(3.0, 4.0, corr_matrix=corr))

    def test_quadratic_form_is_evaluated_as_documented(self):
        corr = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])
        x = np.array([100.0, 200.0, 300.0])
        assert risk_aggregation(100.0, 200.0, 300.0, corr_matrix=corr) == pytest.approx(
            math.sqrt(float(x @ corr @ x)))

    def test_result_is_a_float(self):
        assert isinstance(risk_aggregation(1.0, 1.0, corr_matrix=np.eye(2)), float)

    def test_no_risk_raises_value_error(self):
        with pytest.raises(ValueError, match="No risk is provided"):
            risk_aggregation(corr_matrix=np.eye(2))

    @pytest.mark.parametrize("capitals", [5.0, [5.0]],
                             ids=["scalar", "one-element sequence"])
    def test_fewer_than_two_risks_raises_value_error(self, capitals):
        # `np.atleast_1d` lifts a scalar into a vector, so both spellings of a
        # single risk reach the same guard.
        with pytest.raises(ValueError, match="single scalar"):
            risk_aggregation(capitals, corr_matrix=[[1.0]])

    def test_corr_matrix_shape_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="Correlation matrix shape"):
            risk_aggregation(1.0, 2.0, corr_matrix=np.eye(3))

    def test_negatively_correlated_risks_offset_each_other(self):
        # rho = -1: the aggregate is the absolute difference of the capitals.
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert risk_aggregation(3.0, 4.0, corr_matrix=corr) == pytest.approx(1.0)

    def test_a_negative_off_diagonal_entry_lowers_the_aggregate(self):
        corr = np.array([[1.0, -0.25], [-0.25, 1.0]])
        assert risk_aggregation(100.0, 200.0, corr_matrix=corr) == pytest.approx(200.0)

    def test_a_regulatory_matrix_with_negative_correlations_is_accepted(self):
        # Solvency II life: mortality and longevity are correlated at -0.25.
        assert np.any(CORR_MATRIX_LIFE < 0.0)  # the premise of this test
        capitals = [1.0] * len(CORR_MATRIX_LIFE)
        x = np.array(capitals)
        assert risk_aggregation(*capitals, corr_matrix=CORR_MATRIX_LIFE) == pytest.approx(
            math.sqrt(float(x @ CORR_MATRIX_LIFE @ x)))

    def test_a_non_positive_semidefinite_matrix_raises_value_error(self):
        # Every entry is a legal correlation, but the matrix is not positive
        # semi-definite: the quadratic form of (1, 1, 1) is negative, and
        # `math.sqrt` rejects it.
        corr = np.array([[1.0, -0.9, -0.9], [-0.9, 1.0, -0.9], [-0.9, -0.9, 1.0]])
        assert float(np.ones(3) @ corr @ np.ones(3)) < 0.0  # the premise of this test
        with pytest.raises(ValueError, match="math domain error"):
            risk_aggregation(1.0, 1.0, 1.0, corr_matrix=corr)


class TestRiskNodeStructure:
    def test_name_and_identifier(self):
        node = RiskNode("Non-life")
        assert node.name == "Non-life"
        assert node.identifier == "non_life"

    def test_explicit_identifier_wins(self):
        assert RiskNode("Property", identifier="property_").identifier == "property_"

    @pytest.mark.parametrize("name", ["abc/def", "Market.Spread", ".", ".."],
                             ids=["slash", "dot", "dot-only", "dotdot"])
    def test_reserved_characters_in_name_raise_value_error(self, name):
        with pytest.raises(ValueError, match="contains"):
            RiskNode(name)

    @pytest.mark.parametrize("name, identifier", [
        ("Non-life", "non_life"),
        ("Type 1", "type_1"),
        ("1 Year", "_1_year"),          # digit-led: prefixed to stay a valid kwarg name
        ("2015-35 Module", "_2015_35_module"),
        ("  Mixed  CASE ", "__mixed__case_"),
    ])
    def test_identifier_normalisation(self, name, identifier):
        assert RiskNode(name).identifier == identifier

    def test_a_digit_led_identifier_can_be_used_by_an_agg_func(self):
        # The "_" prefix is what makes this parameter name expressible.
        def agg(_1_year: float, other: float) -> float:
            return _1_year + other

        node = RiskNode("Root")
        node.add_child("1 Year", "Other", agg_func=agg)
        node.goto("1 Year").set_risk_capital(1.0)
        node.goto("Other").set_risk_capital(2.0)
        assert node.risk_capital == 3.0

    @pytest.mark.parametrize("identifier", [1, 0, False], ids=["int", "falsy int", "bool"])
    def test_non_string_identifier_raises_type_error(self, identifier):
        # A falsy non-str identifier used to be swallowed by `identifier or name`.
        with pytest.raises(TypeError, match="expected 'str'"):
            RiskNode("A", identifier=identifier)

    def test_empty_identifier_raises_value_error(self):
        with pytest.raises(ValueError, match="Empty chars"):
            RiskNode("A", identifier="")

    def test_none_identifier_falls_back_to_the_normalised_name(self):
        assert RiskNode("Non-life", identifier=None).identifier == "non_life"

    def test_an_explicit_identifier_is_normalised_like_a_name(self):
        assert RiskNode("Property", identifier="Type 1").identifier == "type_1"

    def test_an_empty_name_is_rejected_by_the_identifier_check(self):
        with pytest.raises(ValueError, match="Empty chars"):
            RiskNode("")

    def test_new_node_is_a_detached_leaf(self):
        node = RiskNode("A")
        assert node.is_root and node.is_leaf and node.parent is None
        assert node.children == [] and node.siblings == [] and node.depth == 0
        assert node.path == "" and node.root is node

    def test_add_sub_risk_accepts_names_and_nodes(self):
        root = RiskNode("T")
        child = RiskNode("B")
        root.add_child("A", child)
        assert root.children == [root.goto("A"), child]
        assert [c.parent for c in root.children] == [root, root]
        assert not root.is_leaf
        assert root.goto("A").depth == 1
        assert root.goto("B").root is root
        assert root.goto("B").siblings == [root.goto("A")]
        assert root.goto("B").path == "B"

    def test_add_sub_risk_rejects_other_types(self):
        with pytest.raises(TypeError, match="Invalid"):
            RiskNode("T").add_child(42)

    def test_link_parent_is_the_mirror_of_link_child(self):
        root, child = RiskNode("T"), RiskNode("C")
        child._link_parent(root)
        assert child.parent is root and child in root.children

    def test_duplicate_name_raises_value_error(self):
        root = RiskNode("T")
        root.add_child("A")
        with pytest.raises(ValueError, match="duplicate name"):
            root.add_child("A")

    def test_duplicate_identifier_raises_value_error(self):
        root = RiskNode("T")
        root.add_child("A-B")
        with pytest.raises(ValueError, match="duplicate identifier"):
            root.add_child("A B")  # both normalise to `a_b`

    def test_relinking_an_attached_child_raises_value_error(self):
        root = RiskNode("T")
        child = RiskNode("A")
        root.add_child(child)
        with pytest.raises(ValueError, match="has parent"):
            root.add_child(child)

    def test_adding_an_ancestor_raises_value_error(self):
        root = RiskNode("T")
        root.add_child("A")
        with pytest.raises(ValueError, match="is root"):
            root.goto("A").add_child(root)

    def test_a_node_cannot_be_linked_to_itself(self):
        root = RiskNode("T")
        with pytest.raises(ValueError, match="is root"):
            root.link_child(root)

    def test_ancestors_descendants_siblings_and_depth(self, tree):
        node = tree.get_node("M/E")
        assert [n.name for n in node.ancestors] == ["M", "T"]  # parent .. root
        assert [n.name for n in node.root.descendants] == ["M", "E", "F", "L", "H"]
        assert [n.name for n in tree.get_node("M").descendants] == ["E", "F"]
        assert [n.name for n in node.siblings] == ["F"]
        assert tree.root.siblings == []
        assert node.depth == 2 and tree.root.depth == 0

    def test_path_is_absolute_from_the_outermost_root(self, tree):
        assert tree.get_node("M/E").path == "M/E"
        assert tree.get_subtree("M").get_node("E").path == "M/E"
        assert tree.root.path == ""

    def test_goto_resolves_paths(self, tree):
        assert tree.root.goto(None) is tree.root
        assert tree.root.goto("") is tree.root
        assert tree.root.goto(".") is tree.root
        assert tree.root.goto("M/E") is tree.get_node("M/E")
        assert tree.root.goto("/M/E") is tree.get_node("M/E")  # leading "/" ignored
        assert tree.get_node("M").goto("E") is tree.get_node("M/E")

    @pytest.mark.parametrize("path", ["..", "M/.."])
    def test_goto_has_no_parent_component(self, tree, path):
        # No name may contain ".", so ".." can never match a child.
        with pytest.raises(ValueError, match="can't goto node"):
            tree.get_node("M/E").goto(path)

    def test_names_may_not_look_like_path_syntax(self):
        root = RiskNode("T")
        with pytest.raises(ValueError, match="contains '.'"):
            root.add_child("..")
        with pytest.raises(ValueError, match="contains '.'"):
            root.add_child(".")
        assert root.children == []

    def test_goto_unknown_child_raises_value_error(self, tree):
        with pytest.raises(ValueError, match="failed at 'X'"):
            tree.root.goto("M/X")

    def test_goto_rejects_non_string_paths(self, tree):
        with pytest.raises(TypeError, match="expected 'str'"):
            tree.root.goto(1)

    def test_truediv_is_goto(self, tree):
        assert tree.root / "M" / "F" is tree.get_node("M/F")


class TestCapitalAggregation:
    def test_leaf_capital_is_aggregated_upwards(self, tree):
        assert tree.get_node("M/E").risk_capital == 1.0
        assert tree.get_node("M").risk_capital == 3.0
        assert tree.get_node("L").risk_capital == 4.0
        assert tree.get_risk_capital() == 7.0

    def test_agg_func_receives_children_keyed_by_identifier(self):
        node = RiskNode("Root")
        def _agg(non_life: float, life: float) -> float:
            return non_life + 2.0 * life
        node.add_child("Non-life", "Life", agg_func=_agg)
        node.goto("Non-life").set_risk_capital(1.0)
        node.goto("Life").set_risk_capital(2.0)
        assert node.risk_capital == 5.0

    def test_explicit_identifier_is_the_keyword(self):
        node = RiskNode("Market")
        def _agg(property_: float, equity: float) -> float:
            return property_ + equity
        node.add_child(RiskNode("Property", identifier="property_"), "Equity", agg_func=_agg)
        node.goto("Property").set_risk_capital(1.0)
        node.goto("Equity").set_risk_capital(2.0)
        assert node.risk_capital == 3.0

    def test_set_risk_capital_on_a_non_leaf_raises_value_error(self, tree):
        with pytest.raises(ValueError, match="non-leaf"):
            tree.set_risk_capital("M", 1.0)

    @pytest.mark.parametrize("value", [3, 3.5, np.float64(3.5), np.float32(3.5),
                                       np.int64(3), np.int32(3), Fraction(7, 2)],
                             ids=["int", "float", "np.float64", "np.float32",
                                  "np.int64", "np.int32", "Fraction"])
    def test_set_risk_capital_accepts_real_numbers(self, tree, value):
        # NumPy scalars matter in practice: capitals read out of a pandas or NumPy
        # pipeline are often `np.int64` / `np.float32`, which an `(int, float)`
        # check would reject.
        tree.set_risk_capital("M/E", value)
        assert tree.get_risk_capital("M/E") == value

    @pytest.mark.parametrize("bad", [None, "1.0", True, Decimal("3.5"), np.array(3.5)],
                             ids=["None", "str", "bool", "Decimal", "0-d array"])
    def test_set_risk_capital_requires_a_number(self, tree, bad):
        with pytest.raises(TypeError, match="Invalid type of risk capital"):
            tree.set_risk_capital("M/E", bad)

    def test_set_risk_capital_rejects_nan(self, tree):
        with pytest.raises(ValueError, match="Invalid value of risk capital"):
            tree.set_risk_capital("M/E", float("nan"))

    def test_set_risk_capital_accepts_negative_and_infinite_values(self, tree):
        tree.set_risk_capital("M/E", -5.0)
        assert tree.get_risk_capital("M/E") == -5.0  # a capital may be an offset
        tree.set_risk_capital("M/E", float("inf"))
        assert tree.get_risk_capital("M/E") == float("inf")  # only NaN is refused

    def test_reading_a_valueless_leaf_raises_value_error(self):
        with pytest.raises(ValueError, match="hasn't been provided"):
            RiskNode("A").risk_capital

    def test_reading_an_internal_node_without_agg_func_raises_value_error(self):
        root = RiskNode("T")
        root.add_child("A")
        with pytest.raises(ValueError, match="aggregation function is None"):
            root.risk_capital

    def test_aggregate_does_nothing_on_a_leaf(self):
        node = RiskNode("A")
        node.aggregate()
        with pytest.raises(ValueError, match="hasn't been provided"):
            node.risk_capital

    def test_setting_a_leaf_clears_the_cache_of_its_ancestors(self, tree):
        assert tree.get_risk_capital() == 7.0  # populates the cache
        tree.set_risk_capital("M/E", 10.0)
        assert tree.root._risk_capital is None
        assert tree.get_node("M")._risk_capital is None
        assert tree.get_risk_capital() == 16.0

    def test_clr_risk_capital_clears_self_and_ancestors_only(self, tree):
        assert tree.get_risk_capital() == 7.0
        tree.get_node("M/F").clr_risk_capital()
        assert tree.get_node("M/F")._risk_capital is None
        assert tree.get_node("M")._risk_capital is None
        assert tree.root._risk_capital is None
        assert tree.get_node("M/E")._risk_capital == 1.0  # sibling kept
        assert tree.get_node("L")._risk_capital == 4.0  # other module kept

    def test_linking_a_valued_child_invalidates_the_parent_cache(self):
        # Regression: `link_child` must clear the parent's cached capital, else a
        # pre-valued sub-module attached later is silently ignored.
        node = RiskNode("P")
        node.add_child("A", "B", agg_func=_sum_agg)
        node.goto("A").set_risk_capital(1.0)
        node.goto("B").set_risk_capital(2.0)
        assert node.risk_capital == 3.0
        extra = RiskNode("C")
        extra.set_risk_capital(10.0)
        node.link_child(extra)
        assert node.risk_capital == 13.0

    def test_a_valued_leaf_stops_reporting_its_value_once_it_has_children(self):
        # The same invalidation seen from the leaf side: the node is internal now,
        # so its old leaf value must not survive as "the" capital.
        node = RiskNode("Q")
        node.set_risk_capital(7.0)
        node.add_child("X", agg_func=_sum_agg)
        assert node._risk_capital is None
        node.goto("X").set_risk_capital(3.0)
        assert node.risk_capital == 3.0

    def test_set_agg_func_twice_raises_value_error(self):
        node = RiskNode("T")
        node.add_child("A", agg_func=_sum_agg)
        with pytest.raises(ValueError, match="already been set"):
            node.set_agg_func(_sum_agg)

    def test_add_sub_risk_is_atomic_when_the_agg_func_is_already_set(self):
        node = RiskNode("S")
        node.add_child("A", agg_func=_sum_agg)
        with pytest.raises(ValueError, match="already been set"):
            node.add_child("B", agg_func=_sum_agg)
        assert [c.name for c in node.children] == ["A"]

    @pytest.mark.parametrize("bad_args", [
        ("A", 42),          # not a RiskNode or a str
        ("A", "A"),         # duplicate name inside the call
        ("A-B", "A B"),     # duplicate identifier inside the call
        ("A.B",),           # reserved character in the name
    ], ids=["bad type", "duplicate name", "duplicate identifier", "reserved character"])
    def test_add_sub_risk_attaches_nothing_when_an_argument_is_rejected(self, bad_args):
        # Staging validates the whole call first, so a rejected argument leaves
        # neither children nor aggregation function behind.
        node = RiskNode("S")
        with pytest.raises((TypeError, ValueError)):
            node.add_child(*bad_args, agg_func=_sum_agg)
        assert node.children == []
        assert node._agg_func is None

    def test_add_sub_risk_rejects_the_same_node_twice(self):
        node = RiskNode("S")
        child = RiskNode("A")
        with pytest.raises(ValueError, match="duplicate object"):
            node.add_child(child, child)
        assert node.children == [] and child.parent is None

    def test_zeroize_sets_every_leaf_to_zero(self, tree):
        tree.zeroize()
        assert all(leaf.risk_capital == 0.0 for leaf in tree.list_leaf_nodes())
        assert tree.get_risk_capital() == 0.0


class TestRiskTree:
    def test_root_from_name_or_node(self):
        assert RiskTree("T").root.name == "T"
        node = RiskNode("T")
        assert RiskTree(node).root is node

    def test_invalid_root_type_raises_type_error(self):
        with pytest.raises(TypeError, match="expected"):
            RiskTree(42)

    def test_is_toptree_and_is_subtree(self, tree):
        assert tree.is_toptree and not tree.is_subtree
        subtree = tree.get_subtree("M")
        assert subtree.is_subtree and not subtree.is_toptree
        assert subtree.get_toptree().root is tree.root

    def test_get_risk_capital_by_path(self, tree):
        assert tree.get_risk_capital() == 7.0
        assert tree.get_risk_capital("") == 7.0
        assert tree.get_risk_capital("M") == 3.0
        assert tree.get_risk_capital("M/E") == 1.0

    def test_get_risk_capital_unknown_path_raises_value_error(self, tree):
        with pytest.raises(ValueError, match="can't goto node"):
            tree.get_risk_capital("Nope")

    def test_get_node_returns_the_node_itself(self, tree):
        assert tree.get_node("M/E") is tree.get_node("M").children[0]

    def test_diversification_is_sum_of_children_minus_the_node(self):
        tree = RiskTree("T")
        tree.root.add_child("A", "B", agg_func=lambda a, b: math.sqrt(a ** 2 + b ** 2))
        tree.batch_set_risk_capital({"A": 3.0, "B": 4.0})
        assert tree.get_risk_diversification() == pytest.approx(2.0)  # 7 - 5
        assert tree.get_risk_diversification("A") == 0.0  # leaf

    def test_subtree_is_a_live_view(self, tree):
        subtree = tree.get_subtree("M")
        assert subtree.get_risk_capital("E") == 1.0  # paths are subtree-relative
        subtree.set_risk_capital("E", 9.0)
        assert tree.get_risk_capital("M/E") == 9.0  # the very same nodes
        assert subtree.get_risk_capital() == 11.0
        assert subtree.get_node("E") is tree.get_node("M/E")

    def test_list_nodes_and_leaves_are_preorder(self, tree):
        assert [n.name for n in tree.list_nodes()] == ["T", "M", "E", "F", "L", "H"]
        assert [n.name for n in tree.list_leaf_nodes()] == ["E", "F", "H"]

    def test_batch_set_risk_capital_accepts_flat_and_nested_dicts(self):
        tree = RiskTree("T")
        tree.root.add_child("M", agg_func=_sum_agg)
        tree.get_node("M").add_child("E", "F", agg_func=_sum_agg)
        tree.batch_set_risk_capital({"M/E": 1.0, "M/F": 2.0})
        assert tree.get_risk_capital() == 3.0
        tree.batch_set_risk_capital({"M": {"E": 10.0, "F": 20.0}})
        assert tree.get_risk_capital() == 30.0

    def test_batch_set_risk_capital_rejects_bad_keys_and_values(self):
        tree = RiskTree("T")
        tree.root.add_child("A", agg_func=_sum_agg)
        with pytest.raises(TypeError, match="expected 'str'"):
            tree.batch_set_risk_capital({1: 2.0})
        with pytest.raises(TypeError, match=r"expected \('float', 'dict'\)"):
            tree.batch_set_risk_capital({"A": "2.0"})

    def test_batch_set_risk_capital_rejects_unknown_and_non_leaf_paths(self, tree):
        with pytest.raises(ValueError, match="non-leaf"):
            tree.batch_set_risk_capital({"M": 1.0})
        with pytest.raises(ValueError, match="can't goto node"):
            tree.batch_set_risk_capital({"M/Nope": 1.0})

    def test_flatten_dict_joins_nested_keys(self):
        assert RiskTree._flatten_dict({"a": {"b": 1.0}, "c": 2.0}) == {"a/b": 1.0, "c": 2.0}
        assert RiskTree._flatten_dict({"a": {"b": 1.0}}, joiner=".") == {"a.b": 1.0}


class TestDisplay:
    def test_display_prints_each_node_with_its_capital(self, tree, capsys):
        tree.display(width=20, precision=1)
        lines = capsys.readouterr().out.splitlines()
        assert [line.split() for line in lines] == [
            ["T", "7.0"], ["M", "3.0"], ["E", "1.0"], ["F", "2.0"], ["L", "4.0"], ["H", "4.0"]]

    def test_display_omits_the_value_of_a_node_without_capital(self, capsys):
        tree = RiskTree("T")
        tree.root.add_child("A", "B", agg_func=_sum_agg)
        tree.set_risk_capital("A", 1.0)
        tree.display(width=20, precision=1)
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].split() == ["T"]  # B has no capital, so T cannot aggregate
        assert lines[1].split() == ["A", "1.0"]
        assert lines[2].split() == ["B"]

    def test_display_indents_by_depth(self, tree, capsys):
        tree.display(width=4, precision=0)  # narrower than the names
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].startswith("T")
        assert lines[1].startswith("    M") and not lines[1].startswith("     ")
        assert lines[2].startswith("        E")


class TestDeepcopy:
    def test_copy_is_independent(self, tree):
        copied = tree.deepcopy()
        copied.set_risk_capital("M/E", 100.0)
        assert copied.get_risk_capital() == 106.0
        assert tree.get_risk_capital() == 7.0
        assert copied.root is not tree.root
        assert copied.root.parent is None
        assert [n.name for n in copied.list_nodes()] == [n.name for n in tree.list_nodes()]
        assert copied.get_node("M").identifier == tree.get_node("M").identifier

    def test_copy_of_a_subtree_is_rebased_on_its_own_root(self, tree):
        copied = tree.get_subtree("M").deepcopy()
        assert copied.is_toptree and copied.root.parent is None
        assert copied.root.path == ""
        assert [n.path for n in copied.list_nodes()] == ["", "E", "F"]
        assert copied.get_risk_capital() == 3.0

    def test_with_value_false_clears_the_leaf_values(self, tree):
        copied = tree.deepcopy(with_value=False)
        assert copied.get_node("M/E")._risk_capital is None
        with pytest.raises(ValueError, match="hasn't been provided"):
            copied.get_risk_capital()

    def test_with_value_true_keeps_the_leaf_values_and_recomputes_the_rest(self, tree):
        assert tree.get_risk_capital() == 7.0  # populate the caches
        assert tree.root._risk_capital == 7.0
        copied = tree.deepcopy()
        assert copied.get_node("M/E")._risk_capital == 1.0  # leaf value carried over
        assert copied.root._risk_capital is None  # internal cache rebuilt lazily
        assert copied.get_risk_capital() == 7.0

    def test_the_aggregation_function_object_is_shared(self):
        agg = _RecordingAgg()
        tree = RiskTree("T")
        tree.root.add_child("A", agg_func=agg)
        tree.set_risk_capital("A", 1.0)
        assert tree.get_risk_capital() == 1.0
        copied = tree.deepcopy()
        copied.set_risk_capital("A", 5.0)
        assert copied.get_risk_capital() == 5.0
        assert agg.seen[-1] == {"a": 5.0}  # the copy called the original object


class TestPreorderTraversal:
    def test_preorder_order(self, tree):
        assert [n.name for n in preorder_traversal(tree.root)] == ["T", "M", "E", "F", "L", "H"]

    def test_single_node(self):
        node = RiskNode("A")
        assert preorder_traversal(node) == [node]


class TestSolvencyModules:
    """The three standard-formula builders are the module's real clients."""

    @pytest.mark.parametrize("make_module", BUILDERS, ids=BUILDER_IDS)
    def test_module_builds_zeroizes_and_aggregates(self, make_module):
        tree = make_module()
        assert tree.is_toptree
        leaves = tree.list_leaf_nodes()
        # `is_zeroize=True` by default: every leaf starts at 0.0, and the whole
        # hierarchy aggregates without any `agg_func` parameter-name mismatch.
        assert leaves and tree.get_risk_capital() == 0.0
        for leaf in leaves:
            leaf.set_risk_capital(1_000_000.0)
        assert tree.get_risk_capital() > 0.0
        tree.zeroize()
        assert tree.get_risk_capital() == 0.0

    @pytest.mark.parametrize("make_module", BUILDERS, ids=BUILDER_IDS)
    def test_submodule_is_built_as_a_detached_tree(self, make_module):
        top = make_module()
        path = top.root.children[0].path
        sub = make_module(submodule=path)
        assert sub.is_toptree and sub.root.name == top.root.children[0].name
        assert [n.name for n in sub.list_nodes()] == [
            n.name for n in top.get_subtree(path).list_nodes()]
        leaf = sub.list_leaf_nodes()[0]
        leaf.set_risk_capital(1.0)
        assert leaf.risk_capital == 1.0
        top_leaf = top.get_node(f"{path}/{leaf.path}")
        assert top_leaf is not leaf
        assert top_leaf.risk_capital == 0.0  # the top module is untouched by the copy

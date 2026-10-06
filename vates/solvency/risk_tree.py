"""
Tree structures for modelling actuarial solvency risk modules.

A solvency capital requirement is normally specified as a hierarchy of risk
modules -- market, counterparty default, life, health, non-life, ... -- each
broken down into sub-risks down to the individual risk factors.  This module
provides the two building blocks of such a hierarchy:

* :class:`RiskNode` -- one node of the hierarchy, holding its name, its
  identifier, the (cached) risk capital and the links to its parent and children.
* :class:`RiskTree` -- a thin wrapper around a root :class:`RiskNode`, adding
  tree-level lookups such as ``get_risk_capital`` or ``get_subtree``.

Capitals are aggregated bottom-up.  Every internal node owns an *aggregation
function* (``agg_func``) that receives the risk capitals of its direct children
as keyword arguments keyed by their ``identifier``, and returns the capital of
the node itself.  The usual implementation is :func:`risk_aggregation`, which
applies the standard-formula correlation matrix to the child capitals::

    sqrt(x' @ C @ x)

where ``x`` is the vector of child capitals and ``C`` their correlation matrix.

Nodes are addressed by "/"-separated paths relative to the root of their tree,
where the root itself has the empty path ``""`` and ``"."`` denotes the current
node (node names therefore contain neither ``"/"`` nor ``"."``)::

    tree = RiskTree(root="C-ROSS MC")
    tree.root.add_sub_risk("Market", agg_func=market_agg)
    tree.set_risk_capital("Market/Interest Rate", 1_000.0)

A ``RiskTree`` is only a *view* on a node and never copies it; use ``deepcopy``
to obtain an independent structure.
"""

import math
import numbers
import numpy as np
from typing import Callable, Self


def risk_aggregation(*args, corr_matrix: np.ndarray) -> float:
    """
    Aggregate risk capitals using a correlation matrix (standard-formula style).

    Computes ``sqrt(x' @ corr_matrix @ x)``, where ``x`` collects the capitals
    passed in.  The risks may be given either as separate positional arguments or
    as a single sequence::

        risk_aggregation(market, life, non_life, corr_matrix=CORR_MATRIX_SCR)
        risk_aggregation([market, life, non_life], corr_matrix=CORR_MATRIX_SCR)

    The result equals the simple sum of the capitals only when every correlation
    is 1; otherwise the aggregation is diversified downwards.  Off-diagonal
    entries may be negative -- the Solvency II life matrix correlates mortality
    and longevity at -0.25 -- which lets opposite risks offset each other.  At
    least two risks are required, since the aggregation of a single risk is the
    identity.

    Args:
        *args: The risk capitals to aggregate, either as separate positional
            arguments or as one sequence of numbers.
        corr_matrix: Square correlation matrix of shape ``(n, n)``, where ``n``
            is the number of risks supplied.

    Returns:
        float: The aggregated risk capital.

    Raises:
        ValueError: If fewer than two risks are supplied, if ``corr_matrix`` is
            not of shape ``(n, n)``, or if the quadratic form is negative (the
            matrix is not positive semi-definite, so not a correlation matrix).
    """
    if len(args) == 0:
        raise ValueError(f"No risk is provided.")
    risk_vector = np.atleast_1d(args[0]) if len(args) == 1 else np.array(args)
    n = len(risk_vector)
    if n == 1:
        raise ValueError(f"Risk ({risk_vector[0]}) is a single scalar, expected vector.")
    corr_matrix = np.array(corr_matrix)
    if corr_matrix.shape != (n, n):
        raise ValueError(f'Correlation matrix shape: {corr_matrix.shape}, expected ({n}, {n}).')
    return math.sqrt(risk_vector @ corr_matrix @ risk_vector.T)


class RiskNode:
    """
    A single node of a solvency risk-module hierarchy.

    A node is either a *leaf* -- a risk factor whose capital is provided directly
    through :meth:`set_risk_capital` -- or an *internal* node, whose capital is
    computed from its children by its aggregation function (see
    :meth:`set_agg_func` and :meth:`aggregate`).

    Children are attached with :meth:`add_sub_risk` / :meth:`link_child`.  The
    name of a node must be unique among its siblings and free of ``"/"`` and
    ``"."``, which are reserved by the path syntax.  Every node also carries an
    ``identifier``: a lower-case, keyword-friendly version of its name, used as
    the keyword argument name when the parent aggregates (``"Non-life"`` ->
    ``"non_life"``).  A leading digit is prefixed by ``"_"`` (``"1 Year"`` ->
    ``"_1_year"``), so an identifier is always a valid parameter name.

    The aggregated capital is cached.  Setting the capital of a leaf clears the
    cache of all its ancestors, so :attr:`risk_capital` recomputes whenever a
    value below it has changed.  A node belongs to at most one tree: linking a
    node that already has a parent raises, which also makes cycles impossible.
    """

    __slots__ = ("_name", "_identifier", "_risk_capital", "_parent", "_children", "_agg_func",)

    def __init__(self, name: str, /, *, identifier: str | None = None):
        """
        Args:
            name: Name of the node, unique among its siblings and free of ``"/"``
                and ``"."``, which are reserved by the path syntax.
            identifier: Keyword-friendly alias of the node, used as the keyword
                argument name when its parent aggregates.  It is normalised like a
                name, and defaults to the normalised ``name`` when ``None``.

        Raises:
            TypeError: If ``identifier`` is not a ``str``.
            ValueError: If ``name`` contains ``"/"`` or ``"."``, or if
                ``identifier`` is empty.
        """
        name = str(name)
        for bad in ("/", "."):
            if bad in name:
                raise ValueError(f"Invalid {name=}: contains '{bad}'.")
        self._name: str = name
        self._identifier: str = self._normalize_identifier(self._name if identifier is None else identifier)
        self._risk_capital: float | None = None
        self._parent: RiskNode | None = None
        self._children: list[RiskNode] = []
        self._agg_func: Callable[..., float] | None = None

    @property
    def name(self) -> str:
        """Name of the node, as given at construction."""
        return self._name

    @property
    def identifier(self) -> str:
        """Keyword-friendly alias of the node, unique among its siblings."""
        return self._identifier

    @property
    def risk_capital(self) -> float:
        """
        Risk capital of the node.

        For a leaf this is the value provided through :meth:`set_risk_capital`;
        for an internal node it is the aggregation of its children, computed and
        cached on first access.

        Raises:
            ValueError: If the node is a leaf without a provided value, or an
                internal node whose children cannot be aggregated.
        """
        if self._risk_capital is None:
            self.aggregate()
        if self._risk_capital is None:
            raise ValueError(f"{self._name}: risk capital hasn't been {'provided' if self.is_leaf else 'aggregated'}.")
        return self._risk_capital

    @property
    def root(self) -> Self:
        """Outermost ancestor of the node (the node itself if it has no parent)."""
        node = self
        while (parent:= node._parent) is not None:
            node = parent
        return node

    @property
    def parent(self) -> Self | None:
        """Parent of the node, or ``None`` for a root."""
        return self._parent

    @property
    def children(self) -> list[Self]:
        """Direct children of the node, in the order they were linked."""
        return self._children

    @property
    def siblings(self) -> list[Self]:
        """Other children of the parent, in order; empty for a root."""
        if self._parent is None:
            return []
        return [x for x in self._parent._children if x is not self]

    @property
    def ancestors(self) -> list[Self]:
        """Ancestors of the node, ordered ``[parent, grandparent, .., root]``."""
        nodes = []
        node = self._parent
        while node is not None:
            nodes.append(node)
            node = node._parent
        return nodes  # [parent, grandparent, .., root]

    @property
    def descendants(self) -> list[Self]:
        """All nodes below the node, in preorder (a child before its own children)."""
        nodes = []
        for c in self._children:
            nodes.append(c)
            nodes.extend(c.descendants)
        return nodes

    @property
    def is_root(self) -> bool:
        """``True`` if the node has no parent."""
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        """``True`` if the node has no children."""
        return len(self._children) == 0

    @property
    def depth(self) -> int:
        """Number of ancestors of the node; ``0`` for a root."""
        return len(self.ancestors)

    @property
    def path(self) -> str:
        """
        Path of the node relative to the root of its tree, e.g. ``"Market/Spread"``.

        The root itself has the empty path ``""``.
        """
        names = []
        node = self
        while node is not None:
            names.append(node._name)
            node = node._parent
        names.reverse()
        return "/".join(names[1:])

    def add_sub_risk(self, *args: Self | str, agg_func: Callable[..., float] | None = None) -> None:
        """
        Attach one or more sub-risks below this node.

        The arguments are validated as a whole before anything is attached: if any
        of them is rejected, neither the children nor the aggregation function of
        this node change.

        Args:
            *args: The children to attach, given either as existing
                :class:`RiskNode` objects or as names from which new nodes are
                created.
            agg_func: Optional aggregation function of this node, set through
                :meth:`set_agg_func`.

        Raises:
            TypeError: If an argument is neither a ``RiskNode`` nor a ``str``.
            ValueError: If this node already has an aggregation function, or if a
                child already has a parent, is a root, or duplicates an existing
                child by object, name or identifier.
        """
        staged = []
        for arg in args:
            if isinstance(arg, RiskNode):
                self._stage_child(arg, staged)
            elif isinstance(arg, str):
                self._stage_child(RiskNode(arg), staged)
            else:
                raise TypeError(f"Invalid {type(arg)=}, expected ('RiskNode', 'str').")
        if agg_func is not None:
            self.set_agg_func(agg_func)
        for node in staged:
            self.link_child(node)

    def _stage_child(self, node: Self, /, staged: list[Self]) -> None:
        """
        Validate ``node`` as a new child of this node and append it to ``staged``.

        The conditions are those of :meth:`link_child`, extended to the children
        already staged by the same :meth:`add_sub_risk` call, so that a whole call
        can be validated -- and abandoned -- before anything is linked.

        Args:
            node: The candidate child.
            staged: The children accepted so far in the current call, appended to
                in place.

        Raises:
            ValueError: If ``node`` cannot become a child of this node.
        """
        children = self._children + staged
        if node._parent is not None:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: has parent '{node._parent._name}'.")
        if node is self.root:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: is root.")
        if node in children:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: duplicate object.")
        if node._name in [c._name for c in children]:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: duplicate name.")
        if node._identifier in [c._identifier for c in children]:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: duplicate identifier.")
        staged.append(node)

    def link_child(self, node: Self, /) -> None:
        """
        Attach ``node`` as the last child of this node.

        Args:
            node: The node to attach.  It must not already have a parent and must
                not be the root of this node's tree, which rules out cycles.

        Raises:
            ValueError: If ``node`` already has a parent, is the root, or
                duplicates an existing child by object, name or identifier.
        """
        if node._parent is not None:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: has parent '{node._parent._name}'.")
        if node is self.root:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: is root.")
        if node in self._children:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: duplicate object.")
        if node._name in [c._name for c in self._children]:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: duplicate name.")
        if node._identifier in [c._identifier for c in self._children]:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: duplicate identifier.")
        self._children.append(node)
        node._parent = self
        self.clr_risk_capital()

    def link_parent(self, node: Self, /) -> None:
        """
        Attach this node below ``node``.

        Equivalent to ``node.link_child(self)``; see :meth:`link_child` for the
        conditions and the exceptions raised.
        """
        node.link_child(self)

    def set_agg_func(self, func: Callable[..., float], /) -> None:
        """
        Set the aggregation function of this node.

        The function is called as ``func(**{child.identifier: child.risk_capital})``
        when the node aggregates, so its parameter names must match the
        identifiers of the children.

        Args:
            func: Callable that takes the capitals of the children as keyword
                arguments and returns the risk capital of this node.

        Raises:
            ValueError: If an aggregation function has already been set.
        """
        if self._agg_func is not None:
            raise ValueError(f"{self._name}: aggregation function has already been set.")
        self._agg_func = func
        self.clr_risk_capital()

    def set_risk_capital(self, value: float, /) -> None:
        """
        Provide the risk capital of a leaf node.

        The cached capital of all ancestors is cleared, so that they recompute on
        their next access.

        Args:
            value (float): The risk capital to store.

        Raises:
            ValueError: If this node is not a leaf, or if ``value`` is NaN.
            TypeError: If ``value`` is not a real number (usually expected `float`).
        """
        if not self.is_leaf:
            raise ValueError(f"{self._name}: non-leaf node rejects set value for risk capital.")
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"Invalid type of risk capital '{type(value)}', expected 'float'.")
        if value != value:  # finiteness guard for NaN
            raise ValueError(f"Invalid value of risk capital: {value!r}.")
        self._risk_capital = value
        if self._parent is not None:
            self._parent.clr_risk_capital()  # cascade

    def clr_risk_capital(self) -> None:
        """
        Clear the cached risk capital of this node and of all its ancestors.

        Only the cache is cleared; leaf values stay untouched and the ancestors
        recompute from their children on their next access.
        """
        self._risk_capital = None
        if self._parent is not None:
            self._parent.clr_risk_capital()  # cascade

    def aggregate(self) -> None:
        """
        Compute and cache the risk capital of this internal node.

        Calls the aggregation function of the node with the capitals of its
        direct children as keyword arguments keyed by their identifiers, and
        stores the result.  Does nothing on a leaf node.

        Raises:
            ValueError: If the node is internal but has no aggregation function,
                or if the capital of a child is not available.
        """
        if self.is_leaf:
            return
        if self._agg_func is None:
            raise ValueError(f"{self._name}: aggregation function is None.")
        kwargs = {c._identifier: c.risk_capital for c in self._children}
        self._risk_capital = self._agg_func(**kwargs)

    def goto(self, path: str | None, /) -> Self:
        """
        Resolve a relative path and return the node it denotes.

        Components are separated by ``"/"``: ``""`` and ``"."`` stay on the
        current node, and any other component matches a direct child by name
        (no name being allowed to contain ``"."`` or ``"/"``).  A leading
        ``"/"`` is ignored, so ``"Market/Spread"``, ``"/Market/Spread"``
        and ``"./Market/Spread"`` are equivalent.

        Args:
            path: The path to resolve, or ``None`` for this node itself.

        Returns:
            RiskNode: The node found at ``path``.

        Raises:
            TypeError: If ``path`` is neither ``None`` nor a ``str``.
            ValueError: If a component has no matching child.
        """
        if path is None:
            return self
        if not isinstance(path, str):
            raise TypeError(f"Invalid type of path: '{type(path)}', expected 'str'.")
        node = self
        for p in path.split("/"):
            if p == "." or p == "":
                pass
            else:
                node = next((c for c in node._children if c._name == p), None)
            if node is None:
                raise ValueError(f"{self._name}: can't goto node: '{path}'; failed at '{p}'.")
        return node

    def __truediv__(self, other: str, /) -> Self:
        """Resolve ``other`` as a path relative to this node: ``node / "Market/Spread"``."""
        return self.goto(other)

    @classmethod
    def _normalize_identifier(cls, chars: str, /) -> str:
        """
        Turn a name into an identifier that can be used as a keyword argument name.

        Lower-cases ``chars`` and replaces every character outside ``[a-z0-9_]``
        by ``"_"`` (``"Non-life"`` -> ``"non_life"``, ``"Type 1"`` -> ``"type_1"``).
        A leading digit is prefixed by ``"_"`` (``"1 Year"`` -> ``"_1_year"``), so
        that the result is always a valid parameter name and usable as a keyword
        when the parent aggregates.

        Args:
            chars: The name to normalise.

        Returns:
            str: The normalised identifier.

        Raises:
            TypeError: If ``chars`` is not a ``str``.
            ValueError: If ``chars`` is empty.
        """
        if not isinstance(chars, str):
            raise TypeError(f"Invalid type: {type(chars)}, expected 'str'.")
        if len(chars) == 0:
            raise ValueError(f"Empty chars is not allowed.")
        chars = chars.lower()
        allowed = [chr(c) for c in range(97, 123)] + [chr(c) for c in range(48, 58)] + ["_"]
        chars = "".join([(c if c in allowed else "_") for c in chars])
        if chars[0] in "0123456789":
            chars = "_" + chars
        return chars

    def zeroize(self) -> None:
        """Set the risk capital of every leaf below this node to ``0.0``."""
        for node in preorder_traversal(self):
            if node.is_leaf:
                node.set_risk_capital(0.0)

    def deepcopy(self, *, with_value: bool = True) -> Self:
        """
        Return an independent copy of this node and of all its descendants.

        The copy keeps the names, the identifiers and the aggregation functions of
        the original nodes, and has no parent, so paths inside it are relative to
        the copy. The aggregation functions are shared rather than copied, hence
        any mutable state they carry stays common to both; capitals are numbers and
        therefore independent. Only the leaf values are carried over, the internal
        ones are recomputed from the copied children on first access.

        Args:
            with_value: If ``True`` (default), copy the risk capitals as they are
                currently cached; if ``False``, copy the structure with all
                capitals cleared.

        Returns:
            RiskNode: The copied node, detached from the original tree.
        """
        copied_node = RiskNode(self._name, identifier=self._identifier)
        copied_node._agg_func = self._agg_func
        copied_node._risk_capital = self._risk_capital if with_value else None
        _path_slice_start = len(self.path)
        for desc in self.descendants:
            copied_desc = RiskNode(desc._name, identifier=desc._identifier)
            copied_desc._agg_func = desc._agg_func
            copied_desc._risk_capital = desc._risk_capital if with_value else None
            copied_desc.link_parent(copied_node.goto(desc._parent.path[_path_slice_start:]))
        return copied_node


class RiskTree:
    """
    A view on a solvency risk-module tree, identified by its root node.

    The tree wraps an existing :class:`RiskNode` without copying it, so structure
    and values are shared with the tree that node belongs to.  Paths passed to the
    methods below are resolved from the root of this tree, hence for a subtree of
    a larger tree they are relative to that subtree and not to the outermost root
    (the :attr:`RiskNode.path` of a node, in contrast, is always absolute within
    the whole structure).  Use :meth:`deepcopy` to obtain an independent
    structure.
    """

    __slots__ = ("_root",)

    def __init__(self, root: RiskNode | str):
        """
        Args:
            root: The root node of the tree, or the name of a new root node.

        Raises:
            TypeError: If ``root`` is neither a ``RiskNode`` nor a ``str``.
        """
        if not isinstance(root, (RiskNode, str)):
            raise TypeError(f"Invalid {type(root)=}, expected ('RiskNode', 'str').")
        self._root: RiskNode = root if isinstance(root, RiskNode) else RiskNode(root)

    @property
    def root(self) -> RiskNode:
        """The root node of the tree."""
        return self._root

    @property
    def is_toptree(self) -> bool:
        """``True`` if the root node has no parent, i.e. the tree is not a subtree."""
        return self._root.is_root

    @property
    def is_subtree(self) -> bool:
        """``True`` if the root node still has a parent, i.e. the tree is a subtree."""
        return not self._root.is_root

    def get_risk_capital(self, path: str | None = None, /) -> float:
        """
        Risk capital of one node of the tree.

        Args:
            path: Path of the node relative to the root, or ``None`` (default)
                for the root itself.

        Returns:
            float: The risk capital of the node.

        Raises:
            ValueError: If ``path`` does not exist, or if the capital of the node
                is not available.
        """
        return self._root.goto(path).risk_capital

    def get_risk_diversification(self, path: str | None = None, /) -> float:
        """
        Diversification benefit at one node of the tree.

        Defined as the sum of the capitals of the direct children minus the
        aggregated capital of the node itself, i.e. the capital that the
        aggregation rule saves compared with adding the sub-risks up.

        Args:
            path: Path of the node relative to the root, or ``None`` (default)
                for the root itself.

        Returns:
            float: The diversification benefit, ``0.0`` for a leaf node.

        Raises:
            ValueError: If ``path`` does not exist, or if a capital is not
                available.
        """
        node = self._root.goto(path)
        if node.is_leaf:
            return 0.0
        return sum([c.risk_capital for c in node.children]) - node.risk_capital

    def get_node(self, path: str, /) -> RiskNode:
        """
        Node at ``path``.

        Args:
            path: Path of the node relative to the root, ``""`` for the root.

        Returns:
            RiskNode: The node found at ``path``.

        Raises:
            ValueError: If ``path`` does not exist.
        """
        return self._root.goto(path)

    def get_toptree(self) -> Self:
        """
        Tree on the outermost ancestor of this tree's root.

        Returns:
            RiskTree: A new tree on the same root node; the node itself is not
                copied.
        """
        return RiskTree(self._root.root)

    def get_subtree(self, path: str, /) -> Self:
        """
        Tree on the node at ``path``.

        Args:
            path: Path of the root of the subtree, relative to the root of this
                tree.

        Returns:
            RiskTree: A live view on the same nodes, not a copy; use
                :meth:`deepcopy` if the subtree must be modified on its own.

        Raises:
            ValueError: If ``path`` does not exist.
        """
        return RiskTree(self._root.goto(path))

    def list_nodes(self) -> list[RiskNode]:
        """All nodes of the tree, in preorder (the root first, then each subtree)."""
        return preorder_traversal(self._root)

    def list_leaf_nodes(self) -> list[RiskNode]:
        """All leaf nodes of the tree, in preorder."""
        return [node for node in self.list_nodes() if node.is_leaf]

    def set_risk_capital(self, path: str, /, value: float) -> None:
        """
        Provide the risk capital of the leaf node at ``path``.

        See :meth:`RiskNode.set_risk_capital` for the conditions and the
        exceptions raised.
        """
        self._root.goto(path).set_risk_capital(value)

    def batch_set_risk_capital(self, value_dict: dict[str, float | dict], /) -> None:
        """
        Provide the risk capitals of several leaves at once.

        Args:
            value_dict: Mapping of leaf paths to values, either flat
                (``{"Market/Equity": 1.0}``) or nested
                (``{"Market": {"Equity": 1.0}}``).  Nested dictionaries are
                flattened into "/"-joined paths.

        Raises:
            TypeError: If a key is not a ``str``, or a value is neither a number
                nor a ``dict``.
            ValueError: If a path does not exist, or does not point to a leaf.
        """
        for path, value in self._flatten_dict(value_dict).items():
            self.set_risk_capital(path, value)

    @classmethod
    def _flatten_dict(cls, nested: dict[str, float | dict], /, joiner: str = "/") -> dict[str, float]:
        """
        Flatten a nested ``{name: value}`` mapping into ``{path: value}``.

        Args:
            nested: The mapping to flatten, whose values are numbers or further
                dictionaries.
            joiner: The string used to join the nested keys into a path.

        Returns:
            dict[str, float]: Flat mapping of joined paths to numbers.

        Raises:
            TypeError: If a key is not a ``str``, or a value is neither a number
                nor a ``dict``.
        """
        flattened: dict[str, float] = {}
        for k, v in nested.items():
            if not isinstance(k, str):
                raise TypeError(f"Invalid {type(k)=}, expected 'str'.")
            if isinstance(v, (float, int)):
                flattened[k] = v
            elif isinstance(v, dict):
                for nk, nv in cls._flatten_dict(v).items():
                    flattened[k + joiner + nk] = nv
            else:
                raise TypeError(f"Invalid {type(v)=}, expected ('float', 'dict').")
        return flattened

    def zeroize(self) -> None:
        """Set the risk capital of every leaf of the tree to ``0.0``."""
        self._root.zeroize()

    def deepcopy(self, *, with_value: bool = True) -> Self:
        """
        Return an independent copy of the whole tree.

        The copied tree shares the aggregation functions of the original one; see
        :meth:`RiskNode.deepcopy`.

        Args:
            with_value: If ``True`` (default), copy the risk capitals as they are
                currently cached; if ``False``, copy the structure with all
                capitals cleared.

        Returns:
            RiskTree: The copied tree, detached from the original one.
        """
        return RiskTree(self._root.deepcopy(with_value=with_value))

    def display(self, *, width: int = 80, precision: int = 2) -> None:
        """
        Print the tree, with the capital of every node that has one.

        Nodes without an available capital -- typically leaves without a provided
        value -- are printed without a number.

        Args:
            width: Column at which a printed value ends; a name reaching beyond
                ``width`` is simply followed by one space.
            precision: Number of decimals of the printed values, which are also
                grouped by thousands (``f"{value:,.{precision}f}"``).
        """
        root_depth = self._root.depth
        for node in preorder_traversal(self._root):
            prefixed_name = f"{'    ' * (node.depth - root_depth)}{node.name}"
            try:
                val = node.risk_capital
                val_width = width - len(prefixed_name)
                print(f"{prefixed_name} {val:>{val_width},.{precision}f}")
            except ValueError:
                print(prefixed_name)


def preorder_traversal(node: RiskNode) -> list[RiskNode]:
    """
    Collect a node and all of its descendants, in preorder.

    Args:
        node: The node to start from.

    Returns:
        list[RiskNode]: ``node`` first, then the subtree of each of its children
            in order.
    """
    nodes = [node]
    for c in node.children:
        nodes.extend(preorder_traversal(c))
    return nodes

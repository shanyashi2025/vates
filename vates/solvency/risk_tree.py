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
function* (``agg_func``) that receives risk capitals as keyword arguments keyed by
their ``identifier``, and returns the capital of the node itself.  Which capitals
those are is the *aggregation scope*: by default (``scope="children"``) those of
the direct children, or, with ``scope="descendants"``, those of every node below,
so that a rule can consume both a sub-module's own aggregated capital and the risk
factors nested inside it.  Every internal node below such a node must be
aggregatable itself, its capital being part of what the node above receives.  The
usual implementation is :func:`risk_aggregation`, which applies the
standard-formula correlation matrix to the capitals it is given::

    sqrt(x' @ C @ x)

where ``x`` is the vector of capitals and ``C`` their correlation matrix.

Nodes are addressed by "/"-separated paths relative to the root of their tree,
where the root itself has the empty path ``""`` denotes the current
node (node names therefore contain no ``"/"``)::

    tree = RiskTree(root="Solvency II SCR")
    tree.grow("", children=("Market", "Life"), agg_func=overall_risk_agg)
    tree.grow("Market", children=("Interest Rate", "Equity"), agg_func=market_agg,
              agg_scope="descendants")
    tree.grow("Market/Interest Rate", children=("Interest Rate Up", "Interest Rate Down"),
              agg_func=max_at_zero)
    tree.set_risk_capital("Market/Interest Rate/Interest Rate Up", 1_000.0)

Once its hierarchy is final, a tree can be locked with
:meth:`RiskTree.lock_structure`: structural changes then raise
``AttributeError``, while capitals stay writable so that the same tree can be
reused for every scenario.  Locking is always whole-tree, and the derived views
of each node (``path``, ``descendants``, ``leaves``, ...) are cached while it
lasts.  :meth:`RiskTree.unlock_structure` gives the structure back.

A ``RiskTree`` is only a *view* on a node and never copies it; use ``deepcopy``
to obtain an independent structure.
"""

import functools
import math
import numbers
import numpy as np
from typing import Callable, Literal, Self


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

def on_structure_locked_rejected(func):
    """Refuse a structural mutator with an ``AttributeError`` while the structure is locked."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_structure_locked:
            raise AttributeError(f"{type(self).__name__} structure is locked: '{func.__name__}' is not allowed.")
        result = func(self, *args, **kwargs)
        return result
    return wrapper

def on_structure_locked_cached(key: str):
    """Cache the result of a derived property in the slot ``key`` while the structure is locked."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.is_structure_locked and (result := getattr(self, key, None)) is not None:
                return result
            result = func(self, *args, **kwargs)
            if self.is_structure_locked:
                setattr(self, key, result)
            return result
        return wrapper
    return decorator


class RiskNode:
    """
    A single node of a solvency risk-module hierarchy.

    A node is either a *leaf* -- a risk factor whose capital is provided directly
    through :meth:`set_risk_capital` -- or an *internal* node, whose capital is
    computed by its aggregation function from the capitals below it: those of its
    direct children, or of all its descendants, depending on the aggregation scope
    (see :meth:`set_agg_func` and :meth:`aggregate`).

    Children are attached with :meth:`add_child`, or, from a tree, with
    :meth:`RiskTree.grow`.  The name of a node must be unique among its
    siblings and free of ``"/"``, which are reserved by the path
    syntax.  Every node also carries an ``identifier``: a lower-case,
    keyword-friendly version of its name, used as the keyword argument name when
    the parent aggregates (``"Non-life"`` -> ``"non_life"``).  A leading digit is
    prefixed by ``"_"`` (``"1 Year"`` -> ``"_1_year"``), so an identifier is
    always a valid parameter name.

    The aggregated capital is cached.  Setting the capital of a leaf clears the
    cache of all its ancestors, so :attr:`risk_capital` recomputes whenever a
    value below it has changed.  A node belongs to at most one tree: adding a
    node that already has a parent raises, which also makes cycles impossible.
    """

    __slots__ = ("_name", "_identifier", "_risk_capital", "_parent", "_children", "_agg_func", "_agg_scope",
                 "_is_structure_locked", "_root", "_siblings", "_ancestors", "_descendants", "_leaves", "_path")

    _derived_caches = ("_root", "_siblings", "_ancestors", "_descendants", "_leaves", "_path")  # cached while locked
    _allowed_agg_scopes = ("children", "descendants")  # _allowed_agg_scopes[0] will be used as default

    def __init__(self, name: str, /, *, identifier: str | None = None):
        """
        Args:
            name: Name of the node, unique among its siblings and free of ``"/"``
                and ``"."``, which are reserved by the path syntax.
            identifier: Keyword-friendly alias of the node, used as the keyword
                argument name when its parent aggregates.  It is normalised like a
                name, and defaults to the normalised ``name`` when ``None``.

        Raises:
            TypeError: If ``name`` is not a ``str``, or if ``identifier`` is not a ``str``.
            ValueError: If ``name`` is empty or contains ``"/"`` or ``"."``, or if
                ``identifier`` is empty.
        """
        self._name: str = self._validate_name(name)
        self._identifier: str = self._normalize_identifier(self._name if identifier is None else identifier)
        self._risk_capital: float | None = None
        self._parent: RiskNode | None = None
        self._children: tuple[RiskNode, ...] = ()
        self._agg_func: Callable[..., float] | None = None
        self._agg_scope: Literal["children", "descendants"] | None = None
        self._is_structure_locked: bool = False

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
        for an internal node it is the aggregation of the nodes its aggregation
        scope selects, computed and cached on first access.

        Raises:
            ValueError: If the node is a leaf without a provided value, or an
                internal node whose capitals cannot be aggregated.
        """
        if self._risk_capital is None:
            self.aggregate()
        if self._risk_capital is None:
            raise ValueError(f"{self._name}: risk capital hasn't been {'provided' if self.is_leaf else 'aggregated'}.")
        return self._risk_capital

    @property
    @on_structure_locked_cached("_root")
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
    def children(self) -> tuple[Self, ...]:
        """Direct children of the node, in the order they were linked."""
        return self._children

    @property
    @on_structure_locked_cached("_siblings")
    def siblings(self) -> tuple[Self, ...]:
        """Other children of the parent, in order; empty for a root."""
        if self._parent is None:
            return ()
        return tuple([x for x in self._parent._children if x is not self])

    @property
    @on_structure_locked_cached("_ancestors")
    def ancestors(self) -> tuple[Self, ...]:
        """Ancestors of the node, ordered ``(parent, grandparent, .., root)``."""
        nodes = []
        node = self._parent
        while node is not None:
            nodes.append(node)
            node = node._parent
        return tuple(nodes)

    @property
    @on_structure_locked_cached("_descendants")
    def descendants(self) -> tuple[Self, ...]:
        """All nodes below the node, in preorder (a child before its own children)."""
        nodes = []
        for c in self._children:
            nodes.append(c)
            nodes.extend(c.descendants)
        return tuple(nodes)

    @property
    @on_structure_locked_cached("_leaves")
    def leaves(self) -> tuple[Self, ...]:
        """Leaf nodes of the subtree rooted at this node. This includes the node
        when it has no children, so that the leaves of a leaf are the node itself.
        """
        return tuple([node for node in (self.descendants + (self, )) if node.is_leaf])

    @property
    def is_root(self) -> bool:
        """``True`` if the node has no parent."""
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        """``True`` if the node has no children."""
        return not self._children

    @property
    def depth(self) -> int:
        """Number of ancestors of the node; ``0`` for a root."""
        return len(self.ancestors)

    @property
    @on_structure_locked_cached("_path")
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

    @on_structure_locked_rejected
    def add_child(self, *args: Self | str) -> None:
        """
        Attach one or more sub-risks below this node.

        The arguments are validated as a whole before anything is attached: if any
        of them is rejected, the children of this node doesn't change.

        Args:
            *args: The children to attach, given either as existing
                :class:`RiskNode` objects or as names from which new nodes are
                created.

        Raises:
            TypeError: If an argument is neither a ``RiskNode`` nor a ``str``.
            ValueError: If a child structure is locked, already has a parent, is a root, or
                duplicates an existing child by object, name or identifier.
            AttributeError: If the structure of this node is locked.
        """
        staged = []
        for arg in args:
            if isinstance(arg, RiskNode):
                self._stage_child(arg, staged)
            elif isinstance(arg, str):
                self._stage_child(RiskNode(arg), staged)
            else:
                raise TypeError(f"Invalid {type(arg)=}, expected ('RiskNode', 'str').")
        self._children += tuple(staged)
        for node in staged:
            node._parent = self
            self.clr_risk_capital()

    def _stage_child(self, node: Self, /, staged: list[Self]) -> None:
        """
        Validate ``node`` as a new child of this node and append it to ``staged``.

        The conditions are those of :meth:`add_child`, extended to the children
        already staged by the same call, so that a whole call can be validated --
        and abandoned -- before anything is attached.

        Args:
            node: The candidate child.
            staged: The children accepted so far in the current call, appended to
                in place.

        Raises:
            ValueError: If ``node`` cannot become a child of this node.
        """
        children = self._children + tuple(staged)
        if node.is_structure_locked:
            raise ValueError(f"{self._name}: can't add '{node._name}' as child: structure is locked.")
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

    @on_structure_locked_rejected
    def set_agg_func(self, func: Callable[..., float], /, *,
                     scope: Literal["children", "descendants"] | None = None) -> None:
        """
        Set the aggregation function of this node.

        The function is called as ``func(**{node.identifier: node.risk_capital})``
        when the node aggregates, so its parameter names must match the identifiers
        of the nodes it is given: the direct children by default, or, with
        ``scope="descendants"``, every node below this one.  The latter lets a rule
        use both a sub-module's own aggregated capital and the risk factors nested
        inside it; the capitals it has no parameter for are passed all the same, so
        a rule that needs only some of them ends in ``**kwargs``.

        Args:
            func: Callable that takes the capitals selected by ``scope`` as keyword
                arguments and returns the risk capital of this node.
            scope: The nodes the function is given: ``"children"`` (default) for
                the direct children only, or ``"descendants"`` for every node below
                this one.

        Raises:
            ValueError: If an aggregation function has already been set, or if
                ``scope`` is neither ``"children"`` nor ``"descendants"``.
            AttributeError: If the structure of this node is locked.
        """
        if self._agg_func is not None:
            raise ValueError(f"{self._name}: aggregation function has already been set.")
        if scope is not None and scope not in self._allowed_agg_scopes:
            raise ValueError(f"Invalid agg scope: {scope!r}, expected {self._allowed_agg_scopes}.")
        self._agg_func = func
        self._agg_scope = scope or self._allowed_agg_scopes[0]
        self.clr_risk_capital()

    @on_structure_locked_rejected
    def grow(self, *, children: tuple[str | Self, ...] | str | Self = tuple(),
             agg_func: Callable[..., float] | None = None, agg_scope: Literal["children", "descendants"] | None = None
             ) -> None:
        """
        Set up the node: attach its children and set its aggregation.

        A convenience for building a hierarchy level by level: the children are
        attached as by :meth:`add_child`, and ``agg_func`` is set as by
        :meth:`set_agg_func`.  Capitals are not touched.

        The call is all-or-nothing: when it raises, neither the children nor the
        aggregation function of the node change.  A node is given an aggregation
        function only once, so setting up the same node twice with ``agg_func``
        raises rather than leaving the second set of children behind.

        Args:
            children: The children to attach, as names or as :class:`RiskNode`
                objects, unpacked into :meth:`add_child`.  A single child may be
                passed on its own -- a bare ``str`` is one child, not one child
                per character -- and anything else is read as a sequence of children.
            agg_func: Aggregation function of the node, or ``None`` (default) to
                leave it unset and provide it later through :meth:`set_agg_func`.
            agg_scope: The aggregation scope of ``agg_func``: ``"children"``
                (default) for its direct children, or ``"descendants"`` for every
                node below it; see :meth:`set_agg_func`.  It is ignored
                when ``agg_func`` is ``None``, there being no aggregation function
                to apply it to.

        Raises:
            ValueError: If the node already has an aggregation function, or if a
                child is rejected -- see :meth:`add_child` for the conditions.
            TypeError: If ``children`` is neither a child nor a sequence of
                children, or if a child is neither a :class:`RiskNode` nor a
                ``str``.
            AttributeError: If the structure of this node is locked.
        """
        if agg_func is not None:
            # Pre-flight check: `set_agg_func` below would otherwise refuse the node
            # only after `add_child` has already attached the children.
            if self._agg_func is not None:
                raise ValueError(f"{self._name}: aggregation function has already been set.")
            if agg_scope is not None and agg_scope not in self._allowed_agg_scopes:
                raise ValueError(f"Invalid agg scope: {agg_scope!r}, expected {self._allowed_agg_scopes}.")
        if isinstance(children, (str, RiskNode)):
            children = (children, )
        else:
            children = tuple(children)
        self.add_child(*children)
        if agg_func is not None:
            self.set_agg_func(agg_func, scope=agg_scope)

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

        Calls the aggregation function of the node with the risk capitals selected
        by its aggregation scope -- its direct children, or all its descendants --
        as keyword arguments keyed by their identifiers, and stores the result.  A
        node aggregating from its descendants needs the identifiers below it to be
        unique, since they become the names of those keyword arguments.  Does
        nothing on a leaf node.

        Raises:
            ValueError: If the node is internal but has no aggregation function, if
                two nodes below a node that aggregates from its descendants share an
                identifier, or if a capital it needs is not available.
            NotImplementedError: If its aggregation scope is neither ``"children"``
                nor ``"descendants"``.  :meth:`set_agg_func` only accepts those, so
                this guards against the scope being written directly.
        """
        if self.is_leaf:
            return
        if self._agg_func is None:
            raise ValueError(f"{self._name}: aggregation function is None.")
        if self._agg_scope == "children":
            capitals = {x._identifier: x.risk_capital for x in self._children}
        elif self._agg_scope == "descendants":
            descendants = self.descendants
            identifiers = [x._identifier for x in descendants]
            duplicates = sorted({i for i in identifiers if identifiers.count(i) > 1})
            if duplicates:
                raise ValueError(f"{self._name}: duplicate identifiers among descendants: {duplicates}.")
            capitals = {x._identifier: x.risk_capital for x in descendants}
        else:
            raise NotImplementedError(f"agg scope: {self._agg_scope}.")
        self._risk_capital = self._agg_func(**capitals)

    def get_child(self, name: str, /) -> Self:
        """
        Get the child node by a given name.

        Args:
            name: The ``name`` to lookup.

        Returns:
            RiskNode: The child node named ``name``.

        Raises:
            ValueError: If current node has no matching child.
        """
        node = next((c for c in self._children if c._name == name), None)
        if node is None:
            raise KeyError(f"{self._name}: has no child node named '{name}'.")
        return node

    def get_descendant(self, key: str, /) -> Self:
        """
        Resolve a relative path and return the descendant node it denotes.

        Components are separated by ``"/"``: ``""`` stays on the
        current node, and any other component matches a direct child by name
        (no name being allowed to contain ```"/"``).  A leading
        ``"/"`` is ignored, so ``"Market/Spread"``, ``"/Market/Spread"``
        and ``"Market/Spread"`` are equivalent.

        Args:
            key: The relative path to resolve, or ``""`` for this node itself.

        Returns:
            RiskNode: The node found at ``key``.

        Raises:
            TypeError: If ``key`` is not a ``str``.
            KeyError: If a component has no matching child.
        """
        if not isinstance(key, str):
            raise TypeError(f"Invalid type of key: '{type(key)}', expected 'str'.")
        node = self
        for k in key.split("/"):
            node = node.get_child(k) if k else node
        return node

    def __truediv__(self, key: str, /) -> Self:
        """Resolve ``key`` as a path relative to this node: ``node / "Market/Spread"``."""
        return self.get_descendant(key)

    @classmethod
    def _validate_name(cls, name: str, /) -> str:
        """
        Args:
            name: Name of the node, unique among its siblings and free of ``"/"``
                and ``"."``, which are reserved by the path syntax.

        Raises:
            TypeError: If ``name`` is not a ``str``.
            ValueError: If ``name`` is empty or contains ``"/"`` or ``"."``.

        Returns:
            str: The validated name.
        """
        if not isinstance(name, str):
            raise TypeError(f"Invalid type of name: {type(name)}, expected 'str'.")
        if name == "" or all(c == " " for c in name):
            raise ValueError(f"Name cannot be empty: {name}.")
        for bad in ("/", "."):
            if bad in name:
                raise ValueError(f"Invalid {name=}: contains '{bad}'.")
        return name

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
        """Set the risk capital of every leaf of this node's subtree -- the node
        itself when it has no children -- to ``0.0``.
        """
        for node in self.leaves:
            node.set_risk_capital(0.0)

    def deepcopy(self, *, with_value: bool = True, lock_structure: bool | None = None) -> Self:
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
            lock_structure: Whether the copy is locked: ``None`` (default) copies
                the state of the original, ``True`` or ``False`` forces it.

        Returns:
            RiskNode: The copied node, detached from the original tree.
        """
        def _copy_stuffs(*, original: RiskNode, copied: RiskNode) -> None:
            copied._agg_func = original._agg_func
            copied._agg_scope = original._agg_scope
            copied._risk_capital = original._risk_capital if with_value else None
        copied_root = RiskNode(self._name, identifier=self._identifier)
        _copy_stuffs(original=self, copied=copied_root)
        offset = len(self.path)
        for desc in self.descendants:
            copied_desc = RiskNode(desc._name, identifier=desc._identifier)
            _copy_stuffs(original=desc, copied=copied_desc)
            copied_root.get_descendant(desc._parent.path[offset:]).add_child(copied_desc)
        lock_structure = self._is_structure_locked if lock_structure is None else lock_structure
        if lock_structure:
            copied_root.lock_structure()
        return copied_root

    @property
    def is_structure_locked(self) -> bool:
        """``True`` if the structure of the tree this node belongs to is locked."""
        return self._is_structure_locked

    def lock_structure(self) -> None:
        """
        Lock the structure of the whole tree this node belongs to.

        Adding children and setting aggregation functions then raise
        ``AttributeError``, while risk capitals stay writable; the derived views
        become cached.  Called on a node rather than on a root, it locks the whole
        tree and not the subtree.
        """
        if not self.is_root:
            self.root.lock_structure()
            return
        self._lock_structure()
        for node in self.descendants:
            node._lock_structure()

    def _reset_derived_caches(self) -> None:
        """Forget every derived view cached by this node."""
        for attr in self._derived_caches:
            setattr(self, attr, None)  # not computed yet / no longer valid

    def _lock_structure(self) -> None:
        """Lock this node only, as part of :meth:`lock_structure`'s walk."""
        if self._is_structure_locked:
            return  # already locked: the structure cannot have changed since
        self._reset_derived_caches()
        self._is_structure_locked = True

    def unlock_structure(self) -> None:
        """
        Unlock the structure of the whole tree this node belongs to.

        The mirror of :meth:`lock_structure`: the derived caches are dropped and
        the hierarchy can be grown again.
        """
        if not self.is_root:
            self.root.unlock_structure()
            return
        self._unlock_structure()
        for node in self.descendants:
            node._unlock_structure()

    def _unlock_structure(self) -> None:
        """Unlock this node only, as part of :meth:`unlock_structure`'s walk."""
        if not self._is_structure_locked:
            return  # already unlocked: the caches are unused until the next lock resets them
        self._reset_derived_caches()
        self._is_structure_locked = False

class RiskTree:
    """
    A view on a solvency risk-module tree, identified by its root node.

    The tree wraps an existing :class:`RiskNode` without copying it, so structure
    and values are shared with the tree that node belongs to.  A hierarchy level is
    built with :meth:`grow`, and capitals are provided with
    :meth:`set_risk_capital` or :meth:`batch_set_risk_capital`.  Paths passed to the
    methods below are resolved from the root of this tree, hence for a subtree of
    a larger tree they are relative to that subtree and not to the outermost root
    (the :attr:`RiskNode.path` of a node, in contrast, is always absolute within
    the whole structure).  Use :meth:`deepcopy` to obtain an independent
    structure.  Once built, :meth:`lock_structure` freezes the hierarchy, so that
    only capitals can change.
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

    def get_risk_capital(self, key: str = "", /) -> float:
        """
        Risk capital of one node of the tree.

        Args:
            key: Path of the node relative to the root, or ``""`` (default)
                for the root itself.

        Returns:
            float: The risk capital of the node.

        Raises:
            ValueError: If ``key`` does not exist, or if the capital of the node
                is not available.
        """
        return self.get_node(key).risk_capital

    def get_risk_diversification(self, key: str = "", /) -> float:
        """
        Diversification benefit at one node of the tree.

        Defined as the sum of the capitals of the direct children minus the
        aggregated capital of the node itself, i.e. the capital that the
        aggregation rule saves compared with adding the sub-risks up.

        This is the saving of an aggregation over the direct children, which is the
        default scope: a node that aggregates from its descendants is not formed
        from its direct children, so there the value is informational only.  A
        direct child that is an internal node without an aggregation function has
        no capital to add up, and reading it raises.

        Args:
            key: Path of the node relative to the root, or ``""`` (default)
                for the root itself.

        Returns:
            float: The diversification benefit, ``0.0`` for a leaf node.

        Raises:
            ValueError: If ``key`` does not exist, or if a capital is not
                available.
        """
        node = self.get_node(key)
        if node.is_leaf:
            return 0.0
        return sum([c.risk_capital for c in node.children]) - node.risk_capital

    def get_node(self, key: str = "", /) -> RiskNode:
        """
        Node at ``key``.

        Args:
            key: Path of the node relative to the root, ``""`` for the root.

        Returns:
            RiskNode: The node found at ``key``.

        Raises:
            KeyError: If ``key`` does not exist.
        """
        return self._root.get_descendant(key)

    @on_structure_locked_rejected
    def grow(self, key: str, /, *, children: tuple[str | RiskNode, ...] | str | RiskNode = tuple(),
             agg_func: Callable[..., float] | None = None, agg_scope: Literal["children", "descendants"] | None = None
             ) -> None:
        """
        Set up the node at ``key`` (``""`` denotes the root): attach its children and set its aggregation.

        See :meth:`RiskNode.grow` for the conditions and the exceptions raised.
        """
        self.get_node(key).grow(children=children, agg_func=agg_func, agg_scope=agg_scope)

    def get_toptree(self) -> Self:
        """
        Tree on the outermost ancestor of this tree's root.

        Returns:
            RiskTree: A new tree on the same root node; the node itself is not
                copied.
        """
        return RiskTree(self._root.root)

    def get_subtree(self, key: str, /) -> Self:
        """
        Tree on the node at ``key``.

        Args:
            key: Path of the root of the subtree, relative to the root of this
                tree.

        Returns:
            RiskTree: A live view on the same nodes, not a copy; use
                :meth:`deepcopy` if the subtree must be modified on its own.

        Raises:
            ValueError: If ``key`` does not exist.
        """
        return RiskTree(self.get_node(key))

    def get_all_nodes(self) -> tuple[RiskNode, ...]:
        """All nodes of the tree, in preorder (the root first, then each subtree)."""
        return (self._root, ) + self._root.descendants

    def get_leaf_nodes(self) -> tuple[RiskNode, ...]:
        """All leaf nodes of the tree, in preorder."""
        return self._root.leaves

    def set_risk_capital(self, key: str, /, value: float) -> None:
        """
        Provide the risk capital of the leaf node at ``key``.

        See :meth:`RiskNode.set_risk_capital` for the conditions and the
        exceptions raised.
        """
        self.get_node(key).set_risk_capital(value)

    def batch_set_risk_capital(self, leaf_values: dict[str, float | dict], /) -> None:
        """
        Provide the risk capitals of several leaves at once.

        Args:
            leaf_values: Mapping of leaf paths to values, either flat
                (``{"Market/Equity": 1.0}``) or nested
                (``{"Market": {"Equity": 1.0}}``).  Nested dictionaries are
                flattened into "/"-joined paths.

        Raises:
            TypeError: If a key is not a ``str``, or a value is neither a number
                nor a ``dict``.
            ValueError: If a key does not exist, or does not point to a leaf.
        """
        for key, value in self._flatten_dict(leaf_values).items():
            self.set_risk_capital(key, value)

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

    def deepcopy(self, *, with_value: bool = True, lock_structure: bool | None = None) -> Self:
        """
        Return an independent copy of the whole tree.

        The copied tree shares the aggregation functions of the original one; see
        :meth:`RiskNode.deepcopy`.

        Args:
            with_value: If ``True`` (default), copy the risk capitals as they are
                currently cached; if ``False``, copy the structure with all
                capitals cleared.
            lock_structure: Whether the copied tree is locked: ``None`` (default)
                copies the state of the original, ``True`` or ``False`` forces it.

        Returns:
            RiskTree: The copied tree, detached from the original one.
        """
        copied_tree = RiskTree(self._root.deepcopy(with_value=with_value, lock_structure=lock_structure))
        return copied_tree

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
        for node in self.get_all_nodes():
            prefixed_name = f"{'    ' * (node.depth - root_depth)}{node.name}"
            try:
                val = node.risk_capital
                val_width = width - len(prefixed_name)
                print(f"{prefixed_name} {val:>{val_width},.{precision}f}")
            except ValueError:
                print(prefixed_name)

    @property
    def is_structure_locked(self) -> bool:
        """``True`` if the structure of this tree is locked."""
        return self._root.is_structure_locked

    def lock_structure(self) -> None:
        """
        Lock the structure of the whole tree this view is on.

        See :meth:`RiskNode.lock_structure`; a view of a subtree locks the
        outermost tree, not the subtree alone.
        """
        self._root.lock_structure()

    def unlock_structure(self) -> None:
        """Unlock the structure; the mirror of :meth:`lock_structure`."""
        self._root.unlock_structure()


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

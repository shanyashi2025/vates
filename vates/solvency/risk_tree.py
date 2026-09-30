import csv
import math
import numpy as np
import warnings
from pathlib import Path
from typing import Callable, Self


def risk_aggregation(*args, corr_matrix: np.ndarray) -> float:
    if len(args) == 0:
        raise ValueError(f"No risk is provided.")
    risk_vector = np.array(args[0]) if len(args) == 1 else np.array(args)
    corr_matrix = np.array(corr_matrix)
    n = len(risk_vector)
    if corr_matrix.shape != (n, n):
        raise ValueError(f'corr matrix shape: {corr_matrix.shape}, expected ({n}, {n}).')
    return math.sqrt(risk_vector @ corr_matrix @ risk_vector.T)


class RiskNode:

    __slots__ = ("_name", "_identifier", "_risk_capital", "_parent", "_children", "_agg_func",)

    def __init__(self, name: str, /, *, identifier: str | None = None):
        self._name: str = name
        self._identifier: str = self._normalize_identifier(identifier or name)
        self._risk_capital: float | None = None
        self._parent: RiskNode | None = None
        self._children: list[RiskNode] = []
        self._agg_func: Callable[[float, ...], float] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def risk_capital(self) -> float:
        if self._risk_capital is None:
            self.aggregate()
        if self._risk_capital is None:
            raise ValueError(f"{self._name}: risk capital hasn't been {'provided' if self.is_leaf else 'aggregated'}.")
        return self._risk_capital

    @property
    def risk_diversification(self) -> float:
        if self.is_leaf:
            return 0.0
        return sum([c.risk_capital for c in self._children]) - self.risk_capital

    @property
    def root(self) -> Self:
        node = self
        while (parent:= node._parent) is not None:
            node = parent
        return node

    @property
    def parent(self) -> Self | None:
        return self._parent

    @property
    def children(self) -> list[Self]:
        return self._children

    @property
    def siblings(self) -> list[Self]:
        if self._parent is None:
            return []
        return [x for x in self._parent._children if x is not self]

    @property
    def descendants(self) -> list[Self]:
        return self.preorder_traversal()[1:]  # slicing [1:] can handle when len(ls) == 1

    @property
    def leaves(self) -> list[Self]:
        nodes = []
        for c in self._children:
            if c.is_leaf:
                nodes.append(c)
            else:
                nodes.extend(c.leaves)
        return nodes

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def ancestors(self) -> tuple[str, ...]:
        names = []
        node = self
        while node is not None:
            names.append(node._name)
            node = node._parent
        names.reverse()
        return tuple(names)  # include root

    @property
    def depth(self) -> int:
        return len(self.ancestors) - 1

    @property
    def path(self) -> str:
        return "/".join(self.ancestors[1:])

    def attach_sub_risk(self, *args: Self) -> None:
        if len(args) == 0:
            raise ValueError(f"Nothing to attach.")
        for node in args:
            if node._parent is not None:
                raise ValueError(f"{self._name}: can't attach '{node._name}' as sub-risk: sub of '{node._parent._name}'.")
            if node is self.root:
                raise ValueError(f"{self._name}: can't attach '{node._name}' as sub-risk: root.")
            if node in self._children:
                raise ValueError(f"{self._name}: can't attach '{node._name}' as sub-risk: duplicate.")
            if node._name in [c._name for c in self._children]:
                raise ValueError(f"{self._name}: can't attach '{node._name}' as sub-risk: duplicate name.")
            if node._identifier in [c._identifier for c in self._children]:
                raise ValueError(f"{self._name}: can't attach '{node._name}' as sub-risk: duplicate identifier.")
            self._children.append(node)
            node._parent = self

    def set_agg_func(self, func) -> None:
        if self._agg_func is not None:
            warnings.warn(f"{self._name}: aggregation function will be reset.")
        self._agg_func = func

    def set_risk_capital(self, value: float | None = None, /) -> None:
        if self.is_leaf:
            if not isinstance(value, float):
                raise TypeError(f"Invalid type of risk capital '{type(value)}', expected 'float'.")
            self._risk_capital = value
        else:
            if value is not None:
                warnings.warn(f"{self._name}: non-leaf node rejects set value for risk capital; reset to 'None' only.")
            self._risk_capital = None  # lazy evaluation: reset only, will calculate when property `risk_capital` is called
        if self._parent is not None:
            self._parent.set_risk_capital()  # cascade

    def aggregate(self) -> None:
        if self.is_leaf:
            return
        if self._agg_func is None:
            raise ValueError(f"{self._name}: aggregation function is None; use `set_agg_func(..)` to set.")
        kwargs = {c._identifier: c.risk_capital for c in self._children}
        self._risk_capital = self._agg_func(**kwargs)

    def select(self, path: str, /) -> Self:
        if path is None:
            return self
        if not isinstance(path, str):
            raise TypeError(f"Invalid type of path: '{type(path)}', expected 'str'.")
        node = self
        for p in path.split("/"):
            if p == "." or p == "":
                pass
            elif p == "..":
                node = node._parent
            else:
                node = next((c for c in node._children if c._name == p), None)
            if node is None:
                raise ValueError(f"{self._name}: can't select: '{path}'; failed at '{p}'.")
        return node

    def __truediv__(self, other: str, /) -> Self:
        return self.select(other)

    def __str__(self) -> str:
        title = "/".join(self.ancestors)
        try:
            return f"RiskNode '{title}' (risk_capital={self.risk_capital:,.2f})"
        except ValueError:
            return f"RiskNode '{title}'"

    @classmethod
    def _normalize_identifier(cls, /, chars: str) -> str:
        chars = chars.lower().replace(" ", "_").replace("-", "_")
        allowed = [chr(c) for c in range(97, 123)] + [chr(c) for c in range(48, 58)] + ["_"]
        return "".join([(c if c in allowed else "_") for c in chars])

    def zeroize(self) -> None:
        for node in self.leaves:
            node.set_risk_capital(0.0)

    def preorder_traversal(self) -> list[Self]:
        nodes = [self]
        for c in self._children:
            nodes.extend(c.preorder_traversal())
        return nodes

    def deepcopy(self, *, with_value: bool = True) -> Self:
        copied_node = RiskNode(self._name, identifier=self._identifier)
        copied_node.set_agg_func(self._agg_func)
        _path_slice_start = len(self.path)
        for desc in self.descendants:
            # copy descendants
            copied_desc = RiskNode(desc._name, identifier=desc._identifier)
            copied_desc.set_agg_func(desc._agg_func)
            copied_node.select(desc._parent.path[_path_slice_start:]).attach_sub_risk(copied_desc)
        if with_value:
            for node in copied_node.leaves:
                node.set_risk_capital(self.select(node.path)._risk_capital)
        return copied_node


class RiskTree:

    __slots__ = ("_root", "name",)

    def __init__(self, /, root: RiskNode | None = None, name: str | None = None):
        if root is None and name is None:
            raise ValueError(f"Must provide at least one of ('root', 'name').")
        self._root: RiskNode = root or RiskNode(name)
        self.name: str = name or self._root.name

    @property
    def root(self) -> RiskNode:
        return self._root

    @property
    def is_subtree(self) -> bool:
        return not self._root.is_root

    @property
    def n_nodes(self) -> int:
        return len(self.preorder_traversal())

    @property
    def n_leaf_nodes(self) -> int:
        return len(self.list_leaf_nodes())

    def get_risk_capital(self, path: str | None = None, /) -> float:
        return self._root.select(path).risk_capital

    def set_risk_capital(self, path: str, /, value: float) -> None:
        self._root.select(path).set_risk_capital(value)

    def get_risk_diversification(self, path: str | None = None, /) -> float:
        return self._root.select(path).risk_diversification

    def select_node(self, path: str, /) -> RiskNode:
        return self._root.select(path)

    def select_subtree(self, path: str, /) -> Self:
        return RiskTree(self._root.select(path))

    def list_nodes(self) -> list[RiskNode]:
        return self.preorder_traversal()

    def list_leaf_nodes(self) -> list[RiskNode]:
        return self._root.leaves

    def zeroize(self) -> None:
        self._root.zeroize()

    def preorder_traversal(self) -> list[RiskNode]:
        return self._root.preorder_traversal()

    def deepcopy(self, *, with_value: bool = True) -> Self:
        return RiskTree(root=self._root.deepcopy(with_value=with_value), name=self.name)

    def display(self, *, print_to_file: str | None = None, padding: str = "", width: int = 80, dp: int = 2) -> None:
        root_depth = self._root.depth
        rows = []

        for node in self._root.preorder_traversal():
            prefixed_name = f"{'    ' * (node.depth - root_depth)}{node.name}"
            try:
                val = node.risk_capital
                val_width = width - len(prefixed_name)
                print(f"{prefixed_name} {val:{padding}>{val_width},.{dp}f}")
                rows.append((prefixed_name, f"{val:.{dp}f}"))
            except ValueError:
                print(prefixed_name)
                rows.append((prefixed_name, ))

        if print_to_file is not None:
            file_path = Path(print_to_file)
            mode = 'a' if file_path.is_file() else 'w'
            with open(file_path, mode=mode, newline='', encoding='utf-8-sig') as csvfile:
                csv.writer(csvfile).writerows(rows)

    def __str__(self) -> str:
        try:
            return f"RiskTree '{self.name}' (risk_capital={self._root.risk_capital:,.2f})"
        except ValueError:
            return f"RiskTree '{self.name}'"

import math
import numpy as np
import warnings
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
    def _parts(self) -> tuple[str, ...]:
        names = []
        node = self
        while node is not None:
            names.append(node._name)
            node = node._parent
        names.reverse()
        return tuple(names)  # root, .., self

    @property
    def ancestors(self) -> tuple[str, ...]:
        return self._parts[:-1]

    @property
    def descendants(self) -> list[Self]:
        nodes = []
        for c in self._children:
            nodes.append(c)
            nodes.extend(c.descendants)
        return nodes

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def depth(self) -> int:
        return len(self._parts) - 1

    @property
    def path(self) -> str:
        return "/".join(self._parts[1:])

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
        if not self.is_leaf:
            raise ValueError(f"{self._name}: non-leaf node rejects set value for risk capital.")
        if not isinstance(value, (float, int)):
            raise TypeError(f"Invalid type of risk capital '{type(value)}', expected 'float'.")
        self._risk_capital = value
        if self._parent is not None:
            self._parent.clr_risk_capital()  # cascade

    def clr_risk_capital(self) -> None:
        self._risk_capital = None
        if self._parent is not None:
            self._parent.clr_risk_capital()  # cascade

    def aggregate(self) -> None:
        if self.is_leaf:
            return
        if self._agg_func is None:
            raise ValueError(f"{self._name}: aggregation function is None; use `set_agg_func(..)` to set.")
        kwargs = {c._identifier: c.risk_capital for c in self._children}
        self._risk_capital = self._agg_func(**kwargs)

    def get_node(self, path: str, /) -> Self:
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
                raise ValueError(f"{self._name}: can't get node: '{path}'; failed at '{p}'.")
        return node

    def __truediv__(self, other: str, /) -> Self:
        return self.get_node(other)

    def __str__(self) -> str:
        title = "/".join(self._parts)
        try:
            return f"RiskNode '{title}' (risk_capital={self.risk_capital:,.2f})"
        except ValueError:
            return f"RiskNode '{title}'"

    @classmethod
    def _normalize_identifier(cls, /, chars: str) -> str:
        chars = chars.lower()
        allowed = [chr(c) for c in range(97, 123)] + [chr(c) for c in range(48, 58)] + ["_"]
        return "".join([(c if c in allowed else "_") for c in chars])

    def zeroize(self) -> None:
        for node in self.descendants:
            if node.is_leaf:
                node.set_risk_capital(0.0)

    def deepcopy(self, *, with_value: bool = True) -> Self:
        copied_node = RiskNode(self._name, identifier=self._identifier)
        copied_node.set_agg_func(self._agg_func)
        _path_slice_start = len(self.path)
        for desc in self.descendants:
            copied_desc = RiskNode(desc._name, identifier=desc._identifier)
            copied_desc.set_agg_func(desc._agg_func)
            copied_node.get_node(desc._parent.path[_path_slice_start:]).attach_sub_risk(copied_desc)
        if with_value:
            for node in copied_node.descendants:
                if node.is_leaf:
                    node.set_risk_capital(self.get_node(node.path)._risk_capital)
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
    def is_toptree(self) -> bool:
        return self._root.is_root

    @property
    def is_subtree(self) -> bool:
        return not self._root.is_root

    @property
    def n_nodes(self) -> int:
        return len(self.list_nodes())

    @property
    def n_leaf_nodes(self) -> int:
        return len(self.list_leaf_nodes())

    def get_risk_capital(self, path: str | None = None, /) -> float:
        return self._root.get_node(path).risk_capital

    def get_risk_diversification(self, path: str | None = None, /) -> float:
        return self._root.get_node(path).risk_diversification

    def get_node(self, path: str, /) -> RiskNode:
        return self._root.get_node(path)

    def get_toptree(self, *, name: str | None = None) -> Self:
        return RiskTree(self._root.root, name)

    def get_subtree(self, path: str, /, *, name: str | None = None) -> Self:
        return RiskTree(self._root.get_node(path), name)

    def list_nodes(self) -> list[RiskNode]:
        return self._preorder(self._root)

    def list_leaf_nodes(self) -> list[RiskNode]:
        return [node for node in self.list_nodes() if node.is_leaf]

    def set_risk_capital(self, path: str, /, value: float) -> None:
        self._root.get_node(path).set_risk_capital(value)

    def batch_set_risk_capital(self, value_dict: dict[str, float | dict], /) -> None:
        for path, value in self._flatten_dict(value_dict).items():
            self.set_risk_capital(path, value)

    @classmethod
    def _flatten_dict(cls, nested: dict[str, float | dict], /, joiner: str = "/") -> dict[str, float]:
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
                raise TypeError(f"Invalid {type(v)=}, expected 'float' or 'dict'.")
        return flattened

    def zeroize(self) -> None:
        self._root.zeroize()

    @classmethod
    def _preorder(cls, node: RiskNode) -> list[RiskNode]:
        nodes = [node]
        for c in node.children:
            nodes.extend(cls._preorder(c))
        return nodes

    def preorder_traversal(self) -> list[RiskNode]:
        return self._preorder(self._root)

    def deepcopy(self, *, with_value: bool = True) -> Self:
        return RiskTree(root=self._root.deepcopy(with_value=with_value), name=self.name)

    def display(self, *, width: int = 80, precision: int = 2) -> None:
        print(f"<RiskTree> '{self.name}':")
        root_depth = self._root.depth
        for node in self._preorder(self._root):
            prefixed_name = f"{'    ' * (node.depth - root_depth)}{node.name}"
            try:
                val = node.risk_capital
                val_width = width - len(prefixed_name)
                print(f"{prefixed_name} {val:>{val_width},.{precision}f}")
            except ValueError:
                print(prefixed_name)

    def __str__(self) -> str:
        try:
            return f"<RiskTree> '{self.name}' (risk_capital={self._root.risk_capital:,.2f})"
        except ValueError:
            return f"<RiskTree> '{self.name}'"

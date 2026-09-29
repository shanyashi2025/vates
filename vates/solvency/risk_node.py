import csv
import math
import numpy as np
import warnings
from datetime import datetime
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

    __slots__ = ("_name", "_slug", "_risk_capital", "_parent", "_children", "_agg_func",)

    def __init__(self, name: str, /, *, slug: str | None = None):
        self._name: str = name
        self._slug: str = self._normalize_identifier(slug or name)
        self._risk_capital: float | None = None
        self._parent: RiskNode | None = None
        self._children: list[RiskNode] = []
        self._agg_func: Callable[[float, ...], float] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def slug(self) -> str:
        return self._slug

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
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def _parts(self) -> tuple[str, ...]:
        names = []
        node = self
        while node is not None:
            names.append(node._name)
            node = node._parent
        names.reverse()
        return tuple(names)  # include root

    @property
    def depth(self) -> int:
        return len(self._parts) - 1

    @property
    def path_from_root(self) -> str:
        return "/".join(self._parts[1:])

    def select(self, path: str, /) -> Self:
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

    def list_leaves(self) -> list[Self]:
        leaves = []
        for c in self._children:
            if c.is_leaf:
                leaves.append(c)
            else:
                leaves.extend(c.list_leaves())
        return leaves

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
            if node._slug in [c._slug for c in self._children]:
                raise ValueError(f"{self._name}: can't attach '{node._name}' as sub-risk: duplicate slug.")
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
                raise ValueError(f"Not allowed to set risk capital for a non-leaf node.")
            self._risk_capital = None  # lazy evaluation: reset only, will calculate when property `risk_capital` is called
        if self._parent is not None:
            self._parent.set_risk_capital()  # cascade

    def aggregate(self) -> float:
        if self.is_leaf:
            return self._risk_capital
        if self._agg_func is None:
            raise ValueError(f"{self._name}: aggregation function is None; use `set_agg_func(..)` to set.")
        kwargs = {c._slug: c.risk_capital for c in self._children}
        self._risk_capital = self._agg_func(**kwargs)
        return self._risk_capital

    @classmethod
    def _normalize_identifier(cls, /, chars: str) -> str:
        chars = chars.lower().replace(" ", "_").replace("-", "_")
        allowed = [chr(c) for c in range(97, 123)] + [chr(c) for c in range(48, 58)] + ["_"]
        return "".join([(c if c in allowed else "_") for c in chars])

    def preorder_traversal(self) -> list[Self]:
        nodes = [self]
        for c in self._children:
            nodes.extend(c.preorder_traversal())
        return nodes

    def copy_tree_structure(self, *, new_name: str = None, new_slug: str = None) -> Self:
        root = self.root
        copied_root = RiskNode(new_name or root._name, slug=new_slug)
        copied_root.set_agg_func(root._agg_func)
        if not root.is_leaf:
            for node in root.preorder_traversal()[1:]:  # the first one is root
                copied_node = RiskNode(node._name, slug=node._slug)
                copied_node.set_agg_func(node._agg_func)
                copied_root.select(node._parent.path_from_root).attach_sub_risk(copied_node)
        return copied_root

    def get_risk_capital(self, path: str | None = None, /) -> float:
        if path is None:
            return self.risk_capital
        return self.select(path).risk_capital

    def __truediv__(self, other: str, /) -> Self:
        return self.select(other)

    def __str__(self) -> str:
        title = "/".join(self._parts)
        try:
            return f"{title}: {self.risk_capital:,.2f}"
        except ValueError:
            return f"{title}"

    def print_tree(self, *, to_file: str | None = None, width: int = 80, dp: int = 2) -> None:
        lines = []
        lines_tuple = []

        for node in self.root.preorder_traversal():
            prefix = "    " * node.depth
            name = f"{prefix}{node._name}"
            try:
                val = node.risk_capital
                num_width = width - len(name)
                lines.append(f"{name} {val:·>{num_width},.{dp}f}")
                lines_tuple.append((name, f"{val:.{dp}f}"))
            except ValueError:
                lines.append(name)
                lines_tuple.append((name, ))

        for line in lines:
            print(line)

        if to_file is not None:
            file_path = Path(to_file)
            mode = 'a' if file_path.is_file() else 'w'
            with open(file_path, mode=mode, newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"==== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - RiskNode.print_tree - started ===="])
                writer.writerows(lines_tuple)
                writer.writerow([f"==== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - RiskNode.print_tree - ended ===="])

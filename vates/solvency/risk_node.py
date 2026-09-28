import numpy as np
import math
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

    __slots__ = ("_name", "_slug", "_risk_charge", "_parent", "_children", "_agg_func",)

    def __init__(self, name: str, /, *, slug: str | None = None):
        self._name: str = name
        self._slug: str = self._normalize_identifier(slug or name)
        self._risk_charge: float | None = None
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
    def risk_charge(self) -> float:
        if self._risk_charge is None:
            self.aggregate()
        if self._risk_charge is None:
            raise ValueError(f"{self._name}: risk charge hasn't been {'provided' if self.is_leaf else 'aggregated'}.")
        return self._risk_charge

    @property
    def risk_diversification(self) -> float:
        if self.is_leaf:
            return 0.0
        return sum([c.risk_charge for c in self._children]) - self.risk_charge

    @property
    def root(self) -> Self:
        obj = self
        while obj._parent is not None:
            obj = obj._parent
        return obj

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
    def parts(self) -> tuple[str, ...]:
        names = [self._name]
        obj = self
        while obj._parent is not None:
            obj = obj._parent
            names.append(obj._name)
        names.reverse()
        return tuple(names)

    @property
    def depth(self) -> int:
        return len(self.parts) - 1

    @property
    def path(self) -> str:
        return "./" + "/".join(self.parts[1:])  # `.` represents root

    def select(self, path: str, /) -> Self:
        if not isinstance(path, str):
            raise TypeError(f"Invalid type of path: '{type(path)}', expected 'str'.")
        obj = self
        for p in path.split("/"):
            if p == "." or p == "":
                pass
            elif p == "..":
                obj = obj._parent
            else:
                obj = next((c for c in obj._children if c._name == p), None)
            if obj is None:
                raise ValueError(f"{self._name}: can't select: '{path}'; failed at '{p}'.")
        return obj

    def list_leaves(self) -> list[Self]:
        leaves = []
        for c in self._children:
            if c.is_leaf:
                leaves.append(c)
            else:
                leaves.extend(c.list_leaves())
        return leaves

    def attach_sub_risk(self, *args: Self) -> Self | list[Self]:
        if len(args) == 0:
            raise ValueError(f"Nothing to attach.")
        attached: list[Self] = []
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
            attached.append(node)
        return attached[0] if len(attached) == 0 else attached

    def set_agg_func(self, func) -> None:
        if self._agg_func is not None:
            warnings.warn(f"{self._name}: aggregation function will be reset.")
        self._agg_func = func

    def update_risk_charge(self, value: float | None = None, /) -> None:
        if not self.is_leaf:
            self._risk_charge = None  # reset only, lazy evaluation will be executed when calling property `risk_charge`
            return
        if not isinstance(value, float):
            raise TypeError(f"Invalid type of risk charge '{type(value)}', expected 'float'.")
        self._risk_charge = value
        if self._parent is not None:
            self._parent.update_risk_charge()  # cascade

    def aggregate(self) -> float:
        if self.is_leaf:
            return self._risk_charge
        if self._agg_func is None:
            raise ValueError(f"{self._name}: aggregation function is None; use `set_agg_func(..)` to set.")
        kwargs = {c._slug: c.risk_charge for c in self._children}
        self._risk_charge = self._agg_func(**kwargs)
        return self._risk_charge

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

    def copy_structure(self, *, new_name: str = None, new_slug: str = None) -> Self:
        if not self.is_root:
            raise ValueError(f"Can't copy structure from a non-root node; use `foo.root.copy_structure(..)`.")
        copied_root = RiskNode(new_name or self._name, slug=new_slug)
        copied_root.set_agg_func(self._agg_func)
        if not self.is_leaf:
            for node in self.preorder_traversal()[1:]:  # the first one is self
                copied_node = RiskNode(node._name, slug=node._slug)
                copied_node.set_agg_func(node._agg_func)
                copied_root.select(node._parent.path).attach_sub_risk(copied_node)
        return copied_root

    def __truediv__(self, other: str, /) -> Self:
        return self.select(other)

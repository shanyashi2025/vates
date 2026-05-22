import math


class AutogradCell:
    """ holds a single scalar value, and automates the backpropagation process to compute its gradient (partial derivative)  """

    __slots__ = ('value', 'grad', '_children', '_local_grads')

    def __init__(self, value, children=(), local_grads=()):
        self.value = value               # scalar value of this node calculated during forward pass
        self.grad = 0                    # derivative w.r.t. this node, calculated in backward pass
        self._children = children        # children of this node in the computation graph
        self._local_grads = local_grads  # local derivative of this node w.r.t. its children

    def __add__(self, other):
        if isinstance(other, AutogradCell):
            return AutogradCell(self.value + other.value, (self, other), (1, 1))
        else:
            return AutogradCell(self.value + other, (self,), (1,))

    def __mul__(self, other):
        if isinstance(other, AutogradCell):
            return AutogradCell(self.value * other.value, (self, other), (other.value, self.value))
        else:
            return AutogradCell(self.value * other, (self,), (other,))

    def __pow__(self, other):
        if isinstance(other, AutogradCell):
            return AutogradCell(pow_val := self.value ** other.value, (self, other),
                                (other.value * self.value ** (other.value - 1), pow_val * math.log(self.value)))
        else:
            return AutogradCell(self.value ** other, (self,), (other * self.value ** (other - 1),))

    def log(self):
        return AutogradCell(math.log(self.value), (self,), (1 / self.value,))

    def exp(self):
        return AutogradCell(exp_val := math.exp(self.value), (self,), (exp_val,))

    def apply_cap(self, other):
        oth_val = other if not isinstance(other, AutogradCell) else other.value
        if self.value <= oth_val:
            return AutogradCell(self.value, (self,), (1,))
        elif oth_val < self.value and isinstance(other, AutogradCell):
            return AutogradCell(oth_val, (other,), (1,))
        else:
            return oth_val

    def apply_floor(self, other):
        oth_val = other if not isinstance(other, AutogradCell) else other.value
        if self.value >= oth_val:
            return AutogradCell(self.value, (self,), (1,))
        elif oth_val > self.value and isinstance(other, AutogradCell):
            return AutogradCell(oth_val, (other,), (1,))
        else:
            return oth_val

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def __lt__(self, other) -> bool:
        if isinstance(other, AutogradCell):
            return self.value < other.value
        else:
            return self.value < other
    def __le__(self, other) -> bool:
        if isinstance(other, AutogradCell):
            return self.value <= other.value
        else:
            return self.value <= other
    def __gt__(self, other) -> bool: return not self <= other
    def __ge__(self, other) -> bool: return not self < other

    def backward(self) -> None:
        topo = AutogradCell.build_topo(self)
        self.grad = 1
        for cell in reversed(topo):
            for child, local_grad in zip(cell._children, cell._local_grads):
                child.grad += local_grad * cell.grad

    @classmethod
    def build_topo(cls, cell, topo = None, visited = None) -> list['AutogradCell']:
        if topo is None: topo = []
        if visited is None: visited = set()
        if cell not in visited:
            visited.add(cell)
            for child in cell._children:
                cls.build_topo(child, topo, visited)
            topo.append(cell)
        return topo

    def get_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__ if hasattr(self, s)}

    def __repr__(self):
        return f"AutogradCell(value={self.value}, grad={self.grad})"
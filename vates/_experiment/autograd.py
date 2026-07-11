import math


class Cell:
    """ holds a single scalar value, and automates the backpropagation process to compute its gradient (partial derivative)  """

    __slots__ = ('value', 'grad', '_children', '_local_grads')

    def __init__(self, value, children=(), local_grads=()):
        self.value = value               # scalar value of this node calculated during forward pass
        self._children = children        # children of this node in the computation graph
        self._local_grads = local_grads  # local derivative of this node w.r.t. its children
        self.grad = {}                   # derivative w.r.t. this node, calculated in backward pass

    def __add__(self, other):
        if isinstance(other, Cell):
            return Cell(self.value + other.value, (self, other), (1, 1))
        else:
            return Cell(self.value + other, (self,), (1,))

    def __mul__(self, other):
        if isinstance(other, Cell):
            return Cell(self.value * other.value, (self, other), (other.value, self.value))
        else:
            return Cell(self.value * other, (self,), (other,))

    def __pow__(self, other):
        if isinstance(other, Cell):
            return Cell(pow_val := self.value ** other.value, (self, other),
                        (other.value * self.value ** (other.value - 1), pow_val * math.log(self.value)))
        else:
            return Cell(self.value ** other, (self,), (other * self.value ** (other - 1),))

    def apply_cap(self, other):
        oth_val = other if not isinstance(other, Cell) else other.value
        if self.value <= oth_val:
            return Cell(self.value, (self,), (1,))
        elif oth_val < self.value and isinstance(other, Cell):
            return Cell(oth_val, (other,), (1,))
        else:
            return oth_val

    def apply_floor(self, other):
        oth_val = other if not isinstance(other, Cell) else other.value
        if self.value >= oth_val:
            return Cell(self.value, (self,), (1,))
        elif oth_val > self.value and isinstance(other, Cell):
            return Cell(oth_val, (other,), (1,))
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
        if isinstance(other, Cell):
            return self.value < other.value
        else:
            return self.value < other
    def __le__(self, other) -> bool:
        if isinstance(other, Cell):
            return self.value <= other.value
        else:
            return self.value <= other
    def __gt__(self, other) -> bool: return not self <= other
    def __ge__(self, other) -> bool: return not self < other

    def backward(self, name) -> None:
        topo = []
        visited = set()
        def build_topo(c):
            if c not in visited:
                visited.add(c)
                for cc in c._children:
                    build_topo(cc)
                topo.append(c)
        build_topo(self)
        self.grad[name] = 1
        for cell in reversed(topo):
            for child, local_grad in zip(cell._children, cell._local_grads):
                if name in child.grad:
                    child.grad[name] += local_grad * cell.grad[name]
                else:
                    child.grad[name] = local_grad * cell.grad[name]

    def get_grad(self, name, default=0.0) -> float:
        return self.grad.get(name, default)

    def get_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__ if hasattr(self, s)}

    def __repr__(self):
        return f"AutogradCell(value={self.value}, grad={self.grad})"

def log(x):
    if isinstance(x, Cell):
        return Cell(math.log(x.value), (x,), (1 / x.value,))
    else:
        return math.log(x)

def exp(x):
    if isinstance(x, Cell):
        return Cell(exp_val := math.exp(x.value), (x,), (exp_val,))
    else:
        return math.exp(x)

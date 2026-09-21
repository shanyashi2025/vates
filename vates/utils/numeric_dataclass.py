from dataclasses import dataclass, fields, replace
from typing import Self

@dataclass(slots=True)
class NumericDataclass:

    def zero(self) -> None:
        for f in fields(self):
            setattr(self, f.name, 0)

    def copy(self) -> Self:
        return replace(self)

    def __add__(self, other) -> Self:
        if type(other) is not type(self):
            return NotImplemented
        return replace(self, **{
            f.name: getattr(self, f.name) + getattr(other, f.name)
            for f in fields(self)
        })

    def __radd__(self, other) -> Self:
        return self.__add__(other)

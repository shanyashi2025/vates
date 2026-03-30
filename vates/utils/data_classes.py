from dataclasses import dataclass
from typing import Self

@dataclass
class NumVarGroup:

    def reset(self) -> None:
        for k, _ in self.__dict__.items():
            self.__dict__[k] = 0

    def copy(self) -> Self:
        new_obj = self.__class__()
        for k, v in self.__dict__.items():
            new_obj.__dict__[k] = v
        return new_obj

    def __add__(self, other) -> Self:
        if isinstance(other, self.__class__):
            new_obj = self.__class__()
            for k, v in self.__dict__.items():
                new_obj.__dict__[k] = v + getattr(other, k)
            return new_obj
        return NotImplemented

    def __radd__(self, other) -> Self:
        return self.__add__(other)

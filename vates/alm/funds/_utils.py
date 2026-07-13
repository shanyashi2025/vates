from dataclasses import dataclass, field

from vates.alm.assets import Asset, Cash
from vates.alm.liabs import Liab


@dataclass(slots=True)
class ALContainer:
    """Asset liability container

    """
    _assets: list[Asset] = field(default_factory=list)
    _liabs: list[Liab] = field(default_factory=list)
    _accum_free_proceeds: float = 0.0

    @property
    def assets(self) -> list[Asset]:
        return self._assets

    @property
    def liabs(self) -> list[Liab]:
        return self._liabs

    @property
    def accum_free_proceeds(self) -> float:
        return self._accum_free_proceeds

    def deposit_free_proceeds(self, amount: float) -> None:
        self._accum_free_proceeds += amount

    def withdrawal_accum_free_proceeds(self) -> float:
        amount = self._accum_free_proceeds
        self._accum_free_proceeds = 0.0
        return amount

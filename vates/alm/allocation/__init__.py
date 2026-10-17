from typing import Callable

from vates.alm.allocation._enums import AssetAllocationStrategy, AssetPurchaseMethod
from vates.alm.allocation._allocation_group import AssetAllocationGroup, AssetAllocationParams


def getRebalanceFunction(key: str | None = None, /) -> Callable:
    if key is None or key == "standard":
        from vates.alm.allocation.rebalance import standard_rebalance
        return standard_rebalance
    raise NotImplementedError(f"Rebalance function '{key}' is not implemented.")

__all__ = [
    'AssetAllocationStrategy',
    'AssetPurchaseMethod',
    'AssetAllocationGroup',
    'AssetAllocationParams',
    'getRebalanceFunction',
]




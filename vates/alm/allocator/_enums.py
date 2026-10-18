from enum import Enum, auto

class AssetAllocationStrategy(Enum):
    """Enum for asset allocation strategies."""
    NO_TRADE = auto()
    BUY_HOLD = auto()
    BUY_SELL = auto()
    UNCONSTRAINED = auto()

class AssetPurchaseMethod(Enum):
    """Enum for asset purchase method."""
    SCALE_UP_EXISTING = auto()
    PURCHASE_PROFILE = auto()
    NOT_USED = auto()

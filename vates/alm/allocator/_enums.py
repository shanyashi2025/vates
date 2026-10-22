from enum import Enum, auto

class AssetAllocationStrategy(Enum):
    """Enum for asset allocation strategies."""
    NO_TRADE = auto()
    BUY_HOLD = auto()
    BUY_SELL = auto()
    UNCONSTRAIN = auto()

class AssetPurchaseMethod(Enum):
    """Enum for asset purchase method."""
    SCALE_UP_EXISTING = auto()
    PURCHASE_PROFILE = auto()
    NOT_USED = auto()

class AssetRebalanceObjective(Enum):
    """Enum for asset rebalance objective."""
    NOT_USED = auto()
    NO_ACTION = auto()
    TO_TARGET_WEIGHT = auto()
    TO_LOWER_WEIGHT = auto()
    TO_UPPER_WEIGHT = auto()

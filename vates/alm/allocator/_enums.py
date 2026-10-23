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

class AssetSizeKind(Enum):
    """Enum for asset sizes."""
    NOT_USED = auto()
    CURRENT = auto()
    TARGET = auto()
    LOWER = auto()
    UPPER = auto()

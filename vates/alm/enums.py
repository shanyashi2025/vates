from enum import Enum, unique

@unique
class AssetBuySellApproach(Enum):
    """Enum for asset buy/sell approach."""
    NO_TRADE = "NO_TRADE"
    BUY_HOLD = "BUY_HOLD"
    BUY_SELL = "BUY_SELL"
    RESIDUAL = "RESIDUAL"


@unique
class AssetPurchaseMethod(Enum):
    """Enum for asset purchase method."""
    NOT_USED = "NOT_USED"
    SCALE_UP_EXISTING = "SCALE_UP_EXISTING"
    PURCHASE_PROFILE = "PURCHASE_PROFILE"

from enum import Enum, unique
import warnings

from vates.alm.assets.cash import Cash
from vates.alm.assets.equity import Equity
from vates.alm.assets.bond_fixed import BondFixed
from vates.alm.assets.derivatives import EquityOption
from vates.alm.assets.builders.bond_fixed_builder import BondFixedBuilder
from vates.alm.assets.builders.equity_option_builder import EquityOptionBuilder

_ASSET_CLS_MAP = {
    "cash": Cash,
    "equity": Equity,
    "bond": BondFixed,
    "bond_fixed": BondFixed,
    "fixed_bond": BondFixed,
    "equity_option": EquityOption
}

_BUILDER_CLS_MAP = {
    BondFixed: BondFixedBuilder,
    EquityOption: EquityOptionBuilder,
}

@unique
class AssetPreCalculation(Enum):
    """Enum for asset pre-calculations."""
    MARKET_SPREAD = 'calibrate_market_spread'
    MARKET_PRICE = "calculate_market_price"
    AMORT_RATE = 'calculate_amort_rate'
    COUPON_RATE = 'derive_coupon_rate'
    INTEREST_RATE = 'derive_interest_rate'
    IMPLIED_VOLATILITY = 'calibrate_implied_volatility'
    RISK_NEUTRALIZATION = 'risk_neutralization'
    FUTURES_PRICE = 'calculate_futures_price'
    CONTRACT_VALUE = 'calculate_contract_value'
    DELIVERY_PRICE = 'derive_delivery_price'


def create_asset(asset_cls,
                 pre_calculations: str | AssetPreCalculation | list[str | AssetPreCalculation] | None = None,
                 **kwargs):
    """
    Factory function to create an asset.

    Args:
        asset_cls: Asset class.
        pre_calculations (str | AssetPreCalculation | list[str | AssetPreCalculation] | None): List of pre-calculations.
        **kwargs: Parameters.

    Returns:
        Asset: The constructed Asset object.

    Raises:
        ValueError: If asset_cls is not a valid asset class name.
    """
    if isinstance(asset_cls, str):
        if asset_cls.lower() in _ASSET_CLS_MAP:
            asset_cls = _ASSET_CLS_MAP[asset_cls.lower()]
        else:
            raise ValueError(f"'{asset_cls}' is not a valid asset class name.")

    if asset_cls in _BUILDER_CLS_MAP:
        return create_asset_by_builder(_BUILDER_CLS_MAP[asset_cls], pre_calculations, **kwargs)

    return asset_cls(**kwargs)


def create_asset_by_builder(builder_cls,
                            pre_calculations: str | AssetPreCalculation | list[str | AssetPreCalculation] | None = None,
                            **kwargs):
    """
    Factory function to create an asset.

    Args:
        builder_cls: Asset builder class.
        pre_calculations (str | AssetPreCalculation | list[str | AssetPreCalculation] | None): List of pre-calculations.
        **kwargs: Parameters.

    Returns:
        Asset: The constructed Asset object.
    """
    builder = builder_cls(**kwargs)

    if pre_calculations is None:
        return builder.build()

    if not isinstance(pre_calculations, list):
        pre_calculations = [pre_calculations]

    if len(pre_calculations) == 0:
        return builder.build()

    for calc in pre_calculations:
        if isinstance(calc, AssetPreCalculation):
            func_name = calc.value
        elif isinstance(calc, str) and calc.upper() in AssetPreCalculation:
            func_name = AssetPreCalculation[calc.upper()]
        else:  # no further validation
            func_name = calc

        func = getattr(builder, func_name, None)
        if callable(func):
            func()
        else:
            warnings.warn(f"{calc} ignored: {func_name} not found in {str(builder)} or not callable.")

    return builder.build()

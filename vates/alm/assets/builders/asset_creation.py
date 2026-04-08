import warnings


_PRECALC_DICT = {
    'market_spread': 'calibrate_market_spread',
    'market_price': 'calculate_market_price',
    'amort_rate': 'calculate_amort_rate',
    'coupon_rate': 'derive_coupon_rate',
    'interest_rate': 'derive_interest_rate',
    'implied_volatility': 'calibrate_implied_volatility',
    'risk_neutralization': 'risk_neutralization',
    'futures_price': 'calculate_futures_price',
    'contract_value': 'calculate_contract_value',
    'delivery_price': 'derive_delivery_price',
}


def create_asset(asset_builder_cls, pre_calculations: str | list[str] | None, **kwargs):
    """
    Factory function to create an asset.

    Args:
        asset_builder_cls: Asset builder class.
        pre_calculations (str | list[str] | None): List of pre-calculations.
        **kwargs: Parameters.

    Returns:
        Asset: The constructed Asset object.

    Raises:
        ValueError: If pre-calculation is not defined.
    """
    builder = asset_builder_cls(**kwargs)

    if type(pre_calculations) == str:
        pre_calculations = [pre_calculations]

    if pre_calculations:
        for calc in pre_calculations:
            if calc not in _PRECALC_DICT: raise ValueError(f"Initial calculation {calc} not defined.")
            func_name = _PRECALC_DICT[calc]
            func = getattr(builder, func_name, None)
            if callable(func):
                func()
            else:
                warnings.warn(f"{asset_builder_cls} {calc} ignored: {func_name} not found in {str(builder)} or not callable.")

    return builder.build()

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

_BUILDER_MAP = {
    BondFixed: BondFixedBuilder,
    EquityOption: EquityOptionBuilder,
}


def create_asset(asset_cls, build_pipeline: str | list[str] | None = None, **kwargs):
    """
    Factory function to create an asset.

    Args:
        asset_cls: Asset class, 'cash', 'equity', 'bond' ('bond_fixed', 'fixed_bond' equivalently), 'equity_option'
        build_pipeline (str | list[str] | None): Build pipeline.
        **kwargs: Parameters.

    Returns:
        Asset: The constructed Asset object.

    Raises:
        ValueError: If `asset_cls` is not a valid asset class name.
    """
    if isinstance(asset_cls, str):
        if asset_cls.lower() in _ASSET_CLS_MAP:
            asset_cls = _ASSET_CLS_MAP[asset_cls.lower()]
        else:
            raise ValueError(f"'{asset_cls}' is not a valid asset class name.")

    if asset_cls in _BUILDER_MAP:
        # return create_asset_by_builder(_BUILDER_MAP[asset_cls], build_pipeline, **kwargs)
        return _BUILDER_MAP[asset_cls](**kwargs).build(build_pipeline)
    else:
        return asset_cls(**kwargs)

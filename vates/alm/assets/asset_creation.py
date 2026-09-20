from vates.alm.assets.cash import Cash
from vates.alm.assets.equity import Equity
from vates.alm.assets.builders.bond_fixed_builder import BondFixedBuilder
from vates.alm.assets.builders.equity_option_builder import EquityOptionBuilder

_DEFAULT_CREATOR_CLASS_DICT = {
    "cash": Cash,
    "equity": Equity,
    "bond": BondFixedBuilder,
    "bond_fixed": BondFixedBuilder,
    "fixed_bond": BondFixedBuilder,
    "equity_option": EquityOptionBuilder,
}


def create_asset(asset_cls, *, build_pipeline: str | list[str] | None = None, pipe_operator: str = "|>",
                 creator_class_dict: dict[str, ...] = None, dynamic_setattrs: dict[str, ...] | None = None, **kwargs):
    """
    Factory function to create an asset.

    Args:
        asset_cls: Asset class, 'cash', 'equity', 'bond' ('bond_fixed', 'fixed_bond' equivalently), 'equity_option'
        build_pipeline (str | list[str] | None): Build pipeline.
        pipe_operator (str): Pipe operator, used when `build_pipeline` is str, defaults to '|>'.
        creator_class_dict (dict[str, ...]): Dict of asset creator class, defaults to `_DEFAULT_CREATOR_CLASS_DICT`
        dynamic_setattrs (dict[str, ...]): Dict of attributes to be dynamically created, defaults to None.
        **kwargs: Parameters.

    Returns:
        Asset: The constructed Asset object.

    Raises:
        ValueError: If `asset_cls` is not a valid asset class name.
    """
    if creator_class_dict is None:
        creator_class_dict = _DEFAULT_CREATOR_CLASS_DICT

    creator_class = creator_class_dict.get(asset_cls) or asset_cls

    if hasattr(creator_class, "build"):
        obj = creator_class(**kwargs).build(build_pipeline, pipe_operator)
    else:
        if build_pipeline:
            raise ValueError(f"{asset_cls}: build_pipeline is provided ({build_pipeline}), "
                             f"but creator class {creator_class} doesn't have 'build' method.")
        obj = creator_class(**kwargs)  # directly construct

    if dynamic_setattrs:
        for key, val in dynamic_setattrs.items():
            setattr(obj, key, val)

    return obj

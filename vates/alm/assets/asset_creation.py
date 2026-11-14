import warnings
from typing import Any, Literal, overload

from vates.global_conf import GLOBAL_CHECK_MODE, CheckMode
from vates.alm.assets.cash import Cash
from vates.alm.assets.bond_fixed import BondFixed
from vates.alm.assets.equity import Equity
from vates.alm.assets.derivatives.equity_option import EquityOption
from vates.alm.assets.builders.bond_fixed_builder import BondFixedBuilder
from vates.alm.assets.builders.equity_option_builder import EquityOptionBuilder

_DEFAULT_CREATOR_CLASS_MAP: dict[str | type, type] = {
    "cash": Cash,
    Cash: Cash,
    "equity": Equity,
    Equity: Equity,
    "bond": BondFixedBuilder,
    "bond_fixed": BondFixedBuilder,
    "fixed_bond": BondFixedBuilder,
    BondFixed: BondFixedBuilder,
    "equity_option": EquityOptionBuilder,
    EquityOption: EquityOptionBuilder,
}

@overload
def create_asset(cls_like: Literal["cash"] | type[Cash], /, *,
                 build_pipeline: list[str | tuple[str, dict]] | None = None,
                 creator_class_dict: dict[str | type, Any] | None = None,
                 check_mode: CheckMode = GLOBAL_CHECK_MODE, **kwargs) -> Cash: ...

@overload
def create_asset(cls_like: Literal["equity"] | type[Equity], /, *,
                 build_pipeline: list[str | tuple[str, dict]] | None = None,
                 creator_class_dict: dict[str | type, Any] | None = None,
                 check_mode: CheckMode = GLOBAL_CHECK_MODE, **kwargs) -> Equity: ...

@overload
def create_asset(cls_like: Literal["bond", "bond_fixed", "fixed_bond"] | type[BondFixed], /, *,
                 build_pipeline: list[str | tuple[str, dict]] | None = None,
                 creator_class_dict: dict[str | type, Any] | None = None,
                 check_mode: CheckMode = GLOBAL_CHECK_MODE, **kwargs) -> BondFixed: ...

@overload
def create_asset(cls_like: Literal["equity_option"] | type[EquityOption], /, *,
                 build_pipeline: list[str | tuple[str, dict]] | None = None,
                 creator_class_dict: dict[str | type, Any] | None = None,
                 check_mode: CheckMode = GLOBAL_CHECK_MODE, **kwargs) -> EquityOption: ...

@overload
def create_asset(cls_like: str | type, /, *,
                 build_pipeline: list[str | tuple[str, dict]] | None = None,
                 creator_class_dict: dict[str | type, Any] | None = None,
                 check_mode: CheckMode = GLOBAL_CHECK_MODE, **kwargs) -> Any: ...

def create_asset(cls_like: str | type, /, *,
                 build_pipeline: list[str | tuple[str, dict]] | None = None,
                 creator_class_dict: dict[str | type, Any] | None = None,
                 check_mode: CheckMode = GLOBAL_CHECK_MODE, **kwargs) -> Any:
    """
    Factory function to create an asset.

    Args:
        cls_like: Asset class, 'cash', 'equity', 'bond' ('bond_fixed', 'fixed_bond' equivalently), 'equity_option'
        build_pipeline (list[str | tuple[str, dict] | None): Build pipeline, values can be either a key (str)
            or a tuple of (key, kwargs (dict)) pairs.
        creator_class_dict (dict[str, ...]): Dict of asset creator class. Lookup order:
            `creator_class_dict.get(cls_like)` > `_DEFAULT_CREATOR_CLASS_MAP.get(cls_like)` > `cls_like`
        check_mode (CheckMode): Check mode for validation against allocation, defaults to `GLOBAL_CHECK_MODE`.
        **kwargs: Parameters.

    Returns:
        Asset: The constructed Asset object.

    Raises:
        ValueError: If `cls_like` is not a valid asset class name.
    """
    cls = (creator_class_dict or {}).get(cls_like) or _DEFAULT_CREATOR_CLASS_MAP.get(cls_like) or cls_like
    if hasattr(cls, "build"):
        return cls(**kwargs).build(build_pipeline)
    if build_pipeline:
        msg = f"{cls_like}: {build_pipeline=} are ignored as creator {cls} doesn't have 'build' method."
        if check_mode == CheckMode.ERROR:
            raise ValueError(msg)
        warnings.warn(msg)
    return cls(**kwargs)  # directly construct

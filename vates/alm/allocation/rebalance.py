import warnings

from vates.global_conf import CheckLevel
from vates.alm.assets import Asset, Cash
from vates.alm._utils import Connector
from vates.alm.allocation._allocation_group import AssetAllocationGroup, AssetAllocationParams, TradeOrder, BuySellAction
from vates.alm.allocation._enums import AssetAllocationStrategy, AssetPurchaseMethod

def standard_rebalance(
    exist_assets: Connector[Asset] | list[Asset],
    *,
    profile_assets: list[Asset] | None = None,
    total_size: float,
    asset_size_basis: str,
    allocation_group_attr_name: str = "allocation_group",
    allocation_group_params: dict[str, AssetAllocationParams],
    check_level: CheckLevel | None = CheckLevel.WARN,
    **kwargs
) -> float:
    """Rebalance assets in the fund to match the target allocation.

    Args:
        exist_assets (Connector[Asset] | list[Asset]): Connector of list of assets to be rebalanced.
        profile_assets (list[Asset] | None): Profile assets for reference.
        total_size (float | dict[str, float]): Total size for allocation, shall correspond to `asset_size_basis`.
        asset_size_basis (str): Asset reporting basis for sizing (e.g. FAV or BSV).
        allocation_group_attr_name (str): Named attribute for asset allocation group, defaults to "allocation_group".
        allocation_group_params (dict[str, AssetAllocationGroupParams]): Parameters for each allocation group.
        check_level (CheckLevel): Check level for validation against allocation, defaults to None.

    Returns:
        float: Net proceeds after the reblancing: proceeds received from disposal (sell), minus proceeds spent
            to purchase (buy).

    """
    # --- step 1: build up the allocation groups ---
    groups: list[AssetAllocationGroup] = _make_sorted_alloc_groups(
        exist_assets=exist_assets,
        profile_assets=profile_assets,
        total_size=total_size,
        asset_size_basis=asset_size_basis,
        allocation_group_attr_name=allocation_group_attr_name,
        allocation_group_params=allocation_group_params,
    )
    res_groups: list[AssetAllocationGroup] = [ag for ag in groups if ag.strategy == AssetAllocationStrategy.RESIDUAL]

    # --- step 2: process non-residual groups in sequence ---
    for ag in groups:
        if ag.strategy != AssetAllocationStrategy.RESIDUAL:
            ag.make_trade_decision()
            ag.execute_trade()

    # --- step 3: process residual groups ---
    size_gap = total_size - sum(ag.current_size for ag in groups)
    _process_residual_groups(residual_groups=res_groups, size_gap=size_gap, tolerance=max(abs(total_size * 1e-6), 0.01))

    # --- step 4: validate if target allocations is satisfied ---
    for ag in groups:
        if not ag.is_allocation_satisfied():
            msg = (
                f"{ag.name}: allocation is not satisfied, {ag.current_size/total_size:.4%}; strategy: {ag.strategy.name}; "
                f"(floor, cap): ({ag.target_weight[0]:.4%}, {ag.target_weight[2]:.4%}); "
                f"purchase method: {ag.purchase_method.name}.")
            if check_level == CheckLevel.ERROR:
                raise ValueError(msg)
            warnings.warn(msg)

    # --- step 4: add the purchased assets ---
    net_proceeds = 0.0
    for ag in groups:
        exist_assets.extend(ag.staged_conn)
        net_proceeds += ag.take_out_net_proceeds()

    return net_proceeds


def _make_sorted_alloc_groups(
    *,
    exist_assets: Connector[Asset] | list[Asset],
    profile_assets: list[Asset] | None,
    total_size: float,
    asset_size_basis: str,
    allocation_group_attr_name: str,
    allocation_group_params: dict[str, AssetAllocationParams],
    require_validate: bool = True
) -> list[AssetAllocationGroup]:
    ag_kwargs: dict[str, dict[str, ...]] = {}
    for name, params in allocation_group_params.items():
        if name in ag_kwargs:
            raise KeyError(f"Duplicate allocation group name: '{name}'.")
        ag_kwargs[name] = {
            "params": params,
            "total_size": total_size,
            "size_basis": asset_size_basis,
            "exist_assets": [],
            "profile_assets": [],
        }

    for asset in exist_assets:
        name = getattr(asset, allocation_group_attr_name, None)
        ag = ag_kwargs.get(name, None)
        if ag is not None:
            ag["exist_assets"].append(asset)

    for asset in profile_assets or []:
        name = getattr(asset, allocation_group_attr_name, None)
        ag = ag_kwargs.get(name, None)
        if ag is not None:
            ag["profile_assets"].append(asset)

    alloc_groups: list[AssetAllocationGroup] = [AssetAllocationGroup(name=key, **val) for key, val in ag_kwargs.items()]
    alloc_groups.sort(key=lambda x: x.sequence)

    if require_validate:
        seen_residual = []
        for i, ag in enumerate(alloc_groups):
            if ag.strategy is AssetAllocationStrategy.RESIDUAL:
                seen_residual.append(i)
            if ag.strategy in (AssetAllocationStrategy.BUY_HOLD, AssetAllocationStrategy.BUY_SELL):
                if ag.target_weight is None:
                    raise ValueError(f"'{ag.name}': '{ag.strategy.value}'; 'target_weight' is 'None'.")
                if ag.purchase_method is None:
                    raise ValueError(f"'{ag.name}': '{ag.strategy.value}'; 'purchase_method' is 'None'.")
                if ag.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING and len(ag.exist_conn) == 0:
                    raise ValueError(f"'{ag.name}': 'SCALE_UP_EXISTING'; no exsiting asset.")
                if ag.purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE and len(ag.profile_conn) == 0:
                    raise ValueError(f"'{ag.name}': 'PURCHASE_PROFILE';  no profile asset.")

        if len(seen_residual) == 0:
            pass
        else:
            if seen_residual[-1] + 1 != len(alloc_groups):
                raise ValueError(f"Last allocation group must be 'RESIDUAL'.")
            if seen_residual[-1] - seen_residual[0] + 1 != len(seen_residual):
                raise ValueError(f"All 'RESIDUAL' allocation groups must be after non-'RESIDUAL'.")

    return alloc_groups


def _process_residual_groups(residual_groups: list[AssetAllocationGroup], size_gap: float, tolerance: float) -> None:
    if abs(size_gap) <= tolerance:
        return

    if len(residual_groups) == 0:
        warnings.warn(f"Target allocation is not met but no residual allocation group to process.")
        return

    current_residual_size: float = sum([ag.current_size for ag in residual_groups])

    if current_residual_size == 0:
        # This would be very edge case when total residual allocation group (usually cash) balance happens to be zero.
        # Utilize cash asset to safely proceed
        _fallback_cash = None
        for asset in [ag.exist_conn for ag in residual_groups]:
            if isinstance(asset, Cash):
                _fallback_cash = asset
                break
        if _fallback_cash is not None:
            _fallback_cash.invest_new_money(size_gap)
        else:
            warnings.warn(f"No cash asset in 'RESIDUAL' group is available for sclaing.")
        return

    # scale residual allocation_group
    if size_gap > 0:
        trade_decn = TradeOrder(action=BuySellAction.BUY_EXIST, arg=size_gap / current_residual_size)
    elif size_gap < 0:
        trade_decn = TradeOrder(action=BuySellAction.SELL, arg=- size_gap / current_residual_size)
    else:
        raise ValueError("Should never get here.")

    for ag in residual_groups:
        ag.execute_trade(trade_decn)

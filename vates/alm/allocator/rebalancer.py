import numpy as np
import pandas as pd
import uuid
import warnings
from typing import Literal

from vates._core import ProjModelEngine, time_synchronized, make_proj_variable
from vates.global_conf import CheckLevel, GLOBAL_CHECK_LEVEL
from vates.alm.assets import Asset, Cash
from vates.alm._utils import Connector
from vates.alm.allocator._allocation_group import _AllocationGroup, AssetAllocationParams, TradeOrder, BuySellAction
from vates.alm.allocator._enums import AssetAllocationStrategy, AssetPurchaseMethod, AssetTradeDirection
from vates.alm.allocator.proceeds_keeper import ProceedsKeeper


def _rebalance_alloc_groups(
    groups: list[_AllocationGroup],
    total_size: float,
    check_level: CheckLevel,
) -> None:
    """Rebalance assets to match the target allocation.

    Args:
        total_size (float): Total size for allocation, shall correspond to `asset_size_basis`.
        check_level (CheckLevel): Check level for validation against allocation.

    """
    uc_groups: list[_AllocationGroup] = [ag for ag in groups if ag.strategy == AssetAllocationStrategy.UNCONSTRAINED]

    if len(uc_groups) >= 1:  # has unconstrained group
        # --- step 1: process non-unconstrained groups in priority ---
        for ag in groups:
            if ag.strategy != AssetAllocationStrategy.UNCONSTRAINED:
                trade_order = _make_desired_trade_decision(ag)
                ag.execute_trade(trade_order)

        # --- step 2: process unconstrained groups ---
        size_gap = total_size - sum(ag.current_size for ag in groups)
        _process_unconstrained_groups(groups=uc_groups, size_gap=size_gap, tolerance=max(abs(total_size * 1e-6), 0.01))

    else:
        raise NotImplementedError()

    # --- validate if target allocations is satisfied ---
    for ag in groups:
        if not ag.is_allocation_satisfied():
            msg = (
                f"{ag.name}: allocation is not satisfied, {ag.current_size/total_size:.4%}; strategy: {ag.strategy.name}; "
                f"(lower, upper): ({ag.weight_range[0]:.4%}, {ag.weight_range[2]:.4%}); "
                f"purchase method: {ag.purchase_method.name}.")
            if check_level == CheckLevel.ERROR:
                raise ValueError(msg)
            warnings.warn(msg)


def _make_desired_trade_decision(group: _AllocationGroup) -> TradeOrder:
    """Make the trade decision for an allocation group.

    Raises:
        ValueError: If an invalid asset buy/sell approach is provided.
    """
    strategy = group.strategy
    if strategy == AssetAllocationStrategy.NO_TRADE:
        return TradeOrder(action=BuySellAction.NOOP)
    elif strategy == AssetAllocationStrategy.BUY_HOLD:
        return _make_desired_trade_decision_buy_hold(group)
    elif strategy == AssetAllocationStrategy.BUY_SELL:
        return _make_desired_trade_decision_buy_sell(group)
    else:
        raise ValueError(f"{group.name}: invalid allocation strategy: {strategy}")


def _make_desired_trade_decision_buy_hold(group: _AllocationGroup) -> TradeOrder:
    if group.strategy != AssetAllocationStrategy.BUY_HOLD:
        raise ValueError(f"'{group.name}': strategy ({group.strategy.name}) is not 'BUY_HOLD'.")

    purchase_method = group.purchase_method
    current_size, profile_size = group.current_size, group.profile_size
    lower_size, target_size, upper_size = group.size_range

    if current_size >= lower_size:
        return TradeOrder(action=BuySellAction.NOOP)

    direct = group.desire_direct.below_range
    if direct == AssetTradeDirection.TO_LOWER_WEIGHT:
        buy_to_size = lower_size
    elif direct == AssetTradeDirection.TO_TARGET_WEIGHT:
        buy_to_size = target_size
    elif direct == AssetTradeDirection.TO_UPPER_WEIGHT:
        buy_to_size = upper_size
    else:
        warnings.warn(f"strategy 'BUY_HOLD', below range desire: {direct.name}, can't buy any amount.")
        return TradeOrder(action=BuySellAction.NOOP)

    if purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
        return TradeOrder(action=BuySellAction.BUY_EXIST, arg=(buy_to_size - current_size) / current_size)
    elif purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
        return TradeOrder(action=BuySellAction.BUY_PROFILE, arg=(buy_to_size - current_size) / profile_size)
    else:
        raise ValueError(f"{group.name}: invalid purchase method {purchase_method} for 'BUY_HOLD'.")


def _make_desired_trade_decision_buy_sell(group: _AllocationGroup) -> TradeOrder:
    if group.strategy != AssetAllocationStrategy.BUY_SELL:
        raise ValueError(f"'{group.name}': strategy ({group.strategy.name}) is not 'BUY_SELL'.")

    purchase_method = group.purchase_method
    current_size, profile_size = group.current_size, group.profile_size
    lower_size, target_size, upper_size = group.size_range

    if current_size < lower_size:
        direct = group.desire_direct.below_range
        if direct == AssetTradeDirection.TO_LOWER_WEIGHT:
            buy_to_size = lower_size
        elif direct == AssetTradeDirection.TO_TARGET_WEIGHT:
            buy_to_size = target_size
        elif direct == AssetTradeDirection.TO_UPPER_WEIGHT:
            buy_to_size = upper_size
        else:
            warnings.warn(f"strategy 'BUY_SELL', below range desire: {direct.name}, can't buy any amount.")
            return TradeOrder(action=BuySellAction.NOOP)
        if purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
            return TradeOrder(action=BuySellAction.BUY_EXIST, arg=(buy_to_size - current_size) / current_size)
        elif purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
            return TradeOrder( action=BuySellAction.BUY_PROFILE, arg=(buy_to_size - current_size) / profile_size)
        else:
            raise ValueError(f"{group.name}: invalid purchase method {purchase_method} for 'BUY_SELL'.")
    elif current_size > upper_size:
        direct = group.desire_direct.above_range
        if direct == AssetTradeDirection.TO_LOWER_WEIGHT:
            sell_to_size = lower_size
        elif direct == AssetTradeDirection.TO_TARGET_WEIGHT:
            sell_to_size = target_size
        elif direct == AssetTradeDirection.TO_UPPER_WEIGHT:
            sell_to_size = upper_size
        else:
            warnings.warn(f"strategy 'BUY_SELL', above range desire: {direct.name}, can't sell any amount.")
            return TradeOrder(action=BuySellAction.NOOP)
        return TradeOrder(action=BuySellAction.SELL, arg=(current_size - sell_to_size) / current_size)
    else:  # lower_size <= current_size <= upper_size
        direct = group.desire_direct.within_range
        if direct == AssetTradeDirection.NO_ACTION:
            return TradeOrder(action=BuySellAction.NOOP)
        elif direct == AssetTradeDirection.TO_LOWER_WEIGHT:
            return TradeOrder(action=BuySellAction.SELL, arg=(current_size - lower_size) / current_size)
        elif direct == AssetTradeDirection.TO_TARGET_WEIGHT:
            if current_size < target_size:
                if purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                    return TradeOrder(action=BuySellAction.BUY_EXIST,
                                      arg=(target_size - current_size) / current_size)
                elif purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                    return TradeOrder(action=BuySellAction.BUY_PROFILE,
                                      arg=(target_size - current_size) / profile_size)
                else:
                    raise ValueError(f"{group.name}: invalid purchase method {purchase_method} for 'BUY_SELL'.")
            else:
                return TradeOrder(action=BuySellAction.SELL, arg=(current_size - target_size) / current_size)
        elif direct == AssetTradeDirection.TO_UPPER_WEIGHT:
            if purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                return TradeOrder(action=BuySellAction.BUY_EXIST, arg=(upper_size - current_size) / current_size)
            elif purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                return TradeOrder(action=BuySellAction.BUY_PROFILE, arg=(upper_size - current_size) / profile_size)
            else:
                raise ValueError(f"{group.name}: invalid purchase method {purchase_method} for 'BUY_SELL'.")
        else:
            warnings.warn(f"strategy 'BUY_SELL', within range desire: {direct.name}, can't buy/sell any amount.")
            return TradeOrder(action=BuySellAction.NOOP)


def _make_sorted_alloc_groups(
    *,
    exist_assets: Connector[Asset] | list[Asset],
    profile_assets: list[Asset] | None,
    total_size: float,
    asset_size_basis: str,
    allocation_group_attr_name: str,
    allocation_group_params: dict[str, AssetAllocationParams],
    require_validate: bool
) -> list[_AllocationGroup]:
    ag_kwargs: dict[str, dict[str, ...]] = {}
    for name, params in allocation_group_params.items():
        if name in ag_kwargs:
            raise KeyError(f"Duplicate allocation group name: '{name}'.")
        ag_kwargs[name] = {
            "strategy": params.strategy,
            "weight_range": params.weight_range,
            "desire_direct": params.desire_direct,
            "purchase_method": params.purchase_method,
            "priority": params.priority,
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

    groups: list[_AllocationGroup] = [_AllocationGroup(name=key, **val) for key, val in ag_kwargs.items()]
    groups.sort(key=lambda x: x.priority)

    if require_validate:
        seen_unconstrained = []
        for i, ag in enumerate(groups):
            if ag.strategy is AssetAllocationStrategy.UNCONSTRAINED:
                seen_unconstrained.append(i)
                if ag.purchase_method != AssetPurchaseMethod.SCALE_UP_EXISTING:
                    warnings.warn(f"'{ag.name}': 'UNCONSTRAINED'; 'purchase_method' is '{ag.purchase_method.name}' "
                                  f"but 'SCALE_UP_EXISTING' will be forcibly used.")
        if len(seen_unconstrained) == 0:
            pass
        else:
            if seen_unconstrained[-1] + 1 != len(groups):
                raise ValueError(f"Last allocation group must be 'UNCONSTRAINED'.")
            if seen_unconstrained[-1] - seen_unconstrained[0] + 1 != len(seen_unconstrained):
                raise ValueError(f"All 'UNCONSTRAINED' allocation groups must be after other groups.")

    return groups


def _process_unconstrained_groups(groups: list[_AllocationGroup], size_gap: float, tolerance: float) -> None:
    if abs(size_gap) <= tolerance:
        return

    if len(groups) == 0:
        warnings.warn(f"Target allocation is not met but no unconstrained allocation group to process.")
        return

    current_size: float = sum([ag.current_size for ag in groups])

    if current_size == 0:
        # This would be very edge case when total unconstrained allocation group (usually cash) balance happens to be zero.
        # Utilize cash asset to safely proceed
        _fallback_cash = None
        for asset in [ag.exist_conn for ag in groups]:
            if isinstance(asset, Cash):
                _fallback_cash = asset
                break
        if _fallback_cash is not None:
            _fallback_cash.invest_new_money(size_gap)
        else:
            warnings.warn(f"No cash asset in 'UNCONSTRAINED' group is available for sclaing.")
        return

    # scale unconstrained allocation_group
    if size_gap > 0:
        trade_order = TradeOrder(action=BuySellAction.BUY_EXIST, arg=size_gap / current_size)
    elif size_gap < 0:
        trade_order = TradeOrder(action=BuySellAction.SELL, arg=- size_gap / current_size)
    else:
        raise ValueError("Should never get here.")

    for ag in groups:
        ag.execute_trade(trade_order)


@time_synchronized("model_engine.time_synchronizer")
class Rebalancer:
    """Manages asset allocation and rebalancing for a fund.

    Groups assets by allocation group and applies rebalance policy to sell/buy
    or scale exposure to meet target allocations with tolerances.

    """
    time: int          # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period  # for type hint only, will be injected by decorator `time_synchronized`

    __slots__ = ('__weakref__', 'time', 'period', 'name', 'group_names',
                 'tdv_fund_size', 'tdv_ag_size_bd', 'tdv_ag_size_ad', 'tdv_ag_wgt_pc_bd', 'tdv_ag_wgt_pc_ad',)

    def __init__(
        self,
        name: str | None = None,
        *,
        model_engine: ProjModelEngine | None = None,
        allocation_group_names: list[str]
    ):
        self.name: str = name or str(uuid.uuid4())
        self.group_names: tuple[str, ...] = tuple(allocation_group_names)

        tdv_kwargs = {"model_engine": model_engine, "owner": self.name, "group": 'rebalance'}
        self.tdv_fund_size = make_proj_variable("fund_size", **tdv_kwargs)
        tdv_kwargs = {"model_engine": model_engine, "owner": self.name, "group": 'rebalance', 'dims': [self.group_names]}
        self.tdv_ag_size_bd = make_proj_variable("size_bd", **tdv_kwargs)
        self.tdv_ag_size_ad = make_proj_variable("size_ad", **tdv_kwargs)
        self.tdv_ag_wgt_pc_bd = make_proj_variable("weight_pc_bd", **tdv_kwargs)
        self.tdv_ag_wgt_pc_ad = make_proj_variable("weight_pc_ad", **tdv_kwargs)

    def __call__(
        self,
        exist_assets: Connector[Asset] | list[Asset],
        *,
        free_proceeds_keeper: ProceedsKeeper | None = None,
        profile_assets: list[Asset] | None = None,
        total_size: float,
        asset_size_basis: str,
        allocation_group_attr_name: str = "allocation_group",
        allocation_group_params: dict[str, AssetAllocationParams],
        min_size_tol: float = 0.0001,
        on_negative_size: Literal["zero", "pass", "as_is"] = "zero",
        check_level: CheckLevel = GLOBAL_CHECK_LEVEL,
        **kwargs
    ) -> float:
        """Rebalance assets in the fund to match the target allocation.

        Args:
            exist_assets (Connector[Asset] | list[Asset]): Connector of list of assets to be rebalanced.
            free_proceeds_keeper (ProceedsKeeper | None): Free proceeds' keeper.
            profile_assets (list[Asset] | None): Profile assets for reference.
            total_size (float): Total size for allocation, shall correspond to `asset_size_basis`.
            asset_size_basis (str): Asset reporting basis for sizing (e.g. FAV or BSV).
            allocation_group_attr_name (str): Named attribute for asset allocation group, defaults to "allocation_group".
            allocation_group_params (dict[str, AssetAllocationGroupParams]): Parameters for each allocation group.
            min_size_tol (float): Tolerance of minimum size, normally to prevent ZeroDivisionError; defaults to 0.0001.
            on_negative_size (Literal["zero", "pass", "as_is"]): When size is negative; defaults to "zero".
            check_level (CheckLevel): Check level for validation against allocation, defaults to `GLOBAL_CHECK_LEVEL`.

        Returns:
            float: Free proceeds after the reblancing: proceeds received from disposal (sell), minus proceeds spent
                to purchase (buy); if a `free_proceeds_keeper` provided, return 0.0.

        """
        t, p = self.time, self.period
        if total_size < 0.0:
            if on_negative_size == "pass":
                return 0.0
            elif on_negative_size == "as_is":
                pass
            else:  # on_negative_size == "zero":
                warnings.warn(f'{p} {self.name}: negative fund size ({total_size:.2f}) will be treated as zero, and '
                              'all existing assets will be sold to reblance.')
                total_size = max(total_size, min_size_tol)
        total_size = -max(total_size, min_size_tol) if total_size < 0.0 else max(total_size, min_size_tol)  # to prevent ZeroDivisionError

        groups: list[_AllocationGroup] = _make_sorted_alloc_groups(
            exist_assets=exist_assets,
            profile_assets=profile_assets,
            total_size=total_size,
            asset_size_basis=asset_size_basis,
            allocation_group_attr_name=allocation_group_attr_name,
            allocation_group_params=allocation_group_params,
            require_validate=(check_level != CheckLevel.BYPASS),
        )

        if set([ag.name for ag in groups]) != set(self.group_names):
            warnings.warn(f"Keys in 'allocation_group_params' don't match 'allocation_group_names'.")

        tdv_index: list[int] = [len(self.group_names) + 1] * len(self.group_names)
        for i, ag in enumerate(groups):
            tdv_index[self.group_names.index(ag.name)] = i

        self.tdv_fund_size[t] = total_size
        current_sizes = [groups[i].current_size for i in tdv_index]
        self.tdv_ag_size_bd[t] = np.array(current_sizes)
        self.tdv_ag_wgt_pc_bd[t] = np.array([v / total_size for v in current_sizes]) * 100

        _rebalance_alloc_groups(groups=groups, total_size=total_size, check_level=check_level)

        _free_proceeds_keeper = free_proceeds_keeper or ProceedsKeeper()
        for ag in groups:
            exist_assets.extend(ag.staged_conn)
            _free_proceeds_keeper += ag.take_out_free_proceeds()

        current_sizes = [groups[i].current_size for i in tdv_index]
        self.tdv_ag_size_ad[t] = np.array(current_sizes)
        self.tdv_ag_wgt_pc_ad[t] = np.array([v / total_size for v in current_sizes]) * 100

        return 0.0 if free_proceeds_keeper is not None else _free_proceeds_keeper.take_out()

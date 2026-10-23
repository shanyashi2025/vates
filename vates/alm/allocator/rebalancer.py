import numpy as np
import pandas as pd
import uuid
import warnings
from typing import Literal, Any

from vates._core import ProjModelEngine, time_synchronized, make_proj_variable
from vates.global_conf import CheckLevel, GLOBAL_CHECK_LEVEL
from vates.alm.assets import Asset, Cash
from vates.alm._utils import Connector
from vates.alm.allocator._allocation_group import _AllocationGroup, AssetAllocationParams, TradeOrder, BuySellAction
from vates.alm.allocator._enums import AssetAllocationStrategy, AssetPurchaseMethod
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
    for ag in groups[:-1]:
        if ag.strategy is AssetAllocationStrategy.UNCONSTRAIN:
            raise ValueError(f"'UNCONSTRAIN' allocation groups must be the last.")
    has_uc_group: bool = groups[-1].strategy is AssetAllocationStrategy.UNCONSTRAIN

    if has_uc_group:  # has unconstrained group
        # --- step 1: process non-unconstrained groups in priority ---
        for ag in groups:
            if ag.strategy != AssetAllocationStrategy.UNCONSTRAIN:
                trade_order = _make_trade_order_on_ask(ag)
                ag.execute_trade(trade_order, on_exception_raise=True)
        # --- step 2: process unconstrained groups ---
        _process_unconstrain_group(group=groups[-1], fill_size=total_size - sum(ag.current_size for ag in groups),
                                   tolerance=max(abs(total_size * 1e-6), 0.01))
    else:  # best effort
        trade_orders = _make_trade_orders_on_best_effort(groups=groups, total_size=total_size)
        for ag, trade_order in zip(groups, trade_orders):
            ag.execute_trade(trade_order, on_exception_raise=True)

    # --- validate if target allocations is satisfied ---
    for ag in groups:
        satisfied, ref_sizes = ag.is_allocation_satisfied()
        if not satisfied:
            msg = (f"{ag.name}: allocation is not satisfied; strategy: {ag.strategy.name}; "
                   f"purchase method: {ag.purchase_method.name}.")
            if abs(total_size) > 1e-8:
                curr, lower, upper = ag.current_size/total_size, ref_sizes[0]/total_size, ref_sizes[1]/total_size
                msg += f" actual: {ag.current_size/total_size:.4%}; (lower, upper): ({lower:.4%}, {upper:.4%})"
            if check_level == CheckLevel.ERROR:
                raise ValueError(msg)
            warnings.warn(msg)


def _make_trade_order_on_ask(group: _AllocationGroup) -> TradeOrder:
    strategy = group.strategy
    exist_size = group.exist_size
    ask_size = group.ask_size
    if ask_size is None:
        raise ValueError(f"'{group.name}': 'ask_size' is None.")
    if strategy == AssetAllocationStrategy.NO_TRADE:
        if abs(ask_size - exist_size) > 1e-8:
            raise ValueError(f"'{group.name}': strategy 'NO_TRADE' cannot buy/sell; ask_size: {ask_size:.2f} != "
                             f"exist_size: {exist_size:.2f}.")
        return TradeOrder(action=BuySellAction.PASS)
    if strategy == AssetAllocationStrategy.BUY_HOLD:
        if ask_size < exist_size and abs(ask_size - exist_size) > 1e-8:
            raise ValueError(f"'{group.name}': strategy 'BUY_HOLD' cannot sell; ask_size: {ask_size:.2f} < "
                             f"exist_size: {exist_size:.2f}.")
    if ask_size > group.current_size:  # buy
        if group.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING and abs(group.exist_size) < 1e-8:
            return TradeOrder(action=BuySellAction.PASS)
        if group.purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE and abs(group.profile_size) < 1e-8:
            return TradeOrder(action=BuySellAction.PASS)
    return _make_trade_order_on_require_size(group=group, require_size=ask_size)


def _make_trade_orders_on_best_effort(groups: list[_AllocationGroup], total_size: float) -> list[TradeOrder]:
    require_sizes = []
    size_corridors = []
    for g in groups:
        strategy = g.strategy
        exist_size = g.exist_size
        ask_size = g.ask_size
        lower, _, upper = g.size_box

        if strategy == AssetAllocationStrategy.UNCONSTRAIN:
            raise ValueError(f"'{g.name}': unexpected 'UNCONSTRAIN'.")

        if strategy == AssetAllocationStrategy.NO_TRADE:
            require_sizes.append(exist_size)
            size_corridors.append((exist_size, exist_size))
        elif strategy == AssetAllocationStrategy.BUY_HOLD:
            if ask_size <= exist_size:  # to sell but not allowed
                require_sizes.append(exist_size)
                size_corridors.append((max(lower, exist_size), max(upper, exist_size)))
            else:  # to buy
                if g.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING and abs(exist_size) < 1e-8:
                    require_sizes.append(exist_size)
                    size_corridors.append((exist_size, exist_size))
                else:
                    require_sizes.append(ask_size)
                    size_corridors.append((max(lower, exist_size), max(upper, exist_size)))
        elif strategy == AssetAllocationStrategy.BUY_SELL:
            if ask_size <= exist_size:  # to sell
                require_sizes.append(ask_size)
                size_corridors.append((lower, upper))
            else:  # to buy
                if g.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING and abs(exist_size) < 1e-8:
                    require_sizes.append(exist_size)
                    size_corridors.append((exist_size, exist_size))
                else:
                    require_sizes.append(ask_size)
                    size_corridors.append((lower, upper))
        else:
            raise ValueError(f"Should never get here {strategy.name=}.")

    for i in range(1, len(groups) + 1):
        size_excl_self = sum(require_sizes) - require_sizes[-i]
        lower, upper = size_corridors[-i]
        if size_excl_self + lower > total_size:
            require_sizes[-i] = lower
        elif size_excl_self + upper < total_size:
            require_sizes[-i] = upper
        else:  # lower <= total_size - front_size <= upper
            require_sizes[-i] = total_size - size_excl_self
            break  # size_excl_self + trial_sizes[-i] == total_size

    return [_make_trade_order_on_require_size(group=g, require_size=s) for g, s in zip(groups, require_sizes)]

def _make_trade_order_on_require_size(group: _AllocationGroup, require_size: float) -> TradeOrder:

    strategy = group.strategy
    purchase_method = group.purchase_method
    current_size, exist_size, profile_size = group.current_size, group.exist_size, group.profile_size
    if strategy == AssetAllocationStrategy.NO_TRADE:
        if abs(require_size - exist_size) > 1e-8:
            warnings.warn(f"'{group.name}': strategy 'NO_TRADE', can not rebalance to another size; "
                          f"require={require_size:,.2f}; exist={exist_size:,.2f}.")
            return TradeOrder(action=BuySellAction.EXCEPTION)
        return TradeOrder(action=BuySellAction.PASS)

    def _make_buy_order(_name: str, _purchase_method: AssetPurchaseMethod, _require_size: float, _current_size: float,
                        _exist_size: float, _profile_size: float) -> TradeOrder:
        if _purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
            if abs(_exist_size) < 1e-8:
                warnings.warn(f"'{_name}': 'SCALE_UP_EXISTING', exist_size = 0.")
                return TradeOrder(action=BuySellAction.EXCEPTION)
            return TradeOrder(action=BuySellAction.BUY_EXIST, arg=(_require_size - _current_size) / _exist_size)
        elif _purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
            if abs(_profile_size) < 1e-8:
                warnings.warn(f"'{_name}': 'PURCHASE_PROFILE', profile_size = 0.")
                return TradeOrder(action=BuySellAction.EXCEPTION)
            return TradeOrder(action=BuySellAction.BUY_PROFILE, arg=(_require_size - _current_size) / _profile_size)
        else:
            warnings.warn(f"'{_name}': invalid purchase method '{_purchase_method}'.")
            return TradeOrder(action=BuySellAction.EXCEPTION)

    if strategy == AssetAllocationStrategy.BUY_HOLD:
        if require_size <= exist_size:
            if abs(require_size - exist_size) > 1e-8:
                warnings.warn(f"'{group.name}': strategy 'BUY_HOLD', can not sell; "
                              f"require={require_size:,.2f}; exist={exist_size:,.2f}.")
                return TradeOrder(action=BuySellAction.EXCEPTION)
            return TradeOrder(action=BuySellAction.PASS)
        return _make_buy_order(_name=group.name, _purchase_method=purchase_method, _require_size=require_size,
                               _current_size=current_size, _exist_size=exist_size, _profile_size=profile_size)
    elif strategy == AssetAllocationStrategy.BUY_SELL:
        if require_size == current_size:
            return TradeOrder(action=BuySellAction.PASS)
        elif require_size < current_size:
            return TradeOrder(action=BuySellAction.SELL, arg=(current_size - require_size) / exist_size)
        else:
            return _make_buy_order(_name=group.name, _purchase_method=purchase_method, _require_size=require_size,
                                   _current_size=current_size, _exist_size=exist_size, _profile_size=profile_size)
    else:
        warnings.warn(f"{group.name}: invalid allocation strategy: {strategy}")
        return TradeOrder(action=BuySellAction.EXCEPTION)


def _make_sorted_alloc_groups(
    *,
    exist_assets: Connector[Asset] | list[Asset],
    profile_assets: list[Asset] | None,
    total_size: float,
    asset_size_basis: str,
    allocation_group_attr_name: str,
    allocation_group_params: dict[str, AssetAllocationParams],
) -> list[_AllocationGroup]:
    ag_kwargs: dict[str, dict[str, Any]] = {}
    for name, params in allocation_group_params.items():
        if name in ag_kwargs:
            raise KeyError(f"Duplicate allocation group name: '{name}'.")
        ag_kwargs[name] = {
            "strategy": params.strategy,
            "weight_box": params.weight_box,
            "ask_rebalance_outcome": params.ask_rebalance_outcome,
            "purchase_method": params.purchase_method,
            "priority": params.priority,
            "total_size": total_size,
            "size_basis": asset_size_basis,
            "exist_assets": [],
            "profile_assets": [],
        }

    for asset in exist_assets:
        name: str | None = getattr(asset, allocation_group_attr_name, None)
        if name is not None and name in ag_kwargs:
            ag_kwargs[name]["exist_assets"].append(asset)

    for asset in profile_assets or []:
        name: str | None = getattr(asset, allocation_group_attr_name, None)
        if name is not None and name in ag_kwargs:
            ag_kwargs[name]["profile_assets"].append(asset)

    groups: list[_AllocationGroup] = [_AllocationGroup(name=key, **val) for key, val in ag_kwargs.items()]
    groups.sort(key=lambda x: x.priority)

    return groups


def _process_unconstrain_group(group: _AllocationGroup, fill_size: float, tolerance: float) -> None:
    if group.strategy != AssetAllocationStrategy.UNCONSTRAIN:
        raise ValueError(f"'{group.name}': invalid strategy '{group.strategy}', expected 'UNCONSTRAIN'.")

    if abs(fill_size) <= tolerance:
        return

    if fill_size > 0 and group.purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
        trade_order = TradeOrder(action=BuySellAction.BUY_PROFILE, arg=fill_size / group.profile_size)
    elif group.exist_size == 0:
        # This would be very edge case when unconstrained group (usually cash) total balance happens to be zero.
        # Utilize cash asset to safely proceed
        _fallback_cash: Cash | None = None
        for asset in group.exist_conn:
            if isinstance(asset, Cash) and asset.is_alive:
                _fallback_cash = asset
                break
        if _fallback_cash is not None:
            _fallback_cash.buy_amount(fill_size)
        else:
            warnings.warn(f"No cash asset in 'UNCONSTRAIN' group is available for scaling.")
        return  # early return
    elif fill_size > 0:  # buy
        if group.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
            trade_order = TradeOrder(action=BuySellAction.BUY_EXIST, arg=fill_size / group.exist_size)
        else:  # PURCHASE_PROFILE is processed above
            raise ValueError(f"{group.name}: invalid purchase method {group.purchase_method} for 'UNCONSTRAIN'.")
    else:  # sell
        trade_order = TradeOrder(action=BuySellAction.SELL, arg=-fill_size / group.exist_size)
    group.execute_trade(trade_order, on_exception_raise=True)


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
            float: Free proceeds after the rebalancing: proceeds received from disposal (sell), minus proceeds spent
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
                              'all existing assets will be sold to rebalance.')
                total_size = max(total_size, min_size_tol)
        total_size = -max(total_size, min_size_tol) if total_size < 0.0 else max(total_size, min_size_tol)  # to prevent ZeroDivisionError

        groups: list[_AllocationGroup] = _make_sorted_alloc_groups(
            exist_assets=exist_assets,
            profile_assets=profile_assets,
            total_size=total_size,
            asset_size_basis=asset_size_basis,
            allocation_group_attr_name=allocation_group_attr_name,
            allocation_group_params=allocation_group_params,
        )

        self.tdv_fund_size[t] = total_size
        tdv_dim_indices: list[int | None] = self._make_tdv_dim_indices(groups=groups)
        sizes = [groups[i].current_size if i is not None else 0.0 for i in tdv_dim_indices]
        self.tdv_ag_size_bd[t] = np.array(sizes)
        self.tdv_ag_wgt_pc_bd[t] = np.array([v / total_size for v in sizes]) * 100

        _rebalance_alloc_groups(groups=groups, total_size=total_size, check_level=check_level)

        _free_proceeds_keeper = free_proceeds_keeper or ProceedsKeeper()
        for ag in groups:
            exist_assets.extend(ag.staged_conn)
            _free_proceeds_keeper += ag.take_out_free_proceeds()

        sizes = [groups[i].current_size if i is not None else 0.0 for i in tdv_dim_indices]
        self.tdv_ag_size_ad[t] = np.array(sizes)
        self.tdv_ag_wgt_pc_ad[t] = np.array([v / total_size for v in sizes]) * 100

        return 0.0 if free_proceeds_keeper is not None else _free_proceeds_keeper.take_out()

    def _make_tdv_dim_indices(self, groups: list[_AllocationGroup]) -> list[int | None]:
        tdv_dim_indices: list[int | None] = [None] * len(self.group_names)
        for i, ag in enumerate(groups):
            name = ag.name
            if name in self.group_names:
                tdv_dim_indices[self.group_names.index(name)] = i
            else:
                warnings.warn(f"Key '{name}' in 'allocation_group_params' doesn't match 'allocation_group_names'.")
        return tdv_dim_indices

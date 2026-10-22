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
    ag = groups[-1]
    uc_group: _AllocationGroup | None = ag if ag.strategy is AssetAllocationStrategy.UNCONSTRAIN else None

    if uc_group is not None:  # has unconstrained group
        # --- step 1: process non-unconstrained groups in priority ---
        for ag in groups:
            if ag.strategy != AssetAllocationStrategy.UNCONSTRAIN:
                trade_order = _make_trade_order_on_primary_objective(ag)
                ag.execute_trade(trade_order)
        # --- step 2: process unconstrained groups ---
        size_gap = total_size - sum(ag.current_size for ag in groups)
        _process_unconstrain_group(group=uc_group, incremental_size=size_gap, tolerance=max(abs(total_size * 1e-6), 0.01))
    else:  # best effort
        trade_orders = _make_trade_orders_on_best_effort(groups=groups, total_size=total_size)
        for ag, trade_order in zip(groups, trade_orders):
            ag.execute_trade(trade_order)

    # --- validate if target allocations is satisfied ---
    for ag in groups:
        if not ag.is_allocation_satisfied():
            weight_range = ag.weight_range
            lower, _, upper = weight_range if weight_range is not None else (None, None, None)
            msg = (
                f"{ag.name}: allocation is not satisfied, {ag.current_size/total_size:.4%}; strategy: {ag.strategy.name}; "
                f"(lower, upper): ({lower:.4%}, {upper:.4%}); purchase method: {ag.purchase_method.name}.")
            if check_level == CheckLevel.ERROR:
                raise ValueError(msg)
            warnings.warn(msg)


def _make_trade_order_on_primary_objective(group: _AllocationGroup) -> TradeOrder:
    strategy = group.strategy
    if strategy == AssetAllocationStrategy.NO_TRADE:
        return TradeOrder(action=BuySellAction.PASS)
    else:
        prim_obj_size = group.primary_objective_size
        if prim_obj_size is None:
            raise ValueError(f"'{group.name}': 'primary_objective_size' is None.")
        return _make_trade_order_on_require(group=group, require_size=prim_obj_size)


def _make_trade_orders_on_best_effort(groups: list[_AllocationGroup], total_size: float) -> list[TradeOrder]:
    require_sizes = []
    size_corridors = []
    for g in groups:
        strategy = g.strategy
        exist_size = g.exist_size
        prim_obj_size = g.primary_objective_size
        lower, _, upper = g.size_range

        if strategy == AssetAllocationStrategy.UNCONSTRAIN:
            raise ValueError(f"'{g.name}': unexpected 'UNCONSTRAIN'.")

        if strategy == AssetAllocationStrategy.NO_TRADE:
            require_sizes.append(exist_size)
            size_corridors.append((exist_size, exist_size))
        elif strategy == AssetAllocationStrategy.BUY_HOLD:
            if prim_obj_size <= exist_size:  # to sell but not allowed
                require_sizes.append(exist_size)
                size_corridors.append((max(lower, exist_size), max(upper, exist_size)))
            else:  # to buy
                if g.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING and abs(exist_size) < 1e-8:
                    require_sizes.append(exist_size)
                    size_corridors.append((exist_size, exist_size))
                else:
                    require_sizes.append(prim_obj_size)
                    size_corridors.append((max(lower, exist_size), max(upper, exist_size)))
        elif strategy == AssetAllocationStrategy.BUY_SELL:
            if prim_obj_size <= exist_size:  # to sell
                require_sizes.append(prim_obj_size)
                size_corridors.append((lower, upper))
            else:  # to buy
                if g.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING and abs(exist_size) < 1e-8:
                    require_sizes.append(exist_size)
                    size_corridors.append((exist_size, exist_size))
                else:
                    require_sizes.append(prim_obj_size)
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

    return [_make_trade_order_on_require(group=g, require_size=s) for g, s in zip(groups, require_sizes)]

def _make_trade_order_on_require(group: _AllocationGroup, require_size: float) -> TradeOrder:

    strategy = group.strategy
    purchase_method = group.purchase_method
    current_size, exist_size, profile_size = group.current_size, group.exist_size, group.profile_size
    if strategy == AssetAllocationStrategy.NO_TRADE:
        if abs(require_size - current_size) > 0.001:
            warnings.warn(f"'{group.name}': strategy 'NO_TRADE', can not rebalance to another size; "
                          f"require={require_size:,.2f}; current={current_size:,.2f}.")
        return TradeOrder(action=BuySellAction.PASS)

    def _make_buy_order(_purchase_method: AssetPurchaseMethod, _require_size: float, _current_size: float,
                        _exist_size: float, _profile_size: float) -> TradeOrder | None:
        if _purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
            return TradeOrder(action=BuySellAction.BUY_EXIST, arg=(_require_size - _current_size) / _exist_size)
        elif _purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
            return TradeOrder(action=BuySellAction.BUY_PROFILE, arg=(_require_size - _current_size) / _profile_size)
        else:
            return None

    if strategy == AssetAllocationStrategy.BUY_HOLD:
        if require_size <= current_size:
            if abs(require_size - current_size) > 0.001:
                warnings.warn(f"'{group.name}': strategy 'BUY_HOLD', can not sell; "
                              f"require={require_size:,.2f}; current={current_size:,.2f}.")
            return TradeOrder(action=BuySellAction.PASS)
        else:
            trade_order = _make_buy_order(_purchase_method=purchase_method, _require_size=require_size,
                                          _current_size=current_size, _exist_size=exist_size, _profile_size=profile_size)
            if trade_order is None:
                raise ValueError(f"{group.name}: invalid purchase method {purchase_method} for 'BUY_HOLD'.")
            return trade_order
    elif strategy == AssetAllocationStrategy.BUY_SELL:
        if require_size == current_size:
            return TradeOrder(action=BuySellAction.PASS)
        elif require_size < current_size:
            return TradeOrder(action=BuySellAction.SELL, arg=(current_size - require_size) / exist_size)
        else:
            trade_order = _make_buy_order(_purchase_method=purchase_method, _require_size=require_size,
                                          _current_size=current_size, _exist_size=exist_size, _profile_size=profile_size)
            if trade_order is None:
                raise ValueError(f"{group.name}: invalid purchase method {purchase_method} for 'BUY_SELL'.")
            return trade_order
    else:
        raise ValueError(f"{group.name}: invalid allocation strategy: {strategy}")


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
            "weight_range": params.weight_range,
            "primary_objective": params.primary_objective,
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


def _process_unconstrain_group(group: _AllocationGroup, incremental_size: float, tolerance: float) -> None:
    if abs(incremental_size) <= tolerance:
        return

    if group.strategy != AssetAllocationStrategy.UNCONSTRAIN:
        raise ValueError(f"'{group.name}': invalid strategy '{group.strategy}', expected 'UNCONSTRAIN'.")

    if group.exist_size == 0:
        # This would be very edge case when unconstrained group (usually cash) total balance happens to be zero.
        # Utilize cash asset to safely proceed
        _fallback_cash: Cash | None = None
        for asset in group.exist_conn:
            if isinstance(asset, Cash) and asset.is_alive:
                _fallback_cash = asset
                break
        if _fallback_cash is not None:
            _fallback_cash.invest_new_money(incremental_size)
        else:
            warnings.warn(f"No cash asset in 'UNCONSTRAIN' group is available for scaling.")
        return

    ratio = incremental_size / group.exist_size

    if ratio > 0:  # buy
        if group.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
            trade_order = TradeOrder(action=BuySellAction.BUY_EXIST, arg=ratio)
        elif group.purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
            trade_order = TradeOrder(action=BuySellAction.BUY_PROFILE, arg=group.exist_size * ratio / group.profile_size)
        else:
            raise ValueError(f"{group.name}: invalid purchase method {group.purchase_method} for 'UNCONSTRAIN'.")
    else:  # sell
        trade_order = TradeOrder(action=BuySellAction.SELL, arg=-ratio)
    group.execute_trade(trade_order)


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

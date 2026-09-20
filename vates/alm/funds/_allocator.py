import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Self

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDimVariable
from vates.global_conf import CheckLevel
from vates.alm.assets import Asset, Cash
from vates.alm.enums import AssetBuySellApproach, AssetPurchaseMethod
from vates.alm.funds._connector import AssetLiabConnector


@dataclass(slots=True)
class TargetWeight:
    """Target allocation weights for an allocation group.

    Attributes:
        target (float): Target weight for the allocation group.
        floor (float): Floor of allowed weight.
        cap (float): Cap of allowed weight.
    """
    target: float
    floor: float | None = None
    cap: float | None = None

    def __post_init__(self):
        if self.floor is None:
            self.floor = self.target
        if self.cap is None:
            self.cap = self.target
        if self.floor > self.target:
            raise ValueError(f"floor: {self.floor:4f} > target: {self.target:.4f}.")
        if self.cap < self.target:
            raise ValueError(f"cap: {self.cap:4f} < target: {self.target:.4f}.")


@dataclass(slots=True, frozen=True)
class TargetSize:
    target: float
    floor: float
    cap: float

    def __post_init__(self):
        if self.floor > self.target:
            raise ValueError(f"floor: {self.floor:4f} > target: {self.target:.4f}.")
        if self.cap < self.target:
            raise ValueError(f"cap: {self.cap:4f} < target: {self.target:.4f}.")

    @classmethod
    def from_target_weight(cls, total_size: float, target_weight: TargetWeight) -> Self:
        return TargetSize(
            target=total_size * target_weight.target,
            floor =total_size * target_weight.floor,
            cap=total_size * target_weight.cap
        )


@unique
class BuySellAction(Enum):
    """Enum for asset buy/sell action."""
    NO_ACTION = auto()
    BUY_SCALE_EXIST = auto()
    BUY_PROFILE = auto()
    SELL = auto()


@dataclass(slots=True, frozen=True)
class TradeOrder:
    action: BuySellAction
    propn: float = 0


class AssetAllocationGroup:
    """Asset allocation group.

    Attributes:
        _name (str): Name of the allocation group.
        _sequence (int): The sequence in which the allocation group will be processed, must be unique and consecutive
            integer starting from 1.
        _buysell_approach (AssetBuySellApproach): Buy/sell approach for this allocation group.
        _purchase_method (AssetPurchaseMethod): Purchase method for this allocation group.
    """

    __slots__ = ("_name", "_sequence", "_buysell_approach", "_purchase_method", "_size_basis", "_target_weight",
                 "_exist_conn", "_profile_conn", "_staged_conn", "_target_size", "_profile_size", "trade_decision")

    def __init__(self, *, name: str, sequence: int, buysell_approach: AssetBuySellApproach | str,
                 purchase_method: AssetPurchaseMethod | str,
                 size_basis: str | None = None, target_weight: TargetWeight | None = None):
        self._name: str = name
        self._sequence: int = sequence
        self._buysell_approach: AssetBuySellApproach = AssetBuySellApproach[buysell_approach.upper()] \
            if isinstance(buysell_approach, str) else buysell_approach
        self._purchase_method: AssetPurchaseMethod = AssetPurchaseMethod[purchase_method.upper()] \
            if isinstance(purchase_method, str) else purchase_method
        self._size_basis: str = size_basis
        self._target_weight: TargetWeight | None = target_weight

        self._exist_conn: AssetLiabConnector | None = None
        self._profile_conn: AssetLiabConnector | None = None
        self._staged_conn: AssetLiabConnector = AssetLiabConnector()
        self._target_size: TargetSize | None = None
        self._profile_size: float | None = None
        self.trade_decision: TradeOrder | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def buysell_approach(self) -> AssetBuySellApproach:
        return self._buysell_approach

    @property
    def purchase_method(self) -> AssetPurchaseMethod:
        return self._purchase_method

    @property
    def size_basis(self) -> str:
        return self._size_basis

    @size_basis.setter
    def size_basis(self, val: str, /):
        if not isinstance(val, str):
            raise TypeError(f"Invalid type of size_basis: {type(val)}, expected 'str'.")
        self._size_basis = val

    @property
    def exist_conn(self) -> AssetLiabConnector:
        return self._exist_conn

    @exist_conn.setter
    def exist_conn(self, conn: AssetLiabConnector, /) -> None:
        if not isinstance(conn, AssetLiabConnector):
            raise TypeError(f"Invalid type of connector: {type(conn)}, expected 'AssetLiabConnector'.")
        self._exist_conn = conn

    @property
    def profile_conn(self) -> AssetLiabConnector:
        return self._profile_conn

    @profile_conn.setter
    def profile_conn(self, conn: AssetLiabConnector, /) -> None:
        if not isinstance(conn, AssetLiabConnector):
            raise TypeError(f"Invalid type of connector: {type(conn)}, expected 'AssetLiabConnector'.")
        self._profile_conn = conn
        self._maybe_refresh_profile_size()

    @property
    def staged_conn(self) -> AssetLiabConnector:
        return self._staged_conn

    @property
    def current_size(self) -> float:
        return self.exist_size + self.staged_size

    @property
    def exist_size(self) -> float:
        if self._size_basis is None:
            warnings.warn(f"size_basis is None, cann't compute size.")
            return 0.0
        else:
            return self._exist_conn.sum_asset(self._size_basis)

    @property
    def staged_size(self) -> float:
        if self._size_basis is None:
            warnings.warn(f"size_basis is None, cann't compute size.")
            return 0.0
        else:
            return self._staged_conn.sum_asset(self._size_basis)

    @property
    def profile_size(self) -> float:
        return self._profile_size or 0.0

    def _maybe_refresh_profile_size(self):
        if self._size_basis is None:
            warnings.warn(f"size_basis is None, cann't compute size.")
            self._profile_size = None
        elif self._profile_conn is None:
            warnings.warn(f"profile_conn is None, cann't refresh exist size.")
            self._profile_size = None
        else:
            self._profile_size = self._profile_conn.sum_asset(self._size_basis)

    @property
    def target_weight(self) -> TargetWeight:
        return self._target_weight

    @property
    def target_size(self) -> TargetSize:
        return self._target_size

    def update_target_size(self, total_size: float, new_target_weight: TargetWeight | None = None) -> None:
        if new_target_weight is not None:
            self._target_weight = new_target_weight
        if self._target_weight is not None:
            self._target_size = TargetSize.from_target_weight(total_size=total_size, target_weight=self._target_weight)
        else:
            self._target_size = None

    def make_trade_decision(self) -> None:
        """Make the trade decision for an allocation group.

        Raises:
            ValueError: If an invalid asset buy/sell approach is provided.
        """
        if self._buysell_approach == AssetBuySellApproach.NO_TRADE:
            self.trade_decision = TradeOrder(action=BuySellAction.NO_ACTION)
        elif self._buysell_approach == AssetBuySellApproach.BUY_HOLD:
            if self.current_size < self._target_size.floor:
                if self._purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                    self.trade_decision = TradeOrder(
                        action=BuySellAction.BUY_SCALE_EXIST,
                        propn=(self._target_size.target - self.current_size) / self.current_size)
                elif self._purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                    self.trade_decision = TradeOrder(
                        action=BuySellAction.BUY_PROFILE,
                        propn=(self._target_size.target - self.current_size) / self.profile_size)
                else:
                    raise ValueError(f'Can not implement purchase method {self._purchase_method} for {self.name}.')
            else:
                self.trade_decision = TradeOrder(action=BuySellAction.NO_ACTION)
        elif self._buysell_approach == AssetBuySellApproach.BUY_SELL:
            if self.current_size < self._target_size.floor:
                if self._purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                    self.trade_decision = TradeOrder(
                        action=BuySellAction.BUY_SCALE_EXIST,
                        propn=(self._target_size.target - self.current_size) / self.current_size)
                elif self._purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                    self.trade_decision = TradeOrder(
                        action=BuySellAction.BUY_PROFILE,
                        propn=(self._target_size.target - self.current_size) / self.profile_size)
                else:
                    raise ValueError(f'Can not implement purchase method {self._purchase_method} for {self.name}.')
            elif self.current_size > self._target_size.cap:
                self.trade_decision = TradeOrder(
                    action=BuySellAction.SELL,
                    propn=(self.current_size - self._target_size.target) / self.current_size)
            else:
                self.trade_decision = TradeOrder(action=BuySellAction.NO_ACTION)
        else:
            raise ValueError(f"Invalid asset buy/sell approach: {self._buysell_approach}")

    def execute_trade(self, trade_order: TradeOrder | None = None) -> None:
        """Execute a trade.
        """
        if trade_order is None:
            trade_order = self.trade_decision
            use_self_trade_decision = True
        else:
            use_self_trade_decision = False

        if trade_order is None:
            return

        net_proceeds: float = 0.0  # proceeds received from disposal (sell), less spent to purchase (buy)
        action, propn = trade_order.action, trade_order.propn

        if action == BuySellAction.NO_ACTION:
            pass
        elif action == BuySellAction.SELL:
            for asset in self.exist_conn.assets:
                mv_bd = asset.market_value
                asset.sell_propn(propn)
                mv_ad = asset.market_value
                net_proceeds += mv_bd - mv_ad
        elif action == BuySellAction.BUY_SCALE_EXIST:
            for asset in self.exist_conn.assets:
                mv_bd = asset.market_value
                asset.buy_propn(propn)
                mv_ad = asset.market_value
                net_proceeds += mv_bd - mv_ad
        elif action == BuySellAction.BUY_PROFILE:
            if self.profile_conn is None:
                raise ValueError("Can't buy assets from empty profile.")
            for profile_asset in self.profile_conn.assets:
                new_asset = profile_asset.scale_profile(scale=propn)
                self.staged_conn.assets.append(new_asset) # append to asset list of the staged
                net_proceeds -= new_asset.market_value

        self.staged_conn.accumulate_free_estate(net_proceeds)

        if use_self_trade_decision:
            self.trade_decision = None  # reset after consumed

    def validate_allocation(self, total_size: float, tolerance: float = 1e-4,) -> bool:
        """Validate if the allocation for a group is satisfied.

        Args:
            total_size (float): Total size.
            tolerance (float): Tolerance for validation (default 0.0001).

        Returns:
            bool: True if the allocation is satisfied, False otherwise.
        """
        if self.buysell_approach in (AssetBuySellApproach.RESIDUAL, AssetBuySellApproach.NO_TRADE):
            return True

        current_weight = self.current_size / total_size

        if self.buysell_approach == AssetBuySellApproach.BUY_HOLD:
            return self.target_weight.floor <= current_weight + tolerance
        elif self.buysell_approach == AssetBuySellApproach.BUY_SELL:
            return self.target_weight.floor - tolerance <= current_weight <= self.target_weight.cap + tolerance
        else:
            raise ValueError(f"{self.name}: invalid asset buy/sell appraoch {self.buysell_approach}.")

    def on_finish_clear(self):
        self._exist_conn = None
        self._profile_conn = None
        self._staged_conn = AssetLiabConnector()
        self._target_size = None
        self._profile_size = None
        self.trade_decision = None


@add_projection_time_synchronizer
class AssetAllocator:
    """Manages asset allocation and rebalancing for a fund.

    Groups assets by allocation group and applies rebalance policy to sell/buy
    or scale exposure to meet target allocations with tolerances.

    Attributes:
        alloc_groups (list[AssetAllocationGroup]): List of allocation group.
    """
    time: int  # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period  # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    _default_alloc_check_level: CheckLevel = CheckLevel.WARN

    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer',
                 'name', 'connector', 'alloc_groups', 'alloc_group_attr', 'asset_report_bases', 'alloc_group_names',
                 'tdv_fund_size', 'tdv_ag_size_bd', 'tdv_ag_size_ad', 'tdv_ag_wgt_pc_bd', 'tdv_ag_wgt_pc_ad',)

    def __init__(
            self,
            *,
            name: str,
            model_engine: ProjModelEngine | None = None,
            connector: AssetLiabConnector,
            allocation_groups: list[AssetAllocationGroup],
            allocation_group_attr: str,
            asset_report_bases: list[str],
    ):
        self.name: str = name
        self.connector: AssetLiabConnector = connector
        self.alloc_groups: list[AssetAllocationGroup] = self._sequence_alloc_groups(allocation_groups)
        self.alloc_group_attr: str = allocation_group_attr
        self.alloc_group_names: list[str] = [ag.name for ag in self.alloc_groups]
        self.asset_report_bases: list[str] = asset_report_bases

        tdv_kwargs = {"model_engine": model_engine, "owner": self.name, "group": 'rebalance'}
        self.tdv_fund_size = TDimVariable("fund_size", **tdv_kwargs)
        self.tdv_ag_size_bd = TDimVariable("size_bd", dims=[self.alloc_group_names], **tdv_kwargs)
        self.tdv_ag_size_ad = TDimVariable("size_ad", dims=[self.alloc_group_names], **tdv_kwargs)
        self.tdv_ag_wgt_pc_bd = TDimVariable("weight_pc_bd", dims=[self.alloc_group_names], **tdv_kwargs)
        self.tdv_ag_wgt_pc_ad = TDimVariable("weight_pc_ad", dims=[self.alloc_group_names], **tdv_kwargs)

    def _sequence_alloc_groups(self, alloc_groups: list[AssetAllocationGroup]) -> list[AssetAllocationGroup]:
        """Sort allocation group by sequence.

            Args:
                alloc_groups: list[RebalancePolicyParams]: List of allocation groups.

            Returns:
                list[str]: Sequential list of allocation groups.
        """
        n_group = len(alloc_groups)
        sorted_list: list[AssetAllocationGroup | None] = [None] * n_group
        min_res_seq: int = n_group
        max_nonres_seq: int = 0

        for ag in alloc_groups:
            seq = ag.sequence
            if seq - 1 not in range(n_group):
                raise ValueError(f'{self.name} {ag.name}: invalid sequence {seq}, expected 1 to {n_group}.')
            if sorted_list[seq - 1] is not None:
                raise ValueError(f'Fund {self.name}: {ag.name} and {sorted_list[seq - 1]} have the same '
                                 f'rebalance sequence {seq}.')
            sorted_list[seq - 1] = ag
            if ag.buysell_approach != AssetBuySellApproach.RESIDUAL:
                max_nonres_seq = max(seq, max_nonres_seq)
            else:
                min_res_seq = min(seq, min_res_seq)

        if min_res_seq <= max_nonres_seq:
            raise ValueError(f'Residual allocation group must be in later sequence than non-residual groups: '
                             f'max non-residual group {sorted_list[max_nonres_seq - 1]}: {max_nonres_seq}, '
                             f'min residual group {sorted_list[min_res_seq - 1]}: {min_res_seq}.')

        return sorted_list

    def rebalance(self, *, total_size: float, asset_size_basis: str | None = None,
                  profile_assets: list[Asset] | None = None, target_weights: dict[str, TargetWeight] | None = None,
                  check_level: CheckLevel | None = None, **kwargs) -> None:
        """Rebalance assets in the fund to match the target allocation.

        Args:
            total_size (float): Total size for allocation.
            asset_size_basis (str): Asset reporting basis for sizing (e.g. FAV or BSV).
            target_weights (dict[str, TargetWeight]): Target weight by allocation group.
            profile_assets (list[Asset]): Profile assets for reference.
            check_level (CheckLevel): Check level for validation against allocation, defaults to None.

        """
        t, p = self.time, self.period
        if total_size < 0:
            warnings.warn(f'{p} {self.name}: negative fund size ({total_size:.2f}) will be treated as zero, and '
                          'all existing assets will be sold to reblance.')
        total_size = max(total_size, 0.0001)  # to prevent ZeroDivisionError

        self._on_enter_rebalance(total_size=total_size, size_basis=asset_size_basis, profile_assets=profile_assets,
                                 target_weights=target_weights)

        self.tdv_fund_size[t] = total_size
        self.tdv_ag_size_bd[t] = np.array([ag.current_size for ag in self.alloc_groups])
        self.tdv_ag_wgt_pc_bd[t] = np.array([ag.current_size / total_size for ag in self.alloc_groups]) * 100

        # --- step 1: process non-residual groups in sequence ---
        for ag in self.alloc_groups:
            if ag.buysell_approach != AssetBuySellApproach.RESIDUAL:
                ag.make_trade_decision()
                ag.execute_trade()

        # --- step 2: process residual groups ---
        size_gap = total_size - sum(ag.current_size for ag in self.alloc_groups)
        self._process_residual_groups(
            size_gap=size_gap, tolerance=max(abs(total_size * 1e-6), 0.01))

        # step 3: validate if target allocations is satisfied
        if check_level is None:
            check_level = self._default_alloc_check_level

        for ag in self.alloc_groups:
            is_satisfied = ag.validate_allocation(total_size=total_size)
            if not is_satisfied:
                msg = (
                    f"{p} | {self.name} | {ag.name}: allocation is not satisfied, {ag.current_size/total_size:.4f}; "
                    f"floor: {ag.target_weight.floor}, cap: {ag.target_weight.cap}; "
                    f"purchase method: {ag.purchase_method.name}; buysell approach: {ag.buysell_approach.name}.")
                if check_level == CheckLevel.ERROR:
                    raise ValueError(msg)
                warnings.warn(msg)

        # step 4: add the purchased assets and proceeds to `self.connector`
        for ag in self.alloc_groups:
            self.connector.assets.extend(ag.staged_conn.assets)
            self.connector.accumulate_free_estate(ag.staged_conn.free_estate)

        self.tdv_ag_size_ad[t] = np.array([ag.current_size for ag in self.alloc_groups])
        self.tdv_ag_wgt_pc_ad[t] = np.array([ag.current_size / total_size for ag in self.alloc_groups]) * 100

        self._on_exit_reblance()

    def _on_enter_rebalance(self, total_size: float, size_basis: str | None, profile_assets: list[Asset] | None,
                            target_weights: dict[str, TargetWeight] | None) -> None:
        profile_assets = profile_assets or []
        target_weights = target_weights or {}

        exist_assets_by_name: dict[str, list] = {ag.name: [] for ag in self.alloc_groups}
        for asset in self.connector.assets:
            name = getattr(asset, self.alloc_group_attr, None)
            if name in exist_assets_by_name:
                exist_assets_by_name[name].append(asset)

        profile_assets_by_name: dict[str, list] = {ag.name: [] for ag in self.alloc_groups}
        for asset in profile_assets:
            name = getattr(asset, self.alloc_group_attr, None)
            if name in profile_assets_by_name:
                profile_assets_by_name[name].append(asset)

        for ag in self.alloc_groups:
            if size_basis is not None:
                ag.size_basis = size_basis
            maybe_new_target_weight = target_weights.get(ag.name, None)
            ag.update_target_size(total_size=total_size, new_target_weight=maybe_new_target_weight)
            ag.exist_conn = AssetLiabConnector(exist_assets_by_name[ag.name])  # auto refresh size
            ag.profile_conn = AssetLiabConnector(profile_assets_by_name[ag.name])  # auto refresh size
            # perform pre-checks
            if ag.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                if len(ag.exist_conn.assets) == 0:
                    raise ValueError(f"'{self.name}'|'{ag.name}': SCALE_UP_EXISTING; no exsiting asset found.")
            if ag.purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                if len(ag.profile_conn.assets) == 0:
                    raise ValueError(f"'{self.name}'|'{ag.name}': PURCHASE_PROFILE;  no profile asset found.")
            if ag.buysell_approach not in (AssetBuySellApproach.NO_TRADE, AssetBuySellApproach.RESIDUAL):
                if ag.target_weight is None:
                    raise ValueError(f"'{self.name}'|'{ag.name}': {ag.buysell_approach}; no target weight provided.")

    def _on_exit_reblance(self) -> None:
        for ag in self.alloc_groups:
            ag.on_finish_clear()

    def _process_residual_groups(self, size_gap: float, tolerance: float) -> None:
        if abs(size_gap) <= tolerance:
            return

        residual_ag_count: int = 0
        current_residual_size: float = 0.0

        for ag in self.alloc_groups:
            if ag.buysell_approach == AssetBuySellApproach.RESIDUAL:
                residual_ag_count += 1
                current_residual_size += ag.current_size

        if residual_ag_count == 0:
            warnings.warn(f"Fund {self.name} has not met target allocation "
                          f"but no residual allocation group to process.")
        elif current_residual_size == 0:
            # This would be very edge case when total residual allocation group (usually cash) balance is zero.
            # Utilize cash asset to safely proceed
            _fallback_cash = None
            for asset in self.connector.assets:
                if isinstance(asset, Cash):
                    asset_ag = getattr(asset, self.alloc_group_attr)
                    ag = next((x for x in self.alloc_groups if x.name == asset_ag), None)
                    if ag.buysell_approach == AssetBuySellApproach.RESIDUAL:
                        _fallback_cash = asset
            if _fallback_cash is not None:
                _fallback_cash.invest_new_money(size_gap)
            else:
                warnings.warn(
                    f"Fund {self.name}: no cash asset (allocation group = residual) is available for sclaing.")
        else:  # scale residual allocation_group
            if size_gap > 0:
                trade_decn = TradeOrder(action=BuySellAction.BUY_SCALE_EXIST, propn=size_gap / current_residual_size)
            elif size_gap < 0:
                trade_decn = TradeOrder(action=BuySellAction.SELL, propn=- size_gap / current_residual_size)
            else:
                raise ValueError("Should never get here.")

            for ag in self.alloc_groups:
                if ag.buysell_approach == AssetBuySellApproach.RESIDUAL:
                    ag.execute_trade(trade_decn)

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.name}'"

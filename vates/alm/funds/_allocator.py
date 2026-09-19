import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from enum import Enum, auto, unique

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDimVariable
from vates.global_conf import CheckLevel
from vates.alm.assets import Asset, Cash
from vates.alm.enums import AssetBuySellApproach, AssetPurchaseMethod
from vates.alm.funds._connector import AssetLiabConnector


@dataclass(slots=True)
class TargetWeight:
    """Target allocation weights for an allocation group.

    Attributes:
        tgt_weight (float): Target weight for the allocation group.
        min_weight (float): Minimum allowed weight.
        max_weight (float): Maximum allowed weight.
    """
    tgt_weight: float
    min_weight: float
    max_weight: float

    def to_size(self, size: float) -> tuple[float, float, float]:
        return size * self.tgt_weight, size * self.min_weight, size * self.max_weight


@unique
class BuySellAction(Enum):
    """Enum for asset buy/sell action."""
    NO_ACTION = auto()
    BUY_SCALE_EXIST = auto()
    BUY_PROFILE = auto()
    SELL = auto()


@dataclass(slots=True)
class TradeDecision:
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

    __slots__ = ("_name", "_sequence", "_buysell_approach", "_purchase_method", "target_weight",
                 "current_conn", "profile_conn", "current_size", "profile_size", "trade_decision")

    _alloc_check: CheckLevel = CheckLevel.WARN

    def __init__(self, *, name: str, sequence: int, buysell_approach: AssetBuySellApproach | str,
                 purchase_method: AssetPurchaseMethod | str,
                 tgt_weight: float | None = None, min_weight: float | None = None, max_weight: float | None = None):
        self._name: str = name
        self._sequence: int = sequence
        self._buysell_approach: AssetBuySellApproach = AssetBuySellApproach[buysell_approach.upper()] \
            if isinstance(buysell_approach, str) else buysell_approach
        self._purchase_method: AssetPurchaseMethod = AssetPurchaseMethod[purchase_method.upper()] \
            if isinstance(purchase_method, str) else purchase_method
        if tgt_weight is None:
            self.target_weight: TargetWeight | None = None
        else:
            self.target_weight: TargetWeight | None = TargetWeight(
                tgt_weight=tgt_weight,
                min_weight=tgt_weight if min_weight is None else min_weight,
                max_weight=tgt_weight if max_weight is None else max_weight
            )
        self.current_conn: AssetLiabConnector | None = None
        self.profile_conn: AssetLiabConnector | None = None
        self.current_size: float | None = None
        self.profile_size: float | None = None
        self.trade_decision: TradeDecision | None = None

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

    def make_trade_decision(self, total_size: float) -> None:
        """Make the trade decision for an allocation group.

        Args:
            total_size (float): Total size.

        Raises:
            ValueError: If an invalid asset buy/sell approach is provided.
        """
        if self.target_weight is not None:
            tgt_size, min_size, max_size = self.target_weight.to_size(total_size)
        else:
            tgt_size, min_size, max_size = None, None, None

        if self._buysell_approach == AssetBuySellApproach.NO_TRADE:
            self.trade_decision = TradeDecision(action=BuySellAction.NO_ACTION)
        elif self._buysell_approach == AssetBuySellApproach.BUY_HOLD:
            if self.current_size < min_size:
                if self._purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                    self.trade_decision = TradeDecision(
                        action=BuySellAction.BUY_SCALE_EXIST,
                        propn=(tgt_size - self.current_size) / self.current_size)
                elif self._purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                    self.trade_decision = TradeDecision(
                        action=BuySellAction.BUY_PROFILE,
                        propn=(tgt_size - self.current_size) / self.profile_size)
                else:
                    raise ValueError(f'Can not implement purchase method {self._purchase_method} for {self.name}.')
            else:
                self.trade_decision = TradeDecision(action=BuySellAction.NO_ACTION)
        elif self._buysell_approach == AssetBuySellApproach.BUY_SELL:
            if self.current_size < min_size:
                if self._purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                    self.trade_decision = TradeDecision(
                        action=BuySellAction.BUY_SCALE_EXIST,
                        propn=(tgt_size - self.current_size) / self.current_size)
                elif self._purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                    self.trade_decision = TradeDecision(
                        action=BuySellAction.BUY_PROFILE,
                        propn=(tgt_size - self.current_size) / self.profile_size)
                else:
                    raise ValueError(f'Can not implement purchase method {self._purchase_method} for {self.name}.')
            elif self.current_size > max_size:
                self.trade_decision = TradeDecision(
                    action=BuySellAction.SELL,
                    propn=(self.current_size - tgt_size) / self.current_size)
            else:
                self.trade_decision = TradeDecision(action=BuySellAction.NO_ACTION)
        else:
            raise ValueError(f"Invalid asset buy/sell approach: {self._buysell_approach}")

    def validate_allocation(self, total_size: float, tolerance: float = 1e-4,
                            check_level: CheckLevel | None = None) -> None:
        """Validate if the allocation for a group meets the target.

        Args:
            total_size (float): Total size.
            tolerance (float): Tolerance for validation (default 0.0001).
            check_level (CheckLevel): Check level for validation against allocation, defaults to None.

        Returns:
            bool: True if the allocation meets the target, False otherwise.
        """
        if check_level is None:
            check_level = self._alloc_check

        if check_level == CheckLevel.BYPASS:
            return

        if self.buysell_approach in (AssetBuySellApproach.RESIDUAL, AssetBuySellApproach.NO_TRADE):
            return

        current_weight = self.current_size / total_size

        if self.buysell_approach == AssetBuySellApproach.BUY_HOLD:
            is_pass = self.target_weight.min_weight <= current_weight + tolerance
        elif self.buysell_approach == AssetBuySellApproach.BUY_SELL:
            is_pass = self.target_weight.min_weight - tolerance <= current_weight <= self.target_weight.max_weight + tolerance
        else:
            raise ValueError(f"{self.name}: invalid asset buy/sell appraoch {self.buysell_approach}.")

        if is_pass:
            return

        msg = (f"{self.name} target allocaion is not met: current={current_weight:.4f}, "
               f"min={self.target_weight.min_weight:.4f}, max={self.target_weight.max_weight:.4f}")
        if check_level == CheckLevel.ERROR:
            raise ValueError(msg)
        else:
            warnings.warn(msg)


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

    def rebalance(self, *, total_size: float, size_basis: str,
                  target_weights: dict[str, TargetWeight] | None = None, profile_assets: list[Asset] | None = None,
                  **kwargs) -> None:
        """Rebalance assets in the fund to match the target allocation.

        Args:
            total_size (float): Total size for allocation.
            size_basis (str): Asset reporting basis for sizing (e.g. FAV or BSV).
            target_weights (dict[str, TargetWeight]): Target weight by allocation group.
            profile_assets (list[Asset] | None): Profile assets for reference.

        """
        t, p = self.time, self.period
        free_proceeds = 0

        self.tdv_fund_size[t] = total_size
        # validate fund size
        if total_size < 0:
            warnings.warn(f'{p} {self.name}: negative fund size ({total_size:.2f}) will be treated as zero, and '
                          'all existing assets will be sold to reblance.')
        total_size = max(total_size, 0.0001)  # to prevent ZeroDivisionError

        # --- step 1: compute existing and profile asset size ---
        self._alloc_groups_create_connect(profile_assets=profile_assets, size_basis=size_basis)  # also calculate size
        # self._alloc_groups_update_current_size(size_basis=size_basis)
        self.tdv_ag_size_bd[t] = np.array([ag.current_size for ag in self.alloc_groups])
        self.tdv_ag_wgt_pc_bd[t] = np.array([ag.current_size / total_size for ag in self.alloc_groups]) * 100

        # --- step 2: cross-validate allocation groups ---
        self._alloc_groups_precheck(target_weights=target_weights)

        # --- step 3: process non-residual groups in sequence ---
        for ag in self.alloc_groups:
            if ag.buysell_approach != AssetBuySellApproach.RESIDUAL:
                ag.make_trade_decision(total_size=total_size)
            else:
                ag.trade_decision = TradeDecision(action=BuySellAction.NO_ACTION)

        for ag in self.alloc_groups:
            free_proceeds += self._execute_trade(alloc_group=ag,)

        # --- step 4: process residual groups ---
        self._alloc_groups_update_current_size(size_basis=size_basis)
        size_gap = total_size - sum(ag.current_size for ag in self.alloc_groups)
        free_proceeds += self._process_residual_groups(
            size_gap=size_gap, tolerance=max(abs(total_size * 1e-6), 0.01))

        # step 5: validate if target allocations met
        self._alloc_groups_update_current_size(size_basis=size_basis)
        for ag in self.alloc_groups:
            ag.validate_allocation(total_size=total_size)
            ag.current_conn = None
            ag.profile_conn = None  # destroy the references to asset instances, otherwise gc can't work

        # step 6: add free proceeds to free estate
        self.connector.accumulate_free_estate(free_proceeds)

        self.tdv_ag_size_ad[t] = np.array([ag.current_size for ag in self.alloc_groups])
        self.tdv_ag_wgt_pc_ad[t] = np.array([ag.current_size / total_size for ag in self.alloc_groups]) * 100

    def _execute_trade(self, *, alloc_group: AssetAllocationGroup) -> float:
        """Execute a trade for a given allocation group.

        Args:
            alloc_group (AssetAllocationGroup): Allocation group.

        Returns:
            float: net proceeds.
        """
        if alloc_group.trade_decision is None:
            return 0.0

        net_proceeds: float = 0.0  # proceeds received from disposal (sell), less spent to purchase (buy)
        action, propn = alloc_group.trade_decision.action, alloc_group.trade_decision.propn

        if action == BuySellAction.NO_ACTION:
            pass
        elif action == BuySellAction.SELL:
            for asset in alloc_group.current_conn.assets:
                mv_bd = asset.market_value
                asset.sell_propn(propn)
                mv_ad = asset.market_value
                net_proceeds += mv_bd - mv_ad
        elif action == BuySellAction.BUY_SCALE_EXIST:
            for asset in alloc_group.current_conn.assets:
                mv_bd = asset.market_value
                asset.buy_propn(propn)
                mv_ad = asset.market_value
                net_proceeds += mv_bd - mv_ad
        elif action == BuySellAction.BUY_PROFILE:
            if alloc_group.profile_conn is None:
                raise ValueError("Can't buy assets from empty profile.")
            for profile_asset in alloc_group.profile_conn.assets:
                new_asset = profile_asset.scale_profile(scale=propn)
                self.connector.assets.append(new_asset)  # append to asset list of the fund
                alloc_group.current_conn.assets.append(new_asset) # append to asset list of the allocation group
                net_proceeds -= new_asset.market_value

        return net_proceeds

    def _alloc_groups_create_connect(self, profile_assets: list[Asset] | None, size_basis: str) -> None:
        name_to_current_conn: dict[str, AssetLiabConnector] = {}
        name_to_profile_conn: dict[str, AssetLiabConnector] = {}
        for ag in self.alloc_groups:
            ag.current_conn = AssetLiabConnector()
            ag.profile_conn = AssetLiabConnector()
            name_to_current_conn[ag.name] = ag.current_conn
            name_to_profile_conn[ag.name] = ag.profile_conn

        for asset in self.connector.assets:
            name = getattr(asset, self.alloc_group_attr, None)
            if name in name_to_current_conn:
                name_to_current_conn[name].assets.append(asset)

        profile_assets = profile_assets or []
        for asset in profile_assets:
            name = getattr(asset, self.alloc_group_attr, None)
            if name in name_to_profile_conn:
                name_to_profile_conn[name].assets.append(asset)

        # also calculate size
        for ag in self.alloc_groups:
            ag.current_size = ag.current_conn.sum_asset(size_basis)
            ag.profile_size = ag.profile_conn.sum_asset(size_basis)

    def _alloc_groups_update_current_size(self, size_basis: str) -> None:
        for ag in self.alloc_groups:
            ag.current_size = ag.current_conn.sum_asset(size_basis)

    def _alloc_groups_precheck(self, target_weights: dict[str, TargetWeight]) -> None:
        for ag in self.alloc_groups:
            if ag.name in target_weights:
                ag.target_weight = target_weights[ag.name]

            if ag.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                if len(ag.current_conn.assets) == 0:
                    raise ValueError(f"'{self.name}'|'{ag.name}': SCALE_UP_EXISTING; no exsiting asset found.")
            if ag.purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                if len(ag.profile_conn.assets) == 0:
                    raise ValueError(f"'{self.name}'|'{ag.name}': PURCHASE_PROFILE;  no profile asset found.")
            if ag.buysell_approach not in (AssetBuySellApproach.NO_TRADE, AssetBuySellApproach.RESIDUAL):
                if ag.target_weight is None:
                    raise ValueError(f"'{self.name}'|'{ag.name}': {ag.buysell_approach}; no target weight provided.")

    def _process_residual_groups(self, size_gap: float, tolerance: float) -> float:
        if abs(size_gap) <= tolerance:
            return 0.0

        residual_ag_count: int = 0
        current_residual_size: float = 0.0
        net_proceeds: float = 0.0

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
            cash_asset = None
            for asset in self.connector.assets:
                if isinstance(asset, Cash):
                    asset_ag = getattr(asset, self.alloc_group_attr)
                    ag = next((x for x in self.alloc_groups if x.name == asset_ag), None)
                    if ag.buysell_approach == AssetBuySellApproach.RESIDUAL:
                        cash_asset = asset
            if cash_asset is not None:
                cash_asset.invest_new_money(size_gap)
                net_proceeds -= size_gap
            else:
                warnings.warn(
                    f"Fund {self.name}: no cash asset (allocation group = residual) is available for sclaing.")
        else:  # scale residual allocation_group
            if size_gap > 0:
                trade_decn = TradeDecision(action=BuySellAction.BUY_SCALE_EXIST,
                                           propn=size_gap / current_residual_size)
            elif size_gap < 0:
                trade_decn = TradeDecision(action=BuySellAction.SELL, propn=- size_gap / current_residual_size)
            else:
                raise ValueError("Should never get here.")

            for ag in self.alloc_groups:
                if ag.buysell_approach == AssetBuySellApproach.RESIDUAL:
                    ag.trade_decision = trade_decn
                    net_proceeds += self._execute_trade(alloc_group=ag)

        return net_proceeds

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.name}'"

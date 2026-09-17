import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from enum import Enum, auto, unique

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDimVariable
from vates.global_conf import STRICTNESS_LEVEL, StrictnessLevel
from vates.alm.assets import Asset, Cash
from vates.alm.enums import AssetBuySellApproach, AssetPurchaseMethod
from vates.alm.funds._utils import AssetLiabConnector


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
                 "current_size", "profile_size", "trade_decision")

    def __init__(self, *, name: str, sequence: int, buysell_approach: AssetBuySellApproach | str,
                 purchase_method: AssetPurchaseMethod | str,
                 tgt_weight: float = 0.0, min_weight: float = 0.0, max_weight: float = 0.0):
        self._name: str = name
        self._sequence: int = sequence
        self._buysell_approach: AssetBuySellApproach = AssetBuySellApproach[buysell_approach.upper()] \
            if isinstance(buysell_approach, str) else buysell_approach
        self._purchase_method: AssetPurchaseMethod = AssetPurchaseMethod[purchase_method.upper()] \
            if isinstance(purchase_method, str) else purchase_method
        self.target_weight: TargetWeight = TargetWeight(
            tgt_weight=tgt_weight, min_weight=min_weight, max_weight=max_weight)
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
        tgt_size, min_size, max_size = self.target_weight.to_size(total_size)

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
                            strictness: StrictnessLevel = STRICTNESS_LEVEL) -> bool:
        """Validate if the allocation for a group meets the target.

        Args:
            total_size (float): Total size.
            tolerance (float): Tolerance for validation (default 0.0001).
            strictness (StrictnessLevel): Strictness for validation, defaults to STRICTNESS_LEVEL.

        Returns:
            bool: True if the allocation meets the target, False otherwise.
        """
        current_weight = self.current_size / total_size

        if self.buysell_approach == AssetBuySellApproach.RESIDUAL or self.buysell_approach == AssetBuySellApproach.NO_TRADE:
            is_pass = True
        elif self.buysell_approach == AssetBuySellApproach.BUY_HOLD:
            is_pass = self.target_weight.min_weight <= current_weight + tolerance
        elif self.buysell_approach == AssetBuySellApproach.BUY_SELL:
            is_pass = self.target_weight.min_weight - tolerance <= current_weight <= self.target_weight.max_weight + tolerance
        else:
            raise ValueError(f"{self.name}: invalid asset buy/sell appraoch {self.buysell_approach}.")

        if not is_pass:
            msg = (f"{self.name} target allocaion is not met: current={current_weight:.4f}, "
                   f"min={self.target_weight.min_weight:.4f}, max={self.target_weight.max_weight:.4f}")
            if strictness == STRICTNESS_LEVEL.ERROR:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        return is_pass

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
                 'name', 'connector', 'alloc_groups', 'alloc_group_attr', 'asset_report_bases',
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
        self.asset_report_bases: list[str] = asset_report_bases

        tdv_kwargs = {"model_engine": model_engine, "owner": self.name, "group": 'rebalance'}
        self.tdv_fund_size = TDimVariable("fund_size", **tdv_kwargs)
        alloc_group_names = [ag.name for ag in self.alloc_groups]
        self.tdv_ag_size_bd = TDimVariable("size_bd", dims=[alloc_group_names], **tdv_kwargs)
        self.tdv_ag_size_ad = TDimVariable("size_ad", dims=[alloc_group_names], **tdv_kwargs)
        self.tdv_ag_wgt_pc_bd = TDimVariable("weight_pc_bd", dims=[alloc_group_names], **tdv_kwargs)
        self.tdv_ag_wgt_pc_ad = TDimVariable("weight_pc_ad", dims=[alloc_group_names], **tdv_kwargs)

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

        size_basis_index: int = self.asset_report_bases.index(size_basis)
        profile_assets = profile_assets or []
        self.tdv_fund_size[t] = total_size

        # validate fund size
        if total_size < 0:
            warnings.warn(f'{p} {self.name}: negative fund size ({total_size:.2f}) will be treated as zero, and '
                          'all existing assets will be sold to reblance.')
        total_size = max(total_size, 0.0001)  # to prevent divide by zero error

        # --- step 1: aggregate existing and profile asset size by allocation group ---
        exist_asset_size, exist_asset_count = self._groupby_sum_asset_size(self.connector.assets, size_basis)
        profile_asset_size, profile_asset_count = self._groupby_sum_asset_size(profile_assets, size_basis)
        for i, ag in enumerate(self.alloc_groups):
            ag.current_size = exist_asset_size[i]
            ag.profile_size = profile_asset_size[i]
        self.tdv_ag_size_bd[t] = np.array([ag.current_size for ag in self.alloc_groups])
        self.tdv_ag_wgt_pc_bd[t] = np.array([ag.current_size / total_size for ag in self.alloc_groups]) * 100

        # --- step 2: cross-validate asset groups ---
        for i, ag in enumerate(self.alloc_groups):
            if ag.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING and exist_asset_count[i] == 0:
                raise ValueError(f"'{self.name}' - '{ag.name}': "
                                 f"purchase method=SCALE_UP_EXISTING but not found in exsiting assets.")
            if ag.purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE and profile_asset_count[i] == 0:
                raise ValueError(f"'{self.name}' - '{ag.name}': "
                                 f"purchase method=PURCHASE_PROFILE but not found in profile.")
            if ag.buysell_approach not in [AssetBuySellApproach.NO_TRADE, AssetBuySellApproach.RESIDUAL] and \
                    ag.name not in target_weights:
                raise ValueError(f"'{self.name}' - '{ag.name}': "
                                 f"buy/sell appraoch={ag.buysell_approach} but not found in target allocation.")
            if ag.name in target_weights:
                ag.target_weight = target_weights[ag.name]

        # --- step 3: process non-residual groups in sequence ---
        for ag in self.alloc_groups:
            if ag.buysell_approach != AssetBuySellApproach.RESIDUAL:
                ag.make_trade_decision(total_size=total_size)
            else:
                ag.trade_decision = TradeDecision(action=BuySellAction.NO_ACTION)

        for ag in self.alloc_groups:
            free_proceeds += self._execute_trade(alloc_group=ag, profile_assets=profile_assets)

        # --- step 4: process residual groups ---
        current_asset_size, _ = self._groupby_sum_asset_size(self.connector.assets, size_basis)
        total_current_size = 0.0
        for i, ag in enumerate(self.alloc_groups):
            ag.current_size = current_asset_size[i]
            total_current_size += ag.current_size
        size_gap = total_size - total_current_size
        free_proceeds += self._process_residual_groups(
            size_gap=size_gap, tolerance=max(abs(total_size * 1e-6), 0.01), profile_assets=profile_assets)

        # step 5: validate if target allocations met
        current_asset_size, _ = self._groupby_sum_asset_size(self.connector.assets, size_basis)
        for i, ag in enumerate(self.alloc_groups):
            ag.current_size = current_asset_size[i]
            ag.validate_allocation(total_size=total_size, strictness=STRICTNESS_LEVEL)

        # step 6: add free proceeds to free estate
        self.connector.accumulate_free_estate(free_proceeds)

        self.tdv_ag_size_ad[t] = np.array([ag.current_size for ag in self.alloc_groups])
        self.tdv_ag_wgt_pc_ad[t] = np.array([ag.current_size / total_size for ag in self.alloc_groups]) * 100

    def _execute_trade(self, *, alloc_group: AssetAllocationGroup, profile_assets: list[Asset] | None) -> float:
        """Execute a trade for a given allocation group.

        Args:
            alloc_group (AssetAllocationGroup): Allocation group.
            profile_assets (list[Asset] | None): Profile assets for rebalance.

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
            for asset in self.connector.assets:
                if getattr(asset, self.alloc_group_attr) == alloc_group.name:
                    mv_bd = asset.market_value
                    asset.sell_propn(propn)
                    mv_ad = asset.market_value
                    net_proceeds += mv_bd - mv_ad
        elif action == BuySellAction.BUY_SCALE_EXIST:
            for asset in self.connector.assets:
                if getattr(asset, self.alloc_group_attr) == alloc_group.name:
                    mv_bd = asset.market_value
                    asset.buy_propn(propn)
                    mv_ad = asset.market_value
                    net_proceeds += mv_bd - mv_ad
        elif action == BuySellAction.BUY_PROFILE:
            if not profile_assets:
                raise ValueError("Can't buy assets from empty profile.")
            for asset in profile_assets:
                if getattr(asset, self.alloc_group_attr) == alloc_group.name:
                    asset.buy_profile_scale(scale=propn)
                    self.connector.assets.append(asset)  # append to list
                    net_proceeds -= asset.market_value

        return net_proceeds

    def _groupby_sum_asset_size(self, assets: list[Asset], size_basis: str,
                                ) -> tuple[list[float], list[int]]:
        """Aggregate asset reported values by allocation group.

        Args:
            assets (list[Asset]): Assets to aggregate.
            size_basis (str): Asset reporting basis for sizing.

        Returns:
            list[float]]: Asset size of each allocation group.
            list[int]: Asset count of each allocation group.
        """
        asset_size: list[float] = [0.0 for ag in self.alloc_groups]
        asset_count: list[int] = [0 for ag in self.alloc_groups]
        key_to_index: dict[str, int] = {item.name: index for index, item in enumerate(self.alloc_groups)}

        for asset in assets:
            key = getattr(asset, self.alloc_group_attr)
            idx = key_to_index.get(key)
            if idx is None:
                raise ValueError(f'Asset {asset.asset_id}: allocation group {key} not included '
                                 f'the fund reblance policy.')
            asset_size[idx] += getattr(asset, size_basis)
            asset_count[idx] += 1

        return asset_size, asset_count

    def _process_residual_groups(self, size_gap: float, profile_assets: list[Asset] | None, tolerance: float) -> float:
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
                    net_proceeds += self._execute_trade(alloc_group=ag, profile_assets=profile_assets)

        return net_proceeds

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.name}'"

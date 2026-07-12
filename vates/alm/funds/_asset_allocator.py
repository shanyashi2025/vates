from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings

from vates._core import ProjModelEngine, TDepVariable
from vates.alm.enums import AssetRepBasis, AssetBuySellApproach, AssetPurchaseMethod
from vates.alm.assets import Asset, Cash
from vates.alm.funds._utils import ALContainer


@dataclass
class RebalancePolicyParams:
    """Rebalance policy parameters.

    Attributes:
        sequence (int): The sequence in which the allocation group will be processed, must be unique and consecutive
            integer starting from 1.
        buysell_approach (AssetBuySellApproach): Buy/sell approach for this allocation group.
        purchase_method (AssetPurchaseMethod): Purchase method for this allocation group.
    """
    sequence: int
    buysell_approach: AssetBuySellApproach
    purchase_method: AssetPurchaseMethod


@dataclass
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


class AssetAllocator:
    """Manages asset allocation and rebalancing for a fund.

    Groups assets by allocation group and applies rebalance policy to sell/buy
    or scale exposure to meet target allocations with tolerances.

    Attributes:
        rebalance_policy (dict[str, RebalancePolicyParams]): Rebalance policy by allocation group.
    """

    def __init__(self, model_engine: ProjModelEngine, fund_id: str, container: ALContainer,
                 rebalance_policy: dict[str, RebalancePolicyParams]):
        model_engine.attach_time_observer(self)
        self.time: int = model_engine.time
        self._start_date: pd.Period = model_engine.START_DATE

        self.fund_id: str = fund_id
        self.container: ALContainer = container
        self.rebalance_policy = rebalance_policy
        self.ag_seq_list = self.list_ag_in_sequence(fund_id, rebalance_policy)

        self.tdv_fund_size: TDepVariable = TDepVariable(model_engine, "fund_size", fund_id, 'rebalance')
        self.tdv_ag_repval_bd: TDepVariable = TDepVariable(model_engine, "ag_repval_bd", fund_id, 'rebalance',
                                                           dims=[self.ag_seq_list, AssetRepBasis])
        self.tdv_ag_repval_ad: TDepVariable = TDepVariable(model_engine, "ag_repval_ad", fund_id, 'rebalance',
                                                           dims=[self.ag_seq_list, AssetRepBasis])
        self.tdv_ag_alloc_pc_bd: TDepVariable = TDepVariable(model_engine, "ag_alloc_pc_bd", fund_id, 'rebalance',
                                                             dims=[self.ag_seq_list])
        self.tdv_ag_alloc_pc_ad: TDepVariable = TDepVariable(model_engine, "ag_alloc_pc_ad", fund_id, 'rebalance',
                                                             dims=[self.ag_seq_list])

    def sync_time(self, subject: ProjModelEngine) -> None:
        self.time = subject.time

    @property
    def period(self) -> pd.Period:
        return self._start_date + self.time

    @staticmethod
    def list_ag_in_sequence(fund_id: str, rebalance_policy: dict[str, RebalancePolicyParams]) -> list[str]:
        """List allocation group in sequence.

            Args:
                fund_id (str): Fund.
                rebalance_policy: dict[str, RebalancePolicyParams]: Rebalance policy by allocation group.

            Returns:
                list[str]: Sequential list of allocation groups.

            Raises:
                ValueError: If timing is invalid or allocations are invalid.
        """
        max_seq = len(rebalance_policy)
        ag_list: list[str | None] = [None] * max_seq
        min_res_seq: int = max_seq
        max_nonres_seq: int = 0

        for ag, policy in rebalance_policy.items():
            seq = policy.sequence
            if seq - 1 not in range(max_seq):
                raise ValueError(f'Fund {fund_id} allocation group {ag}: invalid sequence {seq}, expected 1 to {max_seq}.')
            if ag_list[seq - 1] is not None:
                raise ValueError(f'Fund {fund_id}: both allocation group {ag} and {ag_list[seq - 1]} have the same '
                                 f'rebalance sequence {seq}.')
            ag_list[seq - 1] = ag
            if policy.buysell_approach != AssetBuySellApproach.RESIDUAL:
                max_nonres_seq = max(seq, max_nonres_seq)
            else:
                min_res_seq = min(seq, min_res_seq)

        if min_res_seq <= max_nonres_seq:
            raise ValueError(f'Residual allocation group must be in later sequence than non-residual groups: '
                             f'max non-residual group {ag_list[max_nonres_seq -1]}: {max_nonres_seq}, '
                             f'min residual group {ag_list[min_res_seq-1]}: {min_res_seq}.')

        return ag_list

    def rebalance(self, fund_size: float, size_basis: AssetRepBasis,
                  target_weight: dict[str, TargetWeight], assets_profile: list[Asset] | None=None,
                  **kwargs) -> tuple[float, float]:
        """Rebalance assets in the fund to match the target allocation.

        Args:
            fund_size (float): Total size for allocation.
            size_basis (AssetRepBasis): Basis for sizing (usually FAV or BSV).
            target_weight (dict[str, TargetWeight]): Target allocations weight by group.
            assets_profile (list[Asset] | None): Profile assets for reference.

        Returns:
            tuple[float, float]: (free_proceeds, realized_gain_loss).

        Raises:
            ValueError: If allocations are invalid.
        """
        t, p = self.time, self.period
        free_proceeds, realized_gl = 0, 0
        # initialize fund size
        self.tdv_fund_size[t] = fund_size
        if fund_size < 0:
            warnings.warn(f'{p} {self.fund_id}: negative fund size ({fund_size:.2f}) will be treated as zero whereby '
                          'all existing assets will be sold to reblance.')
        fund_size = max(fund_size, 0.0001)  # to prevent divide by zero error

        # --- step 1: aggregate existing and profile asset reported value by allocation group ---
        # --- existing asset ---
        exist_asset_repval, exist_asset_count = self._group_by_alloc_group(self.container.assets, self.ag_seq_list)
        self.tdv_ag_repval_bd[t] = np.array([val for val in exist_asset_repval.values()])
        exist_asset_value = {key: arr[size_basis.value] for key, arr in exist_asset_repval.items()}
        self.tdv_ag_alloc_pc_bd[t] = self._calculate_ag_weight(exist_asset_value, fund_size) * 100
        # --- profile asset ---
        profile_asset_repval, profile_asset_count = self._group_by_alloc_group(assets_profile, self.ag_seq_list)
        profile_asset_value = {key: arr[size_basis.value] for key, arr in profile_asset_repval.items()}

        # --- step 2: cross-validate asset groups ---
        for ag, policy in self.rebalance_policy.items():
            if policy.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING and exist_asset_count[ag] == 0:
                raise ValueError(f"Fund {self.fund_id} asset allocation group {ag}: "
                                 f"purchase method=SCALE_UP_EXISTING but not found in exsiting assets.")
            if policy.purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE and profile_asset_count[ag] == 0:
                raise ValueError(f"Fund {self.fund_id} asset allocation group {ag}: "
                                 f"purchase method=PURCHASE_PROFILE but not found in profile.")
            if policy.buysell_approach not in [AssetBuySellApproach.NO_TRADE, AssetBuySellApproach.RESIDUAL] and\
                    ag not in target_weight:
                raise ValueError(f"Fund {self.fund_id} asset allocation group {ag}: "
                                 f"buy/sell appraoch={policy.buysell_approach} but not found in target allocation.")
            if ag not in target_weight: # add dummy weights
                target_weight[ag] = TargetWeight(tgt_weight=0, min_weight=0, max_weight=0)

        # --- step 3: process non-residual groups in sequence ---
        ag_trade_decn: dict[str, tuple[str, float]] = {}
        res_ag_exist_value: float = 0.0
        res_ag_count: int = 0

        for ag in self.ag_seq_list:
            policy = self.rebalance_policy[ag]
            buysell_app = policy.buysell_approach
            pur_method = policy.purchase_method
            # obtain the asset value of the allocation group
            ag_exist_value, ag_profile_value = exist_asset_value[ag], profile_asset_value[ag]
            # calculate the target and min/max value
            wgt = target_weight[ag]
            tgt_value = fund_size * wgt.tgt_weight
            min_value = fund_size * wgt.min_weight
            max_value = fund_size * wgt.max_weight

            if buysell_app != AssetBuySellApproach.RESIDUAL:
                ag_trade_decn[ag] = self._make_trade_decision(
                    ag, buysell_app, pur_method, min_value, tgt_value, max_value, ag_exist_value, ag_profile_value
                )
            else:
                res_ag_count += 1
                res_ag_exist_value += ag_exist_value
                ag_trade_decn[ag] = "none", 0.0

        for ag, trade_decn in ag_trade_decn.items():
            if trade_decn[0] == "none":
                continue
            proceeds, rgl = self._trade_asset(ag, trade_decn, assets_profile)
            free_proceeds += proceeds
            realized_gl += rgl

        # --- step 4: process residual groups ---
        exist_asset_repval, _ = self._group_by_alloc_group(self.container.assets, self.ag_seq_list)
        exist_asset_value = {key: arr[size_basis.value] for key, arr in exist_asset_repval.items()}
        total_exist_value = sum(exist_asset_value.values())
        value_gap = fund_size - total_exist_value
        tolerance = max(abs(fund_size * 1e-6), 0.01)

        if abs(value_gap) > tolerance:
            if res_ag_count == 0:
                raise ValueError(f"Fund {self.fund_id} has not met target allocation "
                                 f"but no residual allocation group to process.")

            if res_ag_exist_value == 0:
                # This would be very extreme case when total residual allocation_group (usually cash) balance is zero.
                # Utilize cash asset to safely proceed
                cash_asset = None
                for asset in self.container.assets:
                    if isinstance(asset, Cash) and self.rebalance_policy[
                        asset.allocation_group].buysell_approach == AssetBuySellApproach.RESIDUAL:
                        cash_asset = asset
                        break
                if cash_asset is None:
                    raise ValueError(f"Fund {self.fund_id}: no cash asset (allocation group = residual) is available for sclaing.")
                cash_asset.invest_new_money(value_gap)
                free_proceeds -= value_gap
            else:  # scale residual allocation_group
                if value_gap > 0:
                    trade_decn = "buy_scale_exist", value_gap / res_ag_exist_value
                elif value_gap < 0:
                    trade_decn = "sell", - value_gap / res_ag_exist_value
                else:
                    raise ValueError("Should never get here.")

                for ag, policy in self.rebalance_policy.items():
                    if policy.buysell_approach == AssetBuySellApproach.RESIDUAL:
                        proceeds, rgl = self._trade_asset(ag, trade_decn, assets_profile)
                        free_proceeds += proceeds
                        realized_gl += rgl

        # step 5: validate if target allocations met
        exist_asset_repval, _ = self._group_by_alloc_group(self.container.assets, self.ag_seq_list)
        exist_asset_value = {key: arr[size_basis.value] for key, arr in exist_asset_repval.items()}
        self.tdv_ag_repval_ad[t] = np.array([val for val in exist_asset_repval.values()])
        self.tdv_ag_alloc_pc_ad[t] = self._calculate_ag_weight(exist_asset_value, fund_size) * 100

        for ag, policy in self.rebalance_policy.items():
            buysell_app = policy.buysell_approach
            current_weight = exist_asset_value[ag] / fund_size if ag in exist_asset_value else 0
            wgt = target_weight[ag]
            if not self._validate_allocation(ag, buysell_app, current_weight, wgt.min_weight, wgt.max_weight):
                raise ValueError(
                    f"Fund {self.fund_id}, target allocaion is not met for {ag} at {p}: "
                    f"min={wgt.min_weight: .4f}, max={wgt.max_weight: .4f}, "
                    f"current={current_weight: .4f}.")

        return free_proceeds, realized_gl

    def _trade_asset(self, allocation_group: str, trade_decn: tuple[str, float],
                     assets_profile: list[Asset] | None) -> tuple[float, float]:
        """Execute a trade for a given allocation group.

        Args:
            allocation_group (str): Allocation group to trade.
            trade_decn (tuple[str, float]): Trade decision:
                buysell (str): Buy/sell type (buy_scale_exist/buy_profile/sell).
                propn (float): Proportion of asset to trade.
            assets_profile (list[Asset] | None): Profile assets for rebalance.

        Returns:
            tuple[float, float]: (proceeds, realized_gain_loss).
        """
        buysell, propn = trade_decn

        if buysell not in ['sell', 'buy_scale_exist', 'buy_profile']:
            raise ValueError(f"Invalid asset asset buy/sell: {buysell}")

        proceeds: float = 0.0  # proceeds received from sell, less spent to buy
        rgl: float = 0.0  # realized gain or loss

        if buysell == 'sell':
            for asset in self.container.assets:
                if asset.allocation_group == allocation_group:
                    fav_bd, mv_bd = asset.fav, asset.mv
                    asset.sell_propn(propn)
                    fav_ad, mv_ad = asset.fav, asset.mv
                    proceeds += mv_bd - mv_ad
                    rgl += (mv_bd - mv_ad) - (fav_bd - fav_ad)
        elif buysell == 'buy_scale_exist':
            for asset in self.container.assets:
                if asset.allocation_group == allocation_group:
                    mv_bd = asset.mv
                    asset.buy_propn(propn)
                    mv_ad = asset.mv
                    proceeds += mv_bd - mv_ad
        elif buysell == 'buy_profile':
            if not assets_profile: raise ValueError("Can't buy assets from empty profile.")
            for asset in assets_profile:
                if asset.allocation_group == allocation_group:
                    asset.buy_profile_scale(scale=propn)
                    self.container.assets.append(asset)
                    proceeds -= asset.mv

        return proceeds, rgl

    @staticmethod
    def _calculate_ag_weight(ag_asset_value_dict: dict[str, npt.NDArray[np.float64]], fund_size: float
                             ) -> npt.NDArray[np.float64]:
        """Get the weight array of each allocation groups.

        Args:
            ag_asset_value_dict (dict[str, npt.NDArray[np.float64]]): Asset values (size basis) by allocation group.
            fund_size (float): Fund size.
        """
        if fund_size < -0.01: raise ValueError("Fund size is negative.")
        if fund_size <= 0: return np.zeros([len(ag_asset_value_dict)])
        return np.array([ag_asset_value_dict[ag] / fund_size for i, ag in enumerate(ag_asset_value_dict)])

    @staticmethod
    def _group_by_alloc_group(assets: list[Asset], ag_list: list[str]
                              ) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, int]]:
        """Aggregate asset reported values by allocation group.

        Args:
            assets (list[Asset]): Assets to aggregate.
            ag_list (list[str]): List of allocation group.

        Returns:
            dict[str, npt.NDArray[np.float64]]: Mapping from allocation group to aggregated asset reported value.
            dict[str, int]: Mapping from allocation group to asset counts.
        """
        asset_rep_value: dict = {ag: np.zeros(len(AssetRepBasis)) for ag in ag_list}
        asset_count: dict = {ag: 0 for ag in ag_list}

        for asset in assets:
            ag = asset.allocation_group
            if ag not in asset_rep_value:
                raise ValueError(f'Asset {asset.asset_id} (in fund {asset.fund_id}): allocation group {ag} not included '
                                 f'the fund reblance policy.')
            asset_rep_value[ag] += asset.rep_value
            asset_count[ag] += 1

        return asset_rep_value, asset_count

    @staticmethod
    def _make_trade_decision(allocation_group: str, buysell_app: AssetBuySellApproach, pur_method: AssetPurchaseMethod,
                             min_value: float, tgt_value: float, max_value: float,
                             ag_current_value: float, ag_profile_value: float) -> tuple[str, float]:
        """Make the trade decision for an allocation group.

        Args:
            allocation_group (str): Allocation group.
            buysell_app (AssetBuySellApproach): Buy/sell approach.
            pur_method (AssetPurchaseMethod): Asset purchase method.
            min_value (float): Minimum allowed value.
            tgt_value (float): Target value.
            max_value (float): Maximum allowed value.
            ag_current_value (float): Current value of the allocation group.
            ag_profile_value (float): Profile value of the allocation group.

        Returns:
            tuple[str, float]: (buysell, proportion), where buysell is "buy_profile", "buy_scale_exist", "sell", or "none".

        Raises:
            ValueError: If an invalid asset buy/sell approach is provided.
        """
        if buysell_app == AssetBuySellApproach.NO_TRADE:
            buysell, propn = "none", 0
        elif buysell_app == AssetBuySellApproach.BUY_HOLD:
            if ag_current_value < min_value:
                if pur_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                    buysell, propn = "buy_scale_exist", (tgt_value - ag_current_value) / ag_current_value
                elif pur_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                    buysell, propn = "buy_profile", (tgt_value - ag_current_value) / ag_profile_value
                else:
                    raise ValueError(f'Can not implement purchase method {pur_method} for {allocation_group}.')
            else:
                buysell, propn = "none", 0
        elif buysell_app == AssetBuySellApproach.BUY_SELL:
            if ag_current_value < min_value:
                if pur_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                    buysell, propn = "buy_scale_exist", (tgt_value - ag_current_value) / ag_current_value
                elif pur_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                    buysell, propn = "buy_profile", (tgt_value - ag_current_value) / ag_profile_value
                else:
                    raise ValueError(f'Can not implement purchase method {pur_method} for {allocation_group}.')
            elif ag_current_value > max_value:
                buysell, propn = "sell", (ag_current_value - tgt_value) / ag_current_value
            else:
                buysell, propn = "none", 0
        else:
            raise ValueError(f"Invalid asset buy/sell approach: {buysell_app}")

        return buysell, propn

    @staticmethod
    def _validate_allocation(allocation_group: str, buysell_appraoch: AssetBuySellApproach, current_weight: float,
                             min_weight: float, max_weight: float, tolerance: float = 1e-4) -> bool:
        """Validate if the allocation for a group meets the target.

        Args:
            allocation_group (str): Allocation group.
            buysell_appraoch (AssetBuySellApproach): Strategy to apply.
            current_weight (float): Current weight of the group.
            min_weight (float): Minimum allowed weight.
            max_weight (float): Maximum allowed weight.
            tolerance (float): Tolerance for validation (default 0.0001).

        Returns:
            bool: True if the allocation meets the target, False otherwise.

        Raises:
            ValueError: If an invalid asset buy/sell approach is provided.
        """
        if buysell_appraoch == AssetBuySellApproach.RESIDUAL or buysell_appraoch == AssetBuySellApproach.NO_TRADE:
            target_met = True
        elif buysell_appraoch == AssetBuySellApproach.BUY_HOLD:
            target_met = (min_weight <= current_weight + tolerance)
        elif buysell_appraoch == AssetBuySellApproach.BUY_SELL:
            target_met = (min_weight - tolerance <= current_weight <= max_weight + tolerance)
        else:
            raise ValueError(f"Invalid asset buy/sell appraoch: {buysell_appraoch} for allocation group {allocation_group}")

        return target_met

    def __str__(self) -> str:
        return self.fund_id + "<allocator>"

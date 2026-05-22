from typing import Optional
import pandas as pd
import warnings
import weakref
from enum import Enum, unique

from vates.utils import check_calc_time
from vates.alm.enums import AssetRepBasis
from vates.alm.assets import Asset, Cash
from vates.alm.liabs import Liab
from vates.alm.funds._asset_allocator import AssetAllocator, RebalancePolicyParams, TargetWeight
from vates.alm.funds._fund_calculator import FundCalculator

@unique
class FundSizeType(Enum):
    """Enum for fund size types."""
    FUND = "FUND"
    SURR_VALUE = "SURR_VALUE"
    MATH_RES = "MATH_RES"
    ACCT_VALUE = "ACCT_VALUE"
    ASSET_SHARE = "ASSET_SHARE"
    MAX_AS_MATH = "MAX_AS_MATH"
    MAX_AS_CSV = "MAX_AS_CSV"


class Fund:
    """Investment fund with assets and liabilities.

    Handles asset/liability management, reporting, and rebalance strategies for actuarial projections.

    Attributes:
        fund_id (str): Fund identifier.
        assets (list[Asset]): Assets tracked by the fund.
        primary_cash_asset (Cash | None): Primary cash asset used for residual cash flows.
        liabs (list[Liab]): Liabilities associated with the fund.
        calculator (FundCalculator): Aggregation/returns calculator.
        allocator (AssetAllocator): Asset allocator for rebalancing.
        accum_free_proceeds (float): Accumulated free proceeds pending investment/transfer.
    """

    def __init__(self, model, fund_id: str, rebalance_policy: dict[str, RebalancePolicyParams],
                 asset_categories: list[str]) -> None:
        """
        Initialize a Fund object.

        Args:
            fund_id (str): Fund identifier.
            rebalance_policy (dict[str, RebalancePolicyParams]): Rebalance policy by allocation group.
            asset_categories (list[str]): Asset categories to be reported.
        """
        self._model_ref: weakref.ref = weakref.ref(model)
        self.fund_id = fund_id

        # Asset and liab collections
        self.assets: list[Asset] = []
        self.primary_cash_asset: Cash | None = None
        self.liabs: list[Liab] = []
        self._assembled: bool = False

        self.calculator: FundCalculator = FundCalculator(model, fund_id, self.assets, self.liabs, asset_categories)
        self.allocator: AssetAllocator = AssetAllocator(model, fund_id, self.assets, self.primary_cash_asset, rebalance_policy)

        self.accum_free_proceeds: float = 0.0

        self._lct_dict: dict[str, int] = {}

    @property
    def time(self) -> int | None:
        return self._model_ref().time
    
    @property
    def period(self) -> pd.Period | None:
        return self._model_ref().period

    def add_assets(self, assets: Asset | list[Asset]) -> None:
        """Add assets to the fund.

        Args:
            assets (Asset | list[Asset]): List of assets to add.

        Raises:
            ValueError: If asset already exists in the fund or its `fund_id` mismatches.
        """
        if self._assembled: raise ValueError(f"Fund has already been assembled.")

        if type(assets) != list: assets = [assets]

        for asset in assets:
            if asset in self.assets:
                raise ValueError(f"Asset {asset.asset_id} already exists in fund {self.fund_id}")

            if asset.fund_id != self.fund_id:
                raise ValueError(f"{asset.asset_id}: fund id is {asset.fund_id}, can not add into fund {self.fund_id}")

            self.assets.append(asset)

            if isinstance(asset, Cash) and not self.primary_cash_asset: self.primary_cash_asset = asset

    def add_liabs(self, liabs: Liab | list[Liab]) -> None:
        """Add liabilities to the fund.

        Args:
            liabs (Liab | list[Liab]): List of liabilities to add.

        Raises:
            ValueError: If liability already exists in the fund or its `fund_id` mismatches.
        """
        if self._assembled: raise ValueError(f"Fund has already been assembled.")

        if type(liabs) != list: liabs = [liabs]

        for liab in liabs:
            if liab in self.liabs:
                raise ValueError(f"Liab {liab.liab_id} already exists in fund {self.fund_id}")

            if liab.fund_id != self.fund_id:
                raise ValueError(f"{liab.liab_id}: fund id is {liab.fund_id}, can not add into fund {self.fund_id}")

            self.liabs.append(liab)

    def assemble(self) -> None:
        """Assemble the fund.

        Aggregates asset and liability values, validates primary cash, and marks the fund as assembled.

        Raises:
            ValueError: If no primary cash asset is available.
        """
        if self._assembled: raise ValueError(f"Fund has already been assembled.")

        if not self.primary_cash_asset: raise ValueError("No cash assets available for investment")

        # Aggregate asset value
        self.calculator.aggregate_assets_value("ad")
        self.calculator.aggregate_liabs_value("bd")
        self.calculator.aggregate_liabs_value("ad")

        self._assembled = True

    @check_calc_time({"proc_assets_bd": -1, "proc_assets_ad": -1}, "proc_assets_bd")
    def process_assets_before_dealing(self) -> None:
        """Process asset cash flows and reported values before dealing (bd)."""
        self.calculator.process_assets_before_dealing()
        self.accum_free_proceeds += self.calculator.tdv_totass_cash_flow[self.time]

    @check_calc_time({"proc_liabs_bd": -1, "proc_liabs_ad": -1, "proc_assets_bd": 0}, "proc_liabs_bd")
    def process_liabs_before_dealing(self) -> None:
        """Process liability cash flows and balance sheet variables before dealing (bd)."""
        self.calculator.process_liabs_before_dealing()
        self.accum_free_proceeds += self.calculator.tdv_totliab_cash_flow[self.time]

    @check_calc_time({"proc_assets_ad": -1, "proc_assets_bd": 0, "proc_liabs_bd": 0}, "proc_assets_ad")
    def skip_rebalance(self) -> None:
        """Skip asset rebalance and invest free proceeds into primary cash."""
        t = self.time
        self.calculator.tdv_accum_free_proceeds_bd[t] = self.accum_free_proceeds
        self._invest_accum_free_proceeds_into_cash()  # just invest accum_free_proceeds into primary cash,
                                                      # no other rebalance#, accum_free_proceeds is reset to zero
        for asset in self.assets:
            asset.complete_dealing()
        self.calculator.tdv_accum_free_proceeds_ad[t] = self.accum_free_proceeds # reset to zero
        self.calculator.process_assets_after_dealing()

    def _invest_accum_free_proceeds_into_cash(self) -> None:
        """Invest accumulated free proceeds into the primary cash asset."""
        if not self.primary_cash_asset: raise RuntimeError(f"{self.fund_id}: primary cash asset is not defined.")
        self.primary_cash_asset.invest_new_money(self.accum_free_proceeds)
        self._reset_accum_free_proceeds()

    @check_calc_time({"proc_assets_ad": -1, "proc_assets_bd": 0, "proc_liabs_bd": 0}, "proc_assets_ad")
    def rebalance_assets(self, fund_size_type: FundSizeType, fund_size_basis: AssetRepBasis,
                         target_weight: dict[str, TargetWeight], assets_profile: list[Asset] | None=None, **kwargs
                         ) -> None:
        """Rebalance assets per target allocation and optional profile.

        Args:
            fund_size_type (FundSizeType): Fund size type (FUND, MATH_RES, ASSET_SHARE, etc.).
            fund_size_basis (AssetRepBasis): Basis for sizing against fund (usually FAV or BSV).
            target_weight (dict[str, TargetWeight]): Target allocation by group.
            assets_profile (list[Asset] | None=None): Profile assets for purchases (e.g., bonds).
        """
        t, p = self.time, self.period

        self.calculator.tdv_accum_free_proceeds_bd[t] = self.accum_free_proceeds
        # process rebalance
        fund_size = self._get_fund_size(fund_size_type, fund_size_basis)
        free_proceeds, realized_gl = self.allocator.rebalance(
            fund_size=fund_size,
            size_basis = fund_size_basis,
            target_weight=target_weight,
            assets_profile=assets_profile,
            kwargs=kwargs
        )
        for asset in self.assets:
            asset.complete_dealing()
        self.accum_free_proceeds += free_proceeds
        self.calculator.tdv_accum_free_proceeds_ad[t] = self.accum_free_proceeds
        self.calculator.process_assets_after_dealing()
        # reconcile realized gain/loss
        if abs((rgl := self.calculator.tdv_totass_rgl_ad[p]) - realized_gl) > 0.01: raise ValueError(
            f"Fund {self.fund_id} at {p=} realized gain/loss reconciliation break, "
            f"calculator: {rgl} <> allocator: {realized_gl}")

    def _get_fund_size(self, size_type: FundSizeType, size_basis: AssetRepBasis) -> float:
        """Get the fund size based on the fund size type and basis.

        Args:
            size_type (FundSizeType): Fund size type (FUND, MATH_RES, ASSET_SHARE, etc.).
            size_basis (AssetRepBasis): Asset reporting basis use for rebalance (usually FAV or BSV).

        Returns:
            float: Computed fund size on the requested basis.

        Raises:
            ValueError: If fund size type is invalid.
        """
        t = self.time
        if size_type == FundSizeType.FUND:
            # need to include accum_free_proceeds
            return self.calculator.tdv_totass_rep_value_bd[t][size_basis.value] + self.accum_free_proceeds
        elif size_type == FundSizeType.SURR_VALUE:
            return self.calculator.tdv_tot_surr_val[t]
        elif size_type == FundSizeType.MATH_RES:
            return self.calculator.tdv_tot_math_res[t]
        elif size_type == FundSizeType.ACCT_VALUE:
            return self.calculator.tdv_tot_acct_val_bd[t]
        elif size_type == FundSizeType.ASSET_SHARE:
            return self.calculator.tdv_tot_asset_share_bd[t]
        elif size_type == FundSizeType.MAX_AS_MATH:
            return max(self.calculator.tdv_tot_asset_share_bd[t], self.calculator.tdv_tot_math_res[t])
        elif size_type == FundSizeType.MAX_AS_CSV:
            return max(self.calculator.tdv_tot_asset_share_bd[t], self.calculator.tdv_tot_surr_val[t])

        raise ValueError(f"Invalid fund size type: {size_type}.")

    @check_calc_time({"proc_liabs_ad": -1, "proc_liabs_bd": 0, "proc_assets_ad": 0}, "proc_liabs_ad")
    def process_liabs_after_dealing(self) -> None:
        """Process liability values after dealing (ad)."""
        # note: liab.update_ad() is NOT automatically called here
        self.calculator.process_liabs_after_dealing()

    def _reset_accum_free_proceeds(self) -> None:
        self.accum_free_proceeds = 0.0

    @check_calc_time({"proc_assets_ad": 0})
    def transfer_free_proceeds_to_other(self, other: Optional['Fund']) -> None:
        """Transfer free proceeds to the other fund.

        Args:
            other (Optional[Fund]): Fund to receive proceeds (usually shareholder fund), or None.
        """
        t = self.time

        if other is None:
            pass
        elif type(other) == Fund:
            other.receive_free_proceeds(self.accum_free_proceeds)
        else:
            warnings.warn(f"Invalid {type(other)}, expected <class 'Fund'> ")

        self.calculator.tdv_proceeds_transferred_out[t] = max(self.accum_free_proceeds, 0.0)
        self.calculator.tdv_proceeds_transferred_in[t] = max(- self.accum_free_proceeds, 0.0)
        self._reset_accum_free_proceeds()

    def receive_free_proceeds(self, amount: float) -> None:
        """Receive free proceeds.

        Args:
            amount (float): Amount to receive (can be either positive ornegative).
        """
        t = self.time
        if self.calculator.tdv_proceeds_transferred_in[t] is None:
            self.calculator.tdv_proceeds_transferred_in[t] = max(amount, 0.0)
            self.calculator.tdv_proceeds_transferred_out[t] = max(- amount, 0.0)
        else:
            self.calculator.tdv_proceeds_transferred_in[t] += max(amount, 0.0)
            self.calculator.tdv_proceeds_transferred_out[t] += max(- amount, 0.0)

        self.accum_free_proceeds += amount

    def _rate_of_return_bd(self, t_or_p: int | pd.Period, basis: AssetRepBasis) -> float:
        """Rate of return before dealing.

        Args:
            t_or_p (int | pd.Period): Time period.
            basis (AssetRepBasis): Reporting basis.

        Returns:
            float: Rate of return (decimal) before dealing.
        """
        return self.calculator.tdv_totass_ror_pc_bd[t_or_p][basis.value] / 100

    def _rate_of_return_ad(self, t_or_p: int | pd.Period, basis: AssetRepBasis) -> float:
        """Rate of return after dealing.

        Args:
            t_or_p (int | pd.Period): Time period.
            basis (AssetRepBasis): Reporting basis.

        Returns:
            float: Rate of return (decimal) after dealing.
        """
        return self.calculator.tdv_totass_ror_pc_ad[t_or_p][basis.value] / 100

    def rate_of_return_mv_bd(self, t_or_p: int | pd.Period) -> float:
        """float: Rate of return (MV basis) before dealing."""
        return self._rate_of_return_bd(t_or_p, AssetRepBasis.MV)

    def rate_of_return_fav_bd(self, t_or_p: int | pd.Period) -> float:
        """float: Rate of return (FAV or PL basis) before dealing."""
        return self._rate_of_return_bd(t_or_p, AssetRepBasis.FAV)

    def rate_of_return_mv_ad(self, t_or_p: int | pd.Period) -> float:
        """float: Rate of return (MV basis) after dealing."""
        return self._rate_of_return_ad(t_or_p, AssetRepBasis.MV)

    def rate_of_return_fav_ad(self, t_or_p: int | pd.Period) -> float:
        """float: Rate of return (FAV or PL basis) after dealing."""
        return self._rate_of_return_ad(t_or_p, AssetRepBasis.FAV)

    def __str__(self) -> str:
        return self.fund_id

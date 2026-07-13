import pandas as pd
import warnings
from enum import Enum, unique
from typing import Optional

from vates._core.proj_model_engine import ProjModelEngine
from vates.utils import t_checker
from vates.alm.enums import AssetRepBasis
from vates.alm.assets import Asset, Cash
from vates.alm.liabs import Liab
from vates.alm.funds._asset_allocator import AssetAllocator, RebalancePolicyParams, TargetWeight
from vates.alm.funds._fund_calculator import FundCalculator
from vates.alm.funds._utils import ALContainer

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
        _container (ALContainer): Assets and liabilities held by the fund.
        _primary_cash_asset (Cash | None): Primary cash asset used for residual cash flows.
        _calculator (FundCalculator): Aggregation/returns calculator.
        _allocator (AssetAllocator): Asset allocator for rebalancing.
    """

    def __init__(
        self,
        fund_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        asset_allocator = None,
        rebalance_policy: dict[str, RebalancePolicyParams] = None,
        asset_categories: list[str] = None,
    ) -> None:
        """
        Initialize a Fund object.

        Args:
            fund_id (str): Fund identifier.
            rebalance_policy (dict[str, RebalancePolicyParams]): Rebalance policy by allocation group.
            asset_categories (list[str]): Asset categories to be reported.
        """
        if model_engine is not None:
            model_engine.attach_time_observer(self)
            self.time: int = model_engine.time
            self._start_date: pd.Period = model_engine.START_DATE

        self.fund_id = fund_id
        # Asset and liab collections
        self._container: ALContainer = ALContainer()
        self._primary_cash_asset: Cash | None = None
        self._assembled: bool = False

        self._calculator: FundCalculator = FundCalculator(
            fund_id=fund_id, model_engine=model_engine, container=self._container, asset_categories=asset_categories
        )
        self._allocator: AssetAllocator = asset_allocator or AssetAllocator(
            fund_id=fund_id, model_engine=model_engine, container=self._container, rebalance_policy=rebalance_policy
        )

    def sync_time(self, subject) -> None:
        self.time = subject.time

    @property
    def period(self) -> pd.Period:
        return self._start_date + self.time

    @property
    def assets(self) -> list[Asset]:
        return self._container.assets

    @property
    def liabs(self) -> list[Liab]:
        return self._container.liabs

    def assemble_on_start(self, *, existing_assets: Asset | list[Asset] | None,
                          existing_liabs: Liab | list[Liab] | None = None) -> None:
        """Assemble the fund.

        Aggregates asset and liability values, and marks the fund as assembled.

        Args:
            existing_assets (Asset | list[Asset] | None): Existing assets to be included.
            existing_liabs (Liab | list[Liab] | None): Existing liabilities to be included.

        Raises:

        """
        if self._assembled:
            raise ValueError(f"Fund has already been assembled.")

        if existing_assets is None:
            pass
        elif isinstance(existing_assets, list):
            self._container.assets.extend(existing_assets)
        else:
            self._container.assets.append(existing_assets)

        if existing_liabs is None:
            pass
        elif isinstance(existing_liabs, list):
            self._container.liabs.extend(existing_liabs)
        else:
            self._container.liabs.append(existing_liabs)

        # validate if any duplicate asset objects
        assets_set = set(self._container.assets)
        if len(self._container.assets) != len(assets_set):
            dup_lst = [x for x in assets_set if self._container.assets.count(x) > 1]
            warnings.warn(f"{dup_lst} duplicate asset objects, including '{dup_lst[:min(5, len(dup_lst))]}'.")

        # validate if any duplicate liability objects
        liabs_set = set(self._container.liabs)
        if len(self._container.liabs) != len(liabs_set):
            dup_lst = [x for x in liabs_set if self._container.liabs.count(x) > 1]
            warnings.warn(f"{dup_lst} duplicate liability objects, including '{dup_lst[:min(5, len(dup_lst))]}'.")

        # validate if any profile asset objects
        for asset in self._container.assets:
            if asset.is_profile:
                warnings.warn(f"Unexpected profile asset '{asset}'")

        # Aggregate asset value
        self._calculator.aggregate_assets_value("ad")
        self._calculator.aggregate_liabs_value("bd")
        self._calculator.aggregate_liabs_value("ad")

        self._assembled = True

    @property
    def primary_cash_asset(self) -> Cash | None:
        if self._primary_cash_asset is None:
            for asset in self._container.assets:
                if isinstance(asset, Cash):
                    self._primary_cash_asset = asset
                    break
            if self._primary_cash_asset is None:
                warnings.warn(f"No cash assets available.")
        return self._primary_cash_asset

    @t_checker({"proc_assets_bd": -1, "proc_assets_ad": -1}, "proc_assets_bd")
    def process_assets_before_dealing(self) -> None:
        """Process asset cash flows and reported values before dealing (bd)."""
        self._calculator.process_assets_before_dealing()
        self._container.deposit_free_proceeds(self._calculator.tdv_totass_cash_flow[self.time])

    @t_checker({"proc_liabs_bd": -1, "proc_liabs_ad": -1, "proc_assets_bd": 0}, "proc_liabs_bd")
    def process_liabs_before_dealing(self) -> None:
        """Process liability cash flows and balance sheet variables before dealing (bd)."""
        self._calculator.process_liabs_before_dealing()
        self._container.deposit_free_proceeds(self._calculator.tdv_totliab_cash_flow[self.time])

    @t_checker({"proc_assets_ad": -1, "proc_assets_bd": 0, "proc_liabs_bd": 0}, "proc_assets_ad")
    def skip_rebalance(self) -> None:
        """Skip asset rebalance and invest free proceeds into primary cash."""
        t = self.time
        self._calculator.tdv_accum_free_proceeds_bd[t] = self._container.accum_free_proceeds
        # just invest accum_free_proceeds into primary cash, no other rebalance, accum_free_proceeds is reset to zero
        self.primary_cash_asset.invest_new_money(self._container.withdrawal_accum_free_proceeds())
        for asset in self.assets:
            asset.close_dealing()
        self._calculator.tdv_accum_free_proceeds_ad[t] = self._container.accum_free_proceeds # should be zero
        self._calculator.process_assets_after_dealing()

    @t_checker({"proc_assets_ad": -1, "proc_assets_bd": 0, "proc_liabs_bd": 0}, "proc_assets_ad")
    def rebalance_assets(self, *, fund_size_type: FundSizeType, asset_size_basis: AssetRepBasis,
                         target_weight: dict[str, TargetWeight], assets_profile: list[Asset] | None = None, **kwargs
                         ) -> None:
        """Rebalance assets per target allocation and optional profile.

        Args:
            fund_size_type (FundSizeType): Fund size type (FUND, MATH_RES, ASSET_SHARE, etc.).
            asset_size_basis (AssetRepBasis): Basis for sizing against fund (usually FAV or BSV).
            target_weight (dict[str, TargetWeight]): Target allocation by group.
            assets_profile (list[Asset] | None=None): Profile assets for purchases (e.g., bonds).
        """
        t, p = self.time, self.period

        self._calculator.tdv_accum_free_proceeds_bd[t] = self._container.accum_free_proceeds
        # process rebalance
        fund_size = self._get_fund_size(fund_size_type, asset_size_basis)
        recon_rgl = self._allocator.rebalance(
            fund_size=fund_size,
            asset_size_basis=asset_size_basis,
            target_weight=target_weight,
            assets_profile=assets_profile,
            **kwargs
        )
        for asset in self.assets:
            asset.close_dealing()
        self._calculator.tdv_accum_free_proceeds_ad[t] = self._container.accum_free_proceeds
        self._calculator.process_assets_after_dealing()
        # reconcile realized gain/loss
        if abs((rgl := self._calculator.tdv_totass_rgl_ad[p]) - recon_rgl) > 0.01: raise ValueError(
            f"Fund {self.fund_id} at {p=} realized gain/loss reconciliation break, "
            f"calculator: {rgl} <> allocator: {recon_rgl}")

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
            return self._calculator.tdv_totass_rep_value_bd[t][size_basis.value] + self._container.accum_free_proceeds
        elif size_type == FundSizeType.SURR_VALUE:
            return self._calculator.tdv_tot_surr_val[t]
        elif size_type == FundSizeType.MATH_RES:
            return self._calculator.tdv_tot_math_res[t]
        elif size_type == FundSizeType.ACCT_VALUE:
            return self._calculator.tdv_tot_acct_val_bd[t]
        elif size_type == FundSizeType.ASSET_SHARE:
            return self._calculator.tdv_tot_asset_share_bd[t]
        elif size_type == FundSizeType.MAX_AS_MATH:
            return max(self._calculator.tdv_tot_asset_share_bd[t], self._calculator.tdv_tot_math_res[t])
        elif size_type == FundSizeType.MAX_AS_CSV:
            return max(self._calculator.tdv_tot_asset_share_bd[t], self._calculator.tdv_tot_surr_val[t])

        raise ValueError(f"Invalid fund size type: {size_type}.")

    @t_checker({"proc_liabs_ad": -1, "proc_liabs_bd": 0, "proc_assets_ad": 0}, "proc_liabs_ad")
    def process_liabs_after_dealing(self) -> None:
        """Process liability values after dealing (ad)."""
        # note: liab.update_ad() is NOT automatically called here
        self._calculator.process_liabs_after_dealing()

    # def _reset_accum_free_proceeds(self) -> None:
    #     self._accum_free_proceeds = 0.0

    @t_checker({"proc_assets_ad": 0})
    def transfer_free_proceeds_to_other(self, other: Optional['Fund']) -> None:
        """Transfer free proceeds to the other fund.

        Args:
            other (Optional[Fund]): Fund to receive proceeds (usually shareholder fund), or None.
        """
        t = self.time

        amount = self._container.withdrawal_accum_free_proceeds()
        if other is None:
            pass
        elif type(other) == Fund:
            other.receive_free_proceeds(amount)
        else:
            warnings.warn(f"Invalid {type(other)}, expected <class 'Fund'> ")

        self._calculator.tdv_proceeds_transferred_out[t] = max(amount, 0.0)
        self._calculator.tdv_proceeds_transferred_in[t] = max(- amount, 0.0)

    def receive_free_proceeds(self, amount: float) -> None:
        """Receive free proceeds.

        Args:
            amount (float): Amount to receive (can be either positive ornegative).
        """
        t = self.time
        if self._calculator.tdv_proceeds_transferred_in[t] is None:
            self._calculator.tdv_proceeds_transferred_in[t] = max(amount, 0.0)
            self._calculator.tdv_proceeds_transferred_out[t] = max(- amount, 0.0)
        else:
            self._calculator.tdv_proceeds_transferred_in[t] += max(amount, 0.0)
            self._calculator.tdv_proceeds_transferred_out[t] += max(- amount, 0.0)

        self._container.deposit_free_proceeds(amount)

    def _rate_of_return_bd(self, t_or_p: int | pd.Period, basis: AssetRepBasis) -> float:
        """Rate of return before dealing.

        Args:
            t_or_p (int | pd.Period): Time period.
            basis (AssetRepBasis): Reporting basis.

        Returns:
            float: Rate of return (decimal) before dealing.
        """
        return self._calculator.tdv_totass_ror_pc_bd[t_or_p][basis.value] / 100

    def _rate_of_return_ad(self, t_or_p: int | pd.Period, basis: AssetRepBasis) -> float:
        """Rate of return after dealing.

        Args:
            t_or_p (int | pd.Period): Time period.
            basis (AssetRepBasis): Reporting basis.

        Returns:
            float: Rate of return (decimal) after dealing.
        """
        return self._calculator.tdv_totass_ror_pc_ad[t_or_p][basis.value] / 100

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

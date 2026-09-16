import pandas as pd
import warnings
from enum import Enum, unique
from typing import Optional

from vates._core import ProjModelEngine, add_projection_time_synchronizer
from vates.utils import t_checker
from vates.alm.assets import Asset, Cash
from vates.alm.liabs import Liab
from vates.alm.funds._asset_allocator import AssetAllocator, RebalancePolicyParams, TargetWeight
from vates.alm.funds._fund_calculator import FundCalculator
from vates.alm.funds._utils import AssetLiabConnector, _RateOfReturnIndexer

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


@add_projection_time_synchronizer
class Fund:
    """Investment fund with assets and liabilities.

    Handles asset/liability management, reporting, and rebalance strategies for actuarial projections.

    Attributes:
        fund_id (str): Fund identifier.
        _connector (AssetLiabConnector): Assets and liabilities held by the fund.
        _primary_cash_asset (Cash | None): Primary cash asset used for residual cash flows.
        calculator (FundCalculator): Aggregation/returns calculator.
        _allocator (AssetAllocator): Asset allocator for rebalancing.
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    
    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_tt_dict',
                 'fund_id', '_connector', '_primary_cash_asset', '_assembled', 'calculator', '_allocator',
                 'rate_of_return_mv_bd', 'rate_of_return_mv_ad', 'rate_of_return_fav_bd', 'rate_of_return_fav_ad')

    def __init__(
        self,
        fund_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        asset_allocator = None,
        rebalance_policy: dict[str, RebalancePolicyParams] = None,
        asset_categories: list[str] = None,
        asset_report_bases: list[str] = None,
    ) -> None:
        """
        Initialize a Fund object.

        Args:
            fund_id (str): Fund identifier.
            rebalance_policy (dict[str, RebalancePolicyParams]): Rebalance policy by allocation group.
            asset_categories (list[str]): Asset categories to be reported.
        """
        self.fund_id = fund_id
        # Asset and liab collections
        self._connector: AssetLiabConnector = AssetLiabConnector()
        self._primary_cash_asset: Cash | None = None
        self._assembled: bool = False
        self._asset_report_bases: list[str] = asset_report_bases

        self.calculator: FundCalculator = FundCalculator(
            name=fund_id, model_engine=model_engine, connector=self._connector,
            asset_categories=asset_categories, asset_report_bases=asset_report_bases
        )
        self._allocator: AssetAllocator = asset_allocator or AssetAllocator(
            name=fund_id, model_engine=model_engine, connector=self._connector,
            rebalance_policy=rebalance_policy, asset_report_bases=asset_report_bases
        )

        # rate of return indexers
        self.rate_of_return_bd: _RateOfReturnIndexer = _RateOfReturnIndexer(
            self.calculator.tdv_totass_ror_pc_bd, divby=100)
        self.rate_of_return_ad: _RateOfReturnIndexer = _RateOfReturnIndexer(
            self.calculator.tdv_totass_ror_pc_ad, divby=100)

    @property
    def assets(self) -> list[Asset]:
        return self._connector.assets

    @property
    def liabs(self) -> list[Liab]:
        return self._connector.liabs

    def assemble_on_start(self, *, existing_assets: Asset | list[Asset] | None,
                          existing_liabs: Liab | list[Liab] | None = None) -> None:
        """Assemble the fund.

        Aggregates asset and liability values, and marks the fund as assembled.

        Args:
            existing_assets (Asset | list[Asset] | None): Existing assets to be included.
            existing_liabs (Liab | list[Liab] | None): Existing liabilities to be included.

        """
        if self._assembled:
            warnings.warn(f"Fund has already been assembled.")

        if existing_assets is None:
            pass
        elif isinstance(existing_assets, list):
            self._connector.assets.extend(existing_assets)
        else:
            self._connector.assets.append(existing_assets)

        if existing_liabs is None:
            pass
        elif isinstance(existing_liabs, list):
            self._connector.liabs.extend(existing_liabs)
        else:
            self._connector.liabs.append(existing_liabs)

        # validate if any duplicate asset objects
        assets_set = set(self._connector.assets)
        if len(self._connector.assets) != len(assets_set):
            dup_lst = [x for x in assets_set if self._connector.assets.count(x) > 1]
            warnings.warn(f"{dup_lst} duplicate asset objects, including '{dup_lst[:min(5, len(dup_lst))]}'.")

        # validate if any duplicate liability objects
        liabs_set = set(self._connector.liabs)
        if len(self._connector.liabs) != len(liabs_set):
            dup_lst = [x for x in liabs_set if self._connector.liabs.count(x) > 1]
            warnings.warn(f"{dup_lst} duplicate liability objects, including '{dup_lst[:min(5, len(dup_lst))]}'.")

        # validate if any profile asset objects
        for asset in self._connector.assets:
            if asset.is_profile:
                warnings.warn(f"Unexpected profile asset '{asset}'")

        # Aggregate asset value
        self.calculator.aggregate_assets_value("ad")
        self.calculator.aggregate_liabs_value("bd")
        self.calculator.aggregate_liabs_value("ad")

        self._assembled = True

    @property
    def primary_cash_asset(self) -> Cash | None:
        if self._primary_cash_asset is None:
            for asset in self._connector.assets:
                if isinstance(asset, Cash):
                    self._primary_cash_asset = asset
                    break
            if self._primary_cash_asset is None:
                warnings.warn(f"No cash assets available.")
        return self._primary_cash_asset

    @t_checker({"proc_assets_bd": -1, "proc_assets_ad": -1}, "proc_assets_bd")
    def process_assets_before_dealing(self) -> None:
        """Process asset cash flows and reported values before dealing (bd)."""
        self.calculator.process_assets_before_dealing()
        self._connector.accumulate_free_estate(self.calculator.tdv_totass_cash_flow[self.time])

    @t_checker({"proc_liabs_bd": -1, "proc_liabs_ad": -1, "proc_assets_bd": 0}, "proc_liabs_bd")
    def process_liabs_before_dealing(self) -> None:
        """Process liability cash flows and balance sheet variables before dealing (bd)."""
        self.calculator.process_liabs_before_dealing()
        self._connector.accumulate_free_estate(self.calculator.tdv_totliab_cash_flow[self.time])

    @t_checker({"proc_assets_ad": -1, "proc_assets_bd": 0, "proc_liabs_bd": 0}, "proc_assets_ad")
    def no_action_on_rebalance(self) -> None:
        """Skip asset rebalance, invest free proceeds into primary cash."""
        t = self.time
        self.calculator.tdv_free_estate_bd[t] = self._connector.free_estate
        # just invest free_estate into primary cash, no other action, free_estate is reset to zero
        self.primary_cash_asset.invest_new_money(self._connector.dispose_free_estate())
        for asset in self.assets:
            asset.close_dealing()
        self.calculator.tdv_free_estate_ad[t] = self._connector.free_estate # should be zero
        self.calculator.process_assets_after_dealing()

    @t_checker({"proc_assets_ad": -1, "proc_assets_bd": 0, "proc_liabs_bd": 0}, "proc_assets_ad")
    def rebalance_assets(self, *, fund_size_type: str | FundSizeType, asset_size_basis: str,
                         target_weight: dict[str, TargetWeight], assets_profile: list[Asset] | None = None, **kwargs
                         ) -> None:
        """Rebalance assets per target allocation and optional profile.

        Args:
            fund_size_type (str | FundSizeType): Fund size type (FUND, MATH_RES, ASSET_SHARE, etc.).
            asset_size_basis (str): Basis for sizing against fund (usually FAV or BSV).
            target_weight (dict[str, TargetWeight]): Target allocation by group.
            assets_profile (list[Asset] | None=None): Profile assets for purchases (e.g., bonds).
        """
        t, p = self.time, self.period
        fund_size_type = FundSizeType[fund_size_type.upper()] if isinstance(fund_size_type, str) else fund_size_type

        self.calculator.tdv_free_estate_bd[t] = self._connector.free_estate
        # process rebalance
        fund_size = self._get_fund_size(fund_size_type=fund_size_type, asset_size_basis=asset_size_basis)
        recon_rgl = self._allocator.rebalance(
            fund_size=fund_size,
            asset_size_basis=asset_size_basis,
            target_weight=target_weight,
            assets_profile=assets_profile,
            **kwargs
        )
        for asset in self.assets:
            asset.close_dealing()
        self.calculator.tdv_free_estate_ad[t] = self._connector.free_estate
        self.calculator.process_assets_after_dealing()
        # reconcile realized gain/loss
        size_basis_index = self._asset_report_bases.index(asset_size_basis)
        mv_basis_index = self._asset_report_bases.index("MV")

        val_bd = self.calculator.tdv_totass_rep_value_bd[t][mv_basis_index] - self.calculator.tdv_totass_rep_value_bd[t][size_basis_index]
        val_ad = self.calculator.tdv_totass_rep_value_ad[t][mv_basis_index] - self.calculator.tdv_totass_rep_value_ad[t][size_basis_index]

        if abs((val_bd - val_ad) - recon_rgl) > 0.01:
            warnings.warn(f"Fund {self.fund_id} at {p=} realized gain/loss reconciliation break, "
                          f"calculator: {val_bd - val_ad} != allocator: {recon_rgl}")

    def _get_fund_size(self, *, fund_size_type: FundSizeType, asset_size_basis: str) -> float:
        """Get the fund size based on the fund size type and basis.

        Args:
            fund_size_type (str): Fund size type (FUND, MATH_RES, ASSET_SHARE, etc.).
            asset_size_basis (str): Asset reporting basis use for rebalance (usually FAV or BSV).

        Returns:
            float: Computed fund size on the requested basis.

        Raises:
            ValueError: If fund size type is invalid.
        """
        if fund_size_type == FundSizeType.FUND:
            # return self._connector.get_totass_value(asset_size_basis, include_free_estate=True)
            return self._connector.groupby_sum_asset_report_value(basis=asset_size_basis) + self._connector.free_estate
            # # need to include free_estate
        elif fund_size_type == FundSizeType.SURR_VALUE:
            return self._connector.totliab_surr_value
        elif fund_size_type == FundSizeType.MATH_RES:
            return self._connector.totliab_math_res
        elif fund_size_type == FundSizeType.ACCT_VALUE:
            return self._connector.totliab_acct_value
        elif fund_size_type == FundSizeType.ASSET_SHARE:
            return self._connector.totliab_asset_share
        elif fund_size_type == FundSizeType.MAX_AS_MATH:
            return max(self._connector.totliab_asset_share, self._connector.totliab_math_res)
        elif fund_size_type == FundSizeType.MAX_AS_CSV:
            return max(self._connector.totliab_asset_share, self._connector.totliab_surr_value)
        raise ValueError(f"Unknown fund size type: {fund_size_type}.")

    @t_checker({"proc_liabs_ad": -1, "proc_liabs_bd": 0, "proc_assets_ad": 0}, "proc_liabs_ad")
    def process_liabs_after_dealing(self) -> None:
        """Process liability values after dealing (ad). Note: liab.update_ad() is NOT automatically called here."""
        self.calculator.process_liabs_after_dealing()

    @t_checker({"proc_assets_ad": 0})
    def transfer_free_proceeds_to_other(self, other: Optional['Fund']) -> None:
        """Transfer free proceeds to the other fund.

        Args:
            other (Optional[Fund]): Fund to receive proceeds (usually shareholder fund), or None.
        """
        t = self.time

        amount = self._connector.dispose_free_estate()
        if other is None:
            pass
        elif isinstance(other, Fund):
            other.receive_free_proceeds(amount)
        else:
            warnings.warn(f"Invalid {type(other)}, expected <class 'Fund'> ")

        self.calculator.tdv_proceeds_transferred_out[t] = max(amount, 0.0)
        self.calculator.tdv_proceeds_transferred_in[t] = max(- amount, 0.0)

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

        self._connector.accumulate_free_estate(amount)

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.fund_id}'"

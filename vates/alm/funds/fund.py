import pandas as pd
import warnings
from typing import Callable, Optional

from vates._core import ProjModelEngine, add_projection_time_synchronizer
from vates.utils import maybe_raise_if_ne
from vates.alm.assets import Asset, Cash
from vates.alm.liabs import Liab
from vates.alm.funds._allocator import AssetAllocator, AssetAllocationGroup, TargetWeight
from vates.alm.funds._connector import AssetLiabConnector
from vates.alm.funds._recorder import FundRecorder
from vates.alm.funds._utils import _RateOfReturnIndexer


@add_projection_time_synchronizer
class Fund:
    """Investment fund with assets and liabilities.

    Handles asset/liability management, reporting, and rebalance strategies for actuarial projections.

    Attributes:
        fund_id (str): Fund identifier.
        _connector (AssetLiabConnector): Assets and liabilities held by the fund.
        _primary_cash_asset (Cash | None): Primary cash asset used for residual cash flows.
        _recorder (FundRecorder): Aggregation/returns calculator.
        _allocator (AssetAllocator): Asset allocator for rebalancing.
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    
    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_state',
                 'fund_id', '_connector', '_primary_cash_asset', '_asset_report_bases', '_recorder', '_allocator',
                 'rate_of_return_bd', 'rate_of_return_ad', )

    def __init__(
        self,
        fund_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        asset_allocation_groups: list[AssetAllocationGroup] = None,
        asset_report_bases: list[str] = None,
        asset_categories: list[str] = None,
        asset_category_attr: str = "category",
        asset_allocation_group_attr: str = "allocation_group",
        output_liab_attrs_bd: list[str] = None,
        output_liab_attrs_ad: list[str] = None,
    ) -> None:
        """
        Initialize a Fund object.

        Args:
            fund_id (str): Fund identifier.
            asset_allocation_groups (list[AssetAllocationGroup]): List of asset allocation groups.
            asset_report_bases (list[str]): Asset reporting bases.
            asset_categories (list[str]): Asset categories to be reported.
            asset_category_attr (str): Named attribute for asset category, defaults to "category".
            asset_allocation_group_attr (str): Named attribute for asset allocation group, defaults to "allocation_group".
            output_liab_attrs_bd (list[str]): List of liability attributes to output.
            output_liab_attrs_ad (list[str]): List of liability attributes to output.
        """
        self.fund_id = fund_id
        # Asset and liab collections
        self._connector: AssetLiabConnector = AssetLiabConnector()
        self._primary_cash_asset: Cash | None = None
        self._asset_report_bases: list[str] = asset_report_bases or []

        self._recorder: FundRecorder = FundRecorder(
            name=fund_id, model_engine=model_engine, connector=self._connector,
            asset_report_bases=self._asset_report_bases,
            asset_categories=asset_categories, asset_category_attr=asset_category_attr,
            output_liab_attrs_bd=output_liab_attrs_bd or [], output_liab_attrs_ad=output_liab_attrs_ad or []
        )
        self._allocator: AssetAllocator = AssetAllocator(
            name=fund_id, model_engine=model_engine, connector=self._connector,
            asset_report_bases=self._asset_report_bases,
            allocation_groups=asset_allocation_groups, allocation_group_attr= asset_allocation_group_attr,
        )

        # rate of return indexers
        self.rate_of_return_bd: _RateOfReturnIndexer = _RateOfReturnIndexer(
            self._recorder.tdv_totass_ror_pc_bd, divby=100)
        self.rate_of_return_ad: _RateOfReturnIndexer = _RateOfReturnIndexer(
            self._recorder.tdv_totass_ror_pc_ad, divby=100)

        self._state: tuple[str, int] = ("initialized", self.time or 0)

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
        maybe_raise_if_ne(self._state, ("initialized", self.time))

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
        self._recorder.sum_asset_report_values("ad")
        self._recorder.sum_liab_attrs("bd")
        self._recorder.sum_liab_attrs("ad")

        self._state = ("assembled", self.time)

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

    def process_assets_before_dealing(self) -> None:
        """Process asset cash flows and reported values before dealing (bd)."""
        if self._state != ("assembled", self.time - 1):
            maybe_raise_if_ne(self._state, ("closed", self.time - 1))
        self._recorder.record_asset_before_dealing()
        self._connector.accumulate_free_estate(self._recorder.tdv_totass_cf[self.time])
        self._state = ("proc_assets_bd", self.time)

    def process_liabs_before_dealing(self) -> None:
        """Process liability cash flows and balance sheet variables before dealing (bd)."""
        maybe_raise_if_ne(self._state, ("proc_assets_bd", self.time))
        self._recorder.record_liab_before_dealing()
        self._connector.accumulate_free_estate(self._recorder.tdv_totliab_cf[self.time])
        self._state = ("proc_liabs_bd", self.time)

    def no_action_on_rebalance(self) -> None:
        """Skip asset rebalance, invest free proceeds into primary cash."""
        maybe_raise_if_ne(self._state, ("proc_liabs_bd", self.time))
        t = self.time
        self._recorder.record_free_estate("bd")
        # just invest free_estate into primary cash, no other action, free_estate is reset to zero
        self.primary_cash_asset.invest_new_money(self._connector.dispose_free_estate())
        for asset in self.assets:
            asset.close_dealing()
        self._recorder.record_free_estate("ad")  # free_estate should be zero
        self._recorder.record_asset_after_dealing()
        self._state = ("closed", self.time)

    def rebalance_assets(self, *, total_size: float, asset_size_basis: str,
                         target_weights: dict[str, TargetWeight] | None, profile_assets: list[Asset] | None = None,
                         **kwargs) -> None:
        """Rebalance assets per target allocation and optional profile.

        Args:
            total_size (float): Total size for allocation.
            asset_size_basis (str): Basis for sizing against fund (usually FAV or BSV).
            target_weights (dict[str, TargetWeight]): Target weight by allocation group.
            profile_assets (list[Asset] | None=None): Profile assets for purchases (e.g., bonds).
        """
        maybe_raise_if_ne(self._state, ("proc_liabs_bd", self.time))
        t, p = self.time, self.period

        self._recorder.record_free_estate("bd")
        # process rebalance
        self._allocator.rebalance(
            total_size=total_size,
            size_basis=asset_size_basis,
            target_weights=target_weights,
            profile_assets=profile_assets,
            **kwargs
        )
        for asset in self.assets:
            asset.close_dealing()
        self._recorder.record_free_estate("ad")  # free_estate should be zero
        self._recorder.record_asset_after_dealing()
        self._state = ("closed", self.time)

    def get_size(self, *, func: Callable | None = None, key: str | tuple[str, ...] | dict[str, str],
                 treat_missing_as_0: bool = False) -> float:
        """ Get the fund size of the requested key

        Args:
            func (Callable | None): Function, defaults to None.
            key (str | tuple[str, ...] | dict[str, str]): Requested key, use `asset.<attr_name>` and/or `liab.<attr_name>`
                to indicate an attribute of assets or liabilities.
            treat_missing_as_0 (bool): True if treat value of missing attribute as 0.0, defaults to False.

        Examples:
            1. get_size(key="liab.math_res_if")
            2. get_size(func=lambda x, y: max(x, y), key=("liab.surr_val_bd", "liab.math_res_if")
            3. get_size(func=lambda x, y: max(x, y), key={"x": "liab.surr_val_bd", "y": "liab.math_res_if"})
            4. get_size(func=lambda x, y: x + y, key=(f"asset.FAV", "free_estate"))

        """
        return self._connector.get_size(func=func, key=key, treat_missing_as_0=treat_missing_as_0)

    def process_liabs_after_dealing(self) -> None:
        """Process liability values after dealing (ad). Note: liab.update_ad() is NOT automatically called here."""
        self._recorder.record_liab_after_dealing()

    def transfer_free_estate_to_other(self, other: Optional['Fund']) -> None:
        """Transfer free estate to the other fund.

        Args:
            other (Optional[Fund]): Fund to receive proceeds (usually shareholder fund), or None.
        """
        if other is self:
            raise ValueError(f"Cann\'t transfer to self.")

        amount = self._connector.dispose_free_estate()
        self._recorder.record_free_proceeds_transfer(- amount)

        if isinstance(other, Fund):
            other.receive_free_proceeds(amount)
        else:
            pass

    def receive_free_proceeds(self, amount: float) -> None:
        """Receive free proceeds.

        Args:
            amount (float): Amount to receive (can be either positive ornegative).
        """
        self._connector.accumulate_free_estate(amount)
        self._recorder.record_free_proceeds_transfer(amount)

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.fund_id}'"

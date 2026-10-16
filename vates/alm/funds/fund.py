import pandas as pd
import warnings
from enum import Enum, auto
from typing import Callable, Self

from vates._core import ProjModelEngine, time_synchronized
from vates.utils import Lifecycle, transition
from vates.alm.assets import Asset, Cash
from vates.alm.liabs import Liab
from vates.alm.funds._allocator import AssetAllocator, AssetAllocationGroup
from vates.alm.funds._recorder import AssetRecorder, LiabRecorder, OtherRecorder
from vates.alm.funds._utils import Connector, UninvestedProceeds, _RateOfReturnIndexer

class FundPhase(Enum):
    PROC_ASSET_BD = auto()
    PROC_LIAB_BD = auto()
    REBALANCED = auto()
    PROC_ASSET_AD = auto()
    PROC_LIAB_AD = auto()
    CLOSED = auto()

@time_synchronized("model_engine.time_synchronizer")
class Fund:
    """Investment fund with assets and liabilities.

    Handles asset/liability management, reporting, and rebalance strategies for actuarial projections.

    Attributes:
        fund_id (str): Fund identifier.
        _asset_conn (AssetConnector): Assets held by the fund.
        _liab_conn (LiabConnector): Liabilities held by the fund.
        _uninvested_proceeds (UninvestedProceeds): Balance of uninvested proceeds.
        _primary_cash_asset (Cash | None): Primary cash asset used for residual cash flows.
        _asset_rec (AssetRecorder): Assets recorder (reported value, returns calculator).
        _liab_rec (LiabRecorder): Liabilities recorder.
        _other_rec (OtherRecorder): Other recorder.
        _allocator (AssetAllocator): Asset allocator for rebalancing.
    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`
    
    __slots__ = ('__dict__', '__weakref__', 'time', 'period', '_lc', 'fund_id', '_asset_conn', '_liab_conn',
                 '_uninvested_proceeds', '_primary_cash_asset',
                 '_asset_rec', '_liab_rec', '_other_rec',
                 '_allocator',
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
        self._asset_conn: Connector[Asset] = Connector()
        self._liab_conn: Connector[Liab] = Connector()
        self._uninvested_proceeds: UninvestedProceeds = UninvestedProceeds()
        self._primary_cash_asset: Cash | None = None
        asset_report_bases: list[str] = asset_report_bases or []

        self._asset_rec: AssetRecorder = AssetRecorder(
            name=fund_id, model_engine=model_engine, asset_connector=self._asset_conn,
            asset_report_bases=asset_report_bases, asset_categories=asset_categories, asset_category_attr=asset_category_attr,
        )
        self._liab_rec: LiabRecorder = LiabRecorder(
            name=fund_id, model_engine=model_engine, liab_connector=self._liab_conn,
            output_liab_attrs_bd=output_liab_attrs_bd or [], output_liab_attrs_ad=output_liab_attrs_ad or []
        )
        self._other_rec: OtherRecorder = OtherRecorder(name=fund_id, model_engine=model_engine)

        self._allocator: AssetAllocator = AssetAllocator(
            name=fund_id, model_engine=model_engine, connector=self._asset_conn,
            uninvested_proceeds=self._uninvested_proceeds, asset_report_bases=asset_report_bases,
            allocation_groups=asset_allocation_groups, allocation_group_attr= asset_allocation_group_attr,
        )

        # rate of return indexers
        self.rate_of_return_bd: _RateOfReturnIndexer = _RateOfReturnIndexer(
            self._asset_rec.tdv_totass_ror_pc_bd, divby=100)
        self.rate_of_return_ad: _RateOfReturnIndexer = _RateOfReturnIndexer(
            self._asset_rec.tdv_totass_ror_pc_ad, divby=100)

        self._lc: Lifecycle[FundPhase] = Lifecycle[FundPhase]()

    @property
    def assets(self) -> list[Asset]:
        return self._asset_conn.objs

    @property
    def uninvested_proceeds(self) -> float:
        return self._uninvested_proceeds.value

    @property
    def liabs(self) -> list[Liab]:
        return self._liab_conn.objs

    @property
    def total_asset_cash_flow(self) -> float:
        return self._asset_rec.total_cash_flow

    @property
    def total_liab_cash_flow(self) -> float:
        return self._liab_rec.total_cash_flow

    def assemble_on_start(self, *, existing_assets: Asset | list[Asset] | None,
                          existing_liabs: Liab | list[Liab] | None = None) -> None:
        """Assemble the fund.

        Aggregates asset and liability values, and marks the fund as assembled.

        Args:
            existing_assets (Asset | list[Asset] | None): Existing assets to be included.
            existing_liabs (Liab | list[Liab] | None): Existing liabilities to be included.

        """
        if len(self._lc.snapshot) > 0:
            raise ValueError(f"Fund should assemble before any action, actual: {self._lc._describe_actual()}")

        if existing_assets is None:
            pass
        elif isinstance(existing_assets, list):
            self._asset_conn.extend(existing_assets)
        else:
            self._asset_conn.append(existing_assets)

        if existing_liabs is None:
            pass
        elif isinstance(existing_liabs, list):
            self._liab_conn.extend(existing_liabs)
        else:
            self._liab_conn.append(existing_liabs)

        # validate if any duplicate asset objects
        assets_set = set(self._asset_conn.objs)
        if len(self._asset_conn) != len(assets_set):
            dup_lst = [x for x in assets_set if self._asset_conn.count(x) > 1]
            warnings.warn(f"{dup_lst} duplicate asset objects, including '{dup_lst[:min(5, len(dup_lst))]}'.")

        # validate if any duplicate liability objects
        liabs_set = set(self._liab_conn.objs)
        if len(self._liab_conn) != len(liabs_set):
            dup_lst = [x for x in liabs_set if self._liab_conn.count(x) > 1]
            warnings.warn(f"{dup_lst} duplicate liability objects, including '{dup_lst[:min(5, len(dup_lst))]}'.")

        # validate if any profile asset objects
        for asset in self._asset_conn:
            if asset.is_profile:
                warnings.warn(f"Unexpected profile asset '{asset}'")

        # Aggregate asset value
        self._asset_rec.record_after_dealing(require_investment_return=False)
        self._liab_rec.record_before_dealing()
        self._liab_rec.record_after_dealing()

        self._lc.mark(FundPhase.CLOSED, self.time or 0)

    @property
    def primary_cash_asset(self) -> Cash | None:
        if self._primary_cash_asset is None or not self._primary_cash_asset.is_alive:
            for asset in self._asset_conn:
                if isinstance(asset, Cash) and asset.is_alive:
                    self._primary_cash_asset = asset
                    break
            if self._primary_cash_asset is None:
                warnings.warn(f"No cash assets available.")
        return self._primary_cash_asset

    @transition(require_not=FundPhase.PROC_ASSET_BD, mark=FundPhase.PROC_ASSET_BD)
    def process_assets_before_dealing(self) -> None:
        """Process asset cash flows and reported values before dealing (bd)."""
        self._asset_rec.record_before_dealing()
        self._uninvested_proceeds.increase(self._asset_rec.total_cash_flow)

    @transition(require_not=FundPhase.PROC_LIAB_BD, mark=FundPhase.PROC_LIAB_BD)
    def process_liabs_before_dealing(self) -> None:
        """Process liability cash flows and balance sheet variables before dealing (bd)."""
        self._liab_rec.record_before_dealing()
        self._uninvested_proceeds.increase(self._liab_rec.total_cash_flow)

    @transition(require_not=FundPhase.REBALANCED, mark=FundPhase.REBALANCED)
    def no_action_on_rebalance(self) -> None:
        """Skip asset rebalance, invest free proceeds into primary cash."""
        self._other_rec.record_uninvested_proceeds(self._uninvested_proceeds.value, "bd")
        # just invest free_estate into primary cash, no other action, free_estate is reset to zero
        self.primary_cash_asset.invest_new_money(self._uninvested_proceeds.take_out())
        for asset in self.assets:
            asset.close_dealing()
        self._other_rec.record_uninvested_proceeds(self._uninvested_proceeds.value, "ad")  # free_estate should be zero
        self._asset_rec.record_after_dealing()

    @transition(require_not=FundPhase.REBALANCED, mark=FundPhase.REBALANCED)
    def rebalance_assets(self, *, total_size: float,
                         profile_assets: list[Asset] | None = None, **kwargs) -> None:
        """Rebalance assets per target allocation and optional profile.

        Args:
            total_size (float): Total size for allocation.
            profile_assets (list[Asset]): Profile assets for reference.
        """
        self._other_rec.record_uninvested_proceeds(self._uninvested_proceeds.value, "bd")
        # process rebalance
        self._allocator.rebalance(total_size=total_size, profile_assets=profile_assets, **kwargs)
        for asset in self.assets:
            asset.close_dealing()
        self._other_rec.record_uninvested_proceeds(self._uninvested_proceeds.value, "ad")  # free_estate should be zero
        self._asset_rec.record_after_dealing()

    def get_total(self, *, func: Callable | None = None, key: str | tuple[str, ...] | dict[str, str],
                  allow_not_exist: bool = False) -> float:
        """ Get the fund total value of the requested key

        Args:
            func (Callable | None): Function, defaults to None.
            key (str | tuple[str, ...] | dict[str, str]): Requested key, use `asset.<attr_name>` and/or `liab.<attr_name>`
                to indicate an attribute of assets or liabilities.
            allow_not_exist (bool): True if allow not-exist attribute and return 0.0, defaults to False.

        Examples:
            1. get_total(key="liab.math_res_if")
            2. get_total(func=lambda x, y: max(x, y), key=("liab.surr_val_bd", "liab.math_res_if")
            3. get_total(func=lambda x, y: max(x, y), key={"x": "liab.surr_val_bd", "y": "liab.math_res_if"})
            4. get_total(func=lambda x, y: x + y, key=(f"asset.FAV", "uninvested_proceeds"))

        """
        if func is None:
            return self._get_total(key, allow_not_exist)
        if isinstance(key, str):
            return func(self._get_total(key, allow_not_exist))
        elif isinstance(key, tuple):
            args = tuple([self._get_total(x, allow_not_exist) for x in key])
            return func(*args)
        elif isinstance(key, dict):
            kwargs = {k: self._get_total(v, allow_not_exist) for k, v in key.items()}
            return func(**kwargs)
        raise TypeError(f"Invalid {type(key)=}, expected 'str', 'tuple', 'dict'.")

    def _get_total(self, key: str, allow_not_exist: bool = False) -> float:
        if not isinstance(key, str):
            raise TypeError(f"Invalid {type(key)=}, expected 'str'.")

        if "." not in key:
            return getattr(self, key, 0.0 if allow_not_exist else None)

        conn, name, *_ = key.split(".")
        if conn in ("asset", "assets"):
            return self._asset_conn.sum_attr(name, allow_not_exist=allow_not_exist)
        elif conn in ("liab", "liabs", "liability", 'liabilities'):
            return self._liab_conn.sum_attr(name, allow_not_exist=allow_not_exist)
        else:
            raise ValueError(f"Invalid '{key}', expected 'asset.attr' or 'liab.attr'.")

    @transition(require_all=FundPhase.REBALANCED, require_not=FundPhase.PROC_LIAB_AD, mark=FundPhase.PROC_LIAB_AD)
    def process_liabs_after_dealing(self) -> None:
        """Process liability values after dealing (ad). Note: liab.update_ad() is NOT automatically called here."""
        self._liab_rec.record_after_dealing()

    def take_out_uninvested_proceeds(self, *, transfer_to: Self | None = None) -> float:
        """Transfer free estate to the other fund.

        Args:
            transfer_to (Self | None): Fund to receive proceeds (usually shareholder fund), defaults to None.
        """
        if transfer_to is self:
            raise ValueError(f"{self.fund_id}: cann't transfer to self.")

        amount = self._uninvested_proceeds.take_out()
        self._other_rec.record_free_proceeds_transfer(- amount)

        if isinstance(transfer_to, Fund):
            transfer_to.receive_free_proceeds(amount)
            return 0.0

        return amount

    def receive_free_proceeds(self, amount: float) -> None:
        """Receive free proceeds.

        Args:
            amount (float): Amount to receive (can be either positive ornegative).
        """
        self._uninvested_proceeds.increase(amount)
        self._other_rec.record_free_proceeds_transfer(amount)

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.fund_id}'"

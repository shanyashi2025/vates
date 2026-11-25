import pandas as pd
import warnings
from collections.abc import Iterable
from typing import Literal

from vates.global_conf import GLOBAL_CHECK_MODE, CheckMode
from vates._core import ProjModelEngine, time_synchronized
from vates.alm.allocator import UnallocatedCash
from vates.alm.assets import BaseAsset, Cash
from vates.alm.liabs import BaseLiab
from vates.alm._utils import ObjList
from vates.alm.funds._recorder import AssetRecorder, LiabRecorder, OtherRecorder


_default_on_undefined = "zerofill" if GLOBAL_CHECK_MODE == CheckMode.BYPASS else "error"

@time_synchronized("model_engine")
class FundAccount:
    """Container for assets, liabilities, and account-level cash flows.

    Provides projection-time storage, aggregation, reporting, and interactions
    with unallocated cash for a fund account. Asset dealing and rebalancing are
    intentionally handled outside this class.

    Attributes:
        fund_id (str): Fund identifier.
        _assets (ObjList): Assets held by the fund.
        _liabs (ObjList): Liabilities held by the fund.
        _unalloc_cash (UnallocatedCash): Cash that has not been allocated to a specific asset.
        _primary_cash_asset (Cash | None): Primary cash asset used for residual cash flows.
        _asset_rec (AssetRecorder): Assets recorder (reported value, returns calculator).
        _liab_rec (LiabRecorder): Liabilities recorder.
        _other_rec (OtherRecorder): Other recorder.
    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', 'fund_id', '_assets', '_liabs',
                 '_unalloc_cash', '_primary_cash_asset', '_allocator',
                 '_asset_rec', '_liab_rec', '_other_rec',)

    def __init__(
        self,
        fund_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        asset_report_bases: list[str] | tuple[str, ...] = (),
        asset_categories: list[str] | tuple[str, ...] = (),
        asset_category_attr_name: str = "category",
        output_liab_attrs_bd: list[str] | tuple[str, ...] = (),
        output_liab_attrs_ad: list[str] | tuple[str, ...] = (),
        other_cash_flow_kinds: list[str] | tuple[str, ...] = (),
    ) -> None:
        """
        Initialize a FundAccount.

        Args:
            fund_id (str): Fund identifier.
            model_engine (ProjModelEngine | None): Projection engine used to synchronize time and create output variables.
            asset_report_bases (list[str] | tuple[str, ...]): Reported asset bases.
            asset_categories (list[str] | tuple[str, ...]): Asset categories to report.
            asset_category_attr_name (str): Asset attribute containing the category, defaults to "category".
            output_liab_attrs_bd (list[str] | tuple[str, ...]): Liability attributes to output before dealing.
            output_liab_attrs_ad (list[str] | tuple[str, ...]): Liability attributes to output after dealing.
            other_cash_flow_kinds (list[str] | tuple[str, ...]): Registered account-level cash-flow kinds. A kind can
                be recorded with `apply_other_cash_flow` and retrieved with `get_other_cash_flow`.
        """
        self.fund_id = fund_id
        # Asset and liab collections
        self._assets: ObjList[BaseAsset] = ObjList[BaseAsset]()
        self._liabs: ObjList[BaseLiab] = ObjList[BaseLiab]()
        self._unalloc_cash: UnallocatedCash = UnallocatedCash()
        self._primary_cash_asset: Cash | None = None

        self._asset_rec: AssetRecorder = AssetRecorder(
            name=fund_id, model_engine=model_engine, assets=self._assets,
            asset_report_bases=asset_report_bases, asset_categories=asset_categories,
            asset_category_attr_name=asset_category_attr_name,)
        self._liab_rec: LiabRecorder = LiabRecorder(
            name=fund_id, model_engine=model_engine, liabs=self._liabs,
            output_liab_attrs_bd=output_liab_attrs_bd, output_liab_attrs_ad=output_liab_attrs_ad)
        self._other_rec: OtherRecorder = OtherRecorder(
            name=fund_id, model_engine=model_engine, other_cash_flow_kinds=other_cash_flow_kinds)

    @property
    def assets(self) -> ObjList[BaseAsset]:
        return self._assets

    @property
    def unallocated_cash_balance(self) -> float:
        """Current balance of cash that has not been allocated to an asset."""
        return self._unalloc_cash.balance

    @property
    def unallocated_cash(self) -> UnallocatedCash:
        """Cash balance holder used for free-proceeds and rebalancing operations."""
        return self._unalloc_cash

    def accrue_unallocated_cash_interest(self, rate: float) -> None:
        """Apply an interest rate to the unallocated-cash balance.

        Args:
            rate (float): Periodic interest rate. For example, 0.01 represents 1%.
        """
        self._unalloc_cash.accrue_interest(rate)

    @property
    def liabilities(self) -> ObjList[BaseLiab]:
        return self._liabs

    @property
    def asset_cash_flow(self) -> float:
        return self._asset_rec.total_cash_flow

    @property
    def asset_market_value_bd(self) -> float:
        return self._asset_rec.total_market_value_bd

    @property
    def asset_market_value_ad(self) -> float:
        return self._asset_rec.total_market_value_ad

    def get_asset_category_cash_flow(self, category: str) -> float:
        return self._asset_rec.get_category_cash_flow(category)

    def report_asset_value_bd(self, basis: str, category: str | None = None) -> float:
        return self._asset_rec.report_bd("report_value", basis=basis, category=category)

    def report_investment_return_bd(self, basis: str, category: str | None = None) -> float:
        return self._asset_rec.report_bd("investment_return", basis=basis, category=category)

    def report_rate_of_return_bd(self, basis: str, category: str | None = None) -> float:
        return self._asset_rec.report_bd("rate_of_return", basis=basis, category=category)

    def report_asset_value_ad(self, basis: str, category: str | None = None) -> float:
        return self._asset_rec.report_ad("report_value", basis=basis, category=category)

    def report_investment_return_ad(self, basis: str, category: str | None = None) -> float:
        return self._asset_rec.report_ad("investment_return", basis=basis, category=category)

    def report_rate_of_return_ad(self, basis: str, category: str | None = None) -> float:
        return self._asset_rec.report_ad("rate_of_return", basis=basis, category=category)

    @property
    def liability_cash_flow(self) -> float:
        return self._liab_rec.total_cash_flow

    def add_asset(self, obj: BaseAsset | Iterable[BaseAsset], /) -> None:
        """Add asset object(s) to the fund account.

        Args:
            obj (BaseAsset | Iterable[Asset]): Asset(s) to be included.

        """
        if isinstance(obj, Iterable):
            self._assets.extend(obj)
        else:
            self._assets.append(obj)

    def add_liability(self, obj: BaseLiab | Iterable[BaseLiab], /) -> None:
        """Add liability object(s) to the fund account.

        Args:
            obj (BaseLiab | Iterable[BaseLiab]): Liability(s) to be included.

        """
        if isinstance(obj, Iterable):
            self._liabs.extend(obj)
        else:
            self._liabs.append(obj)

    def record_on_start(self) -> None:
        """Record the initial asset, liability, and free-proceeds values.

        This is normally called at projection time 0.
        """
        self._asset_rec.record_before_dealing(require_cash_flow=False, require_investment_return=False)
        self._asset_rec.record_after_dealing(require_investment_return=False)
        self._liab_rec.record_before_dealing(require_cash_flow=False)
        self._liab_rec.record_after_dealing()
        self._other_rec.record_unalloc_cash(self._unalloc_cash.balance, "bd")
        self._other_rec.record_unalloc_cash(self._unalloc_cash.balance, "ad")
        self._other_rec.record_unalloc_cash(self._unalloc_cash.balance, "eop")

    def _get_primary_cash_asset(self) -> Cash | None:
        if self._primary_cash_asset is None or not self._primary_cash_asset.is_alive("eop"):
            self._primary_cash_asset = None
            for asset in self._assets:
                if isinstance(asset, Cash) and asset.is_alive("eop"):
                    self._primary_cash_asset = asset
                    break
            if self._primary_cash_asset is None:
                warnings.warn(f"No cash available in existing assets.")
        return self._primary_cash_asset

    def record_before_dealing(self, scope: Literal["assets", "liabs"] | list[Literal["assets", "liabs"]]) -> None:
        scope = scope if isinstance(scope, list) else [scope]
        for item in scope:
            if item == "assets":
                self._asset_rec.record_before_dealing()
                self._unalloc_cash += self._asset_rec.total_cash_flow
            elif item == "liabs":
                self._liab_rec.record_before_dealing()
                self._unalloc_cash += self._liab_rec.total_cash_flow
            else:
                raise ValueError(f"Invalid scope '{item}', expected ('assets', 'liabs').")

    def record_unallocated_cash(self, when: Literal["bd", "ad", "eop"]):
        """Record the unallocated-cash balance at the specified projection phase.

        Args:
            when (Literal["bd", "ad", "eop"]): Before dealing, after dealing, or end of period.
        """
        self._other_rec.record_unalloc_cash(self._unalloc_cash.balance, when)

    def get_asset_attr(self, attr_name: str, /, *, on_undefined: Literal["zerofill", "error"] | None = None) -> float:
        """Get the sum of a numeric attribute across all assets.

        Args:
            attr_name (str): Requested asset attribute name.
            on_undefined (Literal["zerofill", "error"] | None): Return zero or raise when the attribute is undefined.

        Returns:
            float: Sum of the requested asset attribute.
        """
        return self._assets.sum_attr(attr_name, on_undefined=on_undefined or _default_on_undefined)

    def get_liability_attr(self, attr_name: str, /, *, on_undefined: Literal["zerofill", "error"] | None = None) -> float:
        """Get the sum of a numeric attribute across all liabilities.

        Args:
            attr_name (str): Requested liability attribute name.
            on_undefined (Literal["zerofill", "error"] | None): Return zero or raise when the attribute is undefined.

        Returns:
            float: Sum of the requested liability attribute.
        """
        return self._liabs.sum_attr(attr_name, on_undefined=on_undefined or _default_on_undefined)

    def record_after_dealing(self, scope: Literal["assets", "liabs"] | list[Literal["assets", "liabs"]]) -> None:
        scope = scope if isinstance(scope, list) else [scope]
        for item in scope:
            if item == "assets":
                self._asset_rec.record_after_dealing()
            elif item == "liabs":
                self._liab_rec.record_after_dealing()
            else:
                raise ValueError(f"Invalid scope '{item}', expected ('assets', 'liabs').")

    def invest_unallocated_cash(self, cash: Cash | None = None) -> None:
        """Invest all unallocated cash into a cash asset.

        Args:
            cash (Cash | None): Cash asset to receive the balance. If omitted, use the primary
                cash asset held by the fund account.
        """
        cash = cash or self._get_primary_cash_asset()
        if cash is not None:
            amount = self._unalloc_cash.balance
            cash.adjust_nominal(amount)
            self._unalloc_cash -= amount
        else:
            warnings.warn(f"Cash asset is neither provided nor found in existing assets.")

    def apply_other_cash_flow(self, amount: float, kind: str | None = None) -> None:
        """Apply an account-level cash flow and update unallocated cash.

        Args:
            amount (float): Cash-flow amount. Positive values increase unallocated cash; negative values decrease it.
            kind (str | None): Registered cash-flow kind used for time-dimensioned reporting. If omitted or
                unregistered, the cash flow changes unallocated cash but is not recorded.
        """
        self._unalloc_cash += amount
        if kind is not None:
            self._other_rec.record_other_cash_flow(kind, amount)

    def get_other_cash_flow(self, kind: str) -> float:
        """Get an account-level cash flow for the current projection time.

        Args:
            kind (str): A kind registered through `other_cash_flow_kinds` on initialization.

        Returns:
            float: Cash-flow amount, or `float("nan")` if `kind` is not recognized.
        """
        return self._other_rec.get_other_cash_flow(kind)

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.fund_id}'"

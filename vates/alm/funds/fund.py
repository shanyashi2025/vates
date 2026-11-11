import pandas as pd
import warnings
from collections.abc import Iterable
from typing import Literal

from vates.global_conf import GLOBAL_CHECK_LEVEL, CheckLevel
from vates._core import ProjModelEngine, time_synchronized
from vates.alm.allocator import ProceedsKeeper
from vates.alm.assets import Asset, Cash
from vates.alm.liabs import Liab
from vates.alm._utils import ObjList
from vates.alm.funds._recorder import AssetRecorder, LiabRecorder, OtherRecorder
from vates.alm.funds._utils import _RateOfReturnIndexer


@time_synchronized("model_engine.time_synchronizer")
class Fund:
    """Investment fund with assets and liabilities.

    Handles asset/liability management, reporting, and rebalance strategies for actuarial projections.

    Attributes:
        fund_id (str): Fund identifier.
        _assets (ObjList): Assets held by the fund.
        _liabs (ObjList): Liabilities held by the fund.
        _free_proceeds_keeper (ProceedsKeeper): Free proceeds' keeper; can be used to invest, pay tax,
            pay shareholder dividends, taken out, etc.
        _primary_cash_asset (Cash | None): Primary cash asset used for residual cash flows.
        _asset_rec (AssetRecorder): Assets recorder (reported value, returns calculator).
        _liab_rec (LiabRecorder): Liabilities recorder.
        _other_rec (OtherRecorder): Other recorder.
    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', 'fund_id', '_assets', '_liabs',
                 '_free_proceeds_keeper', '_primary_cash_asset', '_allocator',
                 '_asset_rec', '_liab_rec', '_other_rec',)

    def __init__(
        self,
        fund_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        asset_report_bases: list[str] | None = None,
        asset_categories: list[str] | None = None,
        asset_category_attr: str = "category",
        output_liab_attrs_bd: list[str] | None = None,
        output_liab_attrs_ad: list[str] | None = None,
    ) -> None:
        """
        Initialize a Fund object.

        Args:
            fund_id (str): Fund identifier.
            asset_report_bases (list[str]): Asset reporting bases.
            asset_categories (list[str]): Asset categories to be reported.
            asset_category_attr (str): Named attribute for asset category, defaults to "category".
            output_liab_attrs_bd (list[str]): List of liability attributes to output.
            output_liab_attrs_ad (list[str]): List of liability attributes to output.
        """
        self.fund_id = fund_id
        # Asset and liab collections
        self._assets: ObjList[Asset] = ObjList[Asset]()
        self._liabs: ObjList[Liab] = ObjList[Liab]()
        self._free_proceeds_keeper: ProceedsKeeper = ProceedsKeeper()
        self._primary_cash_asset: Cash | None = None
        asset_report_bases: list[str] = asset_report_bases or []

        self._asset_rec: AssetRecorder = AssetRecorder(
            name=fund_id, model_engine=model_engine, assets=self._assets,
            asset_report_bases=asset_report_bases, asset_categories=asset_categories, asset_category_attr=asset_category_attr,
        )
        self._liab_rec: LiabRecorder = LiabRecorder(
            name=fund_id, model_engine=model_engine, liabs=self._liabs,
            output_liab_attrs_bd=output_liab_attrs_bd or [], output_liab_attrs_ad=output_liab_attrs_ad or []
        )
        self._other_rec: OtherRecorder = OtherRecorder(name=fund_id, model_engine=model_engine)

    @property
    def assets(self) -> ObjList[Asset]:
        return self._assets

    @property
    def free_proceeds(self) -> float:
        return self._free_proceeds_keeper.balance

    @property
    def free_proceeds_keeper(self) -> ProceedsKeeper:
        return self._free_proceeds_keeper

    @property
    def liabilities(self) -> ObjList[Liab]:
        return self._liabs

    @property
    def total_asset_cash_flow(self) -> float:
        return self._asset_rec.total_cash_flow

    @property
    def total_liability_cash_flow(self) -> float:
        return self._liab_rec.total_cash_flow

    @property
    def rate_of_return_bd(self) -> _RateOfReturnIndexer:
        return self._asset_rec.rate_of_return_bd

    @property
    def rate_of_return_ad(self) -> _RateOfReturnIndexer:
        return self._asset_rec.rate_of_return_ad

    def add_asset(self, obj: Asset | Iterable[Asset], /) -> None:
        """Add asset object(s) to the fund.

        Args:
            obj (Asset | Iterable[Asset]): Asset(s) to be included.

        """
        if isinstance(obj, Iterable):
            self._assets.extend(obj)
        else:
            self._assets.append(obj)

    def add_liability(self, obj: Liab | Iterable[Liab], /) -> None:
        """Add liability object(s) to the fund.

        Args:
            obj (Liab | Iterable[Liab]): Liability(s) to be included.

        """
        if isinstance(obj, Iterable):
            self._liabs.extend(obj)
        else:
            self._liabs.append(obj)

    def record_on_start(self) -> None:
        """ Record on projection start, typically at `time 0`
        """
        self._asset_rec.record_before_dealing(require_cash_flow=False, require_investment_return=False)
        self._asset_rec.record_after_dealing(require_investment_return=False)
        self._liab_rec.record_before_dealing(require_cash_flow=False)
        self._liab_rec.record_after_dealing()
        self._other_rec.record_free_proceeds(self._free_proceeds_keeper.balance,"bd")
        self._other_rec.record_free_proceeds(self._free_proceeds_keeper.balance,"ad")
        self._other_rec.record_free_proceeds(self._free_proceeds_keeper.balance,"end")

    def _get_primary_cash_asset(self) -> Cash | None:
        if self._primary_cash_asset is None or not self._primary_cash_asset.is_alive("end"):
            self._primary_cash_asset = None
            for asset in self._assets:
                if isinstance(asset, Cash) and asset.is_alive("end"):
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
                self._free_proceeds_keeper += self._asset_rec.total_cash_flow
            elif item == "liabs":
                self._liab_rec.record_before_dealing()
                self._free_proceeds_keeper += self._liab_rec.total_cash_flow
            else:
                raise ValueError(f"Invalid scope '{item}', expected ('assets', 'liabs').")

    def record_free_proceeds(self, timing: Literal["bd", "ad", "end"]):
        self._other_rec.record_free_proceeds(self._free_proceeds_keeper.balance, timing)

    def get_total_asset_attr(self, attr_name: str, /, *, on_undefined: Literal["zerofill", "error"] | None = None) -> float:
        """ Get the sum of a numeric attribute of total assets

        Args:
            attr_name (str): Requested attribute name.
            on_undefined (Literal["zerofill", "error"]): Zerofill or raise for undefined attribute, defaults to "zerofill".

        """
        if on_undefined is None:
            on_undefined = "zerofill" if GLOBAL_CHECK_LEVEL == CheckLevel.BYPASS else "error"
        return self._assets.sum_attr(attr_name, on_undefined=on_undefined)

    def get_total_liability_attr(self, attr_name: str, /, *, on_undefined: Literal["zerofill", "error"] | None = None) -> float:
        """ Get the sum of a numeric attribute of total liabilities

        Args:
            attr_name (str): Requested attribute name.
            on_undefined (Literal["zerofill", "error"]): Zerofill or raise for undefined attribute, defaults to "zerofill".

        """
        if on_undefined is None:
            on_undefined = "zerofill" if GLOBAL_CHECK_LEVEL == CheckLevel.BYPASS else "error"
        return self._liabs.sum_attr(attr_name, on_undefined=on_undefined)

    def record_after_dealing(self, scope: Literal["assets", "liabs"] | list[Literal["assets", "liabs"]]) -> None:
        scope = scope if isinstance(scope, list) else [scope]
        for item in scope:
            if item == "assets":
                self._asset_rec.record_after_dealing()
            elif item == "liabs":
                self._liab_rec.record_after_dealing()
            else:
                raise ValueError(f"Invalid scope '{item}', expected ('assets', 'liabs').")

    def invest_free_proceeds_to_cash(self, cash: Cash | None = None) -> None:
        cash = cash or self._get_primary_cash_asset()
        if cash is not None:
            cash.buy_amount(self.free_proceeds_keeper.take_out())
        else:
            warnings.warn(f"Cash asset is neither provided nor found in existing assets.")

    def take_out_free_proceeds(self) -> float:
        """Take out all the free proceeds.

        Returns:
            float: The free proceeds taken out.
        """
        amount = self._free_proceeds_keeper.take_out()
        self._other_rec.record_free_proceeds_transfer(-amount)
        return amount

    def receive_free_proceeds(self, amount: float, /) -> None:
        """Receive free proceeds.

        Args:
            amount (float): Amount to receive, usually positive but negative is acceptable.
        """
        self._free_proceeds_keeper += amount
        self._other_rec.record_free_proceeds_transfer(amount)

    def consume_free_proceeds(self, amount: float, /) -> None:
        """Consume free proceeds.

        Args:
            amount (float): Amount to consume, usually positive but negative is acceptable.
        """
        self.receive_free_proceeds(-amount)

    def pay_tax(self, amount: float, /) -> None:
        self._free_proceeds_keeper -= amount
        self._other_rec.record_pay_tax(amount)

    def receive_capital_contribution(self, amount: float, /) -> None:
        self._free_proceeds_keeper += amount
        self._other_rec.record_receive_capital(amount)

    def pay_dividends_to_shareholders(self, amount: float, /):
        self._free_proceeds_keeper -= amount
        self._other_rec.record_pay_dividends(amount)

    @property
    def tax_paid(self) -> float:
        return self._other_rec.tax_paid

    @property
    def capital_contribution_received(self) -> float:
        return self._other_rec.capital_received

    @property
    def dividends_paid_to_shareholders(self) -> float:
        return self._other_rec.dividends_paid

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.fund_id}'"

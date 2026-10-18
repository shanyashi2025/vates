import numpy as np
import pandas as pd
from enum import Enum, auto

from vates._core import ProjModelEngine, time_synchronized, TDimVariable, make_proj_variable
from vates.global_conf import CHECK_LEVEL, CheckLevel
from vates.utils import Lifecycle, transition
from vates.alm._utils import Connector
from vates.alm.funds._utils import _RateOfReturnIndexer
from vates.alm.assets.asset_base import Asset, AssetPhase
from vates.alm.liabs.liab_base import Liab

class RecorderPhase(Enum):
    ASSET_REC_BD = auto()
    LIAB_REC_BD = auto()
    ASSET_REC_AD = auto()
    LIAB_REC_AD = auto()

def _validate_timing(val, /) -> str:
    if not isinstance(val, str):
        raise TypeError(f"Invalid type of 'timing': {type(val)}, expected 'str'.")
    val = val.lower()
    if val not in ("bd", "ad"):
        raise ValueError(f"Invalid 'timing': {val}, expected ('bd', 'ad').")
    return val


@time_synchronized("model_engine.time_synchronizer")
class AssetRecorder:
    """Performs aggregation and performance calculations for a fund.

    Computes asset aggregates, investment returns, and
    stores time-dimensioned arrays for reporting by class and total.

    Attributes:

    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    __slots__ = (
        '__weakref__', 'time', 'period', '_lc', 'asset_conn',
        'asset_categories', 'asset_category_attr', 'asset_report_bases',
        'totass_cf', 'catass_cf',
        'totass_mv_op', 'totass_mv_bd', 'totass_mv_ad', 'catass_mv_op', 'catass_mv_bd', 'catass_mv_ad',
        'totass_rv_op', 'totass_rv_bd', 'totass_rv_ad', 'catass_rv_op', 'catass_rv_bd', 'catass_rv_ad',
        'tdv_totass_cf', 'tdv_totass_mv_bd', 'tdv_totass_mv_ad', 'tdv_totass_rv_bd', 'tdv_totass_rv_ad',
        'tdv_totass_ret_bd', 'tdv_totass_ror_pc_bd', 'tdv_totass_ret_ad', 'tdv_totass_ror_pc_ad',
        'tdv_catass_cf', 'tdv_catass_rv_bd', 'tdv_catass_rv_ad',
        'tdv_catass_ret_bd', 'tdv_catass_ror_pc_bd', 'tdv_catass_ret_ad', 'tdv_catass_ror_pc_ad',
        'rate_of_return_bd', 'rate_of_return_ad'
    )

    def __init__(
        self,
        *,
        name: str,
        model_engine: ProjModelEngine | None = None,
        asset_connector: Connector[Asset],
        asset_categories: list[str],
        asset_category_attr: str,
        asset_report_bases: list[str],
    ):
        self.asset_conn: Connector[Asset] = asset_connector
        self.asset_categories: list[str] = asset_categories
        self.asset_category_attr: str = asset_category_attr
        self.asset_report_bases: list[str] = asset_report_bases
        self.totass_cf: float | None = None
        self.catass_cf: np.ndarray | None = None  # shape = (categories, )
        self.totass_mv_op: float | None = None
        self.totass_mv_bd: float | None = None
        self.totass_mv_ad: float | None = None
        self.catass_mv_op: np.ndarray | None = None  # shape = (categories, )
        self.catass_mv_bd: np.ndarray | None = None  # shape = (categories, )
        self.catass_mv_ad: np.ndarray | None = None  # shape = (categories, )
        self.totass_rv_op: np.ndarray | None = None  # shape = (report_bases, )
        self.totass_rv_bd: np.ndarray | None = None  # shape = (report_bases, )
        self.totass_rv_ad: np.ndarray | None = None  # shape = (report_bases, )
        self.catass_rv_op: np.ndarray | None = None  # shape = (categories, report_bases)
        self.catass_rv_bd: np.ndarray | None = None  # shape = (categories, report_bases)
        self.catass_rv_ad: np.ndarray | None = None  # shape = (categories, report_bases)
        self._lc: Lifecycle[RecorderPhase] = Lifecycle[RecorderPhase]()

        # Initialize time-dimensioned variables for output
        # dims = None
        create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund')
        self.tdv_totass_cf: TDimVariable = create_tdv("totass_cash_flow")
        self.tdv_totass_mv_bd: TDimVariable = create_tdv("totass_mv_bd")
        self.tdv_totass_mv_ad: TDimVariable = create_tdv("totass_mv_ad")

        if asset_report_bases:
            # dims = asset_report_bases
            create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund',
                                                      dims=[asset_report_bases])
            self.tdv_totass_rv_bd: TDimVariable = create_tdv("totass_rep_value_bd")
            self.tdv_totass_rv_ad: TDimVariable = create_tdv("totass_rep_value_ad")
            self.tdv_totass_ret_bd: TDimVariable = create_tdv("totass_inv_ret_bd")
            self.tdv_totass_ror_pc_bd: TDimVariable = create_tdv("totass_ror_pc_bd")
            self.tdv_totass_ret_ad: TDimVariable = create_tdv("totass_inv_ret_ad")
            self.tdv_totass_ror_pc_ad: TDimVariable = create_tdv("totass_ror_pc_ad")
        if asset_categories:
            # dims = asset_categories
            create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund',
                                                      dims=[asset_categories])
            self.tdv_catass_cf: TDimVariable = create_tdv("asset_cash_flow")
        if asset_categories and asset_report_bases:
            # dims = asset_categories, asset_report_bases
            create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund',
                                                      dims=[asset_categories, asset_report_bases])
            self.tdv_catass_rv_bd: TDimVariable = create_tdv("asset_rep_value_bd")
            self.tdv_catass_rv_ad: TDimVariable = create_tdv("asset_rep_value_ad")
            self.tdv_catass_ret_bd: TDimVariable = create_tdv("asset_inv_ret_bd")
            self.tdv_catass_ror_pc_bd: TDimVariable = create_tdv("asset_ror_pc_bd")
            self.tdv_catass_ret_ad: TDimVariable = create_tdv("asset_inv_ret_ad")
            self.tdv_catass_ror_pc_ad: TDimVariable = create_tdv("asset_ror_pc_ad")

        # rate of return indexers
        self.rate_of_return_bd: _RateOfReturnIndexer = _RateOfReturnIndexer(
            self.tdv_totass_ror_pc_bd, divby=100)
        self.rate_of_return_ad: _RateOfReturnIndexer = _RateOfReturnIndexer(
            self.tdv_totass_ror_pc_ad, divby=100)

    @property
    @transition(require_all=RecorderPhase.ASSET_REC_BD)
    def total_cash_flow(self) -> float:
        return self.totass_cf

    @transition(require_not=RecorderPhase.ASSET_REC_BD, mark=RecorderPhase.ASSET_REC_BD)
    def record_before_dealing(self, require_cash_flow: bool = True, require_investment_return: bool = True) -> None:
        """Record asset before dealing (bd).
        """
        t = self.time

        # Aggregate asset cash flow
        if require_cash_flow:
            self.totass_cf = self.asset_conn.sum_attr("cash_flow")
            self.tdv_totass_cf[t] = self.totass_cf
        if self.asset_categories:
            val_dict = self.asset_conn.sum_attr(
                "cash_flow", group_by=self.asset_category_attr, slice_key=self.asset_categories)
            self.catass_cf = np.array([val_dict[cat] for cat in self.asset_categories])
            self.tdv_catass_cf[t] = self.catass_cf

        # Aggregate asset value and asset cash flow
        self._sum_report_values("bd")

        if not require_investment_return:
            return

        # Calculate reported investment return
        if self.asset_report_bases:
            cash_flow = np.array([self.totass_cf] * len(self.asset_report_bases))  # scalar -> (report_bases, )
            ret, ror = self._compute_investment_return(
                prev_val=self.totass_rv_op, cash_flow=cash_flow, curr_val=self.totass_rv_bd)
            self.tdv_totass_ret_bd[t] = ret
            self.tdv_totass_ror_pc_bd[t] = ror * 100

        if self.asset_categories and self.asset_report_bases:
            n_category = len(self.asset_categories)
            n_basis = len(self.asset_report_bases)
            cash_flow = np.array([[self.catass_cf[i]] * n_basis for i in range(n_category)])  # (categories, ) -> (categories, report_bases)
            ret, ror = self._compute_investment_return(
                prev_val=self.catass_rv_op, cash_flow=cash_flow, curr_val=self.catass_rv_bd)
            self.tdv_catass_ret_bd[t] = ret
            self.tdv_catass_ror_pc_bd[t] = ror * 100

    @transition(require_not=RecorderPhase.ASSET_REC_AD, mark=RecorderPhase.ASSET_REC_AD)
    def record_after_dealing(self, require_investment_return: bool = True) -> None:
        """Record asset after dealing (ad).
        """
        t = self.time
        # Aggregate asset value
        self._sum_report_values("ad")

        if not require_investment_return:
            return

        # Calculate reported investment return
        if self.asset_report_bases:
            cash_flow = np.array([self.totass_cf + self.totass_mv_bd - self.totass_mv_ad] * len(self.asset_report_bases))  # scalar -> (report_bases, )
            ret, ror = self._compute_investment_return(
                prev_val=self.totass_rv_op, cash_flow=cash_flow, curr_val=self.totass_rv_ad)
            self.tdv_totass_ret_ad[t] = ret
            self.tdv_totass_ror_pc_ad[t] = ror * 100

        if self.asset_categories and self.asset_report_bases:
            n_category = len(self.asset_categories)
            n_basis = len(self.asset_report_bases)
            cash_flow = np.array([[self.catass_cf[i] + self.catass_mv_bd[i] - self.catass_mv_ad[i]] * n_basis
                                  for i in range(n_category)])  # (categories, ) -> (categories, report_bases)
            ret, ror = self._compute_investment_return(
                prev_val=self.catass_rv_op, cash_flow=cash_flow, curr_val=self.catass_rv_ad)
            self.tdv_catass_ret_ad[t] = ret
            self.tdv_catass_ror_pc_ad[t] = ror * 100

    def _sum_report_values(self, timing: str) -> None:
        """Aggregate values from all assets in the fund.

        Args:
            timing (str): Timing of aggregation (bd/ad = before/after dealing).

        Raises:
            ValueError: If an asset is not rolled or updated for the current period, or if timing is invalid.
        """
        timing = _validate_timing(timing)

        t = self.time
        if CHECK_LEVEL != CheckLevel.BYPASS:
            phase = AssetPhase.ROLLED if timing == "bd" else AssetPhase.CLOSED
            for asset in self.asset_conn:
                asset.require_lifecycle(phase=phase)

        # 1. always calculate total asset market value
        tot_mv = self.asset_conn.sum_attr("market_value")
        if timing == "bd":
            self.totass_mv_bd, self.tdv_totass_mv_bd[t] = tot_mv, tot_mv
        else: # timing == "ad"
            self.totass_mv_ad, self.tdv_totass_mv_ad[t] = tot_mv, tot_mv

        # 2. category asset market value, shape = (categories, )
        if self.asset_category_attr:
            val_dict = self.asset_conn.sum_attr(
                "market_value", group_by=self.asset_category_attr, slice_key=self.asset_categories)
            cat_mv = np.array([val_dict[cat] for cat in self.asset_categories])
            if timing == "bd":
                self.catass_mv_bd = cat_mv
            else:  # timing == "ad"
                self.catass_mv_ad = cat_mv

        # 3. total asset reported values, shape = (report_bases, )
        if self.asset_report_bases:
            tot_rv = self.asset_conn.sum_attr(self.asset_report_bases)
            if timing == "bd":
                self.totass_rv_bd, self.tdv_totass_rv_bd[t] = tot_rv, tot_rv
            else:  # timing == "ad"
                self.totass_rv_ad, self.tdv_totass_rv_ad[t] = tot_rv, tot_rv

        # 4. category asset reported values, shape = (categories, report_bases)
        if self.asset_report_bases and self.asset_categories:
            val_dict = self.asset_conn.sum_attr(
                self.asset_report_bases, group_by=self.asset_category_attr, slice_key=self.asset_categories)
            cat_rv = np.array([val_dict[cat] for cat in self.asset_categories])
            if timing == "bd":
                self.catass_rv_bd, self.tdv_catass_rv_bd[t] = cat_rv, cat_rv
            else:  # timing == "ad"
                self.catass_rv_ad, self.tdv_catass_rv_ad[t] = cat_rv, cat_rv

    def _on_time_change(self) -> None:
        self.totass_mv_op, self.totass_mv_ad = self.totass_mv_ad, None
        self.catass_mv_op, self.catass_mv_ad = self.catass_mv_ad, None
        self.totass_rv_op, self.totass_rv_ad = self.totass_rv_ad, None
        self.catass_rv_op, self.catass_rv_ad = self.catass_rv_ad, None
        self.totass_cf = None
        self.catass_cf = None

    @staticmethod
    def _compute_investment_return(prev_val: np.ndarray, cash_flow: np.ndarray, curr_val: np.ndarray,
                                   ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate investment return and rate of return element-wise."""
        ret = curr_val + cash_flow - prev_val
        ror = np.zeros_like(ret, dtype=np.result_type(ret, float))
        mask = (ret != 0) & (prev_val != 0)
        ror[mask] = ret[mask] / prev_val[mask]
        return ret, ror


@time_synchronized("model_engine.time_synchronizer")
class LiabRecorder:
    """Performs aggregation and performance calculations for a fund.

    Computes liability aggregates, investment returns, and
    stores time-dimensioned arrays for reporting by class and total.

    Attributes:

    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    __slots__ = ('__weakref__', 'time', 'period', '_lc', 'liab_conn',
                 'totliab_cf', 'output_liab_attrs_bd', 'output_liab_attrs_ad',
                 'tdv_totliab_cf', 'tdv_totliab_attrs_bd', 'tdv_totliab_attrs_ad',)

    def __init__(
        self,
        *,
        name: str,
        model_engine: ProjModelEngine | None = None,
        liab_connector: Connector[Liab],
        output_liab_attrs_bd: list[str],
        output_liab_attrs_ad: list[str],
    ):
        self.liab_conn: Connector[Liab] = liab_connector
        self.output_liab_attrs_bd: list[str] = output_liab_attrs_bd
        self.output_liab_attrs_ad: list[str] = output_liab_attrs_ad
        self.totliab_cf: float | None = None

        self._lc: Lifecycle[RecorderPhase] = Lifecycle[RecorderPhase]()

        # Initialize time-dimensioned variables for output
        create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund')
        self.tdv_totliab_cf: TDimVariable = create_tdv("totliab_cash_flow")
        self.tdv_totliab_attrs_bd: list[TDimVariable] = [
            create_tdv(f"totliab_{name}") for name in self.output_liab_attrs_bd]
        self.tdv_totliab_attrs_ad: list[TDimVariable] = [
            create_tdv(f"totliab_{name}") for name in self.output_liab_attrs_ad]

    @property
    @transition(require_all=RecorderPhase.LIAB_REC_BD)
    def total_cash_flow(self) -> float:
        return self.totliab_cf

    @transition(require_not=RecorderPhase.LIAB_REC_BD, mark=RecorderPhase.LIAB_REC_BD)
    def record_before_dealing(self, require_cash_flow: bool = True) -> None:
        """Record liability before dealing (bd).
        """
        t = self.time

        # Aggregate liability cash flow
        if require_cash_flow:
            self.totliab_cf = sum(liab.cash_flow for liab in self.liab_conn)
            self.tdv_totliab_cf[t] = self.totliab_cf

        # Aggregate liability value
        self._sum_attrs("bd")

    @transition(require_not=RecorderPhase.LIAB_REC_AD, mark=RecorderPhase.LIAB_REC_AD)
    def record_after_dealing(self) -> None:
        """Process liability values and cash flows after dealing (ad).
        """
        self._sum_attrs("ad")

    def _sum_attrs(self, timing: str) -> None:
        """Aggregate values from all liabilities in the fund.

        Args:
            timing (str): Timing of aggregation (bd/ad = before/after dealing).

        Raises:
            ValueError: If a liability is not rolled/updated for the current period, or if timing is invalid.
        """
        timing = _validate_timing(timing)
        pairs = zip(self.output_liab_attrs_bd, self.tdv_totliab_attrs_bd) if timing == "bd" else zip(
            self.output_liab_attrs_ad, self.tdv_totliab_attrs_ad)
        t = self.time
        for attr_name, tdv in pairs:
            tdv[t] = self.liab_conn.sum_attr(attr_name)

    def _on_time_change(self) -> None:
        self.totliab_cf = None


@time_synchronized("model_engine.time_synchronizer")
class OtherRecorder:
    """Performs aggregation and performance calculations for a fund.

    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    __slots__ = ('__weakref__', 'time', 'period', '_lc', 'name',
                 'tax_paid', 'capital_received', 'dividends_paid',
                 'tdv_free_proceeds_bd', 'tdv_free_proceeds_ad', 'tdv_free_proceeds_end',
                 'tdv_proceeds_tran_in', 'tdv_proceeds_tran_out', 'tdv_proceeds_tran_net',
                 'tdv_tax_paid', 'tdv_capital_received', 'tdv_dividends_paid')

    def __init__(self, *, name: str, model_engine: ProjModelEngine | None = None):
        self.tax_paid: float = 0.0
        self.capital_received: float = 0.0
        self.dividends_paid: float = 0.0
        # Initialize time-dimensioned variables for output
        create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund')
        self.tdv_free_proceeds_bd: TDimVariable = create_tdv("free_proceeds_bd")
        self.tdv_free_proceeds_ad: TDimVariable = create_tdv("free_proceeds_ad")
        self.tdv_free_proceeds_end: TDimVariable = create_tdv("free_proceeds_end")
        self.tdv_proceeds_tran_in: TDimVariable = create_tdv("proceeds_transferred_in")
        self.tdv_proceeds_tran_out: TDimVariable = create_tdv("proceeds_transferred_out")
        self.tdv_proceeds_tran_net: TDimVariable = create_tdv("proceeds_transferred_net")
        self.tdv_tax_paid: TDimVariable = create_tdv("tax_paid")
        self.tdv_capital_received: TDimVariable = create_tdv("capital_contribution_received")
        self.tdv_dividends_paid: TDimVariable = create_tdv("dividends_paid_to_shareholders")

    def record_free_proceeds(self, value: float, timing: str) -> None:
        if timing == "bd":
            self.tdv_free_proceeds_bd[self.time] = value
        elif timing == "ad":
            self.tdv_free_proceeds_ad[self.time] = value
        elif timing == "end":
            self.tdv_free_proceeds_end[self.time] = value
        else:
            raise ValueError(f"Invalid timing {timing}, expected ('bd', 'ad', 'end').")

    def record_free_proceeds_transfer(self, amount: float) -> None:
        t = self.time
        amt_in, amt_out = max(amount, 0.0), max(- amount, 0.0)
        self.tdv_proceeds_tran_in[t] += amt_in
        self.tdv_proceeds_tran_out[t] += amt_out
        self.tdv_proceeds_tran_net[t] += amount

    def record_pay_tax(self, amount: float) -> None:
        self.tax_paid += amount
        self.tdv_tax_paid[self.time] += amount

    def record_receive_capital(self, amount: float) -> None:
        self.capital_received += amount
        self.tdv_capital_received[self.time] += amount

    def record_pay_dividends(self, amount: float) -> None:
        self.dividends_paid += amount
        self.tdv_dividends_paid[self.time] += amount

    def _on_time_change(self) -> None:
        self.tax_paid: float = 0.0
        self.capital_received: float = 0.0
        self.dividends_paid: float = 0.0
        t = self.time
        self.tdv_free_proceeds_bd[t] = None  # set value to None, unless `record_free_proceeds` is actively called
        self.tdv_free_proceeds_ad[t] = None
        self.tdv_free_proceeds_end[t] = None

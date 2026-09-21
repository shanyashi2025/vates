import numpy as np
import pandas as pd
from enum import Enum, auto

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDimVariable
from vates.global_conf import CHECK_LEVEL, CheckLevel
from vates.utils import Lifecycle, transition
from vates.alm.funds._connector import AssetLiabConnector
from vates.alm.assets.asset_base import AssetPhase

class RecorderPhase(Enum):
    ASSET_REC_BD = auto()
    LIAB_REC_BD = auto()
    ASSET_REC_AD = auto()
    LIAB_REC_AD = auto()


@add_projection_time_synchronizer
class FundRecorder:
    """Performs aggregation and performance calculations for a fund.

    Computes asset and liability aggregates, investment returns, and
    stores time-dimensioned arrays for reporting by class and total.

    Attributes:

    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`

    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_lc', 'name', 'connector',
                 'asset_categories', 'asset_category_attr', 'asset_report_bases', 'totass_cf', 'catass_cf'
                 'totass_mv_op', 'totass_mv_bd', 'totass_mv_ad', 'catass_mv_op', 'catass_mv_bd', 'catass_mv_ad',
                 'totass_rv_op', 'totass_rv_bd', 'totass_rv_ad', 'catass_rv_op', 'catass_rv_bd', 'catass_rv_ad',
                 'output_liab_attrs_bd', 'output_liab_attrs_ad',
                 'tdv_free_estate_bd', 'tdv_free_estate_ad', 'tdv_proceeds_tran_in', 'tdv_proceeds_tran_out',
                 'tdv_totass_rv_bd', 'tdv_totass_rv_ad', 'tdv_totass_ret_bd', 'tdv_totass_ror_pc_bd',
                 'tdv_totass_ret_ad', 'tdv_totass_ror_pc_ad', 'tdv_catass_cf', 'tdv_catass_rv_bd',
                 'tdv_catass_rv_ad', 'tdv_catass_ret_bd', 'tdv_catass_ror_pc_bd', 'tdv_catass_ret_ad',
                 'tdv_catass_ror_pc_ad',)

    def __init__(
        self,
        *,
        name: str,
        model_engine: ProjModelEngine | None = None,
        connector: AssetLiabConnector,
        asset_categories: list[str],
        asset_category_attr: str,
        asset_report_bases: list[str],
        output_liab_attrs_bd: list[str],
        output_liab_attrs_ad: list[str],
    ):
        self.name: str = name
        self.connector: AssetLiabConnector = connector
        self.asset_categories: list[str] = asset_categories
        self.asset_category_attr: str = asset_category_attr
        self.asset_report_bases: list[str] = asset_report_bases
        self.output_liab_attrs_bd: list[str] = output_liab_attrs_bd
        self.output_liab_attrs_ad: list[str] = output_liab_attrs_ad
        self.totliab_cf: float | None = None
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
        create_tdv = lambda x: TDimVariable(x, model_engine=model_engine, owner=self.name, group='fund')
        self.tdv_totass_cf: TDimVariable = create_tdv("totass_cash_flow")
        self.tdv_totliab_cf: TDimVariable = create_tdv("totliab_cash_flow")
        self.tdv_totliab_attrs_bd: list[TDimVariable] = [
            create_tdv(f"totliab_{name}") for name in self.output_liab_attrs_bd]
        self.tdv_totliab_attrs_ad: list[TDimVariable] = [
            create_tdv(f"totliab_{name}") for name in self.output_liab_attrs_ad]

        self.tdv_free_estate_bd: TDimVariable = create_tdv("free_estate_bd")
        self.tdv_free_estate_ad: TDimVariable = create_tdv("free_estate_ad")
        self.tdv_proceeds_tran_in: TDimVariable = create_tdv("proceeds_transferred_in")
        self.tdv_proceeds_tran_out: TDimVariable = create_tdv("proceeds_transferred_out")
        # dims = asset_report_bases
        create_tdv = lambda x: TDimVariable(x, model_engine=model_engine, owner=self.name, group='fund', dims=[asset_report_bases])
        self.tdv_totass_rv_bd: TDimVariable = create_tdv("totass_rep_value_bd")
        self.tdv_totass_rv_ad: TDimVariable = create_tdv("totass_rep_value_ad")
        self.tdv_totass_ret_bd: TDimVariable = create_tdv("totass_inv_ret_bd")
        self.tdv_totass_ror_pc_bd: TDimVariable = create_tdv("totass_ror_pc_bd")
        self.tdv_totass_ret_ad: TDimVariable = create_tdv("totass_inv_ret_ad")
        self.tdv_totass_ror_pc_ad: TDimVariable = create_tdv("totass_ror_pc_ad")
        # dims = asset_categories
        create_tdv = lambda x: TDimVariable(x, model_engine=model_engine, owner=self.name, group='fund', dims=[asset_categories])
        self.tdv_catass_cf: TDimVariable = create_tdv("asset_cash_flow")
        # dims = asset_categories, asset_report_bases
        create_tdv = lambda x: TDimVariable(x, model_engine=model_engine, owner=self.name, group='fund', dims=[asset_categories, asset_report_bases])
        self.tdv_catass_rv_bd: TDimVariable = create_tdv("asset_rep_value_bd")
        self.tdv_catass_rv_ad: TDimVariable = create_tdv("asset_rep_value_ad")
        self.tdv_catass_ret_bd: TDimVariable = create_tdv("asset_inv_ret_bd")
        self.tdv_catass_ror_pc_bd: TDimVariable = create_tdv("asset_ror_pc_bd")
        self.tdv_catass_ret_ad: TDimVariable = create_tdv("asset_inv_ret_ad")
        self.tdv_catass_ror_pc_ad: TDimVariable = create_tdv("asset_ror_pc_ad")

    @property
    @transition(require_all=RecorderPhase.ASSET_REC_BD)
    def total_asset_cash_flow(self) -> float:
        return self.totass_cf

    @property
    @transition(require_all=RecorderPhase.LIAB_REC_BD)
    def total_liab_cash_flow(self) -> float:
        return self.totliab_cf

    @transition(mark=RecorderPhase.ASSET_REC_BD)
    def record_asset_before_dealing(self) -> None:
        """Record asset before dealing (bd).
        """
        t = self.time

        # Aggregate asset cash flow
        self.totass_cf = self.connector.sum_asset("cash_flow")
        self.catass_cf = np.array(self.connector.groupby_sum_asset(
            "cash_flow", groupby=self.asset_category_attr, in_list=self.asset_categories))
        self.tdv_catass_cf[t] = self.catass_cf
        self.tdv_totass_cf[t] = self.totass_cf

        # Aggregate asset value and asset cash flow
        self.sum_asset_report_values("bd")

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

    @transition(mark=RecorderPhase.ASSET_REC_AD)
    def record_asset_after_dealing(self) -> None:
        """Record asset after dealing (ad).
        """
        t = self.time
        # Aggregate asset value
        self.sum_asset_report_values("ad")

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

    def sum_asset_report_values(self, timing: str) -> None:
        """Aggregate values from all assets in the fund.

        Args:
            timing (str): Timing of aggregation (bd/ad = before/after dealing).

        Raises:
            ValueError: If an asset is not rolled or updated for the current period, or if timing is invalid.
        """
        timing = self._validate_timing(timing)

        t = self.time
        if CHECK_LEVEL != CheckLevel.BYPASS:
            phase = AssetPhase.ROLLED if timing == "bd" else AssetPhase.CLOSED
            for asset in self.connector.assets:
                asset.require_lifecycle(phase=phase)

        # 1. always calculate total asset market value
        tot_mv = self.connector.sum_asset("market_value")

        # 2. category asset market value, shape = (categories, )
        if self.asset_category_attr:
            cat_mv = self.connector.groupby_sum_asset(
                "market_value", groupby=self.asset_category_attr, in_list=self.asset_categories)
            cat_mv = np.array(cat_mv)
        else:
            cat_mv = None

        # 3. total asset reported values, shape = (report_bases, )
        if self.asset_report_bases:
            tot_rv = self.connector.sum_asset(self.asset_report_bases)
        else:
            tot_rv = None

        # 4. category asset reported values, shape = (categories, report_bases)
        if self.asset_report_bases and self.asset_categories:
            cat_rv = self.connector.groupby_sum_asset(
                self.asset_report_bases, groupby=self.asset_category_attr, in_list=self.asset_categories)
            cat_rv = np.array(cat_rv)
        else:
            cat_rv = None

        if timing == "bd":
            self.totass_mv_op = self.totass_mv_ad
            self.catass_mv_op = self.catass_mv_ad
            self.totass_mv_ad = None
            self.catass_mv_ad = None
            self.totass_mv_bd = tot_mv
            self.catass_mv_bd = cat_mv
            self.totass_rv_op = self.totass_rv_ad
            self.catass_rv_op = self.catass_rv_ad
            self.totass_rv_ad = None
            self.catass_rv_ad = None
            self.totass_rv_bd = tot_rv
            self.catass_rv_bd = cat_rv
            self.tdv_totass_rv_bd[t] = self.totass_rv_bd
            self.tdv_catass_rv_bd[t] = self.catass_rv_bd
        else: # timing == "ad"
            self.totass_mv_ad = tot_mv
            self.catass_mv_ad = cat_mv
            self.totass_rv_ad = tot_rv
            self.catass_rv_ad = cat_rv
            self.tdv_totass_rv_ad[t] = self.totass_rv_ad
            self.tdv_catass_rv_ad[t] = self.catass_rv_ad

    @transition(mark=RecorderPhase.LIAB_REC_BD)
    def record_liab_before_dealing(self) -> None:
        """Record liability before dealing (bd).
        """
        t = self.time

        # Aggregate liability cash flow
        self.totliab_cf = sum(liab.cash_flow for liab in self.connector.liabs)
        self.tdv_totliab_cf[t] = self.totliab_cf

        # Aggregate liability value
        self.sum_liab_attrs("bd")

    @transition(mark=RecorderPhase.LIAB_REC_AD)
    def record_liab_after_dealing(self):
        """Process liability values and cash flows after dealing (ad).
        """
        self.sum_liab_attrs("ad")

    def sum_liab_attrs(self, timing: str) -> None:
        """Aggregate values from all liabilities in the fund.

        Args:
            timing (str): Timing of aggregation (bd/ad = before/after dealing).

        Raises:
            ValueError: If a liability is not rolled/updated for the current period, or if timing is invalid.
        """
        timing = self._validate_timing(timing)
        t = self.time
        if timing == "bd":
            for attr_name, tdv in zip(self.output_liab_attrs_bd, self.tdv_totliab_attrs_bd):
                tdv[t] = self.connector.sum_liab(attr_name)
        else: # "ad":
            for attr_name, tdv in zip(self.output_liab_attrs_ad, self.tdv_totliab_attrs_ad):
                tdv[t] = self.connector.sum_liab(attr_name)

    def record_free_estate(self, timing: str) -> None:
        timing = self._validate_timing(timing)
        if timing == "bd":
            self.tdv_free_estate_bd[self.time] = self.connector.free_estate
        else:
            self.tdv_free_estate_ad[self.time] = self.connector.free_estate

    def record_free_proceeds_transfer(self, amount: float) -> None:
        t = self.time
        amt_in = max(amount, 0.0)
        amt_out = max(- amount, 0.0)

        if self.tdv_proceeds_tran_in[t] is None:
            self.tdv_proceeds_tran_in[t] = amt_in
        else:
            self.tdv_proceeds_tran_in[t] += amt_in

        if self.tdv_proceeds_tran_out[t] is None:
            self.tdv_proceeds_tran_out[t] = amt_out
        else:
            self.tdv_proceeds_tran_out[t] += amt_out

    @staticmethod
    def _validate_timing(val, /) -> str:
        if not isinstance(val, str):
            raise TypeError(f"Invalid type of 'timing': {type(val)}, expected 'str'.")
        val = val.lower()
        if val not in ("bd", "ad"):
            raise ValueError(f"Invalid 'timing': {val}, expected ('bd', 'ad').")
        return val

    @staticmethod
    def _compute_investment_return(prev_val: np.ndarray, cash_flow: np.ndarray, curr_val: np.ndarray,
                                   ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate investment return and rate of return element-wise."""
        ret = curr_val + cash_flow - prev_val
        ror = np.zeros_like(ret, dtype=np.result_type(ret, float))
        mask = (ret != 0) & (prev_val != 0)
        ror[mask] = ret[mask] / prev_val[mask]
        return ret, ror

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.name}'"

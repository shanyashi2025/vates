import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Literal

from vates._core import ProjModelEngine, time_synchronized, TDimVariable, TDimArrayVariable, make_proj_variable
from vates.global_conf import GLOBAL_CHECK_MODE, CheckMode
from vates.utils import Lifecycle, transition, maybe_raise_on_compare
from vates.alm._utils import ObjList
from vates.alm.assets.asset_base import BaseAsset, AssetPhase
from vates.alm.liabs.liab_base import BaseLiab

_on_undefined = "zerofill" if GLOBAL_CHECK_MODE == CheckMode.BYPASS else "error"

class RecorderPhase(Enum):
    ASSET_REC_BD = auto()
    LIAB_REC_BD = auto()
    ASSET_REC_AD = auto()
    LIAB_REC_AD = auto()


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
        '__weakref__', 'time', 'period', '_lc', '_objs',
        'categories', 'category_attr', 'report_bases',
        'tot_cf', 'cat_cf',
        'tot_mv_op', 'tot_mv_bd', 'tot_mv_ad', 'cat_mv_op', 'cat_mv_bd', 'cat_mv_ad',
        'tot_rv_op', 'tot_rv_bd', 'tot_rv_ad', 'cat_rv_op', 'cat_rv_bd', 'cat_rv_ad',
        'tot_ret_bd', 'tot_ror_bd', 'tot_ret_ad', 'tot_ror_ad',
        'cat_ret_bd', 'cat_ror_bd', 'cat_ret_ad', 'cat_ror_ad',
        'tdv_tot_cf', 'tdv_tot_mv_bd', 'tdv_tot_mv_ad', 'tdv_tot_rv_bd', 'tdv_tot_rv_ad',
        'tdv_tot_ret_bd', 'tdv_tot_ror_pc_bd', 'tdv_tot_ret_ad', 'tdv_tot_ror_pc_ad',
        'tdv_cat_cf', 'tdv_cat_mv_bd', 'tdv_cat_mv_ad', 'tdv_cat_rv_bd', 'tdv_cat_rv_ad',
        'tdv_cat_ret_bd', 'tdv_cat_ror_pc_bd', 'tdv_cat_ret_ad', 'tdv_cat_ror_pc_ad',
    )

    def __init__(
        self,
        *,
        name: str,
        model_engine: ProjModelEngine | None = None,
        assets: ObjList[BaseAsset],
        asset_categories: list[str] | None,
        asset_category_attr: str,
        asset_report_bases: list[str] | None,
    ):
        self._objs: ObjList[BaseAsset] = assets
        self.categories: list[str] = asset_categories or []
        self.category_attr: str = asset_category_attr
        self.report_bases: list[str] = asset_report_bases or []
        n_cat, n_bas = len(self.categories), len(self.report_bases)
        self.tot_cf: float = float("nan")
        self.cat_cf: np.ndarray = np.empty(n_cat)
        self.tot_mv_op: float = float("nan")
        self.tot_mv_bd: float = float("nan")
        self.tot_mv_ad: float = float("nan")
        self.cat_mv_op: np.ndarray = np.empty(n_cat)
        self.cat_mv_bd: np.ndarray = np.empty(n_cat)
        self.cat_mv_ad: np.ndarray = np.empty(n_cat)
        self.tot_rv_op: np.ndarray = np.empty(n_bas)
        self.tot_rv_bd: np.ndarray = np.empty(n_bas)
        self.tot_rv_ad: np.ndarray = np.empty(n_bas)
        self.tot_ret_bd: np.ndarray = np.empty(n_bas)
        self.tot_ror_bd: np.ndarray = np.empty(n_bas)
        self.tot_ret_ad: np.ndarray = np.empty(n_bas)
        self.tot_ror_ad: np.ndarray = np.empty(n_bas)
        self.cat_rv_op: np.ndarray = np.empty(shape=(n_cat, n_bas))
        self.cat_rv_bd: np.ndarray = np.empty(shape=(n_cat, n_bas))
        self.cat_rv_ad: np.ndarray = np.empty(shape=(n_cat, n_bas))
        self.cat_ret_bd: np.ndarray = np.empty(shape=(n_cat, n_bas))
        self.cat_ror_bd: np.ndarray = np.empty(shape=(n_cat, n_bas))
        self.cat_ret_ad: np.ndarray = np.empty(shape=(n_cat, n_bas))
        self.cat_ror_ad: np.ndarray = np.empty(shape=(n_cat, n_bas))
        self._lc: Lifecycle[RecorderPhase] = Lifecycle[RecorderPhase]()

        # Initialize time-dimensioned variables for output
        # dims = None
        create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund')
        self.tdv_tot_cf: TDimVariable = create_tdv("totass_cash_flow")
        self.tdv_tot_mv_bd: TDimVariable = create_tdv("totass_mv_bd")
        self.tdv_tot_mv_ad: TDimVariable = create_tdv("totass_mv_ad")

        if asset_report_bases:
            # dims = asset_report_bases
            create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund',
                                                      dims=[asset_report_bases])
            self.tdv_tot_rv_bd: TDimArrayVariable = create_tdv("totass_rep_value_bd")
            self.tdv_tot_rv_ad: TDimArrayVariable = create_tdv("totass_rep_value_ad")
            self.tdv_tot_ret_bd: TDimArrayVariable = create_tdv("totass_inv_ret_bd")
            self.tdv_tot_ror_pc_bd: TDimArrayVariable = create_tdv("%totass_ror_bd")
            self.tdv_tot_ret_ad: TDimArrayVariable = create_tdv("totass_inv_ret_ad")
            self.tdv_tot_ror_pc_ad: TDimArrayVariable = create_tdv("%totass_ror_ad")
        if asset_categories:
            # dims = asset_categories
            create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund',
                                                      dims=[asset_categories])
            self.tdv_cat_cf: TDimArrayVariable = create_tdv("asset_cash_flow")
            self.tdv_cat_mv_bd: TDimArrayVariable = create_tdv("asset_mv_bd")
            self.tdv_cat_mv_ad: TDimArrayVariable = create_tdv("asset_mv_ad")
        if asset_categories and asset_report_bases:
            # dims = asset_categories, asset_report_bases
            create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund',
                                                      dims=[asset_categories, asset_report_bases])
            self.tdv_cat_rv_bd: TDimArrayVariable = create_tdv("asset_rep_value_bd")
            self.tdv_cat_rv_ad: TDimArrayVariable = create_tdv("asset_rep_value_ad")
            self.tdv_cat_ret_bd: TDimArrayVariable = create_tdv("asset_inv_ret_bd")
            self.tdv_cat_ror_pc_bd: TDimArrayVariable = create_tdv("%asset_ror_bd")
            self.tdv_cat_ret_ad: TDimArrayVariable = create_tdv("asset_inv_ret_ad")
            self.tdv_cat_ror_pc_ad: TDimArrayVariable = create_tdv("%asset_ror_ad")

    @property
    @transition(require_all=RecorderPhase.ASSET_REC_BD)
    def total_cash_flow(self) -> float:
        return self.tot_cf

    @property
    @transition(require_all=RecorderPhase.ASSET_REC_BD)
    def total_market_value_bd(self) -> float:
        return self.tot_mv_bd

    @property
    @transition(require_all=RecorderPhase.ASSET_REC_AD)
    def total_market_value_ad(self) -> float:
        return self.tot_mv_ad

    @transition(require_all=RecorderPhase.ASSET_REC_BD)
    def cash_flow(self, category: str | None = None) -> float:
        return self.tot_cf if category is None else self.cat_cf[self.categories.index(category)]

    @transition(require_all=RecorderPhase.ASSET_REC_BD)
    def report_bd(self, kind: Literal["report_value", "investment_return", "rate_of_return"],
                  basis: str, category: str | None = None) -> float:
        bas_idx = self.report_bases.index(basis)
        if category is None:
            if kind == "report_value": return self.tot_rv_bd[bas_idx]
            elif kind == "investment_return": return self.tot_ret_bd[bas_idx]
            elif kind == "rate_of_return": return self.tot_ror_bd[bas_idx]
            else: return float("nan")
        else:
            cat_idx = self.categories.index(category)
            if kind == "report_value": return self.cat_rv_bd[cat_idx, bas_idx]
            elif kind == "investment_return": return self.cat_ret_bd[cat_idx, bas_idx]
            elif kind == "rate_of_return": return self.cat_ror_bd[cat_idx, bas_idx]
            else: return float("nan")

    @transition(require_all=RecorderPhase.ASSET_REC_AD)
    def report_ad(self, kind: Literal["report_value", "investment_return", "rate_of_return"],
                  basis: str, category: str | None = None) -> float:
        bas_idx = self.report_bases.index(basis)
        if category is None:
            if kind == "report_value": return self.tot_rv_ad[bas_idx]
            elif kind == "investment_return": return self.tot_ret_ad[bas_idx]
            elif kind == "rate_of_return": return self.tot_ror_ad[bas_idx]
            else: return float("nan")
        else:
            cat_idx = self.categories.index(category)
            if kind == "report_value": return self.cat_rv_ad[cat_idx, bas_idx]
            elif kind == "investment_return": return self.cat_ret_ad[cat_idx, bas_idx]
            elif kind == "rate_of_return": return self.cat_ror_ad[cat_idx, bas_idx]
            else: return float("nan")

    def retrieve(
        self,
        t: int,
        kind: Literal["cash_flow", "market_value_bd", "market_value_ad", "report_value_bd", "report_value_ad",
            "investment_return_bd", "investment_return_ad", "rate_of_return_bd", "rate_of_return_ad"],
        basis: str | None = None, category: str | None = None
    ) -> float:
        maybe_raise_on_compare(t, must_ge=0, must_le=self.time)
        bas_idx = self.report_bases.index(basis) if basis else len(self.report_bases) + 1
        if category is None:
            if kind == "cash_flow": return self.tdv_tot_cf[t]
            elif kind == "market_value_bd": return self.tdv_tot_mv_bd[t]
            elif kind == "market_value_ad": return self.tdv_tot_mv_ad[t]
            elif kind == "report_value_bd": return self.tdv_tot_rv_bd[t][bas_idx]
            elif kind == "report_value_ad": return self.tdv_tot_rv_ad[t][bas_idx]
            elif kind == "investment_return_bd": return self.tdv_tot_ret_bd[t][bas_idx]
            elif kind == "investment_return_ad": return self.tdv_tot_ret_ad[t][bas_idx]
            elif kind == "rate_of_return_bd": return self.tdv_tot_ror_pc_bd[t][bas_idx] / 100
            elif kind == "rate_of_return_ad": return self.tdv_tot_ror_pc_ad[t][bas_idx] / 100
            else: return float("nan")
        else:
            cat_idx = self.categories.index(category)
            if kind == "cash_flow": return self.tdv_cat_cf[t][cat_idx]
            elif kind == "market_value_bd": return self.tdv_cat_mv_bd[t][cat_idx]
            elif kind == "market_value_ad": return self.tdv_cat_mv_ad[t][cat_idx]
            elif kind == "report_value_bd": return self.tdv_cat_rv_bd[t][cat_idx, bas_idx]
            elif kind == "report_value_ad": return self.tdv_cat_rv_ad[t][cat_idx, bas_idx]
            elif kind == "investment_return_bd": return self.tdv_cat_ret_bd[t][cat_idx, bas_idx]
            elif kind == "investment_return_ad": return self.tdv_cat_ret_ad[t][cat_idx, bas_idx]
            elif kind == "rate_of_return_bd": return self.tdv_cat_ror_pc_bd[t][cat_idx, bas_idx] / 100
            elif kind == "rate_of_return_ad": return self.tdv_cat_ror_pc_ad[t][cat_idx, bas_idx] / 100
            else: return float("nan")

    @transition(require_not=RecorderPhase.ASSET_REC_BD, mark=RecorderPhase.ASSET_REC_BD)
    def record_before_dealing(self, require_cash_flow: bool = True, require_investment_return: bool = True) -> None:
        """Record asset before dealing (bd).
        """
        t = self.time

        # Aggregate asset cash flow
        if require_cash_flow:
            self.tot_cf = self._objs.sum_attr("cash_flow", on_undefined=_on_undefined)
            self.tdv_tot_cf[t] = self.tot_cf
        if self.categories:
            val_dict = self._objs.sum_attr("cash_flow", group_by=self.category_attr, group_keys=self.categories,
                                           on_undefined=_on_undefined)
            self.cat_cf = np.array([val_dict.get(cat, 0.0) for cat in self.categories])
            self.tdv_cat_cf[t] = self.cat_cf

        # Aggregate asset value and asset cash flow
        self._sum_report_values("bd")

        if not require_investment_return:
            return

        # Calculate reported investment return
        if self.report_bases:
            income = np.array([self.tot_cf] * len(self.report_bases))  # scalar -> (report_bases, )
            ret, ror = self._compute_investment_return(
                prev_val=self.tot_rv_op, income=income, curr_val=self.tot_rv_bd)
            self.tot_ret_bd, self.tot_ror_bd = ret, ror
            self.tdv_tot_ret_bd[t] = ret
            self.tdv_tot_ror_pc_bd[t] = ror * 100

        if self.categories and self.report_bases:
            n_category = len(self.categories)
            n_basis = len(self.report_bases)
            income = np.array([[self.cat_cf[i]] * n_basis for i in range(n_category)])  # (categories, ) -> (categories, report_bases)
            ret, ror = self._compute_investment_return(
                prev_val=self.cat_rv_op, income=income, curr_val=self.cat_rv_bd)
            self.cat_ret_bd, self.cat_ror_bd = ret, ror
            self.tdv_cat_ret_bd[t] = ret
            self.tdv_cat_ror_pc_bd[t] = ror * 100

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
        if self.report_bases:
            income = np.array([self.tot_cf + self.tot_mv_bd - self.tot_mv_ad] * len(self.report_bases))  # scalar -> (report_bases, )
            ret, ror = self._compute_investment_return(
                prev_val=self.tot_rv_op, income=income, curr_val=self.tot_rv_ad)
            self.tot_ret_ad, self.tot_ror_ad = ret, ror
            self.tdv_tot_ret_ad[t] = ret
            self.tdv_tot_ror_pc_ad[t] = ror * 100

        if self.categories and self.report_bases:
            n_category = len(self.categories)
            n_basis = len(self.report_bases)
            income = np.array([[self.cat_cf[i] + self.cat_mv_bd[i] - self.cat_mv_ad[i]] * n_basis
                                  for i in range(n_category)])  # (categories, ) -> (categories, report_bases)
            ret, ror = self._compute_investment_return(
                prev_val=self.cat_rv_op, income=income, curr_val=self.cat_rv_ad)
            self.cat_ret_ad, self.cat_ror_ad = ret, ror
            self.tdv_cat_ret_ad[t] = ret
            self.tdv_cat_ror_pc_ad[t] = ror * 100

    def _sum_report_values(self, when: Literal["bd", "ad"]) -> None:
        """Aggregate values from all assets in the fund.

        Args:
            when (str): Timing of aggregation (bd/ad = before/after dealing).

        Raises:
            ValueError: If an asset is not rolled or updated for the current period, or if timing is invalid.
        """
        t = self.time
        if GLOBAL_CHECK_MODE != CheckMode.BYPASS:
            phase = AssetPhase.ROLLED if when == "bd" else AssetPhase.CLOSED
            for obj in self._objs:
                obj.require_lifecycle(phase=phase)

        # 1. always calculate total asset market value
        tot_mv = self._objs.sum_attr("market_value", on_undefined=_on_undefined)
        if when == "bd":
            self.tot_mv_bd, self.tdv_tot_mv_bd[t] = tot_mv, tot_mv
        elif when == "ad":
            self.tot_mv_ad, self.tdv_tot_mv_ad[t] = tot_mv, tot_mv
        else:
            raise ValueError(f"Unexpected when: {when}, expecting ('bd', 'ad').")

        # 2. category asset market value, shape = (categories, )
        if self.category_attr:
            val_dict = self._objs.sum_attr("market_value", group_by=self.category_attr, group_keys=self.categories,
                                           on_undefined=_on_undefined)
            cat_mv = np.array([val_dict.get(cat, 0.0) for cat in self.categories])
            if when == "bd":
                self.cat_mv_bd, self.tdv_cat_mv_bd[t] = cat_mv, cat_mv
            elif when == "ad":
                self.cat_mv_ad, self.tdv_cat_mv_ad[t] = cat_mv, cat_mv
            else:
                raise ValueError(f"Unexpected when: {when}, expecting ('bd', 'ad').")

        # 3. total asset reported values, shape = (report_bases, )
        if self.report_bases:
            tot_rv = self._objs.sum_attr(self.report_bases, on_undefined=_on_undefined)
            if when == "bd":
                self.tot_rv_bd, self.tdv_tot_rv_bd[t] = tot_rv, tot_rv
            elif when == "ad":
                self.tot_rv_ad, self.tdv_tot_rv_ad[t] = tot_rv, tot_rv
            else:
                raise ValueError(f"Unexpected when: {when}, expecting ('bd', 'ad').")

        # 4. category asset reported values, shape = (categories, report_bases)
        if self.report_bases and self.categories:
            val_dict = self._objs.sum_attr(self.report_bases, group_by=self.category_attr, group_keys=self.categories,
                                           on_undefined=_on_undefined)
            cat_rv = np.array([val_dict.get(cat, np.zeros(len(self.report_bases))) for cat in self.categories])
            if when == "bd":
                self.cat_rv_bd, self.tdv_cat_rv_bd[t] = cat_rv, cat_rv
            elif when == "ad":
                self.cat_rv_ad, self.tdv_cat_rv_ad[t] = cat_rv, cat_rv
            else:
                raise ValueError(f"Unexpected when: {when}, expecting ('bd', 'ad').")

    def _on_time_change(self) -> None:
        self.tot_mv_op = self.tot_mv_ad
        self.cat_mv_op = self.cat_mv_ad
        self.tot_rv_op = self.tot_rv_ad
        self.cat_rv_op = self.cat_rv_ad

    @staticmethod
    def _compute_investment_return(prev_val: np.ndarray, income: np.ndarray, curr_val: np.ndarray,
                                   ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate investment return and rate of return element-wise."""
        ret = curr_val + income - prev_val
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

    __slots__ = ('__weakref__', 'time', 'period', '_lc', '_objs',
                 'tot_cf', 'output_attrs_bd', 'output_attrs_ad',
                 'tdv_tot_cf', 'tdv_tot_attrs_bd', 'tdv_tot_attrs_ad',)

    def __init__(
        self,
        *,
        name: str,
        model_engine: ProjModelEngine | None = None,
        liabs: ObjList[BaseLiab],
        output_liab_attrs_bd: list[str],
        output_liab_attrs_ad: list[str],
    ):
        self._objs: ObjList[BaseLiab] = liabs
        self.output_attrs_bd: list[str] = output_liab_attrs_bd
        self.output_attrs_ad: list[str] = output_liab_attrs_ad
        self.tot_cf: float = float("nan")

        self._lc: Lifecycle[RecorderPhase] = Lifecycle[RecorderPhase]()

        # Initialize time-dimensioned variables for output
        create_tdv = lambda x: make_proj_variable(x, model_engine=model_engine, owner=name, group='fund')
        self.tdv_tot_cf: TDimVariable = create_tdv("totliab_cash_flow")
        self.tdv_tot_attrs_bd: list[TDimVariable] = [
            create_tdv(f"totliab_{name}") for name in self.output_attrs_bd]
        self.tdv_tot_attrs_ad: list[TDimVariable] = [
            create_tdv(f"totliab_{name}") for name in self.output_attrs_ad]

    @property
    @transition(require_all=RecorderPhase.LIAB_REC_BD)
    def total_cash_flow(self) -> float:
        return self.tot_cf

    @transition(require_not=RecorderPhase.LIAB_REC_BD, mark=RecorderPhase.LIAB_REC_BD)
    def record_before_dealing(self, require_cash_flow: bool = True) -> None:
        """Record liability before dealing (bd).
        """
        t = self.time

        # Aggregate liability cash flow
        if require_cash_flow:
            self.tot_cf = sum(obj.cash_flow for obj in self._objs)
            self.tdv_tot_cf[t] = self.tot_cf

        # Aggregate liability value
        self._sum_attrs("bd")

    @transition(require_not=RecorderPhase.LIAB_REC_AD, mark=RecorderPhase.LIAB_REC_AD)
    def record_after_dealing(self) -> None:
        """Process liability values and cash flows after dealing (ad).
        """
        self._sum_attrs("ad")

    def _sum_attrs(self, when: Literal["bd", "ad"]) -> None:
        """Aggregate values from all liabilities in the fund.

        Args:
            when (str): Timing of aggregation (bd/ad = before/after dealing).

        """
        if when == "bd":
            pairs = zip(self.output_attrs_bd, self.tdv_tot_attrs_bd)
        elif when == "ad":
            pairs = zip(self.output_attrs_ad, self.tdv_tot_attrs_ad)
        else:
            raise ValueError(f"Unexpected when: {when}, expecting ('bd', 'ad').")

        t = self.time
        for attr_name, tdv in pairs:
            tdv[t] = self._objs.sum_attr(attr_name, on_undefined="zerofill")

    def _on_time_change(self) -> None:
        pass


@time_synchronized("model_engine.time_synchronizer")
class OtherRecorder:
    """Performs aggregation and performance calculations for a fund.

    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    __slots__ = ('__weakref__', 'time', 'period', '_lc', 'name',
                 'tax_paid', 'capital_received', 'dividends_paid',
                 'tdv_free_proceeds_bd', 'tdv_free_proceeds_ad', 'tdv_free_proceeds_eop',
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
        self.tdv_free_proceeds_eop: TDimVariable = create_tdv("free_proceeds_eop")
        self.tdv_proceeds_tran_in: TDimVariable = create_tdv("proceeds_transferred_in")
        self.tdv_proceeds_tran_out: TDimVariable = create_tdv("proceeds_transferred_out")
        self.tdv_proceeds_tran_net: TDimVariable = create_tdv("proceeds_transferred_net")
        self.tdv_tax_paid: TDimVariable = create_tdv("tax_paid")
        self.tdv_capital_received: TDimVariable = create_tdv("capital_contribution_received")
        self.tdv_dividends_paid: TDimVariable = create_tdv("dividends_paid_to_shareholders")

    def record_free_proceeds(self, value: float, when: Literal["bd", "ad", "eop"]) -> None:
        if when == "bd":
            self.tdv_free_proceeds_bd[self.time] = value
        elif when == "ad":
            self.tdv_free_proceeds_ad[self.time] = value
        elif when == "eop":
            self.tdv_free_proceeds_eop[self.time] = value
        else:
            pass

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
        self.tdv_free_proceeds_eop[t] = None

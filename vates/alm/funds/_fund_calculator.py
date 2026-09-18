import numpy as np
import pandas as pd

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDimVariable
from vates.global_conf import CHECK_LEVEL, CheckLevel
from vates.utils import maybe_raise_if_ne
from vates.alm.funds._utils import AssetLiabConnector


@add_projection_time_synchronizer
class FundCalculator:
    """Performs aggregation and performance calculations for a fund.

    Computes asset and liability aggregates, investment returns, and
    stores time-dimensioned arrays for reporting by class and total.

    Attributes:

    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    
    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', 'name', 'connector',
                 'asset_categories', 'asset_category_attr', 'asset_report_bases',
                 'output_liab_attrs_bd', 'output_liab_attrs_ad',)

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

        # Initialize time-dimensioned variables for output
        # dims = None
        create_tdv = lambda x: TDimVariable(x, model_engine=model_engine, owner=self.name, group='fund')
        self.tdv_totass_cash_flow: TDimVariable = create_tdv("totass_cash_flow")
        self.tdv_totliab_cash_flow: TDimVariable = create_tdv("totliab_cash_flow")
        self.tdv_totliab_attrs_bd: list[TDimVariable] = [
            create_tdv(f"totliab_{name}") for name in self.output_liab_attrs_bd]
        self.tdv_totliab_attrs_ad: list[TDimVariable] = [
            create_tdv(f"totliab_{name}") for name in self.output_liab_attrs_ad]

        self.tdv_free_estate_bd: TDimVariable = create_tdv("free_estate_bd")
        self.tdv_free_estate_ad: TDimVariable = create_tdv("free_estate_ad")
        self.tdv_proceeds_transferred_in: TDimVariable = create_tdv("proceeds_transferred_in")
        self.tdv_proceeds_transferred_out: TDimVariable = create_tdv("proceeds_transferred_out")
        # dims = asset_report_bases
        create_tdv = lambda x: TDimVariable(x, model_engine=model_engine, owner=self.name, group='fund', dims=[asset_report_bases])
        self.tdv_totass_rep_value_bd: TDimVariable = create_tdv("totass_rep_value_bd")
        self.tdv_totass_rep_value_ad: TDimVariable = create_tdv("totass_rep_value_ad")
        self.tdv_totass_inv_ret_bd: TDimVariable = create_tdv("totass_inv_ret_bd")
        self.tdv_totass_ror_pc_bd: TDimVariable = create_tdv("totass_ror_pc_bd")
        self.tdv_totass_inv_ret_ad: TDimVariable = create_tdv("totass_inv_ret_ad")
        self.tdv_totass_ror_pc_ad: TDimVariable = create_tdv("totass_ror_pc_ad")
        # dims = asset_categories
        create_tdv = lambda x: TDimVariable(x, model_engine=model_engine, owner=self.name, group='fund', dims=[asset_categories])
        self.tdv_asset_cash_flow: TDimVariable = create_tdv("asset_cash_flow")
        # dims = asset_categories, asset_report_bases
        create_tdv = lambda x: TDimVariable(x, model_engine=model_engine, owner=self.name, group='fund', dims=[asset_categories, asset_report_bases])
        self.tdv_asset_rep_value_bd: TDimVariable = create_tdv("asset_rep_value_bd")
        self.tdv_asset_rep_value_ad: TDimVariable = create_tdv("asset_rep_value_ad")
        self.tdv_asset_inv_ret_bd: TDimVariable = create_tdv("asset_inv_ret_bd")
        self.tdv_asset_ror_pc_bd: TDimVariable = create_tdv("asset_ror_pc_bd")
        self.tdv_asset_inv_ret_ad: TDimVariable = create_tdv("asset_inv_ret_ad")
        self.tdv_asset_ror_pc_ad: TDimVariable = create_tdv("asset_ror_pc_ad")

    def process_assets_before_dealing(self) -> None:
        """Process asset values and returns before dealing (bd).
        """
        t = self.time
        if CHECK_LEVEL != CheckLevel.BYPASS:
            for asset in self.connector.assets:
                maybe_raise_if_ne(asset._state, ("rolled", t))

        # Aggregate asset cash flow
        tot_cash_flow = self.connector.groupby_sum_asset_cash_flow()
        cat_cash_flow = self.connector.groupby_sum_asset_cash_flow(groupby=self.asset_category_attr, in_list=self.asset_categories)
        cat_cash_flow = np.array(cat_cash_flow) # convert to np array
        self.tdv_asset_cash_flow[t] = cat_cash_flow
        self.tdv_totass_cash_flow[t] = tot_cash_flow

        # Aggregate asset value and asset cash flow
        self.aggregate_assets_value("bd")

        # Calculate rates of return
        asset_inv_ret = np.zeros((len(self.asset_categories), len(self.asset_report_bases)))
        asset_ror = np.zeros((len(self.asset_categories), len(self.asset_report_bases)))
        totass_inv_ret = np.zeros(len(self.asset_report_bases))
        totass_ror = np.zeros(len(self.asset_report_bases))

        for i in range(len(self.asset_report_bases)):
            totass_inv_ret[i], totass_ror[i] = self._calculate_investment_return(
                prev_val=float(self.tdv_totass_rep_value_ad[t - 1][i]),
                cash_flow=float(self.tdv_totass_cash_flow[t]),
                curr_val=float(self.tdv_totass_rep_value_bd[t][i])
            )

            for j in range(len(self.asset_categories)):
                asset_inv_ret[j, i], asset_ror[j, i] = self._calculate_investment_return(
                    prev_val=(self.tdv_asset_rep_value_ad[t - 1][j, i]),
                    cash_flow=float(self.tdv_asset_cash_flow[t][j]),
                    curr_val=float(self.tdv_asset_rep_value_bd[t][j, i])
                )

        self.tdv_asset_inv_ret_bd[t] = asset_inv_ret
        self.tdv_asset_ror_pc_bd[t] = asset_ror * 100
        self.tdv_totass_inv_ret_bd[t] = totass_inv_ret
        self.tdv_totass_ror_pc_bd[t] = totass_ror * 100

    def process_assets_after_dealing(self) -> None:
        """Summarize asset values and returns after dealing (ad).
        """
        t = self.time
        # Aggregate asset value
        self.aggregate_assets_value("ad")

        # Calculate rates of return
        asset_inv_ret = np.zeros((len(self.asset_categories), len(self.asset_report_bases)))
        asset_ror = np.zeros((len(self.asset_categories), len(self.asset_report_bases)))
        totass_inv_ret = np.zeros(len(self.asset_report_bases))
        totass_ror = np.zeros(len(self.asset_report_bases))
        mv_index = self.asset_report_bases.index("MV")

        for i in range(len(self.asset_report_bases)):
            if i == mv_index:  # asset dealing doesn't impact MV basis
                totass_inv_ret[i] = self.tdv_totass_inv_ret_bd[t][i]
                totass_ror[i] = self.tdv_totass_ror_pc_bd[t][i] / 100
            else:
                gl_from_dealing = ((self.tdv_totass_rep_value_ad[t][i] - self.tdv_totass_rep_value_ad[t][mv_index])
                                   - (self.tdv_totass_rep_value_bd[t][i] - self.tdv_totass_rep_value_bd[t][mv_index]))
                totass_inv_ret[i] = self.tdv_totass_inv_ret_bd[t][i] + gl_from_dealing
                totass_ror[i] = 0 if totass_inv_ret[i] == 0 else totass_inv_ret[i] / self.tdv_totass_rep_value_ad[t - 1][i]

            for j in range(len(self.asset_categories)):
                if i == mv_index:  # asset dealing doesn't impact MV basis
                    asset_inv_ret[j, i] = self.tdv_asset_inv_ret_bd[t][j, i]
                    asset_ror[j, i] = self.tdv_asset_ror_pc_bd[t][j, i] / 100
                else:
                    gl_from_dealing = ((self.tdv_asset_rep_value_ad[t][j, i] - self.tdv_asset_rep_value_ad[t][j, mv_index])
                                       - (self.tdv_asset_rep_value_bd[t][j, i] - self.tdv_asset_rep_value_bd[t][j, mv_index]))
                    asset_inv_ret[j, i] = self.tdv_asset_inv_ret_bd[t][j, i] + gl_from_dealing
                    if asset_inv_ret[j, i] == 0:
                        asset_ror[j, i] = 0
                    elif self.tdv_asset_rep_value_ad[t - 1][j, i] == 0: # this can happen for derivatives such as futures
                        asset_ror[j, i] = 0
                    else:
                        asset_inv_ret[j, i] = asset_inv_ret[j, i] / self.tdv_asset_rep_value_ad[t - 1][j, i]

        self.tdv_asset_inv_ret_ad[t] = asset_inv_ret
        self.tdv_asset_ror_pc_ad[t] = asset_ror * 100
        self.tdv_totass_inv_ret_ad[t] = totass_inv_ret
        self.tdv_totass_ror_pc_ad[t] = totass_ror * 100

    def aggregate_assets_value(self, timing: str) -> None:
        """Aggregate values from all assets in the fund.

        Args:
            timing (str): Timing of aggregation (bd/ad = before/after dealing).

        Raises:
            ValueError: If an asset is not rolled or updated for the current period, or if timing is invalid.
        """
        t = self.time
        if t > 0 and CHECK_LEVEL != CheckLevel.BYPASS:
            s = "rolled" if timing == "bd" else "closed"
            for asset in self.connector.assets:
                maybe_raise_if_ne(asset._state, (s, t))

        tot_rep_value = self.connector.groupby_sum_asset_report_value(basis=self.asset_report_bases)
        cat_rep_value = self.connector.groupby_sum_asset_report_value(
            basis=self.asset_report_bases, groupby=self.asset_category_attr, in_list=self.asset_categories)
        cat_rep_value = np.array(cat_rep_value)  # convert to np array

        if timing == "bd":
            self.tdv_asset_rep_value_bd[t] = cat_rep_value
            self.tdv_totass_rep_value_bd[t] = tot_rep_value
        elif timing == "ad":
            self.tdv_asset_rep_value_ad[t] = cat_rep_value
            self.tdv_totass_rep_value_ad[t] = tot_rep_value
        else:
            raise ValueError(f"Invalid asset aggregation {timing=}.")

    def process_liabs_before_dealing(self) -> None:
        """Process liability values and cash flows before dealing (bd).
        """
        t = self.time
        if CHECK_LEVEL != CheckLevel.BYPASS:
            for liab in self.connector.liabs:
                maybe_raise_if_ne(liab._state, ("rolled", t))

        # Aggregate liability cash flow
        self.tdv_totliab_cash_flow[t] = sum(liab.cash_flow for liab in self.connector.liabs)

        # Aggregate liability value
        self.aggregate_liabs_value("bd")

    def process_liabs_after_dealing(self):
        """Process liability values and cash flows after dealing (ad).
        """
        self.aggregate_liabs_value("ad")

    def aggregate_liabs_value(self, timing: str) -> None:
        """Aggregate values from all liabilities in the fund.

        Args:
            timing (str): Timing of aggregation (bd/ad = before/after dealing).

        Raises:
            ValueError: If a liability is not rolled/updated for the current period, or if timing is invalid.
        """
        t = self.time
        if timing == "bd":
            for attr_name, tdv in zip(self.output_liab_attrs_bd, self.tdv_totliab_attrs_bd):
                tdv[t] = self.connector.sum_liab(attr_name)
        elif timing == "ad":
            for attr_name, tdv in zip(self.output_liab_attrs_ad, self.tdv_totliab_attrs_ad):
                tdv[t] = self.connector.sum_liab(attr_name)
        else:
            raise ValueError(f"Invalid liab aggregation {timing=}.")

    @staticmethod
    def _calculate_investment_return(prev_val: float, cash_flow: float, curr_val: float) -> tuple[float, float]:
        """Calculate investment return and rate of return.

        Args:
            prev_val (float): Previous value of the asset.
            cash_flow (float): Asset cash flow during the period.
            curr_val (float): Current value of the asset.

        Returns:
            tuple[float, float]: (Investment return, rate of return).
        """
        if (curr_val + cash_flow - prev_val) == 0:
            return 0, 0

        ret = curr_val + cash_flow - prev_val
        if prev_val == 0:  # this can happen for derivatives such as futures
            return ret, 0
        else:
            return ret, ret / prev_val

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.name}'"

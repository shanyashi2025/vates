import numpy as np
import pandas as pd

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDepVariable
from vates.utils import t_checker
from vates.alm.enums import AssetRepBasis
from vates.alm.funds._utils import ALContainer


@add_projection_time_synchronizer
class FundCalculator:
    """Performs aggregation and performance calculations for a fund.

    Computes asset and liability aggregates, investment returns, URGL/RGL, and
    stores time-dependent arrays for reporting by class and total.

    Attributes:

    """
    time: int           # for type hint only, will be injected by decorator `has_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `has_time_synchronizer`
    
    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_tt_dict', 'container', 'asset_categories_enum')

    def __init__(
        self,
        *,
        model_engine: ProjModelEngine | None = None,
        container: ALContainer,
        asset_categories: list[str]
    ):
        self.container: ALContainer = container
        self.asset_categories_enum: dict[str, int] = {item: i for i, item in enumerate(asset_categories)}

        # Initialize time-dependent variables used for reporting
        fund_id = self.container.name
        # dims = None
        create_tdv = lambda name: TDepVariable(name, model_engine=model_engine, owner=fund_id, group='fund')
        self.tdv_totass_cash_flow: TDepVariable = create_tdv("totass_cash_flow")
        self.tdv_totass_urgl_bd: TDepVariable = create_tdv("totass_urgl_bd")
        self.tdv_totass_urgl_ad: TDepVariable = create_tdv("totass_urgl_ad")
        self.tdv_totass_rgl_ad: TDepVariable = create_tdv("totass_rgl_ad")
        self.tdv_totliab_cash_flow: TDepVariable = create_tdv("totliab_cash_flow")
        self.tdv_tot_num_pols: TDepVariable = create_tdv("tot_no_pols_if")
        self.tdv_tot_surr_val: TDepVariable = create_tdv("tot_surr_val_if")
        self.tdv_tot_math_res: TDepVariable = create_tdv("tot_math_res_if")
        self.tdv_tot_acct_val_bd: TDepVariable = create_tdv("tot_acct_val_if_bd")
        self.tdv_tot_acct_val_ad: TDepVariable = create_tdv("tot_acct_val_if_ad")
        self.tdv_tot_asset_share_bd: TDepVariable = create_tdv("tot_asset_share_if_bd")
        self.tdv_tot_asset_share_ad: TDepVariable = create_tdv("tot_asset_share_if_ad")
        self.tdv_free_estate_bd: TDepVariable = create_tdv("free_estate_bd")
        self.tdv_free_estate_ad: TDepVariable = create_tdv("free_estate_ad")
        self.tdv_proceeds_transferred_in: TDepVariable = create_tdv("proceeds_transferred_in")
        self.tdv_proceeds_transferred_out: TDepVariable = create_tdv("proceeds_transferred_out")
        # dims = AssetRepBasis
        create_tdv = lambda name: TDepVariable(name, model_engine=model_engine, owner=fund_id, group='fund', dims=[AssetRepBasis])
        self.tdv_totass_rep_value_bd: TDepVariable = create_tdv("totass_rep_value_bd")
        self.tdv_totass_rep_value_ad: TDepVariable = create_tdv("totass_rep_value_ad")
        self.tdv_totass_inv_ret_bd: TDepVariable = create_tdv("totass_inv_ret_bd")
        self.tdv_totass_ror_pc_bd: TDepVariable = create_tdv("totass_ror_pc_bd")
        self.tdv_totass_inv_ret_ad: TDepVariable = create_tdv("totass_inv_ret_ad")
        self.tdv_totass_ror_pc_ad: TDepVariable = create_tdv("totass_ror_pc_ad")
        # dims = AssetClass
        create_tdv = lambda name: TDepVariable(name, model_engine=model_engine, owner=fund_id, group='fund', dims=[asset_categories])
        self.tdv_asset_cash_flow: TDepVariable = create_tdv("asset_cash_flow")
        self.tdv_asset_urgl_bd: TDepVariable = create_tdv("asset_urgl_bd")
        self.tdv_asset_urgl_ad: TDepVariable = create_tdv("asset_urgl_ad")
        self.tdv_asset_rgl_ad: TDepVariable = create_tdv("asset_rgl_ad")
        # dims = AssetClass,AssetRepBasis
        create_tdv = lambda name: TDepVariable(name, model_engine=model_engine, owner=fund_id, group='fund', dims=[asset_categories, AssetRepBasis])
        self.tdv_asset_rep_value_bd: TDepVariable = create_tdv("asset_rep_value_bd")
        self.tdv_asset_rep_value_ad: TDepVariable = create_tdv("asset_rep_value_ad")
        self.tdv_asset_inv_ret_bd: TDepVariable = create_tdv("asset_inv_ret_bd")
        self.tdv_asset_ror_pc_bd: TDepVariable = create_tdv("asset_ror_pc_bd")
        self.tdv_asset_inv_ret_ad: TDepVariable = create_tdv("asset_inv_ret_ad")
        self.tdv_asset_ror_pc_ad: TDepVariable = create_tdv("asset_ror_pc_ad")

    @t_checker({"proc_assets_bd": -1, "proc_assets_ad": -1}, "proc_assets_bd")
    def process_assets_before_dealing(self) -> None:
        """Process asset values and returns before dealing (bd).
        """
        t = self.time

        # Aggregate asset value and asset cash flow
        self._aggregate_assets_cash_flow()
        self.aggregate_assets_value("bd")

        # Calculate rates of return
        asset_inv_ret = np.zeros((len(self.asset_categories_enum), len(AssetRepBasis)))
        asset_ror = np.zeros((len(self.asset_categories_enum), len(AssetRepBasis)))
        totass_inv_ret = np.zeros(len(AssetRepBasis))
        totass_ror = np.zeros(len(AssetRepBasis))

        for i in range(len(AssetRepBasis)):
            totass_inv_ret[i], totass_ror[i] = self._calculate_investment_return(
                float(self.tdv_totass_rep_value_ad[t - 1][i]),
                float(self.tdv_totass_cash_flow[t]),
                float(self.tdv_totass_rep_value_bd[t][i])
            )

            for j in range(len(self.asset_categories_enum)):
                asset_inv_ret[j, i], asset_ror[j, i] = self._calculate_investment_return(
                    float(self.tdv_asset_rep_value_ad[t - 1][j, i]),
                    float(self.tdv_asset_cash_flow[t][j]),
                    float(self.tdv_asset_rep_value_bd[t][j, i])
                )

        self.tdv_asset_inv_ret_bd[t] = asset_inv_ret
        self.tdv_asset_ror_pc_bd[t] = asset_ror * 100
        self.tdv_totass_inv_ret_bd[t] = totass_inv_ret
        self.tdv_totass_ror_pc_bd[t] = totass_ror * 100

    @t_checker({"proc_assets_ad": -1, "proc_assets_bd": 0, "proc_liabs_bd": 0}, "proc_assets_ad")
    def process_assets_after_dealing(self) -> None:
        """Summarize asset values and returns after dealing (ad).
        """
        t = self.time

        # Aggregate asset value
        self.aggregate_assets_value("ad")

        # Calculate realized gain / loss
        self.tdv_asset_rgl_ad[t] = self.tdv_asset_urgl_bd[t] - self.tdv_asset_urgl_ad[t]
        self.tdv_totass_rgl_ad[t] = self.tdv_totass_urgl_bd[t] - self.tdv_totass_urgl_ad[t]

        # Calculate rates of return
        asset_inv_ret = np.zeros((len(self.asset_categories_enum), len(AssetRepBasis)))
        asset_ror = np.zeros((len(self.asset_categories_enum), len(AssetRepBasis)))
        totass_inv_ret = np.zeros(len(AssetRepBasis))
        totass_ror = np.zeros(len(AssetRepBasis))
        mv_index = AssetRepBasis.MV.value

        for i in range(len(AssetRepBasis)):
            if i == mv_index:  # asset dealing doesn't impact MV basis
                totass_inv_ret[i] = self.tdv_totass_inv_ret_bd[t][i]
                totass_ror[i] = self.tdv_totass_ror_pc_bd[t][i] / 100
            else:
                gl_from_dealing = ((self.tdv_totass_rep_value_ad[t][i] - self.tdv_totass_rep_value_ad[t][mv_index])
                                   - (self.tdv_totass_rep_value_bd[t][i] - self.tdv_totass_rep_value_bd[t][mv_index]))
                totass_inv_ret[i] = self.tdv_totass_inv_ret_bd[t][i] + gl_from_dealing
                totass_ror[i] = 0 if totass_inv_ret[i] == 0 else totass_inv_ret[i] / self.tdv_totass_rep_value_ad[t - 1][i]

            for j in range(len(self.asset_categories_enum)):
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

    def _aggregate_assets_cash_flow(self) -> None:
        """Aggregate cash flows from all assets.

        Raises:
            ValueError: If any asset is not rolled for the current period.
        """
        t = self.time
        cls_cash_flow = np.zeros(len(self.asset_categories_enum))
        tot_cash_flow = 0.0

        for asset in self.container.assets:
            if asset.last_roll_forward != t:
                raise ValueError(f"Asset {asset.asset_id} is not rolled on {t} ({self.period}).")

            cash_flow = asset.cash_flow
            cls_cash_flow[self.asset_categories_enum[asset.asset_category],] += cash_flow
            tot_cash_flow += cash_flow

        self.tdv_asset_cash_flow[t] = cls_cash_flow
        self.tdv_totass_cash_flow[t] = tot_cash_flow

    def aggregate_assets_value(self, timing: str) -> None:
        """Aggregate values from all assets in the fund.

        Args:
            timing (str): Timing of aggregation (bd/ad = before/after dealing).

        Raises:
            ValueError: If an asset is not rolled or updated for the current period, or if timing is invalid.
        """
        t = self.time
        cls_rep_value = np.zeros([len(self.asset_categories_enum), len(AssetRepBasis)])
        tot_rep_value = np.zeros(len(AssetRepBasis))

        for asset in self.container.assets:
            if asset.last_roll_forward != t:
                raise ValueError(f"Asset {asset.asset_id} is not rolled on {t} ({self.period}).")

            if timing == "ad" and asset.last_dealing != t:
                raise ValueError(f"Asset {asset.asset_id} is not updated after dealing (ad) on {t} ({self.period}).")

            rep_value = asset.rep_value
            cls_rep_value[self.asset_categories_enum[asset.asset_category],] += rep_value
            tot_rep_value += rep_value

        cls_urgl = cls_rep_value[:, AssetRepBasis.MV.value] - cls_rep_value[:, AssetRepBasis.FAV.value]
        tot_urgl = tot_rep_value[AssetRepBasis.MV.value] - tot_rep_value[AssetRepBasis.FAV.value]

        if timing == "bd":
            self.tdv_asset_rep_value_bd[t] = cls_rep_value
            self.tdv_totass_rep_value_bd[t] = tot_rep_value
            self.tdv_asset_urgl_bd[t] = cls_urgl
            self.tdv_totass_urgl_bd[t] = tot_urgl
        elif timing == "ad":
            self.tdv_asset_rep_value_ad[t] = cls_rep_value
            self.tdv_totass_rep_value_ad[t] = tot_rep_value
            self.tdv_asset_urgl_ad[t] = cls_urgl
            self.tdv_totass_urgl_ad[t] = tot_urgl
        else:
            raise ValueError(f"Invalid asset aggregation {timing=}.")

    @t_checker({"proc_liabs_bd": -1, "proc_liabs_ad": -1, "proc_assets_bd": 0}, "proc_liabs_bd", )
    def process_liabs_before_dealing(self) -> None:
        """Process liability values and cash flows before dealing (bd).
        """
        # Aggregate liability value and cash flow
        self._aggregate_liabs_cash_flow()
        self.aggregate_liabs_value("bd")

    def _aggregate_liabs_cash_flow(self) -> None:
        """Aggregate cash flows from all liabilities.

        Raises:
            ValueError: If any liability is not rolled for the current period.
        """
        t = self.time
        tot_cash_flow = 0.0

        for liab in self.container.liabs:
            if liab.last_roll_forward != t:
                raise ValueError(f"Liab {liab.liab_id} is not rolled on {t} ({self.period}).")
            tot_cash_flow += liab.cash_flow

        self.tdv_totliab_cash_flow[t] = tot_cash_flow

    @t_checker({"proc_liabs_ad": -1, "proc_liabs_bd": 0, "proc_assets_ad": 0}, "proc_liabs_ad", )
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
            tot_num_pols = 0.0
            tot_surr_val = 0.0
            tot_math_res = 0.0
            tot_acct_value = 0.0
            tot_asset_share = 0.0

            for liab in self.container.liabs:
                if liab.last_roll_forward != t:
                    raise ValueError(f"Liab {liab.liab_id} is not rolled on {t} ({self.period}).")

                tot_num_pols += liab.num_pols
                tot_surr_val += liab.surr_val
                tot_math_res += liab.math_res
                tot_acct_value += liab.acct_value
                tot_asset_share += liab.asset_share

            self.tdv_tot_num_pols[t] = tot_num_pols
            self.tdv_tot_surr_val[t] = tot_surr_val
            self.tdv_tot_math_res[t] = tot_math_res
            self.tdv_tot_acct_val_bd[t] = tot_acct_value
            self.tdv_tot_asset_share_bd[t] = tot_asset_share

        elif timing == "ad":
            tot_acct_value = 0.0
            tot_asset_share = 0.0

            for liab in self.container.liabs:
                if liab.last_roll_forward != t:
                    raise ValueError(f"Liab {liab.liab_id} is not rolled on {t} ({self.period}).")

                if liab.last_update_ad != t:
                    raise ValueError(f"Liab {liab.liab_id} is not updated after dealing (ad) on t={t} ({self.period}).")

                tot_acct_value += liab.acct_value
                tot_asset_share += liab.asset_share

            self.tdv_tot_acct_val_ad[t] = tot_acct_value
            self.tdv_tot_asset_share_ad[t] = tot_asset_share

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
        return f"{type(self).__name__} - '{self.container.name}'"

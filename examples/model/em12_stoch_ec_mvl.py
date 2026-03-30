import numpy as np

from vates import ProjModelEngine, StochExecutor, ConstVariable, alm
from vates.utils import df_to_karray

from local_package import (
    load_file_df,
    build_esg_karr,
    build_yield_curves,
    # build_credit_bands,
    build_equity_indices,
    update_yield_curves,
    # update_credit_bands,
    update_equity_indices,
    update_market_info,
    update_esg_this_month,
    FundrebalanceParams,
    build_rebalance_policy,
    build_all_existing_assets,
    build_liabs,
    fund_assets_roll_forward,
    fund_liabs_roll_forward,
    rebalance_this_month,
    build_all_profile_assets,
    build_target_allocation,
    liabs_update_ad,
)

class ECMVL(ProjModelEngine):
    """
    Example stochastic EC MVL model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize tables
        df = self.read_csv("_file_names", index_col="table")
        filename_dict = {idx: row[self.SCENARIO] for idx, row in df.iterrows() if idx not in ('esg', )}
        file_read_config = self.load_json("_file_read_config")
        self.file_df_dict = load_file_df(self.read_csv, filename_dict, file_read_config)

        # process epl
        self.epl_karr = df_to_karray(self.file_df_dict['epl'], col_index_name='date')
        del self.file_df_dict['epl']

        # initialize economic variables
        self.esg_step = 12
        esg_df = self.read_parquet(df.loc['esg', self.SCENARIO], filter_dict={'SIMULATION': self.SIMULATION})
        esg_df.drop('SIMULATION', axis=1, inplace=True)  # drop SIMULATION

        self.yield_curves, self.yield_curve_esg_helper = build_yield_curves(self, self.file_df_dict['yield_curves'])
        self.yield_curve_karr = build_esg_karr(esg_df, self.yield_curve_esg_helper)
        # self.credit_bands, self.credit_band_esg_helper = build_credit_bands(self, self.file_df_dict['credit_bands'])
        # self.credit_band_karr = build_esg_karr(esg_df, self.credit_band_esg_helper)
        self.equity_indices, self.equity_index_esg_helper = build_equity_indices(self, self.file_df_dict['equity_indices'])
        self.equity_indidex_karr = build_esg_karr(esg_df, self.equity_index_esg_helper)

        deflators_df = esg_df[(esg_df["CLASS"]=='VALN') & (esg_df["MEASURE"]=='DEF')].copy()
        deflators_df.set_index(['ECONOMY', 'CLASS', 'MEASURE', 'TERM'], inplace=True)
        self.deflators_karr = df_to_karray(deflators_df)

        self.market_info = alm.econs.MarketInfo(self,'general_market_info')
        self.market_dict = {
            'currencies': [],
            'equity_indices': self.equity_indices,
            'yield_curves': self.yield_curves,
            'credit_bands': [], # self.credit_bands,
            'market_info': self.market_info,
        }
        # funds
        self.funds = []
        self.fund_rebalance_params: dict = {}
        self.bel_dict: dict[str, ConstVariable] = {}

    def time_zero_calculations(self):
        # ------------------------------------------------------------------------------
        # [01] update market variables
        # ------------------------------------------------------------------------------
        self._update_market_variables()

        # ------------------------------------------------------------------------------
        # [02] initialize funds
        # ------------------------------------------------------------------------------       
        df = self.file_df_dict['funds']
        for idx, row in df.iterrows():
            # step 1: create fund instance
            fund_id = str(idx)
            self.fund_rebalance_params[fund_id] = FundrebalanceParams(
                size_type=alm.funds.FundSizeType[row["fund_size_type"].upper()],
                size_basis=alm.AssetRepBasis[row["fund_size_basis"].upper()],
                rebalance_freq=row["fund_rebalance_freq"]  # 1=A, 2=H, 4=Q, 12=M, 0=SKIP
            )
            rebalance_policy = build_rebalance_policy(self.file_df_dict['rebalance_policy'].copy(), fund_id)
            fund = alm.funds.Fund(self, fund_id, rebalance_policy, row["asset_classes_reported"].split(';'))
            self.funds.append(fund)

            # step 2: create existing asset instances and add to fund
            assets = build_all_existing_assets(self, self.file_df_dict, self.market_dict, fund.fund_id)['all']
            fund.add_assets(assets)

            # step 3: create existing liab instances and add to fund
            liabs = build_liabs(self, self.file_df_dict['liabs'].copy(), fund_id, currencies=[])
            fund.add_liabs(liabs)

            # step 4 (last step): assemble the fund
            fund.assemble()

    def in_time_calculations(self):
        t, p = self.time, self.period
        # --------------------------------------------------------------------------
        # [01] update economic variables
        # --------------------------------------------------------------------------
        self._update_market_variables(1 if p.year == self.START_YEAR else self.esg_step)

        # --------------------------------------------------------------------------
        # [02] policyholder fund calculations
        # --------------------------------------------------------------------------
        skip_dcf = True if p.year > self.START_YEAR and (p.month % self.esg_step != 0) else False

        for fund in self.funds:
            # step 1: assets roll forward
            fund_assets_roll_forward(fund, skip_dcf=skip_dcf)

            # step 2: liabs roll forward
            fund_liabs_roll_forward(
                fund=fund, epl_karr=self.epl_karr,
                as_inv_ret=fund.rate_of_return_fav_bd(t),
                as_cf_ret=0 # specify the rate
            )

            # step 3: rebalance / rebalance assets
            rebalance_freq = self.fund_rebalance_params[fund.fund_id].rebalance_freq
            if rebalance_this_month(p.month, rebalance_freq):
                # initialize assets profile
                str_date = str(p.year * 100 + p.month)
                profile_assets = build_all_profile_assets(self, self.file_df_dict, self.market_dict, fund.fund_id)['all']
                target_allocation = build_target_allocation(self.file_df_dict['asset_allocation'].copy(),
                                                            fund.fund_id, str_date)
                fund.rebalance_assets(
                    fund_size_type=self.fund_rebalance_params[fund.fund_id].size_type,
                    fund_size_basis=self.fund_rebalance_params[fund.fund_id].size_basis,
                    target_weight=target_allocation,
                    assets_profile=profile_assets
                )
            else:
                fund.skip_rebalance()

            # step 4: liabs update after dealing (ad)
            liabs_update_ad(fund=fund)

            # step 5: transfer accumulated free proceeds to shareholder fund
            fund.transfer_free_proceeds_to_other(None)

    def post_time_calculations(self):
        deflators = np.zeros(self.MAX_T + 1)
        index_tuple = ('CNY', 'VALN', 'DEF', 0)
        row_index = self.deflators_karr.key_pos_pairs[0][index_tuple]
        col_ki_pairs = self.deflators_karr.key_pos_pairs[1]
        k = 0
        for i in range(self.MAX_T + 1):
            cal_date = self.START_DATE + i
            if self.esg_step == 1 or cal_date.year == self.START_YEAR: # for monthly step
                col_index = col_ki_pairs[str(cal_date.year * 100 + cal_date.month)]
                deflators[i] = self.deflators_karr[row_index, col_index]
            else:
                if (cal_date.month % self.esg_step) == 1:
                    d0 = deflators[i - 1]
                    col_index = col_ki_pairs[str(cal_date.year * 100 + cal_date.month + self.esg_step - 1)]
                    d1 = self.deflators_karr[row_index, col_index]
                    k = (d1 / d0) ** (1 / self.esg_step)
                deflators[i] = deflators[i - 1] * k

        for fund in self.funds:
            fund_id = fund.fund_id
            self.bel_dict[fund_id] = ConstVariable(self, 'BEL', fund_id, 'fund')
            self.bel_dict[fund_id][0] = - np.dot(fund.calculator.tdv_totliab_cash_flow.result, deflators)

    def _update_market_variables(self, esg_step: int=1):
        is_update, col_lookup = update_esg_this_month(self.period, esg_step)
        update_yield_curves(self.yield_curves, self.yield_curve_esg_helper, self.yield_curve_karr, col_lookup, is_update)
        # update_credit_bands(self.credit_bands, self.credit_band_esg_helper, self.credit_band_karr, col_lookup, is_update)
        update_equity_indices(self.equity_indices, self.equity_index_esg_helper, self.equity_indidex_karr, col_lookup,
                              is_update, esg_step=esg_step)
        update_market_info(self.market_info, self.file_df_dict['market_info'], self.yield_curves)


if __name__ == "__main__":
    from vates import cli_run
    cli_run(ECMVL, stoch_cls=StochExecutor)
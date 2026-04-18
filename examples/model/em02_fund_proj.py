import vates as vt
from local_package import (
    load_file_df,
    build_esg_karr,
    build_yield_curves,
    build_credit_bands,
    build_equity_indices,
    update_yield_curves,
    update_credit_bands,
    update_equity_indices,
    update_market_info,
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
    output_aging_assets,
)

class FundModel(vt.ProjModelEngine):
    """
    Performs fund level projection.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize tables
        df = self.read_csv("_file_names", index_col="table")
        filename_dict = {idx: row[self.SCENARIO] for idx, row in df.iterrows()}
        file_read_config = self.load_json("_file_read_config")
        self.file_df_dict = load_file_df(self.read_csv, filename_dict, file_read_config)
        # process epl
        self.epl_karr = vt.df_to_kr(self.file_df_dict['epl'], col_index_name='date')
        # initialize economic variables
        self.yield_curves, self.yield_curve_esg_helper = build_yield_curves(self, self.file_df_dict['yield_curves'])
        self.yield_curve_karr = build_esg_karr(self.file_df_dict['esg'], self.yield_curve_esg_helper)
        self.credit_bands, self.credit_band_esg_helper = build_credit_bands(self, self.file_df_dict['credit_bands'])
        self.credit_band_karr = build_esg_karr(self.file_df_dict['esg'], self.credit_band_esg_helper)
        self.equity_indices, self.equity_index_esg_helper = build_equity_indices(self, self.file_df_dict['equity_indices'])
        self.equity_indidex_karr = build_esg_karr(self.file_df_dict['esg'], self.equity_index_esg_helper)
        self.market_info = vt.alm.econs.MarketInfo(self, 'general_market_info')
        self.market_dict = {
            'currencies': [],
            'equity_indices': self.equity_indices,
            'yield_curves': self.yield_curves,
            'credit_bands': self.credit_bands,
            'market_info': self.market_info,
        }
        # funds
        self.funds = []
        self.ph_funds = []
        self.sh_fund = None
        self.fund_rebalance_params: dict = {}
        # epl and esg: data have been stored in numarr, delete df for memory efficiency
        del self.file_df_dict['epl']
        del self.file_df_dict['esg']

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
            fund_type = row["fund_type"]
            self.fund_rebalance_params[fund_id] = FundrebalanceParams(
                size_type=vt.alm.funds.FundSizeType[row["fund_size_type"].upper()],
                size_basis=vt.alm.AssetRepBasis[row["fund_size_basis"].upper()],
                rebalance_freq=row["fund_rebalance_freq"]  # 1=A, 2=H, 4=Q, 12=M, 0=SKIP
            )
            rebalance_policy = build_rebalance_policy(self.file_df_dict['rebalance_policy'].copy(), fund_id)
            fund = vt.alm.funds.Fund(self, fund_id, rebalance_policy, row["asset_classes_reported"].split(';'))
            self.funds.append(fund)
            if fund_type.lower() not in ('sh', 'shf', 'shareholder'):
                self.ph_funds.append(fund)
            else:
                if self.sh_fund: raise ValueError("Duplicated shareholder fund.")
                self.sh_fund = fund

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
        str_date = str(p.year * 100 + p.month)
        # --------------------------------------------------------------------------
        # [01] update economic variables
        # --------------------------------------------------------------------------
        self._update_market_variables()

        # --------------------------------------------------------------------------
        # [02] policyholder fund calculations
        # --------------------------------------------------------------------------
        for fund in self.ph_funds:
            # step 1: assets roll forward
            fund_assets_roll_forward(fund)

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
            fund.transfer_free_proceeds_to_other(self.sh_fund)

        # --------------------------------------------------------------------------
        # [03] shareholder fund calculation
        # --------------------------------------------------------------------------
        fund = self.sh_fund
        # step 1: assets roll forward
        fund_assets_roll_forward(fund)

        # step 2: (dummy step) liabs roll forward
        fund_liabs_roll_forward(fund)

        # step 3: rebalance / rebalance assets
        rebalance_freq = self.fund_rebalance_params[fund.fund_id].rebalance_freq
        if rebalance_this_month(p.month, rebalance_freq):
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

        # step 4: (dummy step) liabs update after dealing (ad)
        liabs_update_ad(fund=fund)

        # --- output aging assets ---
        if 'aging_assets_output_config' in self.file_df_dict and (
                df := self.file_df_dict['aging_assets_output_config']) is not None and (
                date_index := p.year * 100 + p.month) in df.index:
            all_assets = []
            for fund in self.funds:
                all_assets.extend(fund.assets)
            output_aging_assets(all_assets, df, date_index, self.workspace_directory)

    def post_time_calculations(self):
        pass

    def _update_market_variables(self):
        col_lookup = str(self.period.year * 100 + self.period.month)
        update_yield_curves(self.yield_curves, self.yield_curve_esg_helper, self.yield_curve_karr, col_lookup)
        update_credit_bands(self.credit_bands, self.credit_band_esg_helper, self.credit_band_karr, col_lookup)
        update_equity_indices(self.equity_indices, self.equity_index_esg_helper, self.equity_indidex_karr, col_lookup)
        update_market_info(self.market_info, self.file_df_dict['market_info'], self.yield_curves)

if __name__ == "__main__":
    vt.cli.run_model(FundModel)

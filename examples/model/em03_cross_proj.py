import numpy as np

from vates import ProjModelEngine, alm, KeyedArray
from vates.utils import df_to_karray
from vates.solvency.cn_cross2 import (
    MinCapUnit,
    MinCapInputer,
    AccountType,
    MinCapConsolidator,
    interest_risk_discount_curve
)

from local_package import (
    load_file_df,
    build_esg_karr,
    build_yield_curves,
    build_credit_bands,
    update_yield_curves,
    update_credit_bands,
    build_all_existing_assets,
)


class CROSSMinCapProj(ProjModelEngine):
    """
    C-ROSS minimum capital projection model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize tables
        df = self.read_csv("_file_names", index_col="table")
        filename_dict = {idx: row[self.SCENARIO] for idx, row in df.iterrows()}
        file_read_config = self.load_json("_file_read_config")
        self.file_df_dict = load_file_df(self.read_csv, filename_dict, file_read_config)
        # initialize market variables
        self.yield_curves, self.yield_curve_esg_helper = build_yield_curves(self, self.file_df_dict['yield_curves'])
        self.yield_curve_karr = build_esg_karr(self.file_df_dict['esg'], self.yield_curve_esg_helper)
        self.credit_bands, self.credit_band_esg_helper = build_credit_bands(self, self.file_df_dict['credit_bands'])
        self.credit_band_karr = build_esg_karr(self.file_df_dict['esg'], self.credit_band_esg_helper)
        self.market_dict = {'currencies': [], 'equity_indices': [], 'market_info': [],
                            'yield_curves': self.yield_curves, 'credit_bands': self.credit_bands,}
        # 60-day moving average of government bond yield curve
        self.gby_60d_ma_curve = next((x for x in self.yield_curves if x.curve_id == 'gby_60d_ma'), None)

        # list of asset file
        self.asset_filename_list = ['assets_cash', 'assets_equity', 'assets_bond',]

        # initialize dictionary of mc unit and mc input for each fund
        self.mc_unit_dict: dict[str, MinCapUnit] = {}
        self.mc_input_dict: dict[str, MinCapInputer] = {}

        for fund_id in (df := self.file_df_dict['funds']).index:
            cross_account_type = AccountType(df.loc[fund_id, 'cross_account_type'].upper())
            self.mc_unit_dict[fund_id] = MinCapUnit(self, fund_id, cross_account_type)
            self.mc_input_dict[fund_id] = MinCapInputer()

        # initialize the company result
        self.company_mc = MinCapConsolidator(self, 'company', [v for _, v in self.mc_unit_dict.items()])

        # epl
        self.epl_karr = df_to_karray(df=self.file_df_dict['epl'], unpack_multi_index=True, col_index_name='date')
        del self.file_df_dict['epl']

    def time_zero_calculations(self):
        pass

    def in_time_calculations(self):
        p = self.period

        if (date_index := p.year * 100 + p.month) not in self.file_df_dict['aging_assets_input_filelist'].index:
            return

        # --- (1) reset asset mc variables ---
        for _, inputer in self.mc_input_dict.items():
            inputer.reset()

        # --- (2) process economic assumptions ---
        self._update_market_variables()
        gby_60d_ma = np.array([self.gby_60d_ma_curve.spot_rates[i * 12] for i in range(41)]) # strip year data
        cross_intba, cross_intup, cross_intdn = interest_risk_discount_curve(gby_60d_ma)
        cross_intba_spot = _interp_monthly_spot(cross_intba)
        cross_intup_spot = _interp_monthly_spot(cross_intup)
        cross_intdn_spot = _interp_monthly_spot(cross_intdn)

        # --- (3) build asset objects that are in-force as at the time point ---
        df = self.file_df_dict['aging_assets_input_filelist']
        asset_df_dict = {
            item: self.read_csv(df.loc[date_index, item], keep_default_na=False, allow_not_found=True) for
            item in self.asset_filename_list
        }
        assets_dict = build_all_existing_assets(self, asset_df_dict, self.market_dict, None)

        # --- (4) calculate asset mc input ---
        mc_factor_equity = self.file_df_dict['cross_mc_factor'].loc[date_index, 'mc_factor_equity']
        mc_factor_spread = self.file_df_dict['cross_mc_factor'].loc[date_index, 'mc_factor_spread']
        for asset in assets_dict['all']:
            fund_id = asset.fund_id
            type_asset = type(asset)
            if type_asset == alm.assets.Equity:
                self.mc_input_dict[fund_id].mc_equity += asset.mv * mc_factor_equity
            elif type_asset == alm.assets.BondFixed:
                self.mc_input_dict[fund_id].aa_int_base += asset.pricer.calculate_market_price(p, cross_intba_spot) * asset.units
                self.mc_input_dict[fund_id].aa_int_up += asset.pricer.calculate_market_price(p, cross_intup_spot) * asset.units
                self.mc_input_dict[fund_id].aa_int_dn += asset.pricer.calculate_market_price(p, cross_intdn_spot) * asset.units
                self.mc_input_dict[fund_id].mc_spread += asset.mv * mc_factor_spread
            # elif type_asset == ...:
            #     ...

        # --- (5) collect liability mc input ---
        date_col=str(p.year * 100 + p.month)
        for _, row in self.file_df_dict['liabs'].iterrows():
            liab_id, fund_id = row[["liab_id", "fund_id"]]
            _get_cross_liab_mc_input_from_epl_df(
                mc_input=self.mc_input_dict[fund_id],
                epl_karr=self.epl_karr,
                liab_id=liab_id,
                date_col=date_col
            )

        # --- (6) calculate minimum capital ---
        for fund_id, unit in self.mc_unit_dict.items():
            unit.calculate_minimum_capital(self.mc_input_dict[fund_id])
        self.company_mc.calculate_minimum_capital()

    def post_time_calculations(self):
        pass

    def _update_market_variables(self):
        col_lookup = str(self.period.year * 100 + self.period.month)
        update_yield_curves(self.yield_curves, self.yield_curve_esg_helper, self.yield_curve_karr, col_lookup)
        update_credit_bands(self.credit_bands, self.credit_band_esg_helper, self.credit_band_karr, col_lookup)


def _interp_monthly_spot(spot_in: np.ndarray) -> np.ndarray:
    from vates.utils import convert_spot_to_fwrd, curve_interp, convert_fwrd_to_spot
    terms = np.arange(len(spot_in)) * 12 # term in month
    # modify the code to implement other curve interpolation method
    fwrd_in = convert_spot_to_fwrd(spot_in, 'A')
    fwrd_interp = curve_interp(terms, fwrd_in, "next")
    spot_out = convert_fwrd_to_spot(fwrd_interp, 'M')
    return spot_out


def _get_cross_liab_mc_input_from_epl_df(mc_input: MinCapInputer, epl_karr:KeyedArray, liab_id: str, date_col: str):
    mc_input.pv_base += epl_karr.loc[liab_id, 'pv_base', date_col]
    mc_input.pv_mortality += epl_karr.loc[liab_id, 'pv_mortality', date_col]
    mc_input.pv_catastrophe += epl_karr.loc[liab_id, 'pv_catastrophe', date_col]
    mc_input.pv_longevity += epl_karr.loc[liab_id, 'pv_longevity', date_col]
    mc_input.pv_morb_incidence += epl_karr.loc[liab_id, 'pv_morb_incidence', date_col]
    mc_input.pv_morb_trend += epl_karr.loc[liab_id, 'pv_morb_trend', date_col]
    mc_input.pv_health += epl_karr.loc[liab_id, 'pv_health', date_col]
    mc_input.pv_other_loss += epl_karr.loc[liab_id, 'pv_other_loss', date_col]
    mc_input.pv_expense += epl_karr.loc[liab_id, 'pv_expense', date_col]
    mc_input.pv_lapse_up += epl_karr.loc[liab_id, 'pv_lapse_up', date_col]
    mc_input.pv_lapse_dn += epl_karr.loc[liab_id, 'pv_lapse_dn', date_col]
    mc_input.pv_lapse_mass += epl_karr.loc[liab_id, 'pv_lapse_mass', date_col]
    mc_input.pv_int_base += epl_karr.loc[liab_id, 'pv_int_base', date_col]
    mc_input.pv_int_up += epl_karr.loc[liab_id, 'pv_int_up', date_col]
    mc_input.pv_int_dn += epl_karr.loc[liab_id, 'pv_int_dn', date_col]
    mc_input.pv_la_lower_limit += epl_karr.loc[liab_id, 'pv_la_lower_limit', date_col]


if __name__ == '__main__':
    from vates import cli_run
    cli_run(CROSSMinCapProj)

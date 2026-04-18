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
    build_all_existing_assets,
)

class AssetModel(vt.ProjModelEngine):
    """
    Performs projection to run off existing assets.
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
        # assets
        self.assets = []
        # epl: data have been stored in numarr, delete df for memory efficiency
        del self.file_df_dict['esg']

    def time_zero_calculations(self):
        self._update_market_variables()
        self.assets = build_all_existing_assets(self, self.file_df_dict, self.market_dict, None)['all']

    def in_time_calculations(self):
        self._update_market_variables()
        for asset in self.assets:
            asset.roll_forward()
            asset.complete_dealing()

    def post_time_calculations(self):
        pass

    def _update_market_variables(self):
        col_lookup = str(self.period.year * 100 + self.period.month)
        update_yield_curves(self.yield_curves, self.yield_curve_esg_helper, self.yield_curve_karr, col_lookup)
        update_credit_bands(self.credit_bands, self.credit_band_esg_helper, self.credit_band_karr, col_lookup)
        update_equity_indices(self.equity_indices, self.equity_index_esg_helper, self.equity_indidex_karr, col_lookup)
        update_market_info(self.market_info, self.file_df_dict['market_info'], self.yield_curves)

if __name__ == "__main__":
    vt.cli.run_model(AssetModel)

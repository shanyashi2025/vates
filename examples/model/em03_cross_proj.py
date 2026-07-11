import json
import numpy as np
import sys
from vates import ProjModelEngine, KeyedArray, kr_from_df, alm
from vates.solvency import cn_cross2
from local_package import (
    load_file_df,
    build_esg_master,
    build_all_existing_assets,
)

def cross_model(start_year: int, start_month: int, end_year: int, scenario: str, workspace_directory: str,
                input_directories: list[str], results_directory: str | None = None,
                model_name: str = "cross_model", model_description: str = "C-ROSS minimum capital projection."):
    model = ProjModelEngine(name=model_name, description=model_description)
    model.set_run_config(
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=12,
        scenario=scenario,
        workspace_directory=workspace_directory,
        input_directories=input_directories,
        results_directory=results_directory
    )

    df = model.read_csv("_file_names.csv", index_col="table")
    filename_dict = {idx: row[model.SCENARIO] for idx, row in df.iterrows()}
    file_read_args = model.load_json("_file_read_config.json")
    file_df_dict = load_file_df(model.read_csv, filename_dict, file_read_args, exclude=["esg_params", "esg"])

    # epl
    epl = kr_from_df(file_df_dict['epl'])
    del file_df_dict['epl']

    # build esg master
    esg_master = build_esg_master(
        model_engine=model,
        esg_params=model.load_json(filename_dict["esg_params"]),
        esg_df=model.read_csv(filename_dict["esg"])
    )

    # 60-day moving average of government bond yield curve
    gby_60d_ma_curve = next((x.econ_obj for x in esg_master.yield_curves if x.econ_obj.curve_id == 'gby_60d_ma'), None)
    if gby_60d_ma_curve is None:
        raise ValueError(f"'gby_60d_ma' is not found in {filename_dict["esg_params"]}.")

    # list of asset file
    asset_filename_list = ['assets_cash', 'assets_equity', 'assets_bond',]

    # initialize dictionary of mc unit and mc inputer for each fund
    mc_unit_dict: dict[str, cn_cross2.MinCapUnit] = {}
    mc_inputer_dict: dict[str, cn_cross2.MinCapInputer] = {}

    df = file_df_dict['funds']
    for fund_id in df.index:
        cross_account_type = cn_cross2.AccountType(df.loc[fund_id, 'cross_account_type'].upper())
        mc_unit_dict[fund_id] = cn_cross2.MinCapUnit(model, fund_id, cross_account_type)
        mc_inputer_dict[fund_id] = cn_cross2.MinCapInputer()

    # initialize the company result
    company_mc = cn_cross2.MinCapConsolidator(model, 'company', [v for _, v in mc_unit_dict.items()])

    liabs_df = file_df_dict['liabs']
    aging_assets_input_filelist_df = file_df_dict['aging_assets_input_filelist']
    cross_mc_factor_df = file_df_dict['cross_mc_factor']

    # ==================================================================================================================
    @model.bind_proj_func
    def cross_min_cap_projection():
        t, p = model.time, model.period
        date_index = p.year * 100 + p.month

        if date_index not in aging_assets_input_filelist_df.index:
            return

        # --- (1) reset asset mc variables ---
        for _, inputer in mc_inputer_dict.items():
            inputer.reset()

        # --- (2) process economic assumptions ---
        esg_master.update_econ_data(p)
        gby_60d_ma = np.array([gby_60d_ma_curve.spot_rates[i * 12] for i in range(41)])  # strip year data
        cross_intba, cross_intup, cross_intdn = cn_cross2.interest_risk_discount_curve(gby_60d_ma)
        cross_intba_spot = _interp_monthly_spot(cross_intba)
        cross_intup_spot = _interp_monthly_spot(cross_intup)
        cross_intdn_spot = _interp_monthly_spot(cross_intdn)

        # --- (3) build asset objects that are in-force as at the time point ---
        aging_assets_df_dict = {
            item: model.read_csv(
                aging_assets_input_filelist_df.at[date_index, item], keep_default_na=False, allow_not_found=True) for
            item in asset_filename_list
        }
        aging_assets_dict = build_all_existing_assets(model, aging_assets_df_dict, esg_master, None)

        # --- (4) calculate asset mc input ---
        mc_factor_equity = cross_mc_factor_df.at[date_index, 'mc_factor_equity']
        mc_factor_spread = cross_mc_factor_df.at[date_index, 'mc_factor_spread']
        for asset in aging_assets_dict['all']:
            type_asset = type(asset)
            mc_inputer = mc_inputer_dict[asset.fund_id]
            if type_asset == alm.assets.Equity:
                mc_inputer.mc_equity += asset.mv * mc_factor_equity
            elif type_asset == alm.assets.BondFixed:
                mc_inputer.aa_int_base += asset.pricer.calculate_market_price(p, cross_intba_spot) * asset.units
                mc_inputer.aa_int_up += asset.pricer.calculate_market_price(p, cross_intup_spot) * asset.units
                mc_inputer.aa_int_dn += asset.pricer.calculate_market_price(p, cross_intdn_spot) * asset.units
                mc_inputer.mc_spread += asset.mv * mc_factor_spread
            # elif type_asset == ...:
            #     ...

        # --- (5) collect liability mc input ---
        for _, row in liabs_df.iterrows():
            _liab_mc_inputer_add_from_epl(
                mc_inputer=mc_inputer_dict[row["fund_id"]],
                epl=epl,
                liab_id=row["liab_id"],
                date_col=date_index
            )

        # --- (6) calculate minimum capital ---
        for key, unit in mc_unit_dict.items():
            unit.calculate_minimum_capital(mc_inputer_dict[key])
        company_mc.calculate_minimum_capital()

    # ==================================================================================================================

    model.run()


def _interp_monthly_spot(spot_in: np.ndarray) -> np.ndarray:
    from vates.utils import convert_spot_to_fwrd, curve_interp, convert_fwrd_to_spot
    terms = np.arange(len(spot_in)) * 12 # term in month
    # modify the code to implement other curve interpolation method
    fwrd_in = convert_spot_to_fwrd(spot_in, 'A')
    fwrd_interp = curve_interp(terms, fwrd_in, "next")
    spot_out = convert_fwrd_to_spot(fwrd_interp, 'M')
    return spot_out


def _liab_mc_inputer_add_from_epl(mc_inputer: cn_cross2.MinCapInputer, epl: KeyedArray, liab_id: str, date_col):
    date_col = str(date_col)
    mc_inputer.pv_base += epl.at[liab_id, 'pv_base', date_col]
    mc_inputer.pv_mortality += epl.at[liab_id, 'pv_mortality', date_col]
    mc_inputer.pv_catastrophe += epl.at[liab_id, 'pv_catastrophe', date_col]
    mc_inputer.pv_longevity += epl.at[liab_id, 'pv_longevity', date_col]
    mc_inputer.pv_morb_incidence += epl.at[liab_id, 'pv_morb_incidence', date_col]
    mc_inputer.pv_morb_trend += epl.at[liab_id, 'pv_morb_trend', date_col]
    mc_inputer.pv_health += epl.at[liab_id, 'pv_health', date_col]
    mc_inputer.pv_other_loss += epl.at[liab_id, 'pv_other_loss', date_col]
    mc_inputer.pv_expense += epl.at[liab_id, 'pv_expense', date_col]
    mc_inputer.pv_lapse_up += epl.at[liab_id, 'pv_lapse_up', date_col]
    mc_inputer.pv_lapse_dn += epl.at[liab_id, 'pv_lapse_dn', date_col]
    mc_inputer.pv_lapse_mass += epl.at[liab_id, 'pv_lapse_mass', date_col]
    mc_inputer.pv_int_base += epl.at[liab_id, 'pv_int_base', date_col]
    mc_inputer.pv_int_up += epl.at[liab_id, 'pv_int_up', date_col]
    mc_inputer.pv_int_dn += epl.at[liab_id, 'pv_int_dn', date_col]
    mc_inputer.pv_la_lower_limit += epl.at[liab_id, 'pv_la_lower_limit', date_col]


def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        kwargs = json.load(file)
    cross_model(**kwargs)


if __name__ == '__main__':
    main()

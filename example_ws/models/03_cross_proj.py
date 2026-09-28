import numpy as np
from dataclasses import dataclass

from vates import ProjModelEngine
from vates.utils import KeyedArray
from vates.finmath import convert_interest_rates, interpolate_interest_rates
from vates.solvency.cn_cross2 import MinCapUnit, MinCapConsolidator, AccountType, interest_risk_discount_curve, AdditiveRiskCharge

from company_package import (
    run_with_json_config,
    load_file_df,
    EsgMaster,
    AssetMaster,
)


@dataclass
class MinCapUnderlyingInput:
    pv_base: float = 0.0
    pv_mortality: float = 0.0
    pv_catastrophe: float = 0.0
    pv_longevity: float = 0.0
    pv_morb_incidence: float = 0.0
    pv_morb_trend: float = 0.0
    pv_health: float = 0.0
    pv_other_loss: float = 0.0
    pv_expense: float = 0.0
    pv_lapse_up: float = 0.0
    pv_lapse_dn: float = 0.0
    pv_lapse_mass: float = 0.0
    pv_int_base: float = 0.0
    pv_int_up: float = 0.0
    pv_int_dn: float = 0.0
    pv_la_lower_limit: float = 0.0
    aa_int_base: float = 0.0
    aa_int_up: float = 0.0
    aa_int_dn: float = 0.0
    mc_non_life: float = 0.0
    mc_equity: float = 0.0
    mc_real_estate: float = 0.0
    mc_overseas_fixed_income: float = 0.0
    mc_overseas_equity: float = 0.0
    mc_exchange_rate: float = 0.0
    mc_spread: float = 0.0
    mc_counterparty_default: float = 0.0


def cross_model(start_year: int, start_month: int, end_year: int, scenario: str, input_directories: list[str],
                workspace_directory: str | None = None, results_directory: str | None = None,
                slug: str = "cross_model", description: str = "C-ROSS minimum capital projection"):
    model = ProjModelEngine(
        slug=slug,
        description=f"{description}, scenario: '{scenario}', from {start_year}/{start_month} to {end_year}/12."
    )
    model.configure_run(
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
    epl = KeyedArray.from_df(file_df_dict['epl'])
    del file_df_dict['epl']

    # build esg master
    esg_master = EsgMaster.from_df(
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

    # initialize dictionary of mc unit for each fund
    mc_units: list[MinCapUnit] = []
    df = file_df_dict['funds']
    for index in df.index:
        req_la = AccountType(df.loc[index, 'cross_account_type'].upper()) in (AccountType.PAR, AccountType.UNIV)
        mc_units.append(MinCapUnit(name=index, model_engine=model, require_loss_absorbency=AccountType(df.loc[index, 'cross_account_type'].upper()) in (AccountType.PAR, AccountType.UNIV)))

    # initialize the company result
    company_mc = MinCapConsolidator(name='company', model_engine=model, min_cap_units=mc_units)

    df = file_df_dict['liabs']
    liabs_dict: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        fund_id = row["fund_id"]
        liab_id = row["liab_id"]
        if fund_id in liabs_dict:
            liabs_dict[fund_id].append(liab_id)
        else:
            liabs_dict[fund_id] = [liab_id]

    aging_assets_input_filelist_df = file_df_dict['aging_assets_input_filelist']
    cross_mc_factor_df = file_df_dict['cross_mc_factor']

    # ==================================================================================================================
    @model.bind_projection
    def cross_min_cap_projection():
        t, p = model.time, model.period
        esg_master.update_econ_data(period=p)
        date_index = p.year * 100 + p.month

        if date_index not in aging_assets_input_filelist_df.index:
            return

        # --- (1) process economic assumptions ---
        gby_60d_ma = np.array([gby_60d_ma_curve.spot_rates[i * 12] for i in range(41)])  # strip year data
        cross_intba, cross_intup, cross_intdn = interest_risk_discount_curve(gby_60d_ma=gby_60d_ma)
        cross_intba_spot = _interp_monthly_spot(cross_intba)
        cross_intup_spot = _interp_monthly_spot(cross_intup)
        cross_intdn_spot = _interp_monthly_spot(cross_intdn)

        # --- (2) build asset objects that are in-force as at the time point ---
        aging_assets_df_dict = {
            item: model.read_csv(
                aging_assets_input_filelist_df.at[date_index, item], keep_default_na=False, allow_not_found=True)
            for item in asset_filename_list
        }
        aging_assets_master_dict = {
            item.name: AssetMaster.existing_from_df(
                aging_assets_df_dict, model_engine=model, econs=esg_master, fund_id=item.name)
            for item in mc_units
        }

        # --- (3) calculate minimum capital for each fund ---
        mc_factor_equity = cross_mc_factor_df.at[date_index, 'mc_factor_equity']
        mc_factor_spread = cross_mc_factor_df.at[date_index, 'mc_factor_spread']
        for mc_unit in mc_units:
            name = mc_unit.name
            # --- (3.0) initialize an MinCapUnderlyingInput instance ---
            mc_underlying_input = MinCapUnderlyingInput()

            # --- (3.1) calculate asset mc ---
            aging_assets_master = aging_assets_master_dict[name]
            # --- (3.1.1) equity risk mc ---
            for asset in aging_assets_master.equity_ls:
                mc_underlying_input.mc_equity += asset.market_value * mc_factor_equity
            # --- (3.1.2) interest rate risk mc ---
            for asset in aging_assets_master.fixed_bond_ls:
                mc_underlying_input.aa_int_base += asset.pricer.calculate_market_price(p, cross_intba_spot) * asset.units
                mc_underlying_input.aa_int_up += asset.pricer.calculate_market_price(p, cross_intup_spot) * asset.units
                mc_underlying_input.aa_int_dn += asset.pricer.calculate_market_price(p, cross_intdn_spot) * asset.units
                mc_underlying_input.mc_spread += asset.market_value * mc_factor_spread

            # --- (3.2) collect liability mc input ---
            liab_ls = liabs_dict.get(name, None) or []
            date_col = str(date_index)
            for item in liab_ls:
                mc_underlying_input.pv_base += epl.at[item, 'pv_base', date_col]
                mc_underlying_input.pv_mortality += epl.at[item, 'pv_mortality', date_col]
                mc_underlying_input.pv_catastrophe += epl.at[item, 'pv_catastrophe', date_col]
                mc_underlying_input.pv_longevity += epl.at[item, 'pv_longevity', date_col]
                mc_underlying_input.pv_morb_incidence += epl.at[item, 'pv_morb_incidence', date_col]
                mc_underlying_input.pv_morb_trend += epl.at[item, 'pv_morb_trend', date_col]
                mc_underlying_input.pv_health += epl.at[item, 'pv_health', date_col]
                mc_underlying_input.pv_other_loss += epl.at[item, 'pv_other_loss', date_col]
                mc_underlying_input.pv_expense += epl.at[item, 'pv_expense', date_col]
                mc_underlying_input.pv_lapse_up += epl.at[item, 'pv_lapse_up', date_col]
                mc_underlying_input.pv_lapse_dn += epl.at[item, 'pv_lapse_dn', date_col]
                mc_underlying_input.pv_lapse_mass += epl.at[item, 'pv_lapse_mass', date_col]
                mc_underlying_input.pv_int_base += epl.at[item, 'pv_int_base', date_col]
                mc_underlying_input.pv_int_up += epl.at[item, 'pv_int_up', date_col]
                mc_underlying_input.pv_int_dn += epl.at[item, 'pv_int_dn', date_col]
                mc_underlying_input.pv_la_lower_limit += epl.at[item, 'pv_la_lower_limit', date_col]

            # --- (3.3) calculate minimum capital ---
            additive_risk_charge = AdditiveRiskCharge(
                mortality=max(mc_underlying_input.pv_mortality - mc_underlying_input.pv_base, 0),
                catastrophe=max(mc_underlying_input.pv_catastrophe - mc_underlying_input.pv_base, 0),
                longevity=max(mc_underlying_input.pv_longevity - mc_underlying_input.pv_base, 0),
                morb_incidence=max(mc_underlying_input.pv_morb_incidence - mc_underlying_input.pv_base, 0),
                morb_trend=max(mc_underlying_input.pv_morb_trend - mc_underlying_input.pv_base, 0),
                health=max(mc_underlying_input.pv_health - mc_underlying_input.pv_base, 0),
                other_loss=max(mc_underlying_input.pv_other_loss - mc_underlying_input.pv_base, 0),
                expense=max(mc_underlying_input.pv_expense - mc_underlying_input.pv_base, 0),
                lapse_up=mc_underlying_input.pv_lapse_up - mc_underlying_input.pv_base,
                lapse_down=mc_underlying_input.pv_lapse_dn - mc_underlying_input.pv_base,
                mass_lapse=mc_underlying_input.pv_lapse_mass - mc_underlying_input.pv_base,
                non_life=mc_underlying_input.mc_non_life,
                interest_rate_up=(mc_underlying_input.pv_int_up - mc_underlying_input.aa_int_up) - (
                        mc_underlying_input.pv_int_base - mc_underlying_input.aa_int_base),
                interest_rate_down=(mc_underlying_input.pv_int_dn - mc_underlying_input.aa_int_dn) - (
                        mc_underlying_input.pv_int_base - mc_underlying_input.aa_int_base),
                equity=mc_underlying_input.mc_equity,
                real_estate=mc_underlying_input.mc_real_estate,
                overseas_fixed_income=mc_underlying_input.mc_overseas_fixed_income,
                overseas_equity=mc_underlying_input.mc_overseas_equity,
                exchange_rate=mc_underlying_input.mc_exchange_rate,
                spread=mc_underlying_input.mc_spread,
                counterparty_default=mc_underlying_input.mc_counterparty_default,
            )
            mc_unit.calculate(
                additive_risk_charge=additive_risk_charge,
                la_pv_base=mc_underlying_input.pv_base if mc_unit.require_loss_absorbency else 0.0,
                la_pv_lower_limit=mc_underlying_input.pv_la_lower_limit if mc_unit.require_loss_absorbency else 0.0,
            )

        # --- (4) consolidate company minimum capital ---
        company_mc.consolidate()

    # ==================================================================================================================

    model.run()
    return model


def _interp_monthly_spot(spot_in: np.ndarray) -> np.ndarray:
    terms = np.arange(len(spot_in)) * 12 # term in month
    # modify the code to implement other curve interpolation method
    fwrd_in = convert_interest_rates(spot_in, interval_unit="Y", from_what="spot", to_what="forward")
    fwrd_interp = interpolate_interest_rates(terms, fwrd_in, method="next")
    spot_out = convert_interest_rates(fwrd_interp, interval_unit="M", from_what="forward", to_what="spot")
    return spot_out


def main():
    run_with_json_config(cross_model)


if __name__ == '__main__':
    main()

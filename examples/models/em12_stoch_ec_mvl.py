import json
import numpy as np
import pandas as pd
import sys
from vates import StochExecutor, ProjModelEngine, KeyedArray, ConstVariable
from company_package import (
    load_file_df,
    EsgMaster,
    AssetMaster,
    FundMaster,
    build_liabs,
    fund_assets_roll_forward,
    fund_liabs_roll_forward,
    fund_reblance_if_needed,
    liabs_update_ad,
)


def fund_projection(model: ProjModelEngine, esg_params: dict, esg_filename: str, assets_df_dict: dict,
                    liabs_df: pd.DataFrame, funds_df: pd.DataFrame, rebalance_policy_df: pd.DataFrame,
                    epl: KeyedArray, asset_allocation_df: pd.DataFrame):
    try:
        import pyarrow.dataset as ds
    except ImportError:
        raise ImportError(f"need to install `pyarrow` library (pip install pyarrow)")

    t, p = model.time, model.period
    str_date = str(p.year * 100 + p.month)

    if t == 0:
        # build esg master
        filepath = model.get_filepath(esg_filename)
        dataset = ds.dataset(filepath, format="parquet")
        table = dataset.scanner(filter=ds.field("SIMULATION") == model.SIMULATION).to_table()
        esg_df = table.to_pandas().drop('SIMULATION', axis=1)  # drop SIMULATION column

        model.esg_master = EsgMaster.from_df(
            model_engine=model,
            esg_params=esg_params,
            esg_df=esg_df,
        )
        model.esg_master.update_econ_data(p)
        
        deflators_df = esg_df[(esg_df["CLASS"]=='VALN') & (esg_df["MEASURE"]=='DEF')].set_index(['ECONOMY', 'CLASS', 'MEASURE', 'TERM'])
        model.deflators_kr = KeyedArray.from_df(deflators_df)
        model.deflators = np.zeros(model.MAX_T + 1)
        model.deflators[0] = model.deflators_kr.at[('CNY', 'VALN', 'DEF', 0), str_date]

        # build fund master
        model.fund_master = FundMaster.from_df(
            df=funds_df,
            model_engine=model,
            rebalance_policy_df=rebalance_policy_df,
        )

        for fund in model.fund_master.funds:
            fund_id = fund.fund_id
            existing_assets = AssetMaster.existing_from_df(
                assets_df_dict, model_engine=model, econs=model.esg_master, fund_id=fund_id
            ).all
            if fund is not model.fund_master.sh_fund:
                existing_liabs = build_liabs(model, liabs_df, fund_id, currencies=[])
            else:
                existing_liabs = None
            fund.assemble_on_start(existing_assets=existing_assets, existing_liabs=existing_liabs)

    else:  # t > 1
        esg_step = 1 if p.year == model.START_YEAR else 12
        model.esg_master.update_econ_data(p, esg_step=esg_step)
        if esg_step == 1:
            model.deflators[t] = model.deflators_kr.at[('CNY', 'VALN', 'DEF', 0), str_date]
        elif (p.month % esg_step) == 1:
            d0 = model.deflators[t - 1]
            d1 = model.deflators_kr.at[('CNY', 'VALN', 'DEF', 0), str(p.year * 100 + p.month + esg_step - 1)]
            k = (d1 / d0) ** (1 / esg_step)
            for i in range(esg_step):
                model.deflators[t + i] = model.deflators[t + i - 1] * k

        for fund in model.fund_master.funds:  # ensure that sh_fund sit in the last
            fund_id = fund.fund_id
            rebalance_params = model.fund_master.rebalance_params_dict[fund_id]

            # step 1: assets roll forward
            update_mv_price = (p.year == model.START_YEAR) or (p.month % esg_step == 0)
            fund_assets_roll_forward(fund, update_mv_price=update_mv_price)

            # step 2: liabs roll forward
            fund_liabs_roll_forward(
                fund=fund,
                epl=epl,
                as_inv_ret=fund.rate_of_return_fav_bd[t],
                as_cf_ret=0  # specify the rate
            )

            # step 3: rebalance if needed
            fund_reblance_if_needed(
                model_engine=model,
                fund=fund,
                rebalance_params=rebalance_params,
                assets_df_dict=assets_df_dict,
                econs=model.esg_master,
                asset_allocation_df=asset_allocation_df
            )

            # step 4: liabs update after dealing (ad)
            liabs_update_ad(fund=fund)

            # step 5: transfer accumulated free proceeds to shareholder fund
            fund.transfer_free_proceeds_to_other(None)  # shareholder fund is None

    if t == model.MAX_T:
        model.bel_dict = {}
        for fund in model.fund_master.funds:
            fund_id = fund.fund_id
            model.bel_dict[fund_id] = ConstVariable('BEL', model_engine=model, owner=fund_id, group='fund')
            model.bel_dict[fund_id][...] = - np.dot(fund.calculator.tdv_totliab_cash_flow.result, model.deflators)


def stoch_ec_mvl(simulations: str, start_year: int, start_month: int, end_year: int, scenario: str, max_workers: int,
                 workspace_directory: str, input_directories: list[str], results_directory: str | None = None,
                 model_name: str = "stoch_ec_mvl", description: str = "Stochastic EC MVL model"):
    stoch = StochExecutor(
        model_name=model_name,
        description=f"{description}, simulations: {simulations}, scenario: '{scenario}', "
                    f"from {start_year}/{start_month} to {end_year}/12."
    )
    stoch.configure_run(
        simulations=simulations,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=12,
        scenario=scenario,
        max_workers=max_workers,
        workspace_directory=workspace_directory,
        input_directories=input_directories,
        results_directory=results_directory
    )
    stoch.bind_projection(fund_projection)

    df = stoch.read_csv("_file_names.csv", index_col="table")
    filename_dict = {idx: row[stoch.SCENARIO] for idx, row in df.iterrows()}
    file_read_args = stoch.load_json("_file_read_config.json")
    file_df_dict = load_file_df(stoch.read_csv, filename_dict, file_read_args, exclude=["esg_params", "esg"])
    epl = KeyedArray.from_df(file_df_dict['epl'])
    del file_df_dict['epl']

    assets_df_dict = file_df_dict
    asset_allocation_df = file_df_dict["asset_allocation"]
    liabs_df = file_df_dict["liabs"]

    _ = stoch.run(
        projection_args={
            "esg_params": stoch.load_json(filename_dict["esg_params"]),
            "esg_filename": filename_dict["esg"],
            "assets_df_dict": assets_df_dict,
            "liabs_df": liabs_df,
            "epl": epl,
            "funds_df": file_df_dict['funds'],
            "rebalance_policy_df": file_df_dict['rebalance_policy'],
            "asset_allocation_df": asset_allocation_df,
        }
    )



def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        kwargs = json.load(file)
    stoch_ec_mvl(**kwargs)


if __name__ == "__main__":
    main()

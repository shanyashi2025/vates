import json
import sys
from vates import ProjModelEngine, KeyedArray
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
    output_aging_assets,
)


def fund_model(start_year: int, start_month: int, end_year: int, scenario: str, workspace_directory: str,
               input_directories: list[str], results_directory: str | None = None,
               model_name: str = "fund_model", description: str = "Fund level projection"):
    model = ProjModelEngine(
        model_name=model_name,
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
    epl = KeyedArray.from_df(file_df_dict['epl'])
    del file_df_dict['epl']

    # build esg master
    esg_master = EsgMaster.from_df(
        model_engine=model,
        esg_params=model.load_json(filename_dict["esg_params"]),
        esg_df=model.read_csv(filename_dict["esg"])
    )
    # build fund master
    fund_master = FundMaster.from_df(
        df=file_df_dict['funds'],
        model_engine=model,
        rebalance_policy_df=file_df_dict['rebalance_policy'],
    )

    assets_df_dict = file_df_dict
    asset_allocation_df = file_df_dict["asset_allocation"]
    liabs_df = file_df_dict["liabs"]
    aging_assets_output_config_df = file_df_dict.get("aging_assets_output_config")

    @model.bind_projection
    def fund_projection():
        t, p = model.time, model.period

        if t == 0:
            for fund in fund_master.funds:
                fund_id = fund.fund_id
                existing_assets = AssetMaster.existing_from_df(
                    assets_df_dict, model_engine=model, econs=esg_master, fund_id=fund_id
                ).all
                if fund is not fund_master.sh_fund:
                    existing_liabs = build_liabs(model, liabs_df, fund_id, currencies=[])
                else:
                    existing_liabs = None
                fund.assemble_on_start(existing_assets=existing_assets, existing_liabs=existing_liabs)

        else:
            for fund in fund_master.ph_funds + ([fund_master.sh_fund] if fund_master.sh_fund else []):  # ensure that sh_fund sit in the last
                fund_id = fund.fund_id
                rebalance_params = fund_master.rebalance_params_dict[fund_id]
                is_sh_fund = fund is fund_master.sh_fund

                # step 1: assets roll forward
                fund_assets_roll_forward(fund)

                # step 2: liabs roll forward
                if not is_sh_fund:
                    fund_liabs_roll_forward(
                        fund=fund,
                        epl=epl,
                        as_inv_ret=fund.rate_of_return_fav_bd[t],
                        as_cf_ret=0  # specify the rate
                    )
                else:
                    fund_liabs_roll_forward(fund)

                # step 3: rebalance if needed
                fund_reblance_if_needed(
                    model_engine=model,
                    fund=fund,
                    rebalance_params=rebalance_params,
                    assets_df_dict=assets_df_dict,
                    econs=esg_master,
                    asset_allocation_df=asset_allocation_df
                )

                # step 4: liabs update after dealing (ad)
                liabs_update_ad(fund=fund)

                # step 5: transfer accumulated free proceeds to shareholder fund
                if not is_sh_fund:
                    fund.transfer_free_proceeds_to_other(fund_master.sh_fund)

        # --- output aging assets ---
        if aging_assets_output_config_df is not None and (
                p.year * 100 + p.month) in aging_assets_output_config_df.index:
            all_assets = []
            for fund in fund_master.funds:
                all_assets.extend(fund.assets)
            output_aging_assets(
                assets=all_assets,
                df_config=aging_assets_output_config_df,
                date_index=p.year * 100 + p.month,
                out_rootfolder=str(model.workspace_directory_path)
            )

    _ = model.run()


def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        kwargs = json.load(file)
    fund_model(**kwargs)


if __name__ == "__main__":
    main()

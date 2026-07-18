import json
import sys
from vates import ProjModelEngine
from bespoke_package import load_file_df, EsgMaster, AssetMaster


def asset_model(start_year: int, start_month: int, end_year: int, scenario: str, workspace_directory: str,
                input_directories: list[str], results_directory: str | None = None,
                model_name: str = "asset_model_inner", description: str = "Run off existing assets"):
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

    # build esg master
    esg_master = EsgMaster.from_df(
        model_engine=model,
        esg_params=model.load_json(filename_dict["esg_params"]),
        esg_df=model.read_csv(filename_dict["esg"])
    )

    assets_df_dict = {
        "assets_cash": file_df_dict["assets_cash"],
        "assets_bond": file_df_dict.get("assets_bond"),
        "assets_equity": file_df_dict.get("assets_equity"),
        "bond_provided_cash_flow": file_df_dict.get("bond_provided_cash_flow"),
    }

    assets = []

    @model.bind_projection
    def assets_projection():
        esg_master.update_econ_data(model.period)
        if model.time == 0:
            assets[:] = AssetMaster.existing_from_df(assets_df_dict, model_engine=model, econs=esg_master).all
        else:
            for asset in assets:
                asset.roll_forward()
                asset.close_dealing()

    runlog = model.run()
    # print(json.dumps(runlog, indent=4), "\n")


def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        kwargs = json.load(file)
    asset_model(**kwargs)


if __name__ == "__main__":
    main()

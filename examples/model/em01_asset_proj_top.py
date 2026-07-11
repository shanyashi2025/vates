import json
import sys
from vates import ProjModelEngine
from local_package import load_file_df, build_esg_master, EsgMaster, build_all_existing_assets


def assets_projection(model: ProjModelEngine, assets_df_dict: dict, esg_master: EsgMaster):
    esg_master.update_econ_data(model.period)
    if model.time == 0:
        model.assets = build_all_existing_assets(model, assets_df_dict, esg_master, None)['all']
    else:
        for asset in model.assets:
            asset.roll_forward()
            asset.close_dealing()


def asset_model(start_year: int, start_month: int, end_year: int, scenario: str, workspace_directory: str,
                input_directories: list[str], results_directory: str | None = None,
                model_name: str = "asset_model_top", model_description: str = "Run off existing assets"):
    model = ProjModelEngine(name=model_name, description=model_description)
    model.bind_proj_func(assets_projection)
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

    # build esg master
    esg_master = build_esg_master(
        model_engine=model,
        esg_params=model.load_json(filename_dict["esg_params"]),
        esg_df=model.read_csv(filename_dict["esg"])
    )

    _ = model.run(proj_func_args={
        "assets_df_dict": file_df_dict,
        "esg_master": esg_master
    })


def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        kwargs = json.load(file)
    asset_model(**kwargs)


if __name__ == "__main__":
    main()

from vates import ProjModelEngine
from vates.alm import Cash
from company_package import load_file_df, EconMaster, AssetMaster, run_with_json_config

try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False

def assets_projection(model: ProjModelEngine, assets_df_dict: dict, econ_master: EconMaster):
    econ_master.update()
    if model.time == 0:
        model.assets = AssetMaster.existing_from_df(assets_df_dict, model_engine=model, econs=econ_master).all
    else:
        for asset in model.assets:
            if isinstance(asset, Cash):
                if asset.nominal >= 0:
                    ret_id = getattr(asset, "positive_cash_balance_ret_id")
                else:
                    ret_id = getattr(asset, "negative_cash_balance_ret_id")
                ret = (1 + econ_master.econ_asm[ret_id]) ** (1 / 12) - 1
                asset.roll_forward(rate_of_return=ret)
            else:
                asset.roll_forward()
            asset.close_dealing()


def asset_model(start_year: int, start_month: int, end_year: int, scenario: str, input_directories: list[str],
                workspace_directory: str | None = None, results_directory: str | None = None,
                slug: str = "asset_model_top", description: str = "Run off existing assets"):
    model = ProjModelEngine(
        slug=slug,
        description=f"{description}, scenario: '{scenario}', from {start_year}/{start_month} to {end_year}/12."
    )
    model.bind_projection(assets_projection)
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
    econ_master = EconMaster.from_df(
        model_engine=model,
        esg_params=model.load_json(filename_dict["esg_params"]),
        esg_df=model.read_csv(filename_dict["esg"]),
        econ_asm_df=file_df_dict["econ_asm"],
    )

    assets_df_dict = {
        "assets_cash": file_df_dict["assets_cash"],
        "assets_bond": file_df_dict.get("assets_bond"),
        "assets_equity": file_df_dict.get("assets_equity"),
        "bond_provided_cash_flow": file_df_dict.get("bond_provided_cash_flow"),
    }

    if file_df_dict.get("assets_equity_option") is not None:
        if has_scipy:
            assets_df_dict["assets_equity_option"] = file_df_dict.get("assets_equity_option")
        else:
            print(f"'assets_equity_option' is excluded to avoid 'ImportError', if you want to include equity option, "
                  f"please install 'scipy' library (`pip install scipy`).")

    model.run(
        projection_args={
            "assets_df_dict": assets_df_dict,
            "econ_master": econ_master
        }
    )
    return model


def main():
    run_with_json_config(asset_model)


if __name__ == "__main__":
    main()

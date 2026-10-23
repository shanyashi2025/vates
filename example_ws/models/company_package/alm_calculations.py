import pandas as pd
from dataclasses import replace

from vates import ProjModelEngine
from vates.alm import Fund, AssetAllocationParams
from vates.utils import KeyedArray

from .setup_objs_alm import FundRebalanceParams
from .asset_master import AssetMaster
from .econ_master import EsgMaster

def rebalance_this_month(cal_month: int, rebalance_freq: int) -> bool:
    """
    Check if asset rebalance is required for this month.

    Args:
        cal_month (int): Calendar month.
        rebalance_freq (int): rebalance frequency.

    Returns:
        bool: True if rebalance is required, False otherwise.
    """
    if rebalance_freq not in (0, 1, 2, 4, 12):
        raise ValueError(f"Invalid fund rebalance frequency {rebalance_freq}.")
    if rebalance_freq == 0:
        return False
    return cal_month % (12 / rebalance_freq) == 0


def fund_assets_roll_forward(fund: Fund, **kwargs) -> None:
    for asset in fund.assets:
        asset.roll_forward(**kwargs)

    fund.record_before_dealing(scope="assets")


def fund_liabs_roll_forward(fund: Fund, epl: KeyedArray | None = None, **kwargs) -> None:
    if fund.liabs is None or len(fund.liabs) == 0:
        fund.record_before_dealing(scope="liabs")
        return

    t, p = fund.time, fund.period
    date_col = str(p.year * 100 + p.month)

    for liab in fund.liabs:
        liab_id = liab.liab_id

        no_pols_if = epl.at[liab_id, "no_pols_if", date_col]
        math_res_if = epl.at[liab_id, "math_res_if", date_col]
        surr_val_if = epl.at[liab_id, "surr_val_if", date_col]
        prem_inc = epl.at[liab_id, "prem_inc", date_col]
        comm_out = epl.at[liab_id, "comm_out", date_col]
        exp_out = epl.at[liab_id, "exp_out", date_col]
        death_out = epl.at[liab_id, "death_out", date_col]
        crben_out = epl.at[liab_id, "crben_out", date_col]
        ann_out = epl.at[liab_id, "ann_out", date_col]
        surr_out = epl.at[liab_id, "surr_out", date_col]
        div_out = epl.at[liab_id, "div_out", date_col]
        invexp_out = 0 #epl.at[liab_id, "invexp_out", date_col]
        mat_out = epl.at[liab_id, "mat_out", date_col]

        liab_type = getattr(liab, 'liab_type', "")
        if liab_type == 'Par_CD':
            as_inv_ret = kwargs["as_inv_ret"]
            as_cf_ret = kwargs.get("as_cf_ret", None)
            as_cf_ret = as_cf_ret if as_cf_ret is not None else as_inv_ret
            cash_flow = prem_inc - comm_out - exp_out - death_out - crben_out - ann_out - surr_out - mat_out \
                        - div_out - invexp_out
            asset_share_prev = liab.asset_share_if_ad
            asset_share_if_bd = asset_share_prev * (1 + as_inv_ret) + cash_flow * (1 + as_cf_ret)
            liab.roll_forward(
                cash_flow=cash_flow,
                prem_inc=prem_inc,
                no_pols_if=no_pols_if,
                math_res_if=math_res_if,
                surr_val_if=surr_val_if,
                asset_share_prev=asset_share_prev,
                asset_share_if_bd=asset_share_if_bd,
            )
        elif liab_type == 'Par_CD_Flex':
            as_inv_ret = kwargs["as_inv_ret"]
            as_cf_ret = kwargs.get("as_cf_ret", None)
            as_cf_ret = as_cf_ret if as_cf_ret is not None else as_inv_ret
            cash_flow = prem_inc - comm_out - exp_out - death_out - ann_out - surr_out - mat_out - invexp_out
            math_res_prev = liab.math_res_if
            div_out = max(cash_flow * (1 + as_cf_ret) + math_res_prev * as_inv_ret - (math_res_if - math_res_prev),
                          0) * 0.7
            cash_flow -= div_out
            liab.roll_forward(
                cash_flow=cash_flow,
                prem_inc=prem_inc,
                no_pols_if=no_pols_if,
                math_res_if=math_res_if,
                surr_val_if=surr_val_if,
                asset_share_if_bd=0,
            )
        else:
            cash_flow = prem_inc - comm_out - exp_out - death_out - crben_out - ann_out - surr_out - mat_out \
                        - div_out - invexp_out
            liab.roll_forward(
                cash_flow=cash_flow,
                prem_inc=prem_inc,
                no_pols_if=no_pols_if,
                math_res_if=math_res_if,
                surr_val_if=surr_val_if,
                asset_share_if_bd=0,
            )

    fund.record_before_dealing(scope="liabs")


def liabs_close_dealing(fund: Fund) -> None:
    if fund.liabs:
        for liab in fund.liabs:
            t, p = fund.time, fund.period
            if getattr(liab, 'liab_type', "") == 'Par_CD':
                as_rgl_ret = fund.rate_of_return_ad[t, "FAV"] - fund.rate_of_return_bd[t, "FAV"]
                asset_share_if_ad = liab.asset_share_if_bd + liab.asset_share_prev * as_rgl_ret
                liab.close_dealing(asset_share_if_ad=asset_share_if_ad)
            else:
                liab.close_dealing(asset_share_if_ad=0.0)

    fund.record_after_dealing(scope="liabs")


def fund_reblance_if_needed(model_engine: ProjModelEngine, fund: Fund, rebalance_params: FundRebalanceParams,
                            assets_df_dict: dict, econs: dict | EsgMaster, asset_mix_df: pd.DataFrame):
    fund_id = fund.fund_id
    period = fund.period

    fund.record_free_proceeds("bd")

    if rebalance_this_month(period.month, rebalance_params.rebalance_freq):
        profile_assets = AssetMaster.profile_from_df(
            assets_df_dict, model_engine=model_engine, econs=econs, fund_id=fund_id
        ).all
        size_type = rebalance_params.fund_size_type.upper()
        asset_size_basis = rebalance_params.asset_size_basis
        total_size = get_fund_size(fund, size_type=size_type, asset_size_basis=asset_size_basis)
        allocation_group_params: dict[str, AssetAllocationParams] = {}
        weight_box = build_weight_box(asset_mix_df, fund_id, str(period.year * 100 + period.month))
        for key, params in rebalance_params.allocation_params.items():
            allocation_group_params[key] = replace(params, weight_box=weight_box.get(key, None))

        _ = rebalance_params.rebalance_func(
            exist_assets=fund.assets,
            free_proceeds_keeper=fund.free_proceeds_keeper,
            profile_assets=profile_assets,
            total_size=total_size,
            asset_size_basis=asset_size_basis,
            allocation_group_params=allocation_group_params,
        )

    else:
        fund.invest_free_proceeds_to_cash()

    for asset in fund.assets:
        asset.close_dealing()

    fund.record_free_proceeds("ad")
    fund.record_after_dealing(scope="assets")

def get_fund_size(fund: Fund, size_type: str, *, asset_size_basis: str) -> float:
    if size_type == "FUND":
        return fund.get_total(key=f"asset.{asset_size_basis}") + fund.free_proceeds
        # return fund.get_total(func=lambda x, y: x + y, key=(f"asset.{asset_size_basis}", "uninvested_proceeds"))
    elif size_type == "MATH_RES":
        return fund.get_total(key="liab.math_res_if")
    elif size_type == "SURR_VALUE":
        return fund.get_total(key="liab.surr_val_if")
    elif size_type == "ACCT_VALUE":
        return fund.get_total(key="liab.acct_val_if")
    elif size_type == "ASSET_SHARE":
        return fund.get_total(key="liab.asset_share_if_bd")
    elif size_type == "MAX_AS_MATH":
        return fund.get_total(func=lambda x, y: max(x, y), key=("liab.asset_share_if_bd", "liab.math_res_if"))
    elif size_type == "MAX_AS_CSV":
        return fund.get_total(func=lambda x, y: max(x, y), key=("liab.asset_share_if_bd", "liab.math_res_if"))
    else:
        raise ValueError(f"Invalid {size_type=}.")


def build_weight_box(df: pd.DataFrame, fund_id: str, date_col: str) -> dict[str, tuple[float, float, float]]:
    """
    Build a dictionary of target allocations for a fund from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing allocation data.
        fund_id (str): Fund id the target allocations for.
        date_col (str): String date column to lookup.

    Returns:
        dict[str, tuple[float, float, float]]: Mapping of allocation group to
            a three-element tuple (lower, target, upper) of weights.
    """
    weight_box: dict[str, tuple[float, float, float]] = {}

    df_flt: pd.DataFrame = df.copy()
    if fund_id is not None: df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

    for _, row in df_flt.iterrows():
        key = row["allocation_group"]
        target = row[date_col] / 100
        lower = target - abs(row["allow_downward_drift_pc"]) / 100
        upper = target + row["allow_upward_drift_pc"] / 100
        weight_box[key] = lower, target, upper

    return weight_box

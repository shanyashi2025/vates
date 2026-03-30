from vates import KeyedArray
from vates.alm.funds import Fund

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
    elif cal_month % (12 / rebalance_freq) == 0:
        return True
    else:
        return False


def fund_assets_roll_forward(fund: Fund, **kwargs) -> None:
    for asset in fund.assets:
        asset.roll_forward(**kwargs)

    fund.process_assets_before_dealing()


def fund_liabs_roll_forward(fund: Fund, **kwargs) -> None:
    if fund.liabs is None or len(fund.liabs) == 0:
        fund.process_liabs_before_dealing()
        return

    t, p = fund.time, fund.period
    epl_karr: KeyedArray = kwargs['epl_karr']
    as_inv_ret = kwargs['as_inv_ret']
    as_cf_ret = kwargs['as_cf_ret']
    date_col = str(p.year * 100 + p.month)

    for liab in fund.liabs:
        liab_id = liab.liab_id

        no_pols_if = epl_karr.loc[liab_id, "no_pols_if", date_col]
        math_res_if = epl_karr.loc[liab_id, "math_res_if", date_col]
        surr_val_if = epl_karr.loc[liab_id, "surr_val_if", date_col]
        prem_inc = epl_karr.loc[liab_id, "prem_inc", date_col]
        comm_out = epl_karr.loc[liab_id, "comm_out", date_col]
        exp_out = epl_karr.loc[liab_id, "exp_out", date_col]
        death_out = epl_karr.loc[liab_id, "death_out", date_col]
        crben_out = epl_karr.loc[liab_id, "crben_out", date_col]
        ann_out = epl_karr.loc[liab_id, "ann_out", date_col]
        surr_out = epl_karr.loc[liab_id, "surr_out", date_col]
        div_out = epl_karr.loc[liab_id, "div_out", date_col]
        invexp_out = 0 #epl_karr.loc[liab_id, "invexp_out", date_col]
        mat_out = epl_karr.loc[liab_id, "mat_out", date_col]
        acct_value_if = 0.0  # for universal life and unit-linked products
        asset_share_if = 0.0  # for participating products

        liab_type = getattr(liab, 'liab_type', None)
        if liab_type == 'Par_CD':
            cash_flow = prem_inc - comm_out - exp_out - death_out - crben_out - ann_out - surr_out - mat_out \
                        - div_out - invexp_out
            asset_share_if = liab.arr_asset_share_ad[t - 1] * (1 + as_inv_ret) + cash_flow * (1 + as_cf_ret)
        elif liab_type == 'Par_CD_Flex':
            cash_flow = prem_inc - comm_out - exp_out - death_out - ann_out - surr_out - mat_out - invexp_out
            math_res_prev = liab.math_res
            div_out = max(cash_flow * (1 + as_cf_ret) + math_res_prev * as_inv_ret - (math_res_if - math_res_prev),
                          0) * 0.7
            cash_flow -= div_out
        else:
            cash_flow = prem_inc - comm_out - exp_out - death_out - crben_out - ann_out - surr_out - mat_out \
                        - div_out - invexp_out

        liab.roll_forward(
            cash_flow=cash_flow,
            prem_inc=prem_inc,
            no_pols_if=no_pols_if,
            math_res_if=math_res_if,
            surr_val_if=surr_val_if,
            acct_value_if=acct_value_if,
            asset_share_if=asset_share_if
        )

    fund.process_liabs_before_dealing()


def liabs_update_ad(fund: Fund) -> None:
    if fund.liabs:
        for liab in fund.liabs:
            t, p = fund.time, fund.period
            if getattr(liab, 'liab_type', None) == 'Par_CD':
                as_rgl_ret = fund.rate_of_return_fav_ad(t) - fund.rate_of_return_fav_bd(t)
                asset_share_if = liab.arr_asset_share_bd[t] + liab.arr_asset_share_ad[t - 1] * as_rgl_ret
                liab.update_ad(asset_share_if=asset_share_if)
            else:
                liab.update_ad()

    fund.process_liabs_after_dealing()

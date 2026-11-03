import pandas as pd
from typing import List

from vates import ProjModelEngine
from vates.alm import Currency, Liab, ExternalLiab
from vates.utils import KeyedArray

def build_liabs(model_engine: ProjModelEngine, df: pd.DataFrame, fund_id: str | None,
                currencies: List['Currency']) -> List['Liab']:
    """
    Build liability objects from a DataFrame and add them to a fund if provided.

    Args:
        model_engine: Model engine object.
        df (pd.DataFrame): DataFrame containing liability data.
        fund_id (str | None): Fund id to filter, or None.
        currencies (list): List of Currency objects.

    Returns:
        list: List of Liab objects.
    """
    liabs = []

    df_flt: pd.DataFrame = df.copy()
    if fund_id is not None: df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

    for idx, row in df_flt.iterrows():
        currency_id = row["currency_id"]
        currency = next((x for x in currencies if x.currency_id == currency_id), None)
        liab_type = row["liab_type"]

        liab = ExternalLiab(
            model_engine=model_engine,
            liab_id=row["liab_id"],
            liab_type=liab_type,
            fund_id=row["fund_id"],
            currency=currency,
            entry_date=pd.Period(row["entry_date"], freq='M'),
            no_pols_if=row["no_pols_if"],
            math_res_if=row["math_res_if"],
            surr_val_if=row["surr_val_if"],
            asset_share_if_bd=row["asset_share_if"],
            asset_share_if_ad=row["asset_share_if"],
            output_attrs_bd=row["output_vars_bd"].split(';') if row["output_vars_bd"] != 'none' else (),
            output_attrs_ad=row["output_vars_ad"].split(';') if row["output_vars_ad"] != 'none' else (),
        )
        # register hooks
        if liab_type == 'Par_CD':
            liab.register_roll_forward_hook(par_cd_roll_forward_hook)
            liab.register_close_dealing_hook(par_cd_close_dealing_hook)
        elif liab_type == 'Par_CD_Flex':
            liab.register_roll_forward_hook(par_cd_flex_roll_forward_hook)
        else:
            liab.register_roll_forward_hook(liab_roll_forward_hook)

        # append to list
        liabs.append(liab)

    return liabs

def par_cd_roll_forward_hook(obj) -> None:
    obj.asset_share_prev = obj.asset_share_if_ad
    # read epl
    liab_id: str = obj.liab_id
    epl: KeyedArray = obj.epl
    date_col: str = str(obj.period.year * 100 + obj.period.month)
    obj.no_pols_if = epl.at[liab_id, "no_pols_if", date_col]
    obj.math_res_if = epl.at[liab_id, "math_res_if", date_col]
    obj.surr_val_if = epl.at[liab_id, "surr_val_if", date_col]
    obj.prem_inc = epl.at[liab_id, "prem_inc", date_col]
    obj.comm_out = epl.at[liab_id, "comm_out", date_col]
    obj.exp_out = epl.at[liab_id, "exp_out", date_col]
    obj.death_out = epl.at[liab_id, "death_out", date_col]
    obj.crben_out = epl.at[liab_id, "crben_out", date_col]
    obj.ann_out = epl.at[liab_id, "ann_out", date_col]
    obj.surr_out = epl.at[liab_id, "surr_out", date_col]
    obj.div_out = epl.at[liab_id, "div_out", date_col]
    obj.invexp_out = 0  # epl.at[liab_id, "invexp_out", date_col]
    obj.mat_out = epl.at[liab_id, "mat_out", date_col]
    # calculate cash flow and asset share
    as_cf_ret = getattr(obj, "as_cf_ret", None)
    if as_cf_ret is None:
        as_cf_ret = obj.inv_ret_bd
    obj.cash_flow = (obj.prem_inc - obj.comm_out - obj.exp_out - obj.death_out - obj.crben_out - obj.ann_out
                     - obj.surr_out - obj.mat_out - obj.div_out - obj.invexp_out)
    obj.asset_share_if_bd = obj.asset_share_prev * (1 + obj.inv_ret_bd) + obj.cash_flow * (1 + as_cf_ret)

def par_cd_close_dealing_hook(obj) -> None:
    ret_diff = obj.inv_ret_ad - obj.inv_ret_bd
    obj.asset_share_if_ad = obj.asset_share_if_bd + obj.asset_share_prev * ret_diff

def par_cd_flex_roll_forward_hook(obj) -> None:
    obj.math_res_prev = obj.math_res_if
    # read epl
    liab_id: str = obj.liab_id
    epl: KeyedArray = obj.epl
    date_col: str = str(obj.period.year * 100 + obj.period.month)
    obj.no_pols_if = epl.at[liab_id, "no_pols_if", date_col]
    obj.math_res_if = epl.at[liab_id, "math_res_if", date_col]
    obj.surr_val_if = epl.at[liab_id, "surr_val_if", date_col]
    obj.prem_inc = epl.at[liab_id, "prem_inc", date_col]
    obj.comm_out = epl.at[liab_id, "comm_out", date_col]
    obj.exp_out = epl.at[liab_id, "exp_out", date_col]
    obj.death_out = epl.at[liab_id, "death_out", date_col]
    obj.crben_out = epl.at[liab_id, "crben_out", date_col]
    obj.ann_out = epl.at[liab_id, "ann_out", date_col]
    obj.surr_out = epl.at[liab_id, "surr_out", date_col]
    obj.div_out = epl.at[liab_id, "div_out", date_col]
    obj.invexp_out = 0  # epl.at[liab_id, "invexp_out", date_col]
    obj.mat_out = epl.at[liab_id, "mat_out", date_col]
    # calculate cash flow and asset share
    as_cf_ret = getattr(obj, "as_cf_ret", None)
    if as_cf_ret is None:
        as_cf_ret = obj.inv_ret_bd
    cash_flow_bf_div = (obj.prem_inc - obj.comm_out - obj.exp_out - obj.death_out - obj.ann_out - obj.surr_out
                        - obj.mat_out - obj.invexp_out)
    obj.div_out = max(cash_flow_bf_div * (1 + as_cf_ret) + obj.math_res_prev * obj.inv_ret_bd -
                      (obj.math_res_if - obj.math_res_prev), 0) * 0.7
    obj.cash_flow = cash_flow_bf_div - obj.div_out

def liab_roll_forward_hook(obj) -> None:
    # read epl
    liab_id: str = obj.liab_id
    epl: KeyedArray = obj.epl
    date_col: str = str(obj.period.year * 100 + obj.period.month)
    obj.no_pols_if = epl.at[liab_id, "no_pols_if", date_col]
    obj.math_res_if = epl.at[liab_id, "math_res_if", date_col]
    obj.surr_val_if = epl.at[liab_id, "surr_val_if", date_col]
    obj.prem_inc = epl.at[liab_id, "prem_inc", date_col]
    obj.comm_out = epl.at[liab_id, "comm_out", date_col]
    obj.exp_out = epl.at[liab_id, "exp_out", date_col]
    obj.death_out = epl.at[liab_id, "death_out", date_col]
    obj.crben_out = epl.at[liab_id, "crben_out", date_col]
    obj.ann_out = epl.at[liab_id, "ann_out", date_col]
    obj.surr_out = epl.at[liab_id, "surr_out", date_col]
    obj.div_out = epl.at[liab_id, "div_out", date_col]
    obj.invexp_out = 0  # epl.at[liab_id, "invexp_out", date_col]
    obj.mat_out = epl.at[liab_id, "mat_out", date_col]
    # calculate cash flow
    obj.cash_flow = (obj.prem_inc - obj.comm_out - obj.exp_out - obj.death_out - obj.crben_out - obj.ann_out
                     - obj.surr_out - obj.mat_out - obj.div_out - obj.invexp_out)

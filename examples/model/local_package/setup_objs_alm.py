from typing import List, Dict
import pandas as pd
from dataclasses import dataclass

from vates import ProjModelEngine
from vates.alm import AssetRepBasis, AssetBuySellApproach, AssetPurchaseMethod
from vates.alm.econs import Currency
from vates.alm.liabs import Liab, ExtProjLiab
from vates.alm.funds import Fund, FundSizeType, RebalancePolicyParams, TargetWeight


@dataclass
class FundRebalanceParams:
    """Parameters for fund rebalance.

    Attributes:
        size_type (FundSizeType): Fund size type (FUND, MATH_RES, ASSET_SHARE, etc.).
        size_basis (AssetRepBasis): The basis for assets to match the fund size (usually FAV).
        rebalance_freq (int): rebalance frequency (1=A, 2=H, 4=Q, 12=M, 0=SKIP).
    """
    size_type: FundSizeType
    size_basis: AssetRepBasis
    rebalance_freq: int

@dataclass
class FundMaster:
    funds: list[Fund]
    rebalance_params_dict: dict[str, FundRebalanceParams] | None = None
    ph_funds: list[Fund] | None = None
    sh_fund: Fund | None = None

def build_fund_master(model_engine: ProjModelEngine, funds_df: pd.DataFrame, rebalance_policy_df: pd.DataFrame
                      ) -> FundMaster:
    funds = []
    ph_funds = []
    sh_fund = None
    rebalance_params_dict: dict = {}

    for idx, row in funds_df.iterrows():
        fund_id = str(idx)
        fund = Fund(
            model_engine,
            fund_id=fund_id,
            rebalance_policy=build_rebalance_policy(rebalance_policy_df, fund_id),
            asset_categories=row["asset_classes_reported"].split(';')
        )
        if row["fund_type"].lower() not in ('sh', 'shf', 'shareholder'):
            ph_funds.append(fund)
        else:
            if sh_fund is not None:
                raise ValueError("Duplicated shareholder fund.")
            sh_fund = fund
        funds.append(fund)
        rebalance_params_dict[fund_id] = FundRebalanceParams(
            size_type=FundSizeType[row["fund_size_type"].upper()],
            size_basis=AssetRepBasis[row["fund_size_basis"].upper()],
            rebalance_freq=row["fund_rebalance_freq"]  # 1=A, 2=H, 4=Q, 12=M, 0=SKIP
        )

    return FundMaster(funds=funds, ph_funds=ph_funds, sh_fund=sh_fund, rebalance_params_dict=rebalance_params_dict)


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
        liab_class = row["liab_class"].lower()

        if liab_class == 'extprojliab':
            liab = ExtProjLiab(
                model_engine=model_engine,
                liab_id=row["liab_id"],
                fund_id=row["fund_id"],
                currency=currency,
                entry_date=pd.Period(row["entry_date"], freq='M'),
                no_pols_if=row["no_pols_if"],
                math_res_if=row["math_res_if"],
                surr_val_if=row["surr_val_if"],
                asset_share_if=row["asset_share_if"]
            )
        else:
            raise ValueError(f"Invalid liab class {liab_class}.")
        # dynamically create attribute(s)
        setattr(liab, 'liab_type', row["liab_type"])
        # append to list
        liabs.append(liab)

    return liabs


def build_rebalance_policy(df: pd.DataFrame, fund_id: str) -> Dict[str, 'RebalancePolicyParams']:
    """
    Build a dictionary of rebalance policy for a fund from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing rebalance policy parameters.
        fund_id (str): Fund the rebalance policy for.

    Returns:
        dict: Mapping of allocation group to AssetStrategyParams.
    """
    rebalance_policy: dict = {}

    df_flt: pd.DataFrame = df.copy()
    df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

    for idx, row in df_flt.iterrows():
        allocation_group = row["allocation_group"]
        sequence = row["sequence"]
        buysell_approach = AssetBuySellApproach(row["buysell_approach"].upper())
        purchase_method = AssetPurchaseMethod(row["purchase_method"].upper())

        rebalance_policy[allocation_group] = RebalancePolicyParams(
            sequence=sequence,
            buysell_approach=buysell_approach,
            purchase_method=purchase_method,
        )

    return rebalance_policy


def build_target_allocation(df: pd.DataFrame, fund_id: str, date_col: str) -> Dict[str, 'TargetWeight']:
    """
    Build a dictionary of target allocations for a fund from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing allocation data.
        fund_id (str): Fund id the target allocations for.
        date_col (str): String date column to lookup.

    Returns:
        dict: Mapping of allocation group to TargetAllocation.
    """
    target_allocation: dict = {}

    df_flt: pd.DataFrame = df.copy()
    if fund_id is not None: df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

    for _, row in df_flt.iterrows():
        allocation_group = row["allocation_group"]
        tgt_wgt = row[date_col] / 100
        min_wgt = tgt_wgt + row["lower_allow_pc"] / 100
        max_wgt = tgt_wgt + row["upper_allow_pc"] / 100
        target_allocation[allocation_group] = TargetWeight(tgt_weight=tgt_wgt, min_weight=min_wgt, max_weight=max_wgt)

    return target_allocation

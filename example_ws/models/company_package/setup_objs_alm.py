from dataclasses import dataclass
from typing import List, Dict, Self
import pandas as pd

from vates import ProjModelEngine
from vates.alm import Currency, Liab, ExtProjLiab, Fund, AssetAllocationGroup


@dataclass
class FundRebalanceParams:
    """Parameters for fund rebalance.

    Attributes:
        fund_size_type (str): Fund size type (FUND, MATH_RES, ASSET_SHARE, etc.), corresponds to Enum `alm.FundSizeType`.
        asset_size_basis (str): The basis for assets to match the fund size (usually FAV), corresponds to Enum `alm.AssetRepBasis`.
        rebalance_freq (int): rebalance frequency (1=A, 2=H, 4=Q, 12=M, 0=No).
    """
    fund_size_type: str
    asset_size_basis: str
    rebalance_freq: int


class FundMaster:

    def __init__(
        self,
        *,
        funds: list[Fund] | None = None,
        ph_funds: list[Fund] | None = None,
        sh_fund: Fund | None = None,
        rebalance_params_dict: dict[str, FundRebalanceParams] | None = None,

    ):
        self.funds: list[Fund] | None = funds or []
        self.rebalance_params_dict: dict[str, FundRebalanceParams] | None = rebalance_params_dict
        self.ph_funds: list[Fund] | None = ph_funds
        self.sh_fund: Fund | None = sh_fund

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        asset_allocation_groups_df: pd.DataFrame
    ) -> Self:
        funds = []
        ph_funds = []
        sh_fund = None
        rebalance_params_dict: dict = {}

        for idx, row in df.iterrows():
            fund_id = str(idx)
            fund = Fund(
                fund_id=fund_id,
                model_engine=model_engine,
                asset_allocation_groups=cls.build_asset_allocation_groups_from_df(asset_allocation_groups_df, fund_id=fund_id),
                asset_categories=row["asset_categories"].split(';'),
                asset_report_bases=row["asset_report_bases"].split(';'),
            )
            if "fund_type" in row:
                if row["fund_type"].lower() not in ('sh', 'shf', 'shareholder'):
                    ph_funds.append(fund)
                else:
                    if sh_fund is not None:
                        raise ValueError("Duplicated shareholder fund.")
                    sh_fund = fund
            funds.append(fund)
            rebalance_params_dict[fund_id] = FundRebalanceParams(
                fund_size_type=row["fund_size_type"].upper(),
                asset_size_basis=row["asset_size_basis"].upper(),
                rebalance_freq=row["fund_rebalance_freq"]  # 1=A, 2=H, 4=Q, 12=M, 0=SKIP
            )

        return FundMaster(funds=funds, ph_funds=ph_funds, sh_fund=sh_fund, rebalance_params_dict=rebalance_params_dict)

    @classmethod
    def build_asset_allocation_groups_from_df(cls, df: pd.DataFrame, *, fund_id: str) -> list[AssetAllocationGroup]:
        """
        Build a list of allocation groups for a fund from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing rebalance policy parameters.
            fund_id (str): Fund the rebalance policy for.

        Returns:
            list[AssetAllocationGroup]: List of allocation groups.
        """
        allocation_groups: list[AssetAllocationGroup] = []

        df_flt: pd.DataFrame = df.copy()
        df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

        for idx, row in df_flt.iterrows():
            alloc_group = AssetAllocationGroup(
                name=row["allocation_group"],
                sequence=row["sequence"],
                buysell_approach=row["buysell_approach"],
                purchase_method=row["purchase_method"],
            )
            allocation_groups.append(alloc_group)

        return allocation_groups

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

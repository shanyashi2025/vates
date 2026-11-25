import json
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Self

from vates import ProjModelEngine
from vates.alm import FundAccount, AssetAllocationParams, Rebalancer

from .asset_master import AssetMaster
from .econ_master import EconMaster


@dataclass
class FundRebalanceParams:
    """Parameters for fund rebalance.

    Attributes:
        fund_size_type (str): Fund size type (FUND, MATH_RES, ASSET_SHARE, etc.), corresponds to Enum `alm.FundSizeType`.
        asset_size_basis (str): The basis for assets to match the fund size (usually FAV), corresponds to Enum `alm.AssetRepBasis`.
        rebalance_freq (int): Rebalance frequency (1=A, 2=H, 4=Q, 12=M, 0=No).
        allocation_params (dict[str, AssetAllocationParams]): AssetAllocationParams of each allocation group; contains
            only static parameters, target weights is not included and will be assigned each month.
    """
    fund_size_type: str
    asset_size_basis: str
    rebalance_freq: int
    allocation_params: dict[str, AssetAllocationParams]
    rebalance_func: Callable


class FundMaster:

    def __init__(
        self,
        *,
        funds: list[FundAccount] | None = None,
        ph_funds: list[FundAccount] | None = None,
        sh_fund: FundAccount | None = None,
        rebalance_params_dict: dict[str, FundRebalanceParams] | None = None,

    ):
        self.funds: list[FundAccount] = funds or []
        self.rebalance_params_dict: dict[str, FundRebalanceParams] | None = rebalance_params_dict
        self.ph_funds: list[FundAccount] = ph_funds or []
        self.sh_fund: FundAccount | None = sh_fund

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
            fund = FundAccount(
                fund_id=fund_id,
                model_engine=model_engine,
                asset_categories=str(row["asset_categories"]).split(';'),
                asset_report_bases=str(row["asset_report_bases"]).split(';'),
                output_liab_attrs_bd=str(row["output_liab_vars_bd"]).split(';') if row["output_liab_vars_bd"] != 'none' else (),
                output_liab_attrs_ad=str(row["output_liab_vars_ad"]).split(';') if row["output_liab_vars_ad"] != 'none' else (),
                other_cash_flow_kinds=("proceeds_transfer",)
            )
            if row["fund_type"].lower() not in ('sh', 'shf', 'shareholder'):
                ph_funds.append(fund)
            else:
                if sh_fund is not None:
                    raise ValueError("Duplicated shareholder fund.")
                sh_fund = fund
            funds.append(fund)

            allocation_params = cls.build_asset_allocation_params_from_df(asset_allocation_groups_df, fund_id=fund_id)
            allocation_group_names = list(allocation_params.keys())
            allocation_group_names.sort(key=lambda x: allocation_params[x].priority)  # sorted by priority
            rebalance_params_dict[fund_id] = FundRebalanceParams(
                fund_size_type=str(row["fund_size_type"]).upper(),
                asset_size_basis=str(row["asset_size_basis"]).upper(),
                rebalance_freq=row["fund_rebalance_freq"],  # 1=A, 2=H, 4=Q, 12=M, 0=SKIP
                allocation_params=allocation_params,
                rebalance_func=Rebalancer(name=fund_id, model_engine=model_engine, allocation_group_names=allocation_group_names)
            )

        return cls(funds=funds, ph_funds=ph_funds, sh_fund=sh_fund, rebalance_params_dict=rebalance_params_dict)

    @classmethod
    def build_asset_allocation_params_from_df(cls, df: pd.DataFrame, *, fund_id: str) -> dict[str, AssetAllocationParams]:
        """
        Build a dict of allocation params of each group for a fund from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing rebalance policy parameters.
            fund_id (str): Fund the rebalance policy for.

        Returns:
            dict[str, AssetAllocationParams]: Dict of allocation params of each group.
        """
        params: dict[str, AssetAllocationParams] = {}

        df_flt: pd.DataFrame = df.copy()
        df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

        for idx, row in df_flt.iterrows():
            name = row["allocation_group"]
            if name in params:
                raise KeyError(f"Duplicate 'allocation_group': '{name}'; fund '{fund_id}'.")
            params[name] = AssetAllocationParams.create(
                priority=row["priority"],
                dealing_strategy=row["dealing_strategy"],
                purchase_method=row["purchase_method"],
                rebalance_preference=json.loads(row["rebalance_preference"]),
            )

        return params


def _rebalance_this_month(cal_month: int, rebalance_freq: int) -> bool:
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


def fund_rebalance(model_engine: ProjModelEngine, fund: FundAccount, rebalance_params: FundRebalanceParams,
                   assets_df_dict: dict, econs: EconMaster, asset_mix_df: pd.DataFrame):
    fund_id = fund.fund_id
    period = fund.period

    fund.record_unallocated_cash("bd")

    if _rebalance_this_month(period.month, rebalance_params.rebalance_freq):
        profile_assets = AssetMaster.profile_from_df(
            assets_df_dict, model_engine=model_engine, econs=econs, fund_id=fund_id
        ).all
        size_type = rebalance_params.fund_size_type.upper()
        asset_size_basis = rebalance_params.asset_size_basis
        total_size = get_fund_size(fund, size_type=size_type, asset_size_basis=asset_size_basis)
        allocation_group_params = rebalance_params.allocation_params
        weight_triplet = build_weight_triplet(asset_mix_df, fund_id, str(period.year * 100 + period.month))
        for key, params in allocation_group_params.items():
            params.weight_triplet = weight_triplet.get(key, None)

        _ = rebalance_params.rebalance_func(
            exist_assets=fund.assets,
            unallocated_cash=fund.unallocated_cash,
            profile_assets=profile_assets,
            total_size=total_size,
            asset_size_basis=asset_size_basis,
            allocation_group_params=allocation_group_params,
        )

    else:
        fund.invest_unallocated_cash()

    for asset in fund.assets:
        asset.close_dealing()

    fund.record_unallocated_cash("ad")
    fund.record_after_dealing(scope="assets")


def get_fund_size(fund: FundAccount, size_type: str, *, asset_size_basis: str) -> float:
    if size_type == "FUND":
        return fund.get_asset_attr(asset_size_basis) + fund.unallocated_cash_balance
    elif size_type == "MATH_RES":
        return fund.get_liability_attr("math_res_if")
    elif size_type == "SURR_VALUE":
        return fund.get_liability_attr("surr_val_if")
    elif size_type == "ACCT_VALUE":
        return fund.get_liability_attr("acct_val_if")
    elif size_type == "ASSET_SHARE":
        return fund.get_liability_attr("asset_share_if_bd")
    elif size_type == "MAX_AS_MATH":
        return max(fund.get_liability_attr("asset_share_if_bd"), fund.get_liability_attr("math_res_if"))
    elif size_type == "MAX_AS_CSV":
        return max(fund.get_liability_attr("asset_share_if_bd"), fund.get_liability_attr("surr_val_if"))
    else:
        raise ValueError(f"Invalid {size_type=}.")


def build_weight_triplet(df: pd.DataFrame, fund_id: str, date_col: str) -> dict[str, tuple[float, float, float]]:
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
    weight_triplet: dict[str, tuple[float, float, float]] = {}

    df_flt: pd.DataFrame = df.copy()
    if fund_id is not None: df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

    for _, row in df_flt.iterrows():
        key = row["allocation_group"]
        target = row[date_col] / 100
        lower = target - abs(row["allow_downward_drift_pc"]) / 100
        upper = target + row["allow_upward_drift_pc"] / 100
        weight_triplet[key] = lower, target, upper

    return weight_triplet

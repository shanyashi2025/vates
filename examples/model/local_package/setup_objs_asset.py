from typing import List
import pandas as pd

from vates.alm import AssetClassification
from vates.alm.assets import (
    Asset,
    Cash,
    BondFixed,
    Equity,
    create_asset,
    BondFixedBuilder,
)
from vates.alm.econs import YieldCurve, CreditBand, EquityIndex, Currency, MarketInfo


ASSET_CLASSES_REPORTED_MAPPING = {
    'cash': 'CASH',
    'equity': 'EQUITY',
    'fixed_bond': 'BOND',
}


def build_all_existing_assets(model, df_dict: dict[str, pd.DataFrame], market_dict: dict[str, ...],
                              fund_id: str | None) -> dict[str, list[...]]:
    """
    Build existing assets objects from a dictionary of DataFrame and add them to a fund if provided.

    Args:
        model: Model object.
        df_dict (dict[str, pd.DataFrame]): Dictionary of DataFrame containing asset data.
        market_dict (dict[str, ...]): Dictionary of market variables.
        fund_id (str | None): Fund id to filter the df, or None.

    Returns:
        dict[str, list[Asset]]: Dictionary of list of asset objects.
    """
    currencies = market_dict['currencies']
    equity_indices = market_dict['equity_indices']
    yield_curves = market_dict['yield_curves']
    credit_bands = market_dict['credit_bands']
    market_info = market_dict['market_info']

    equity_ls = []
    bond_ls = []

    cash_ls = build_assets_cash(
        model=model, df=df_dict['assets_cash'].copy(), fund_id=fund_id,
        currencies=currencies, market_info=market_info
    )
    if df_dict.get(name := 'assets_equity', None) is not None:
        equity_ls = build_assets_equity(
            model=model, df=df_dict[name].copy(), fund_id=fund_id,
            currencies=currencies, equity_indices=equity_indices
        )
    if df_dict.get(name := 'assets_bond', None) is not None:
        bond_ls = build_assets_fixed_bond(
            model=model, df=df_dict[name].copy(), fund_id=fund_id,
            redemp_schedule_df= df_dict.get('redemp_schedule', None),
            currencies=currencies, yield_curves=yield_curves, credit_bands=credit_bands
        )

    assets_all = cash_ls + equity_ls + bond_ls

    return {
        'all': assets_all,
        'cash': cash_ls,
        'equity': equity_ls,
        'bond': bond_ls,
    }


def build_all_profile_assets(
        model, df_dict: dict[str, pd.DataFrame], market_dict: dict[str, ...], fund_id: str
        ) -> dict[str, list[Asset]]:
    """
    Build profile assets objects from a dictionary of DataFrame.

    Args:
        model: Model object.
        df_dict (dict[str, pd.DataFrame]): Dictionary of DataFrame containing asset data.
        market_dict (dict[str, ...]): Dictionary of market variables.
        fund_id (str): Fund id to filter the df.

    Returns:
        dict[str, list[Asset]]: Dictionary of list of asset objects.
    """
    currencies = market_dict['currencies']
    equity_indices = market_dict['equity_indices']
    yield_curves = market_dict['yield_curves']
    credit_bands = market_dict['credit_bands']

    equity_ls = []
    bond_ls = []

    if df_dict.get(name := 'profile_equity', None) is not None:
        equity_ls = build_profile_equity(
            model=model, df=df_dict[name].copy(), fund_id=fund_id,
            currencies=currencies, equity_indices=equity_indices
        )
    if df_dict.get(name := 'profile_bond', None) is not None:
        bond_ls = build_profile_fixed_bond(
            model=model, df=df_dict[name].copy(), fund_id=fund_id,
            currencies=currencies, yield_curves=yield_curves, credit_bands=credit_bands
        )

    profile_all = equity_ls + bond_ls

    return {
        'all': profile_all,
        'equity': equity_ls,
        'bond': bond_ls,
    }


def build_assets_cash(model, df: pd.DataFrame, fund_id: str | None,
                      currencies: List['Currency'], market_info: MarketInfo) -> List['Cash']:
    """
    Build cash asset objects from a DataFrame and add them to a fund if provided.

    Args:
        model: Model object.
        df (pd.DataFrame): DataFrame containing cash asset data.
        fund_id (str | None): Fund id to filter the df, or None.
        currencies (list): List of Currency objects.
        market_info (MarketInfo): MarketInfo object.

    Returns:
        list: List of Cash asset objects.
    """
    cash_list = []

    df_flt: pd.DataFrame = df.copy()
    if fund_id is not None: df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

    for _, row in df_flt.iterrows():
        currency_id = row["currency_id"]
        currency = next((x for x in currencies if x.currency_id == currency_id), None)
        # create instance
        cash = Cash(
            model=model,
            asset_id=row["asset_id"],
            asset_class=ASSET_CLASSES_REPORTED_MAPPING["cash"],
            fund_id=row["fund_id"],
            allocation_group=row["allocation_group"],
            currency=currency,
            nominal=row["nominal"],
            market_info=market_info,
            ret_id=row["positive_cash_balance_ret_id"],
            ret_id_short_pos=row["negative_cash_balance_ret_id"]
        )
        # dynamically create attribute(s)
        for col in df_flt.columns:
            if col.startswith('tag__'):
                setattr(cash, col[5:], row[col])
        # append to list
        cash_list.append(cash)

    return cash_list


def build_assets_fixed_bond(model, df: pd.DataFrame, fund_id: str | None,
                            redemp_schedule_df: pd.DataFrame, currencies: List['Currency'],
                            yield_curves: List['YieldCurve'], credit_bands: List['CreditBand'],
                            ) -> List[BondFixed]:
    """
    Build bond asset objects from a DataFrame and add them to a fund if provided.

    Args:
        model: Model object.
        df (pd.DataFrame): DataFrame containing bond asset dataF.
        fund_id (str | None): Fund id to filter the df, or None.
        redemp_schedule_df (pd.DataFrame): DataFrame of redemption schedules.
        currencies (list): List of Currency objects.
        yield_curves (list): List of YieldCurve objects.
        credit_bands (list): List of CreditBand objects.

    Returns:
        list: List of Bond asset objects.
    """
    fixed_bond_list = []

    df_flt: pd.DataFrame = df.copy()
    if fund_id is not None: df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

    for _, row in df_flt.iterrows():
        currency_id = row["currency_id"]
        currency = next((x for x in currencies if x.currency_id == currency_id), None)
        rf_curve_id = row["rf_curve_id"]
        rf_curve = next((x for x in yield_curves if x.curve_id == rf_curve_id), None)
        credit_band_id = row["credit_band_id"]
        credit_band = next((x for x in credit_bands if x.band_id == credit_band_id), None)
        redemp_sched_id = row["redemp_sched_id"]
        redemp_sched = redemp_schedule_df[redemp_sched_id].values if redemp_sched_id.lower() != 'none' else None
        pre_calc = row["pre_calculation"]

        # create instance
        fixed_bond = create_asset(
            model=model,
            asset_builder_cls=BondFixedBuilder,
            pre_calculations=pre_calc.split(';') if pre_calc.lower() != 'none' else None,
            asset_id=row["asset_id"],
            asset_class=ASSET_CLASSES_REPORTED_MAPPING['fixed_bond'],
            fund_id=row["fund_id"],
            allocation_group=row["allocation_group"],
            currency=currency,
            issue_date=pd.Period(row["issue_date"], freq='M'),
            maturity_date=pd.Period(row["maturity_date"], freq='M'),
            coupon_rate=row["coupon_rate"],
            coupon_freq=row["coupon_freq"],
            face_value=row["face_value"],
            redemp_sched=redemp_sched,
            units=row["units"],
            classification=AssetClassification(row["asset_classification"]),
            rf_curve=rf_curve,
            credit_band=credit_band,
            abv_price=row["abv_price_dirty"],
            mv_price=row["mv_price_dirty"],
            market_spread=row["market_spread"],
            is_profile=False
        )
        # dynamically create attribute(s)
        if redemp_sched_id.lower() != 'none':
            setattr(fixed_bond, 'redemp_sched_id', redemp_sched_id)
        for col in df_flt.columns:
            if col.startswith('tag__'):
                setattr(fixed_bond, col[5:], row[col])
        # append to list
        fixed_bond_list.append(fixed_bond)

    return fixed_bond_list


def build_profile_fixed_bond(model, df: pd.DataFrame, fund_id: str, currencies: List['Currency'],
                             yield_curves: List['YieldCurve'], credit_bands: List['CreditBand']) -> List[BondFixed]:
    """
    Build a profile of bond assets for a fund from a DataFrame.

    Args:
        model: Model object.
        df (pd.DataFrame): DataFrame containing bond profile data.
        fund_id (str): Fund id to filter the df.
        currencies (list): List of Currency objects.
        yield_curves (list): List of YieldCurve objects.
        credit_bands (list): List of CreditBand objects.

    Returns:
        list: List of Bond asset objects for the profile.
    """
    fixed_bond_list = []

    df_flt: pd.DataFrame = df.copy()
    df_flt = df_flt.loc[(df["fund_id"] == fund_id)]
    p: pd.Period = model.period
    str_cal_ym = str(p.year * 100 + p.month)

    # initialize assets profile - bond
    for _, row in df_flt.iterrows():
        # read profile information
        currency_id = row["currency_id"]
        currency = next((x for x in currencies if x.currency_id == currency_id), None)
        rf_curve_id = row["rf_curve_id"]
        rf_curve = next((x for x in yield_curves if x.curve_id == rf_curve_id), None)
        credit_band_id = row["credit_band_id"]
        credit_band = next((x for x in credit_bands if x.band_id == credit_band_id), None)

        fixed_bond = create_asset(
            model=model,
            asset_builder_cls=BondFixedBuilder,
            pre_calculations=['coupon_rate'],
            asset_id=f"{str_cal_ym}{row["_asset_id"]}",
            asset_class=ASSET_CLASSES_REPORTED_MAPPING['fixed_bond'],
            fund_id=fund_id,
            allocation_group=row["allocation_group"],
            currency=currency,
            issue_date=p,
            maturity_date=p + row["maturity_term_y"] * 12,
            coupon_freq=row["coupon_freq"],
            face_value=row["face_value"],
            redemp_sched=None,
            units=row["units"],
            classification=AssetClassification(row["asset_classification"]),
            rf_curve=rf_curve,
            credit_band=credit_band,
            abv_price=row["face_value"],
            mv_price=row["face_value"],
            market_spread=row["market_spread"],
            is_profile=True
        )
        # dynamically create attribute(s)
        for col in df_flt.columns:
            if col.startswith('tag__'):
                setattr(fixed_bond, col[5:], row[col])
        # append to list
        fixed_bond_list.append(fixed_bond)

    return fixed_bond_list


def build_assets_equity(model, df: pd.DataFrame, fund_id: str | None,
                        currencies: List['Currency'], equity_indices: List['EquityIndex']) -> List['Equity']:
    """
    Build equity asset objects from a DataFrame and add them to a fund if provided.

    Args:
        model: Model object.
        df (pd.DataFrame): DataFrame containing equity asset data.
        fund_id (str | None): Fund id to filter the df, or None.
        currencies (list): List of Currency objects.
        equity_indices (list): List of EquityIndex objects.

    Returns:
        list: List of Equity asset objects.
    """
    equity_list = []

    df_flt: pd.DataFrame = df.copy()
    if fund_id is not None: df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

    for _, row in df_flt.iterrows():
        currency_id = row["currency_id"]
        currency = next((x for x in currencies if x.currency_id == currency_id), None)
        equity_index_id = row["equity_index_id"]
        equity_index = next((x for x in equity_indices if x.index_id == equity_index_id), None)
        # create instance
        equity = Equity(
            model=model,
            asset_id=row["asset_id"],
            asset_class=ASSET_CLASSES_REPORTED_MAPPING["equity"],
            fund_id=row["fund_id"],
            allocation_group=row["allocation_group"],
            classification=AssetClassification(row["asset_classification"]),
            currency=currency,
            mv=row["mv"],
            fav=row["fav"],
            equity_index=equity_index,
            is_profile=False
        )
        # dynamically create attribute(s)
        for col in df_flt.columns:
            if col.startswith('tag__'):
                setattr(equity, col[5:], row[col])
        # append to list
        equity_list.append(equity)

    return equity_list


def build_profile_equity(model, df: pd.DataFrame, fund_id: str,
                         currencies: List['Currency'], equity_indices: List['EquityIndex']) -> List['Equity']:
    """
    Build a profile of equity assets for a fund from a DataFrame.

    Args:
        model: Model object.
        df (pd.DataFrame): DataFrame containing equity asset data.
        fund_id (str): Fund id to filter the df.
        currencies (list): List of Currency objects.
        equity_indices (list): List of EquityIndex objects.

    Returns:
        list: List of Equity asset objects for the profile.
    """
    equity_list = []

    df_flt: pd.DataFrame = df.copy()
    df_flt = df_flt.loc[(df["fund_id"] == fund_id)]
    p: pd.Period = model.period
    str_cal_ym = str(p.year * 100 + p.month)

    for _, row in df_flt.iterrows():
        currency_id = row["currency_id"]
        currency = next((x for x in currencies if x.currency_id == currency_id), None)
        equity_index_id = row["equity_index_id"]
        equity_index = next((x for x in equity_indices if x.index_id == equity_index_id), None)
        # create instance
        equity = Equity(
            model=model,
            asset_id=f"{str_cal_ym}{row["_asset_id"]}",
            asset_class=ASSET_CLASSES_REPORTED_MAPPING['equity'],
            fund_id=fund_id,
            allocation_group=row["allocation_group"],
            classification=AssetClassification(row["asset_classification"]),
            currency=currency,
            mv=row["amount"],
            fav=row["amount"],
            equity_index=equity_index,
            is_profile=True
        )
        # dynamically create attribute(s)
        for col in df_flt.columns:
            if col.startswith('tag__'):
                setattr(equity, col[5:], row[col])
        # append to list
        equity_list.append(equity)

    return equity_list

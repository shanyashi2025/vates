import math
import warnings
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

from vates import KeyedArray, kr_from_df
from vates.utils import curve_interp, parse_str_to_int_list
from vates.alm.econs import YieldCurve, CreditBand, EquityIndex, Currency, MarketInfo


@dataclass
class EsgInfo:
    """
    Dataclass representing an ESG info.

    Attributes:
        economy (str)
        class_ (str)
        measure (str)
        term (int | list[int] | None)
        term_type (str | None): Y=year, M=month
        interp_method (str | None): The interpolation method for the curve data.
        compound_freq (str | None): The compounding frequency for the curve data.
    """
    economy: str
    class_: str
    measure: str | None = None
    term: int | list[int] = 0
    term_type: str | None = None
    interp_method: str | None = None
    compound_freq: str | None = None
    ipos_economy: int = 0
    ipos_class: int = 0
    ipos_measure: int = 0
    ipos_term: int | list[int] = 0


def update_esg_this_month(p: pd.Period, esg_step: int) -> tuple[bool, str | None]:
    """
    Args:
        p (pd.Period): Time period.
        esg_step (int): ESG file step in months.

    Returns:
        tuple[bool, str]: True to update at this month, date column string to lookup
    """
    if esg_step == 1:
        return True, str(p.year * 100 + p.month)
    elif esg_step in (3, 6, 12):
        if p.month % esg_step == 1:  # read esg at the first month of the period with period end date
            return True, str(p.year * 100 + p.month + esg_step - 1)
        else:
            return False, None
    else:
        raise ValueError(f'Invalid esg_step: {esg_step}')


def build_esg_kr(esg_df: pd.DataFrame, esg_helper_dict: Dict[str, 'EsgInfo']) -> KeyedArray | None:
    df = esg_df[esg_df["CLASS"].isin([v.class_ for k, v in esg_helper_dict.items()])].copy()
    df.set_index(['ECONOMY', 'CLASS', 'MEASURE', 'TERM'], inplace=True)
    kr = kr_from_df(df=df, unpack_multi_index=True, col_index_name='date_col')
    _set_esg_helper_intpos_index(esg_helper_dict, kr)
    return kr


def _set_esg_helper_intpos_index(esg_helper_dict: Dict[str, 'EsgInfo'], esg_kr: KeyedArray) -> None:
    for key, info in esg_helper_dict.items():
        if info.measure is None:
            continue

        economy, class_, measure, term = info.economy, info.class_, info.measure, info.term

        info.ipos_economy = esg_kr.key_to_pos(dim='ECONOMY', key=economy)
        info.ipos_class = esg_kr.key_to_pos(dim='CLASS', key=class_)
        info.ipos_measure = esg_kr.key_to_pos(dim='MEASURE', key=measure)

        if isinstance(term, int):
            info.ipos_term = esg_kr.key_to_pos(dim='TERM', key=term)
        elif isinstance(term, list):
            info.ipos_term = [esg_kr.key_to_pos(dim='TERM', key=x) for x in term]
        else:
            raise TypeError(f"{key} term: type {type(info.term)} is not allowed, expected 'int' or 'list'.")


def build_yield_curves(model, yield_curves_df: pd.DataFrame) -> tuple[List['YieldCurve'], Dict[str, 'EsgInfo']]:
    """
    Build yield curve objects and their ESG helpers from a DataFrame.

    Args:
        model: Model object.
        yield_curves_df (pd.DataFrame): DataFrame containing yield curve configuration.

    Returns:
        tuple: (List of YieldCurve objects, ESG helper)
    """
    yield_curves = []
    esg_helper = {}

    for idx, row in yield_curves_df.iterrows():
        # create instance
        obj = YieldCurve(model=model, curve_id=str(idx))
        yield_curves.append(obj)

        # initialize esg helper
        economy = row["esg_economy"]
        class_ = row["esg_class"]
        measure = row["esg_measure"]
        term = _parse_esg_term_list(row["esg_term"])
        term_type = row["term_type"]
        interp_method = row["interpolation_method"]

        esg_helper[idx] = EsgInfo(economy, class_, measure, term=term, term_type=term_type, interp_method=interp_method)

    return yield_curves, esg_helper


def _parse_esg_term_list(esg_term_str: str) -> list[int]:
    term_lst = parse_str_to_int_list(esg_term_str, separator=';', joinner='-', sort_list='asc')
    if (entry:= term_lst[0]) <= 0:
        raise ValueError(f"Esg term entry {entry} <= 0, expected positive integer.")
    if len(term_lst) > 1200:
        warnings.warn(f"Number of esg term nodes exceeds 1200.")
    return term_lst


def update_yield_curves(yield_curve_list: List['YieldCurve'] | YieldCurve, esg_helper: Dict[str, 'EsgInfo'],
                        esg_data: KeyedArray, date_col: str, is_update: bool=True) -> None:
    """
    Update yield curve(s) with ESG data.

    Args:
        yield_curve_list (YieldCurve or list of YieldCurve): Curve(s) to update.
        esg_helper (dict): ESG helper info for each curve.
        esg_data (KeyedArray): ESG data.
        date_col (str): Date column to lookup.
        is_update (bool): True if update is required.
    """
    if type(yield_curve_list) != list: yield_curve_list = [yield_curve_list]
    if len(yield_curve_list) == 0: return
    for yield_curve in yield_curve_list:
        if is_update:
            ipos_date = esg_data.key_to_pos("date_col", date_col)
            update_single_yield_curve(yield_curve, esg_helper, esg_data, ipos_date)
        else:
            yield_curve.skip_update()


def update_single_yield_curve(yield_curve: YieldCurve, esg_helper: Dict[str, 'EsgInfo'],
                              esg_data: KeyedArray, ipos_date: int) -> None:
    """
    Update risk-free yield curve with ESG data.

    Args:
        yield_curve (YieldCurve): Curve to update.
        esg_helper (dict): ESG helper info for each curve.
        esg_data (KeyedArray): ESG data.
        ipos_date (int): Integer-position index of date column.
    """
    curve_id = yield_curve.curve_id
    # retrieve esg info from esg_helper
    helper = esg_helper[curve_id]
    economy, class_, measure = helper.economy, helper.class_, helper.measure
    term_list, term_type, interp_method = helper.term, helper.term_type, helper.interp_method
    ipos_economy, ipos_class, ipos_measure, ipos_term = helper.ipos_economy, helper.ipos_class, helper.ipos_measure, helper.ipos_term

    data = np.array(esg_data[ipos_economy, ipos_class, ipos_measure, ipos_term, ipos_date])

    # convert term to months
    if term_type == "Y":
        term = np.insert(arr=np.array(term_list) * 12, obj=0, values=0)
    elif term_type == "M":
        term = np.insert(arr=np.array(term_list), obj=0, values=0)
    else:
        raise ValueError(f"Invalid {term_type=}, expected 'Y' or 'M'.")

    # determine rate type and process curve data
    if measure == "PRICE":
        data = np.insert(arr=data, obj=0, values=1)  # let discount_factor = 1 at term 0
        yield_curve.disc_factors = curve_interp(term, data, interp_method)
    elif measure == "SPOT":
        data = np.insert(arr=data, obj=0, values=0)  # let spot_rate = 0 at term 0
        yield_curve.spot_rates = curve_interp(term, data, interp_method)
    else:
        raise ValueError(f"{curve_id}: invalid yield curve input type: MEASURE = {measure}")


def build_credit_bands(model, credit_bands_df: pd.DataFrame) -> tuple[List['CreditBand'], Dict[str, 'EsgInfo']]:
    """
    Build credit band objects and their ESG helpers from a DataFrame.

    Args:
        model: Model object.
        credit_bands_df (pd.DataFrame): DataFrame containing credit band configuration.

    Returns:
        tuple: (List of CreditBand objects, ESG helper)
    """
    credit_bands = []
    esg_helper = {}

    for idx, row in credit_bands_df.iterrows():
        # create instance
        obj = CreditBand(model=model, band_id=str(idx))
        credit_bands.append(obj)

        # initialize esg helper
        economy = row["esg_economy"]
        class_ = row["esg_class"]
        measure_default = row["esg_measure_default"]
        default_comp_freq = row["default_compound_frequency"]
        measure_recovery = row["esg_measure_recovery"]
        measure_spotmult = row["esg_measure_spotmult"]
        measure_spread = row["esg_measure_spread"]

        esg_helper[f"{idx}|default"] = EsgInfo(economy, class_, measure_default, compound_freq=default_comp_freq)
        esg_helper[f"{idx}|recovery"] = EsgInfo(economy, class_, measure_recovery)
        esg_helper[f"{idx}|spotmult"] = EsgInfo(economy, class_, measure_spotmult)
        esg_helper[f"{idx}|spread"] = EsgInfo(economy, class_, measure_spread)

    return credit_bands, esg_helper


def update_credit_bands(credit_band_list: List['CreditBand'] | CreditBand, esg_helper: Dict[str, 'EsgInfo'],
                        esg_data: KeyedArray, date_col: str, is_update: bool=True) -> None:
    """
    Update credit band(s) with ESG data.

    Args:
        credit_band_list (CreditBand or list of CreditBand): credit band(s) to update.
        esg_helper (dict): ESG helper info for each credit band.
        esg_data (KeyedArray): ESG data.
        date_col (str): Date column to lookup.
        is_update (bool): True if update is required.
    """
    if type(credit_band_list) != list: credit_band_list = [credit_band_list]
    if len(credit_band_list) == 0: return
    for credit_band in credit_band_list:
        if is_update:
            ipos_date = esg_data.key_to_pos("date_col", date_col)
            update_single_credit_band(credit_band, esg_helper, esg_data, ipos_date)
        else:
            credit_band.skip_update()


def update_single_credit_band(credit_band: CreditBand, esg_helper: Dict[str, 'EsgInfo'],
                              esg_data: KeyedArray, ipos_date: int) -> None:
    """
    Update credit band(s) with ESG data.

    Args:
        credit_band (CreditBand): credit band(s) to update.
        esg_helper (dict): ESG helper info for each credit band.
        esg_data (KeyedArray): ESG data.
        ipos_date (int): Integer-position index of date column.
    """
    band_id = credit_band.band_id

    # --- credit default ---
    helper = esg_helper[f"{band_id}|default"]
    economy, class_, measure, term = helper.economy, helper.class_, helper.measure, 0
    ipos_economy, ipos_class, ipos_measure, ipos_term = helper.ipos_economy, helper.ipos_class, helper.ipos_measure, helper.ipos_term
    compound_freq = helper.compound_freq
    if measure == "DEFAULT_PC":
        pod = esg_data[ipos_economy, ipos_class, ipos_measure, ipos_term, ipos_date] / 100
        if compound_freq == 'annual':
            prop_of_default_ac = pod
        elif compound_freq == 'monthly':
            prop_of_default_ac = 1 - (1 - pod) ** 12
        else:
            raise ValueError(f"{band_id}: compounding frequency of default {compound_freq} not defined.")
    else:
        raise ValueError(f"{band_id}: invalid credit measure for default: {measure}")

    # --- credit recovery ---
    helper = esg_helper[f"{band_id}|recovery"]
    economy, class_, measure, term = helper.economy, helper.class_, helper.measure, 0
    ipos_economy, ipos_class, ipos_measure, ipos_term = helper.ipos_economy, helper.ipos_class, helper.ipos_measure, helper.ipos_term
    if measure == "RECOVERY_PC":
        recovery_rate = esg_data[ipos_economy, ipos_class, ipos_measure, ipos_term, ipos_date] / 100
    else:
        raise ValueError(f"{band_id}: invalid credit measure for recovery: {measure}")

    # --- credit spot mult ---
    helper = esg_helper[f"{band_id}|spotmult"]
    economy, class_, measure, term = helper.economy, helper.class_, helper.measure, 0
    ipos_economy, ipos_class, ipos_measure, ipos_term = helper.ipos_economy, helper.ipos_class, helper.ipos_measure, helper.ipos_term
    if measure == "SPOT_MULT_PC":
        credit_spotmult = esg_data[ipos_economy, ipos_class, ipos_measure, ipos_term, ipos_date] / 100
    else:
        raise ValueError(f"{band_id}: invalid credit measure for spot mult: {measure}")

    # --- credit spread ---
    helper = esg_helper[f"{band_id}|spread"]
    economy, class_, measure, term = helper.economy, helper.class_, helper.measure, 0
    ipos_economy, ipos_class, ipos_measure, ipos_term = helper.ipos_economy, helper.ipos_class, helper.ipos_measure, helper.ipos_term
    if measure == "SPREAD_BP":
        credit_spread = esg_data[ipos_economy, ipos_class, ipos_measure, ipos_term, ipos_date] / 10000
    else:
        raise ValueError(f"{band_id}: invalid credit measure for spread: {measure}")

    credit_band.update(
        prop_of_default_ac=float(prop_of_default_ac),
        recovery_rate=float(recovery_rate),
        credit_spotmult=np.full(shape=13, fill_value=credit_spotmult),
        credit_spread=np.full(shape=13, fill_value=credit_spread),
    )


def build_equity_indices(model, equity_indices_df: pd.DataFrame) -> tuple[List['EquityIndex'], Dict[str, 'EsgInfo']]:
    """
    Build equity index objects and their ESG helpers from a DataFrame.

    Args:
        model: Model object.
        equity_indices_df (pd.DataFrame): DataFrame containing equity index configuration.

    Returns:
        tuple: (List of EquityIndex objects, ESG helper)
    """
    equity_indices = []
    esg_helper = {}

    for idx, row in equity_indices_df.iterrows():
        # create instance
        obj = EquityIndex(model=model, index_id=str(idx))
        equity_indices.append(obj)

        # initialize esg helper
        economy = row["esg_economy"]
        class_ = row["esg_class"]
        measure_tri = row["esg_measure_tri"]
        emeasure_dvy = row["esg_measure_dvy"]
        dvy_comp_freq = row["dvy_compound_frequency"]

        esg_helper[f"{idx}|tri"] = EsgInfo(economy, class_, measure_tri)
        esg_helper[f"{idx}|dvy"] = EsgInfo(economy, class_, emeasure_dvy, compound_freq=dvy_comp_freq)

    return equity_indices, esg_helper


def update_equity_indices(equity_index_list: List['EquityIndex'] | EquityIndex, esg_helper: Dict[str, 'EsgInfo'],
                          esg_data: KeyedArray, date_col: str, is_update: bool = True, esg_step: int = 1) -> None:
    """
    Update equity index(es) with ESG data.

    Args:
        equity_index_list (EquityIndex or list of EquityIndex): Equity index(es) to update.
        esg_helper (dict): ESG helper info for each equity index.
        esg_data (KeyedArray): ESG data.
        date_col (str): Date column to lookup.
        is_update (bool): True if update.
        esg_step (int): ESG file step in months.
    """
    if type(equity_index_list) != list: equity_index_list = [equity_index_list]
    if len(equity_index_list) == 0: return
    for equity_index in equity_index_list:
        if is_update:
            ipos_date = esg_data.key_to_pos("date_col", date_col)
            update_single_equity_index(equity_index, esg_helper, esg_data, ipos_date, esg_step)
        else:
            equity_index.compound_growth()


def update_single_equity_index(equity_index: EquityIndex,esg_helper: Dict[str, 'EsgInfo'],
                               esg_data: KeyedArray, ipos_date: int, esg_step: int) -> None:
    """
    Update equity index(es) with ESG data.

    Args:
        equity_index (EquityIndex): Equity index(es) to update.
        esg_helper (dict): ESG helper info for each equity index.
        esg_data (KeyedArray): ESG data.
        ipos_date (int): Integer-position index of date column.
        esg_step (int): ESG file step in months.
    """
    eqi_id = equity_index.index_id
    p = equity_index.period
    # retrieve esg info from esg_helper

    # --- total return index ---
    helper = esg_helper[f"{eqi_id}|tri"]
    economy, class_, measure, term = helper.economy, helper.class_, helper.measure, 0
    ipos_economy, ipos_class, ipos_measure, ipos_term = helper.ipos_economy, helper.ipos_class, helper.ipos_measure, helper.ipos_term

    if measure == "RET_IDX":
        total_return_index = esg_data[ipos_economy, ipos_class, ipos_measure, ipos_term, ipos_date]
        if esg_step != 1:  # convert 'index as at period end' to 'index as at first month of the period'
            mon_tot_ret = (total_return_index / equity_index.tdv_tot_return_index[p - 1]) ** (1 / esg_step)
            total_return_index = equity_index.tdv_tot_return_index[p - 1] * mon_tot_ret
    else:
        raise ValueError(f"{eqi_id}: invalid equity index measure for total return index: {measure}")

    # --- dividend yield ---
    helper = esg_helper[f"{eqi_id}|dvy"]
    economy, class_, measure, term, compound_freq = helper.economy, helper.class_, helper.measure, 0, helper.compound_freq
    ipos_economy, ipos_class, ipos_measure, ipos_term = helper.ipos_economy, helper.ipos_class, helper.ipos_measure, helper.ipos_term

    if measure == "RNY_PC":
        dvy = esg_data[ipos_economy, ipos_class, ipos_measure, ipos_term, ipos_date] / 100
        if compound_freq == 'annual':
            dividend_yield_ac = dvy
        elif compound_freq == 'monthly':
            dividend_yield_ac = (1 + dvy) ** 12 - 1
        elif compound_freq == 'continuous':
            dividend_yield_ac = math.exp(dvy * 12) - 1
        else:
            raise ValueError(f"{eqi_id}: compounding frequency of dividend yield {compound_freq} not defined.")
    else:
        raise ValueError(f"{eqi_id}: invalid equity index measure for dividend yield: {measure}")

    equity_index.update(total_return_index, dividend_yield_ac)


def build_currencies(model, currencies_df: pd.DataFrame) -> tuple[List['Currency'], Dict[str, 'EsgInfo']]:
    """
    Build currency objects and their ESG helpers from a DataFrame.

    Args:
        model: Model object.
        currencies_df (pd.DataFrame): DataFrame containing currency configuration.

    Returns:
        tuple: (List of Currency objects, ESG helper)
    """
    currencies = []
    esg_helper = {}
    for idx, row in currencies_df.iterrows():
        # create instance
        obj = Currency(model=model, currency_id=str(idx))
        currencies.append(obj)

        # initialize esg helper
        # esg_helper[idx] = EsgInfo(economy=str(idx), class_, measure)

    return currencies, esg_helper


def update_currencies(currency_list: List['Currency'] | Currency, esg_helper: Dict[str, 'EsgInfo'],
                      esg_data: KeyedArray, date_col: str, is_update: bool = True, esg_step: int = 1) -> None:
    """
    Update currency(ies) with ESG data.

    Args:
        currency_list (Currency or list of Currency): Currency(ies) to update.
        esg_helper (dict): ESG helper info for each currency.
        esg_data (np.ndarray): ESG data.
        date_col (str): Date column to lookup.
        is_update (bool): True if update.
        esg_step (int): ESG file step in months.
    """
    if type(currency_list) != list: currency_list = [currency_list]
    if len(currency_list) == 0: return

    for currency in currency_list:
        if is_update:
            ipos_date = esg_data.key_to_pos("date_col", date_col)
            update_single_currency(currency, esg_helper, esg_data, ipos_date, esg_step)
        else:
            p = currency.period
            fx_rate = currency.tdv_fx_rate[p - 1] ** 2 / currency.tdv_fx_rate[p - 2]
            currency.update(fx_rate)


def update_single_currency(currency: Currency, esg_helper: Dict[str, 'EsgInfo'],
                           esg_data: KeyedArray, ipos_date: int, esg_step: int) -> None:
    """
    Update currency(ies) with ESG data.

    Args:
        currency (Currency or list of Currency): Currency(ies) to update.
        esg_helper (dict): ESG helper info for each currency.
        esg_data (np.ndarray): ESG data.
        ipos_date (int): Integer-position index of date column.
        esg_step (int): ESG file step in months.
    """
    currency.update(fx_rate=1.0)  # read from esg once there are foreign currencies


def update_market_info(market_info: MarketInfo, market_data_df: pd.DataFrame,
                       yield_curves: List['YieldCurve'], date_col: str | None = None) -> None:
    """
    Update currency(ies) with ESG data.

    Args:
        market_info (MarketInfo): Market info to update.
        market_data_df (pd.DataFrame): Market data.
        yield_curves (List['YieldCurve']): List of yield curve, to update risk-free short rates.
        date_col (str | None): String date column to lookup.
    """
    t, p = market_info.time, market_info.period
    if date_col is None:
        date_col = str(p.year * 100 + p.month)

    market_data = {item: market_data_df.loc[item, date_col] for item in market_data_df.index}
    for curve in yield_curves:
        if (item := curve.curve_id) not in market_data:
            market_data[f'{item}:short_rate'] = market_info.data.get(f'{item}:short_rate_next', 0)
            if curve.last_update != t:
                warnings.warn(f'{item} is not updated on {t} ({p}), skip update the corresponding short rate.')
                continue
            market_data[f'{item}:short_rate_next'] = (1 / curve.disc_factors[1]) ** 12 - 1  # convert to annual effective rates
    market_info.update(market_data)

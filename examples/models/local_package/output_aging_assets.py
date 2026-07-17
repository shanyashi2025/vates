import pandas as pd
import os
import csv
import warnings

from vates.alm import Cash, BondFixed, Equity

def output_aging_assets(assets: list, df_config: pd.DataFrame, date_index: int | str, out_rootfolder: str) -> None:

    header_dict: dict[str, list[str]] = {
        'cash': [
            'asset_id', 'currency_id', 'fund_id', 'allocation_group', 'nominal', 'positive_cash_balance_ret_id',
            'negative_cash_balance_ret_id',
        ],

        'equity': [
            'asset_id', 'is_profile', 'currency_id', 'fund_id', 'allocation_group', 'asset_classification', 'mv', 'fav',
            'equity_index_id', 'purchase_date',
        ],

        'bond': [
            'asset_id', 'is_profile', 'units', 'currency_id', 'fund_id', 'allocation_group', 'asset_classification',
            'issue_date', 'maturity_date', 'coupon_rate', 'coupon_freq', 'face_value', 'provided_cash_flow_id',
            'mv_price_dirty', 'market_spread', 'abv_price_dirty', 'amort_rate', 'rf_curve_id', 'credit_band_id',
            'purchase_date', 'pre_calculation',
        ],

    }

    out_folder = os.path.join(out_rootfolder, df_config.loc[date_index, 'output_folder'])
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    out_file_dict = {
        'cash': None if 'assets_cash' not in df_config else df_config.loc[date_index, 'assets_cash'],
        'equity': None if 'assets_equity' not in df_config else  df_config.loc[date_index, 'assets_equity'],
        'bond': None if 'assets_bond' not in df_config else  df_config.loc[date_index, 'assets_bond'],
    }

    for k, v in out_file_dict.items():
        if v is None: continue
        if not v.endswith('.csv'):  v += '.csv'
        out_file_dict[k] = os.path.join(out_folder, v)

    header_written = {}

    for asset in assets:
        if not asset.is_alive:
            continue

        if type(asset) is Cash:
            if not header_written.get('cash', False):
                _write_output_file_header(out_file_dict['cash'], header_dict['cash'])
                header_written['cash'] = True
            output_asset_cash(asset, out_file_dict['cash'], header_dict['cash'])
        elif type(asset) is BondFixed:
            if not header_written.get('bond', False):
                _write_output_file_header(out_file_dict['bond'], header_dict['bond'])
                header_written['bond'] = True
            output_asset_fixed_bond(asset, out_file_dict['bond'], header_dict['bond'])
        elif type(asset) is Equity:
            if not header_written.get('equity', False):
                _write_output_file_header(out_file_dict['equity'], header_dict['equity'])
                header_written['equity'] = True
            output_asset_equity(asset, out_file_dict['equity'], header_dict['equity'])
        else:
            warnings.warn(f"Output aging assets for asset class '{type(asset)}' not defined, output ingnored.")

def _write_output_file_header(out_file: str, header_list: list[str]) -> None:
    with open(out_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header_list)

def _write_output_file_content(out_file: str, data_list: list) -> None:
    with open(out_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_list)

def output_asset_cash(cash: Cash, out_file: str, header: list[str]) -> None:
    data_dict = dict.fromkeys(header, None)  # dict.fromkeys() preserves the order of keys
    data_dict['asset_id'] = cash.asset_id
    data_dict['currency_id'] = 'none' if cash.currency is None else cash.currency.currency_id
    data_dict['fund_id'] = cash.fund_id
    data_dict['allocation_group'] = cash.allocation_group
    data_dict['nominal'] = cash.nominal
    data_dict['positive_cash_balance_ret_id'] = cash.ret_id
    data_dict['negative_cash_balance_ret_id'] = cash.ret_id_short_pos
    _write_output_file_content(out_file, list(data_dict.values()))

def output_asset_equity(equity: Equity, out_file: str, header: list[str]) -> None:
    data_dict = dict.fromkeys(header, None)  # dict.fromkeys() preserves the order of keys
    data_dict['asset_id'] = equity.asset_id
    data_dict['is_profile'] = equity.is_profile
    data_dict['currency_id'] = 'none' if equity.currency is None else equity.currency.currency_id
    data_dict['fund_id'] = equity.fund_id
    data_dict['allocation_group'] = equity.allocation_group
    data_dict['asset_classification'] = equity.classification.value
    data_dict['mv'] = equity.mv
    data_dict['fav'] = equity.fav
    data_dict['equity_index_id'] = getattr(equity, '_equity_index').index_id
    data_dict['purchase_date'] = equity.purchase_date
    _write_output_file_content(out_file, list(data_dict.values()))

def output_asset_fixed_bond(bond: BondFixed, out_file: str, header: list[str]) -> None:
    data_dict = dict.fromkeys(header, None)  # dict.fromkeys() preserves the order of keys
    data_dict['asset_id'] = bond.asset_id
    data_dict['is_profile'] = bond.is_profile
    data_dict['units'] = bond.units
    data_dict['currency_id'] = 'none' if bond.currency is None else bond.currency.currency_id
    data_dict['fund_id'] = bond.fund_id
    data_dict['allocation_group'] = bond.allocation_group
    data_dict['asset_classification'] = bond.classification.value
    params = getattr(bond, '_params')
    data_dict['issue_date'] = params.issue_date
    data_dict['maturity_date'] = params.maturity_date
    data_dict['coupon_freq'] = params.coupon_freq
    data_dict['face_value'] = params.face_value
    data_dict['provided_cash_flow_id'] = getattr(bond, 'provided_cash_flow_id', 'none')
    data_dict['mv_price_dirty'] = bond.mv_price
    data_dict['market_spread'] = bond.market_spread
    data_dict['abv_price_dirty'] = bond.abv_price
    data_dict['amort_rate'] = bond.amort_rate
    credit_band = getattr(bond, '_credit_band', None)
    data_dict['credit_band_id'] = 'none' if credit_band is None else credit_band.band_id
    data_dict['purchase_date'] = bond.purchase_date
    data_dict['pre_calculation'] = 'none'
    data_dict['coupon_rate'] = params.coupon_rate
    data_dict['rf_curve_id'] = getattr(bond, '_rf_curve').curve_id
    _write_output_file_content(out_file, list(data_dict.values()))

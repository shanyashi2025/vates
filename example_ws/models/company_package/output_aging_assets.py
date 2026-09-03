import csv
import os
import pandas as pd
import warnings
from dataclasses import dataclass, astuple, fields
from pathlib import Path
from typing import Self

from vates.alm import Cash, BondFixed, Equity

@dataclass
class AgingCash:
    asset_id: str
    currency_id: str
    fund_id: str
    allocation_group: str
    nominal: float
    positive_cash_balance_ret_id: str
    negative_cash_balance_ret_id: str

    @classmethod
    def from_asset_object(cls, cash: Cash) -> Self:
        return AgingCash(
            asset_id=cash.asset_id,
            currency_id='none' if cash.currency is None else cash.currency.currency_id,
            fund_id=cash.fund_id,
            allocation_group=cash.allocation_group,
            nominal=cash.nominal,
            positive_cash_balance_ret_id=cash.ret_id,
            negative_cash_balance_ret_id=cash.ret_id_short_pos,
        )

@dataclass
class AgingEquity:
    asset_id: str
    is_profile: bool
    currency_id: str
    fund_id: str
    allocation_group: str
    asset_classification: str
    mv: float
    fav: float
    equity_index_id: str
    purchase_date: pd.Period

    @classmethod
    def from_asset_object(cls, equity: Equity) -> Self:
        return AgingEquity(
            asset_id=equity.asset_id,
            is_profile=equity.is_profile,
            currency_id='none' if equity.currency is None else equity.currency.currency_id,
            fund_id=equity.fund_id,
            allocation_group=equity.allocation_group,
            asset_classification=equity.classification.value,
            mv=equity.mv,
            fav=equity.fav,
            equity_index_id=getattr(equity, '_equity_index').index_id,
            purchase_date=equity.purchase_date,
        )

@dataclass
class AgingBond:
    asset_id: str
    is_profile: bool
    units: float
    currency_id: str
    fund_id: str
    allocation_group: str
    asset_classification: str
    issue_date: pd.Period
    maturity_date: pd.Period
    coupon_rate: float
    coupon_freq: int
    face_value: float
    provided_cash_flow_id: str
    mv_price_dirty: float
    market_spread: float
    abv_price_dirty: float
    amort_rate: float
    rf_curve_id: str
    credit_band_id: str
    purchase_date: pd.Period

    @classmethod
    def from_asset_object(cls, bond: BondFixed) -> Self:
        params = getattr(bond, '_params')
        credit_band = getattr(bond, '_credit_band', None)
        return AgingBond(
            asset_id=bond.asset_id,
            is_profile=bond.is_profile,
            units=bond.units,
            currency_id='none' if bond.currency is None else bond.currency.currency_id,
            fund_id=bond.fund_id,
            allocation_group=bond.allocation_group,
            asset_classification=bond.classification.value,
            issue_date=params.issue_date,
            maturity_date=params.maturity_date,
            coupon_freq=params.coupon_freq,
            face_value=params.face_value,
            provided_cash_flow_id=getattr(bond, 'provided_cash_flow_id', 'none'),
            mv_price_dirty=bond.mv_price,
            market_spread=bond.market_spread,
            abv_price_dirty=bond.abv_price,
            amort_rate=bond.amort_rate,
            credit_band_id='none' if credit_band is None else credit_band.band_id,
            purchase_date=bond.purchase_date,
            coupon_rate=params.coupon_rate,
            rf_curve_id=getattr(bond, '_rf_curve').curve_id,
        )

class AgingAssetFileManager:

    def __init__(self, output_file_path: Path | str, is_delete_existing: bool = True):
        self.output_file_path: Path = self.get_output_file_path_ready(output_file_path, is_delete_existing)
        self._is_file_created: bool = False

    @staticmethod
    def get_output_file_path_ready(output_file_path: Path | str, is_delete_existing: bool) -> Path:
        if not str(output_file_path).lower().endswith('.csv'):
            output_file_path = Path(str(output_file_path) + '.csv')
        output_file_path.resolve().parent.mkdir(parents=True, exist_ok=True)
        if output_file_path.is_file() and is_delete_existing:
            os.remove(output_file_path)
        return output_file_path

    def writerow(self, asset_dataclass_obj: dataclass):
        if self.output_file_path is None:
            raise ValueError(f"output_file_path is not specified.")

        if not self._is_file_created:
            with open(self.output_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                field_names = tuple(f.name for f in fields(asset_dataclass_obj))
                writer.writerow(field_names)
            self._is_file_created = True

        with open(self.output_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(astuple(asset_dataclass_obj))

def output_aging_assets(assets: list, df_config: pd.DataFrame, date_index: int | str, out_rootfolder: str,
                        is_delete_existing: bool = True) -> None:
    aging_asset_dataclasses: dict[type[Cash | Equity | BondFixed], type[AgingCash, AgingEquity, AgingBond]] = {
        Cash: AgingCash,
        Equity: AgingEquity,
        BondFixed: AgingBond,
    }

    output_folder_path: Path = Path(out_rootfolder) / df_config.loc[date_index, 'output_folder']
    aging_asset_file_managers: dict[type[Cash | Equity | BondFixed], AgingAssetFileManager] = {}
    asset_filename_map: dict[type[Cash | Equity | BondFixed], str] = {
        Cash: 'assets_cash',
        Equity: 'assets_equity',
        BondFixed: 'assets_bond',
    }
    for key, val in asset_filename_map.items():
        if val in df_config:
            aging_asset_file_managers[key] = AgingAssetFileManager(
                output_file_path=output_folder_path / df_config.loc[date_index, val],
                is_delete_existing=is_delete_existing
            )

    for asset in assets:
        if not asset.is_alive:
            continue

        asset_type = type(asset)
        aging_asset_data_cls = aging_asset_dataclasses.get(asset_type)
        aging_asset_file_mgr = aging_asset_file_managers.get(asset_type)
        if aging_asset_data_cls is None:
            warnings.warn(f"Asset class '{asset_type}' not defined, output ingnored.")
            continue
        if aging_asset_file_mgr is None:
            warnings.warn(f"Aging asset file not specified for '{asset_filename_map[asset_type]}', output ingnored.")
            continue

        aging_asset_file_mgr.writerow(aging_asset_data_cls.from_asset_object(asset))

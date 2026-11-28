import pandas as pd
from collections.abc import Iterable
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Self

from vates.alm import Cash, BondFixed, Equity

def _df_to_dc[T](df: pd.DataFrame, cls: type[T]) -> list[T]:
    names = {f.name for f in fields(cls)}
    cols = [c for c in df.columns if c in names]
    return [cls(**rec) for rec in df[cols].to_dict("records")]

@dataclass
class CashRecord:
    valuation_date: pd.Period
    asset_id: str
    currency_id: str
    fund_id: str
    allocation_group: str
    nominal: float
    positive_cash_balance_ret_id: str
    negative_cash_balance_ret_id: str
    ifrs_classification: str = "FVTPL"
    local_classification: str = "FVTPL"
    stat_classification: str = "FVTPL"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> list[Self]:
        df["valuation_date"] = pd.PeriodIndex(df["valuation_date"], freq="M")
        return _df_to_dc(df, cls)

    @classmethod
    def from_objects(cls, objs: Iterable[Cash]) -> list[Self]:
        return [
            cls(valuation_date=obj.period,
                asset_id=obj.asset_id,
                currency_id='none' if obj.currency is None else obj.currency.currency_id,
                fund_id=getattr(obj, "fund_id"),
                allocation_group=getattr(obj, "allocation_group"),
                nominal=obj.nominal,
                positive_cash_balance_ret_id=getattr(obj, "positive_cash_balance_ret_id"),
                negative_cash_balance_ret_id=getattr(obj, "negative_cash_balance_ret_id"),
                ifrs_classification=getattr(obj, "ifrs_classification"),
                local_classification=getattr(obj, "local_classification"),
                stat_classification=getattr(obj, "stat_classification"),)
            for obj in objs
        ]


@dataclass
class EquityRecord:
    valuation_date: pd.Period
    asset_id: str
    is_profile: bool
    currency_id: str
    fund_id: str
    allocation_group: str
    ifrs_classification: str
    local_classification: str
    stat_classification: str
    market_value: float
    historical_cost: float
    equity_index_id: str
    entry_date: pd.Period

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> list[Self]:
        if "entry_date" not in df.columns:
            df["entry_date"] = df["valuation_date"]
        for col in ("valuation_date", "entry_date"):
            df[col] = pd.PeriodIndex(df[col], freq="M")
        if "historical_cost" not in df.columns:
            df["historical_cost"] = df["market_value"]
        return _df_to_dc(df, cls)

    @classmethod
    def from_objects(cls, objs: Iterable[Equity]) -> list[Self]:
        return [
            cls(valuation_date=obj.period,
                asset_id=obj.asset_id,
                is_profile=obj.is_profile,
                currency_id='none' if obj.currency is None else obj.currency.currency_id,
                fund_id=getattr(obj, "fund_id"),
                allocation_group=getattr(obj, "allocation_group"),
                ifrs_classification=getattr(obj, 'ifrs_classification'),
                local_classification=getattr(obj, 'local_classification'),
                stat_classification=getattr(obj, 'stat_classification'),
                market_value=obj.market_value,
                historical_cost=obj.historical_cost,
                equity_index_id=getattr(obj, '_equity_index').index_id,
                entry_date=obj.entry_date, )
            for obj in objs
        ]


@dataclass
class BondRecord:
    valuation_date: pd.Period
    asset_id: str
    is_profile: bool
    qty: float
    currency_id: str
    fund_id: str
    allocation_group: str
    ifrs_classification: str
    local_classification: str
    stat_classification: str
    issue_date: pd.Period
    maturity_date: pd.Period
    coupon_freq: int
    face_value: float
    mv_price_dirty: float
    abv_price_dirty: float
    rf_curve_id: str
    spread_id: str
    default_risk_id: str
    entry_date: pd.Period
    build_pipeline: str = ""
    provided_cash_flow_id: str = "none"
    coupon_rate: float = float("nan")
    amortization_rate: float = float("nan")
    extra_spread: float = float("nan")
    credit_spread: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> list[Self]:
        if "entry_date" not in df.columns:
            df["entry_date"] = df["valuation_date"]
        for col in ("valuation_date", "issue_date", "maturity_date", "entry_date"):
            df[col] = pd.PeriodIndex(df[col], freq="M")
        return _df_to_dc(df, cls)

    @classmethod
    def from_objects(cls, objs: Iterable[BondFixed]) -> list[Self]:
        ls = []
        for obj in objs:
            params = getattr(obj, '_params')
            spread = getattr(obj, '_spread', None)
            default_risk = getattr(obj, '_default_risk', None)
            rec = cls(
                valuation_date=obj.period,
                asset_id=obj.asset_id,
                is_profile=obj.is_profile,
                qty=obj.qty,
                currency_id='none' if obj.currency is None else obj.currency.currency_id,
                fund_id=getattr(obj, "fund_id"),
                allocation_group=getattr(obj, "allocation_group"),
                ifrs_classification=getattr(obj, 'ifrs_classification'),
                local_classification=getattr(obj, 'local_classification'),
                stat_classification=getattr(obj, 'stat_classification'),
                issue_date=params.issue_date,
                maturity_date=params.maturity_date,
                coupon_freq=params.coupon_freq,
                face_value=params.face_value,
                provided_cash_flow_id=getattr(obj, 'provided_cash_flow_id', 'none'),
                mv_price_dirty=obj.mv_price,
                extra_spread=obj.extra_spread,
                abv_price_dirty=obj.abv_price,
                amortization_rate=obj.amortization_rate,
                spread_id='none' if spread is None else spread.spread_id,
                default_risk_id='none' if default_risk is None else default_risk.default_risk_id,
                entry_date=obj.entry_date,
                coupon_rate=params.coupon_rate,
                rf_curve_id=getattr(obj, '_rf_curve').curve_id,
                credit_spread=getattr(obj, 'credit_spread'),
            )
            ls.append(rec)
        return ls

def _dc_to_df(dc_objs: Iterable, columns=None) -> pd.DataFrame:
    return pd.DataFrame.from_records([asdict(o) for o in dc_objs], columns=columns)

def _objects_to_df(cls, objs: Iterable, columns=None) -> pd.DataFrame:
    return _dc_to_df(dc_objs=cls.from_objects(objs), columns=columns)

def output_aging_assets(assets: list, config_df: pd.DataFrame, date_index: int | str, output_rootpath: Path | str
                        ) -> None:
    output_dir_path: Path = Path(output_rootpath) / str(config_df.at[date_index, 'output_folder'])
    dc_map = {Cash: CashRecord, Equity: EquityRecord, BondFixed: BondRecord}
    col_map = {Cash: 'assets_cash', Equity: 'assets_equity', BondFixed: 'assets_bond'}

    obj_ls_dict = {Cash: [], Equity: [], BondFixed: []}
    for asset in assets:
        if not asset.is_alive("eop"):
            continue
        atype = type(asset)
        if atype in obj_ls_dict:
            obj_ls_dict[atype].append(asset)

    for atype, obj_ls in obj_ls_dict.items():
        col = col_map[atype]
        if col not in config_df.columns:
            continue
        filename = str(config_df.at[date_index, col])
        if not filename.endswith('.csv'):
            filename += '.csv'
        df = _objects_to_df(cls=dc_map[atype], objs=obj_ls)
        df.to_csv(output_dir_path / filename, index=False, encoding='utf-8-sig')

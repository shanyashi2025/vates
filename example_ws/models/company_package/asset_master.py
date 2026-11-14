import math
import numpy as np
import pandas as pd
from enum import Enum, unique
from typing import NamedTuple

from vates import ProjModelEngine
from vates.alm import (create_asset, BaseAsset, Cash, BondFixed, Equity, EquityOption, YieldCurve, Spread, DefaultRisk,
                       EquityIndex, Currency)
from .asset_serialization import CashRecord, BondRecord, EquityRecord
from .econ_master import EconMaster

ASSET_CATEGORY_MAPPING = {
    'cash': 'CASH',
    'equity': 'EQUITY',
    'fixed_bond': 'BOND',
    'equity_option': 'EQ_DERIV'
}

@unique
class IFRSClassification(Enum):
    FVTPL = "FVTPL"
    FVOCI = "FVOCI"
    AC = "AC"

class IFRSReportItem(NamedTuple):
    IFRS_BS: str
    IFRS_PL: str

@unique
class LocalClassification(Enum):
    FVTPL = "FVTPL"
    FVOCI = "FVOCI"
    AC = "AC"

class LocalReportItem(NamedTuple):
    LOCAL_BS: str
    LOCAL_PL: str

@unique
class StatClassification(Enum):
    FVTPL = "FVTPL"
    FVOCI = "FVOCI"
    AC = "AC"

class StatReportItem(NamedTuple):
    BSV: str
    FAV: str

class AssetReportItems:

    _ifrs_report_mapping: dict[tuple, IFRSReportItem] = {
        ('cash', IFRSClassification.FVTPL): IFRSReportItem(IFRS_BS="market_value", IFRS_PL="market_value"),
        ('equity', IFRSClassification.FVTPL): IFRSReportItem(IFRS_BS="market_value", IFRS_PL="market_value"),
        ('equity', IFRSClassification.FVOCI): IFRSReportItem(IFRS_BS="market_value", IFRS_PL="purchase_cost"),
        ('fixed_bond', IFRSClassification.FVTPL): IFRSReportItem(IFRS_BS="market_value", IFRS_PL="market_value"),
        ('fixed_bond', IFRSClassification.FVOCI): IFRSReportItem(IFRS_BS="market_value", IFRS_PL="amortized_book_value"),
        ('fixed_bond', IFRSClassification.AC): IFRSReportItem(IFRS_BS="amortized_book_value", IFRS_PL="amortized_book_value"),
        ('equity_option', IFRSClassification.FVTPL): IFRSReportItem(IFRS_BS="market_value", IFRS_PL="market_value"),
    }

    _local_report_mapping: dict[tuple, LocalReportItem] = {
        ('cash', LocalClassification.FVTPL): LocalReportItem(LOCAL_BS="market_value", LOCAL_PL="market_value"),
        ('equity', LocalClassification.FVTPL): LocalReportItem(LOCAL_BS="market_value", LOCAL_PL="market_value"),
        ('equity', LocalClassification.FVOCI): LocalReportItem(LOCAL_BS="market_value", LOCAL_PL="purchase_cost"),
        ('fixed_bond', LocalClassification.FVTPL): LocalReportItem(LOCAL_BS="market_value", LOCAL_PL="market_value"),
        ('fixed_bond', LocalClassification.FVOCI): LocalReportItem(LOCAL_BS="market_value", LOCAL_PL="amortized_book_value"),
        ('fixed_bond', LocalClassification.AC): LocalReportItem(LOCAL_BS="amortized_book_value", LOCAL_PL="amortized_book_value"),
        ('equity_option', LocalClassification.FVTPL): LocalReportItem(LOCAL_BS="market_value", LOCAL_PL="market_value"),
    }

    _stat_report_mapping: dict[tuple, StatReportItem] = {
        ('cash', StatClassification.FVTPL): StatReportItem(BSV="market_value", FAV="market_value"),
        ('equity', StatClassification.FVTPL): StatReportItem(BSV="market_value", FAV="market_value"),
        ('equity', StatClassification.FVOCI): StatReportItem(BSV="market_value", FAV="purchase_cost"),
        ('fixed_bond', StatClassification.FVTPL): StatReportItem(BSV="market_value", FAV="market_value"),
        ('fixed_bond', StatClassification.FVOCI): StatReportItem(BSV="market_value", FAV="amortized_book_value"),
        ('fixed_bond', StatClassification.AC): StatReportItem(BSV="amortized_book_value", FAV="amortized_book_value"),
        ('equity_option', StatClassification.FVTPL): StatReportItem(BSV="market_value", FAV="market_value"),
    }

    _unique_report_items: dict[tuple, dict[str, str]] = {}

    @classmethod
    def to_dict(cls, asset_cls: str, ifrs_classification: str, local_classification: str,
                stat_classification: str) -> dict[str, str]:
        key = (asset_cls, ifrs_classification, local_classification, stat_classification)
        if key in cls._unique_report_items:
            return cls._unique_report_items[key]

        _ifrs_classification = IFRSClassification[ifrs_classification.upper()]
        _local_classification = LocalClassification[local_classification.upper()]
        _stat_classification = StatClassification[stat_classification.upper()]

        _ifrs_rep_item = cls._ifrs_report_mapping[(asset_cls, _ifrs_classification)]
        _local_rep_item = cls._local_report_mapping[(asset_cls, _local_classification)]
        _stat_rep_item = cls._stat_report_mapping[(asset_cls, _stat_classification)]

        report_items: dict[str, str] = {
            "MV": "market_value",
            **_ifrs_rep_item._asdict(),
            **_local_rep_item._asdict(),
            **_stat_rep_item._asdict(),
        }

        cls._unique_report_items[key] = report_items
        return report_items


class AssetMaster:

    pipe_operator: str = "|>"

    def __init__(
        self,
        *,
        cash_ls: list[Cash] | None = None,
        fixed_bond_ls: list[BondFixed] | None = None,
        equity_ls: list[Equity] | None = None,
        equity_option_ls: list[EquityOption] | None = None
    ):
        self.cash_ls: list[Cash] = cash_ls or []
        self.fixed_bond_ls: list[BondFixed] = fixed_bond_ls or []
        self.equity_ls: list[Equity] = equity_ls or []
        self.equity_option_ls: list[EquityOption] = equity_option_ls or []

    @property
    def all(self) -> list[BaseAsset]:
        return self.cash_ls + self.fixed_bond_ls + self.equity_ls + self.equity_option_ls

    @classmethod
    def existing_from_df(
        cls,
        df_dict: dict[str, pd.DataFrame],
        *,
        model_engine: ProjModelEngine,
        econs: EconMaster,
        fund_id: str | None = None
    ) -> 'AssetMaster':
        """
        Build existing assets objects from a dictionary of DataFrame and add them to a fund if provided.

        Args:
            model_engine: Model engine object.
            df_dict (dict[str, pd.DataFrame]): Dictionary of DataFrame containing asset data.
            econs (EsgMaster): Economic variables.
            fund_id (str | None): Fund id to filter the df, or None.

        Returns:
            An AssetMaster object.
        """
        currencies: list[Currency] = [item.econ_obj for item in econs.currencies]
        equity_indices: list[EquityIndex] = [item.econ_obj for item in econs.equity_indices]
        yield_curves: list[YieldCurve] = [item.econ_obj for item in econs.yield_curves]
        spreads: list[Spread] = [item.econ_obj for item in econs.spreads]
        default_risks: list[DefaultRisk] = [item.econ_obj for item in econs.default_risks]

        cash_ls = cls.build_assets_cash_from_df(
            model_engine=model_engine, df=df_dict['assets_cash'].copy(), fund_id=fund_id,
            currencies=currencies
        )
        if df_dict.get(name := 'assets_equity', None) is not None:
            equity_ls = cls.build_assets_equity_from_df(
                model_engine=model_engine, df=df_dict[name].copy(), fund_id=fund_id,
                currencies=currencies, equity_indices=equity_indices
            )
        else:
            equity_ls = []
        if df_dict.get(name := 'assets_bond', None) is not None:
            fixed_bond_ls = cls.build_assets_fixed_bond_from_df(
                model_engine=model_engine, df=df_dict[name].copy(), fund_id=fund_id,
                provided_cash_flow_df= df_dict.get('bond_provided_cash_flow', None),
                currencies=currencies, yield_curves=yield_curves, spreads=spreads, default_risks=default_risks,
            )
        else:
            fixed_bond_ls = []
        if df_dict.get(name := 'assets_equity_option', None) is not None:
            equity_option_ls = cls.build_assets_equity_option_from_df(
                model_engine=model_engine, df=df_dict[name].copy(), fund_id=fund_id,
                currencies=currencies, equity_indices=equity_indices, yield_curves=yield_curves
            )
        else:
            equity_option_ls = []

        return AssetMaster(cash_ls=cash_ls, fixed_bond_ls=fixed_bond_ls, equity_ls=equity_ls, equity_option_ls=equity_option_ls)

    @classmethod
    def profile_from_df(
        cls,
        df_dict: dict[str, pd.DataFrame],
        *,
        model_engine: ProjModelEngine,
        econs: EconMaster,
        fund_id: str | None = None,
    ) -> 'AssetMaster':
        """
        Build profile assets objects from a dictionary of DataFrame.

        Args:
            model_engine: Model engine object.
            df_dict (dict[str, pd.DataFrame]): Dictionary of DataFrame containing asset data.
            econs (EsgMaster): Economic variables.
            fund_id (str): Fund id to filter the df.

        Returns:
            dict[str, list[BaseAsset]]: Dictionary of list of asset objects.
        """
        currencies = [item.econ_obj for item in econs.currencies]
        equity_indices = [item.econ_obj for item in econs.equity_indices]
        yield_curves = [item.econ_obj for item in econs.yield_curves]
        spreads = [item.econ_obj for item in econs.spreads]
        default_risks = [item.econ_obj for item in econs.default_risks]

        equity_ls = []
        fixed_bond_ls = []

        if df_dict.get(name := 'profile_equity', None) is not None:
            equity_ls = cls.build_profile_equity_from_df(
                model_engine=model_engine, df=df_dict[name].copy(), fund_id=fund_id,
                currencies=currencies, equity_indices=equity_indices
            )
        if df_dict.get(name := 'profile_bond', None) is not None:
            fixed_bond_ls = cls.build_profile_fixed_bond_from_df(
                model_engine=model_engine, df=df_dict[name].copy(), fund_id=fund_id,
                currencies=currencies, yield_curves=yield_curves, spreads=spreads, default_risks=default_risks
            )

        return AssetMaster(fixed_bond_ls=fixed_bond_ls, equity_ls=equity_ls)

    @classmethod
    def build_assets_cash_from_df(
        cls,
        df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        fund_id: str | None,
        currencies: list['Currency'],
    ) -> list:
        """
        Build cash asset objects from a DataFrame and add them to a fund if provided.

        Args:
            model_engine: Model engine object.
            df (pd.DataFrame): DataFrame containing cash asset data.
            fund_id (str | None): Fund id to filter the df, or None.
            currencies (list): List of Currency objects.

        Returns:
            list: List of Cash asset objects.
        """
        ls = []

        df: pd.DataFrame = df.copy()
        if fund_id is not None:
            df = df.loc[(df["fund_id"] == fund_id)]
        df["asset_id"] = df.index
        if "valuation_date" not in df.columns:
            p = model_engine.period or model_engine.START_DATE
            df["valuation_date"] = pd.PeriodIndex([p] * len(df), freq="M")

        records = CashRecord.from_df(df)

        for rec in records:
            # create instance
            obj = create_asset(
                "cash",
                model_engine=model_engine,
                currency=next((x for x in currencies if x.currency_id == rec.currency_id), None),
                category=ASSET_CATEGORY_MAPPING["cash"],
                attr_aliases=AssetReportItems.to_dict(
                    asset_cls="cash",
                    ifrs_classification=rec.ifrs_classification,
                    local_classification=rec.local_classification,
                    stat_classification=rec.stat_classification,
                ),
                **rec.to_dict(),
            )
            # append to list
            ls.append(obj)

        return ls

    @classmethod
    def build_assets_fixed_bond_from_df(
        cls,
        df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        fund_id: str | None,
        provided_cash_flow_df: pd.DataFrame | None = None,
        currencies: list['Currency'],
        yield_curves: list['YieldCurve'],
        spreads: list['Spread'],
        default_risks: list['DefaultRisk'],
    ) -> list:
        """
        Build bond asset objects from a DataFrame and add them to a fund if provided.

        Args:
            model_engine: Model object.
            df (pd.DataFrame): DataFrame containing bond asset data.
            fund_id (str | None): Fund id to filter the df, or None.
            provided_cash_flow_df (pd.DataFrame): DataFrame of provided cash flow.
            currencies (list): List of Currency objects.
            yield_curves (list): List of YieldCurve objects.
            spreads (list): List of Spread objects.
            default_risks (list): List of DefaultRisk objects.

        Returns:
            list: List of Bond asset objects.
        """
        ls = []

        df: pd.DataFrame = df.copy()
        if fund_id is not None:
            df = df.loc[(df["fund_id"] == fund_id)]
        df["asset_id"] = df.index
        df["is_profile"] = False
        if "valuation_date" not in df.columns:
            p = model_engine.period or model_engine.START_DATE
            df["valuation_date"] = pd.PeriodIndex([p] * len(df), freq="M")

        records = BondRecord.from_df(df)

        for rec in records:
            fixed_cash_flow = cls.match_bond_fixed_cash_flow(
                cash_flow_id=rec.provided_cash_flow_id,
                df=provided_cash_flow_df,
                n_months = (rec.maturity_date - rec.issue_date).n,
            )

            if rec.build_pipeline != "":
                build_pipeline: list[str | tuple[str, dict]] = rec.build_pipeline.split(cls.pipe_operator)
                for i, item in enumerate(build_pipeline):
                    if item == "prob_of_default":
                        build_pipeline[i] = ("prob_of_default", {"credit_spread": rec.credit_spread})
            else:
                build_pipeline = []

            kwargs = rec.to_dict()
            kwargs.pop("build_pipeline")

            # create instance
            obj = create_asset(
                "fixed_bond",
                model_engine=model_engine,
                build_pipeline=build_pipeline,
                fixed_cash_flow=fixed_cash_flow,
                abv_price=rec.abv_price_dirty,
                mv_price=rec.mv_price_dirty,
                currency=next((x for x in currencies if x.currency_id == rec.currency_id), None),
                rf_curve=next((x for x in yield_curves if x.curve_id == rec.rf_curve_id), None),
                spread=next((x for x in spreads if x.spread_id == rec.spread_id), None),
                default_risk=next((x for x in default_risks if x.default_risk_id == rec.default_risk_id), None),
                category=ASSET_CATEGORY_MAPPING["fixed_bond"],
                attr_aliases=AssetReportItems.to_dict(
                    asset_cls="fixed_bond",
                    ifrs_classification=rec.ifrs_classification,
                    local_classification=rec.local_classification,
                    stat_classification=rec.stat_classification,
                ),
                **kwargs,
            )
            # append to list
            ls.append(obj)

        return ls

    @classmethod
    def match_bond_fixed_cash_flow(
        cls,
        cash_flow_id: str,
        df: pd.DataFrame | None,
        *,
        n_months: int = 0,
    ) -> dict[str, np.ndarray] | None:
        if cash_flow_id.lower() == "none":
            return None

        if df is None:
            raise ValueError(f"Require cash_flow_id: '{cash_flow_id}', but data is not provided.")

        principal = np.zeros(n_months)
        coupon = np.zeros(n_months)

        df = df.loc[(df["provided_cash_flow_id"] == cash_flow_id)].astype(
            {"month": int, "principal": float, "coupon": float}).set_index("month")

        for i in range(n_months):
            month = i + 1
            if month in df.index:
                principal[i] = df.at[month, "principal"]
                coupon[i] = df.at[month, "coupon"]

        return {"principal": principal, "coupon": coupon}

    @classmethod
    def build_profile_fixed_bond_from_df(
        cls,
        df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        fund_id: str | None,
        currencies: list['Currency'],
        yield_curves: list['YieldCurve'],
        spreads: list['Spread'],
        default_risks: list['DefaultRisk'],
    ) -> list:
        """
        Build a profile of bond assets for a fund from a DataFrame.

        Args:
            model_engine: Model object.
            df (pd.DataFrame): DataFrame containing bond profile data.
            fund_id (str): Fund id to filter the df.
            currencies (list): List of Currency objects.
            yield_curves (list): List of YieldCurve objects.
            spreads (list): List of Spread objects.
            default_risks (list): List of DefaultRisk objects.

        Returns:
            list: List of Bond asset objects for the profile.
        """
        ls = []

        df: pd.DataFrame = df.copy()
        if fund_id is not None:
            df = df.loc[(df["fund_id"] == fund_id)]
        p: pd.Period = model_engine.period
        str_cal_ym = str(p.year * 100 + p.month)

        df["asset_id"] = str_cal_ym + df.index
        df["is_profile"] = True
        df["valuation_date"] = pd.PeriodIndex([p] * len(df), freq="M")
        df["issue_date"] = pd.PeriodIndex([p] * len(df), freq="M")
        df["maturity_date"] = df["issue_date"] + df["maturity_term_y"] * 12
        df["mv_price_dirty"] = df["face_value"]  # purchase at par
        df["abv_price_dirty"] = df["face_value"]

        records = BondRecord.from_df(df)

        for rec in records:
            kwargs = rec.to_dict()
            kwargs.pop("build_pipeline")

            obj = create_asset(
                "fixed_bond",
                model_engine=model_engine,
                build_pipeline=["coupon_rate"],
                abv_price=rec.abv_price_dirty,
                mv_price=rec.mv_price_dirty,
                currency=next((x for x in currencies if x.currency_id == rec.currency_id), None),
                rf_curve=next((x for x in yield_curves if x.curve_id == rec.rf_curve_id), None),
                spread=next((x for x in spreads if x.spread_id == rec.spread_id), None),
                default_risk=next((x for x in default_risks if x.default_risk_id == rec.default_risk_id), None),
                category=ASSET_CATEGORY_MAPPING["fixed_bond"],
                attr_aliases=AssetReportItems.to_dict(
                    asset_cls="fixed_bond",
                    ifrs_classification=rec.ifrs_classification,
                    local_classification=rec.local_classification,
                    stat_classification=rec.stat_classification,
                ),
                **kwargs,
            )
            # append to list
            ls.append(obj)

        return ls

    @classmethod
    def build_assets_equity_from_df(
        cls,
        df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        fund_id: str | None,
        currencies: list['Currency'],
        equity_indices: list['EquityIndex']
    ) -> list:
        """
        Build equity asset objects from a DataFrame and add them to a fund if provided.

        Args:
            model_engine: Model object.
            df (pd.DataFrame): DataFrame containing equity asset data.
            fund_id (str | None): Fund id to filter the df, or None.
            currencies (list): List of Currency objects.
            equity_indices (list): List of EquityIndex objects.

        Returns:
            list: List of Equity asset objects.
        """
        ls = []

        df: pd.DataFrame = df.copy()
        if fund_id is not None:
            df = df.loc[(df["fund_id"] == fund_id)]
        df["asset_id"] = df.index
        df["is_profile"] = False
        if "valuation_date" not in df.columns:
            p = model_engine.period or model_engine.START_DATE
            df["valuation_date"] = pd.PeriodIndex([p] * len(df), freq="M")

        records = EquityRecord.from_df(df)

        for rec in records:
            # create instance
            obj = create_asset(
                "equity",
                model_engine=model_engine,
                currency=next((x for x in currencies if x.currency_id == rec.currency_id), None),
                equity_index=next((x for x in equity_indices if x.index_id == rec.equity_index_id), None),
                category=ASSET_CATEGORY_MAPPING["equity"],
                attr_aliases=AssetReportItems.to_dict(
                    asset_cls="equity",
                    ifrs_classification=rec.ifrs_classification,
                    local_classification=rec.local_classification,
                    stat_classification=rec.stat_classification,
                ),
                **rec.to_dict(),
            )
            # append to list
            ls.append(obj)

        return ls

    @classmethod
    def build_profile_equity_from_df(
        cls,
        df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        fund_id: str | None,
        currencies: list['Currency'],
        equity_indices: list['EquityIndex']
    ) -> list:
        """
        Build a profile of equity assets for a fund from a DataFrame.

        Args:
            model_engine: Model object.
            df (pd.DataFrame): DataFrame containing equity asset data.
            fund_id (str): Fund id to filter the df.
            currencies (list): List of Currency objects.
            equity_indices (list): List of EquityIndex objects.

        Returns:
            list: List of Equity asset objects for the profile.
        """
        ls = []

        df: pd.DataFrame = df.copy()
        if fund_id is not None:
            df = df.loc[(df["fund_id"] == fund_id)]
        p: pd.Period = model_engine.period
        str_cal_ym = str(p.year * 100 + p.month)
        df["asset_id"] = str_cal_ym + df.index
        df["is_profile"] = True
        df["valuation_date"] = pd.PeriodIndex([p] * len(df), freq="M")
        df["market_value"] = df["amount"]

        records = EquityRecord.from_df(df)

        for rec in records:
            # create instance
            obj = create_asset(
                "equity",
                model_engine=model_engine,
                currency=next((x for x in currencies if x.currency_id == rec.currency_id), None),
                equity_index=next((x for x in equity_indices if x.index_id == rec.equity_index_id), None),
                category=ASSET_CATEGORY_MAPPING["equity"],
                attr_aliases=AssetReportItems.to_dict(
                    asset_cls="equity",
                    ifrs_classification=rec.ifrs_classification,
                    local_classification=rec.local_classification,
                    stat_classification=rec.stat_classification,
                ),
                **rec.to_dict(),
            )
            # append to list
            ls.append(obj)

        return ls

    @classmethod
    def build_assets_equity_option_from_df(
        cls,
        df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        fund_id: str | None,
        currencies: list['Currency'],
        equity_indices: list['EquityIndex'],
        yield_curves: list['YieldCurve']
    ) -> list:
        """
        Build equity option asset objects from a DataFrame and add them to a fund if provided.

        Args:
            model_engine: Model engine object.
            df (pd.DataFrame): DataFrame containing equity asset data.
            fund_id (str | None): Fund id to filter the df, or None.
            currencies (list): List of Currency objects.
            equity_indices (list): List of EquityIndex objects.
            yield_curves (list): List of YieldCurve objects.

        Returns:
            list: List of EquityOption asset objects.
        """
        ls = []

        df: pd.DataFrame = df.copy()
        if fund_id is not None:
            df = df.loc[(df["fund_id"] == fund_id)]

        for asset_id, row in df.iterrows():
            currency_id = row["currency_id"]
            currency = next((x for x in currencies if x.currency_id == currency_id), None)
            equity_index_id = row["equity_index_id"]
            equity_index = next((x for x in equity_indices if x.index_id == equity_index_id), None)
            rf_curve_id = row["rf_curve_id"]
            rf_curve = next((x for x in yield_curves if x.curve_id == rf_curve_id), None)
            ifrs_classification = row["ifrs_classification"]
            local_classification = row["local_classification"]
            stat_classification = row["stat_classification"]
            if "build_pipeline" in df.columns:
                build_pipeline: list[str] | None = str(row["build_pipeline"]).split(cls.pipe_operator)
            else:
                build_pipeline = None

            # create instance
            obj = create_asset(
                "equity_option",
                model_engine=model_engine,
                build_pipeline=build_pipeline,
                asset_id=asset_id,
                attr_aliases=AssetReportItems.to_dict(
                    asset_cls="equity_option",
                    ifrs_classification=ifrs_classification,
                    local_classification=local_classification,
                    stat_classification=stat_classification,
                ),
                currency=currency,
                call_put=row["call_put"],
                exercise_date=pd.Period(row["exercise_date"], freq='M'),
                qty=row["qty"],
                price=row["price"],
                stock_price=row["stock_price"],
                strike_price=row["strike_price"],
                equity_index=equity_index,
                rf_curve=rf_curve,
                std_dev=row["std_dev"],
                is_pay_dividend=(row["is_pay_dividend"].lower() in ('y', 'yes')),
                q=math.log(1 + 0.2 / 100),  # implement the logic
                is_profile=False,
                ifrs_classification=ifrs_classification,
                local_classification=local_classification,
                stat_classification=stat_classification,
                category=ASSET_CATEGORY_MAPPING["equity_option"],
                allocation_group=row["allocation_group"],
                fund_id=row["fund_id"],
            )
            # append to list
            ls.append(obj)

        return ls

import numpy as np
import pandas as pd
from typing import Self, NamedTuple

from vates import ProjModelEngine
from vates.alm import create_asset, Asset, Cash, BondFixed, Equity, EquityOption, YieldCurve, CreditBand, EquityIndex, Currency, MarketInfo
from .econ_master import EsgMaster

ASSET_CATEGORY_MAPPING = {
    'cash': 'CASH',
    'equity': 'EQUITY',
    'fixed_bond': 'BOND',
    'equity_option': 'EQ_DERIV'
}

class AssetReportBasisKey(NamedTuple):
    asset_cls: str
    ifrs_classification: str
    local_classification: str
    stat_classification: str

class AssetMaster:

    _unique_report_basis_to_attr: dict[AssetReportBasisKey, dict[str, str]] = {}

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

    @classmethod
    def get_report_basis_to_attr(cls, asset_cls: str, ifrs_classification: str, local_classification: str,
                                 stat_classification: str) -> dict[str, str]:
        key = AssetReportBasisKey(
            asset_cls=asset_cls,
            ifrs_classification=ifrs_classification,
            local_classification=local_classification,
            stat_classification=stat_classification,
        )
        if key in cls._unique_report_basis_to_attr:
            return cls._unique_report_basis_to_attr[key]

        report_bases: dict[str, str] = {"MV": "market_value", }

        classification = ifrs_classification.upper()
        if classification == "FVTPL":
            report_bases["IFRS_PL"] = "market_value"
            report_bases["IFRS_BS"] = "market_value"
        elif classification == "FVOCI":
            if asset_cls == "fixed_bond":
                report_bases["IFRS_PL"] = "amortized_book_value"
            elif asset_cls == "equity":
                report_bases["IFRS_PL"] = "purchase_cost"
            else:
                raise ValueError(f"'{asset_cls} can not be classified as 'FVOCI' under IFRS basis.")
            report_bases["IFRS_BS"] = "market_value"
        elif classification == "AC":
            if asset_cls == "fixed_bond":
                report_bases["IFRS_PL"] = "amortized_book_value"
                report_bases["IFRS_BS"] = "amortized_book_value"
            else:
                raise ValueError(f"'{asset_cls} can not be classified as 'AC' under IFRS basis.")
        else:
            raise ValueError(f"Invalid IFRS basis {classification=}, expected: ('FVTPL', 'FVOCI', 'AC').")

        classification = local_classification.upper()
        if classification == "FVTPL":
            report_bases["LOCAL_PL"] = "market_value"
            report_bases["LOCAL_BS"] = "market_value"
        elif classification == "FVOCI":
            if asset_cls == "fixed_bond":
                report_bases["LOCAL_PL"] = "amortized_book_value"
            elif asset_cls == "equity":
                report_bases["LOCAL_PL"] = "purchase_cost"
            else:
                raise ValueError(f"'{asset_cls} can not be classified as 'FVOCI' under Local basis.")
            report_bases["LOCAL_BS"] = "market_value"
        elif classification == "AC":
            if asset_cls == "fixed_bond":
                report_bases["LOCAL_PL"] = "amortized_book_value"
                report_bases["LOCAL_BS"] = "amortized_book_value"
            else:
                raise ValueError(f"'{asset_cls} can not be classified as 'AC' under Local basis.")
        else:
            raise ValueError(f"Invalid Local basis {classification=}, expected: ('FVTPL', 'FVOCI', 'AC').")

        classification = stat_classification.upper()
        if classification in ("FVTPL", "TRADING", "HFT"):
            report_bases["FAV"] = "market_value"
            report_bases["BSV"] = "market_value"
        elif classification in ("FVOCI", "AFS"):
            if asset_cls == "fixed_bond":
                report_bases["FAV"] = "amortized_book_value"
            elif asset_cls == "equity":
                report_bases["FAV"] = "purchase_cost"
            else:
                raise ValueError(f"'{asset_cls} can not be classified as 'FVOCI' under Allocation basis.")
            report_bases["BSV"] = "market_value"
        elif classification in ("AC", "HTM"):
            if asset_cls == "fixed_bond":
                report_bases["FAV"] = "amortized_book_value"
                report_bases["BSV"] = "amortized_book_value"
            else:
                raise ValueError(f"'{asset_cls} can not be classified as 'AC' under Allocation basis.")
        else:
            raise ValueError(f"Invalid Allocation basis {classification=}, expected: ('FVTPL', 'FVOCI', 'AC').")

        cls._unique_report_basis_to_attr[key] = report_bases
        return report_bases

    @property
    def all(self) -> list[Asset]:
        return self.cash_ls + self.fixed_bond_ls + self.equity_ls + self.equity_option_ls

    @classmethod
    def existing_from_df(
        cls,
        df_dict: dict[str, pd.DataFrame],
        *,
        model_engine: ProjModelEngine,
        econs: dict[str, ...] | EsgMaster,
        fund_id: str | None = None
    ) -> Self:
        """
        Build existing assets objects from a dictionary of DataFrame and add them to a fund if provided.

        Args:
            model_engine: Model engine object.
            df_dict (dict[str, pd.DataFrame]): Dictionary of DataFrame containing asset data.
            econs (dict[str, ...] | EsgMaster): Economic variables.
            fund_id (str | None): Fund id to filter the df, or None.

        Returns:
            An AssetMaster object.
        """
        if isinstance(econs, EsgMaster):
            currencies = [item.econ_obj for item in econs.currencies]
            equity_indices = [item.econ_obj for item in econs.equity_indices]
            yield_curves = [item.econ_obj for item in econs.yield_curves]
            credit_bands = [item.econ_obj for item in econs.credit_bands]
            market_info = econs.market_info.econ_obj
        elif isinstance(econs, dict):
            currencies = econs['currencies']
            equity_indices = econs['equity_indices']
            yield_curves = econs['yield_curves']
            credit_bands = econs['credit_bands']
            market_info = econs['market_info']
        else:
            raise TypeError(f"Invalid type of 'market_dict': {econs}, expected 'dict' or 'EsgMaster'.")

        cash_ls = cls.build_assets_cash_from_df(
            model_engine=model_engine, df=df_dict['assets_cash'].copy(), fund_id=fund_id,
            currencies=currencies, market_info=market_info
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
                currencies=currencies, yield_curves=yield_curves, credit_bands=credit_bands
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
        econs: dict[str, ...] | EsgMaster,
        fund_id: str | None = None,
    ) -> Self:
        """
        Build profile assets objects from a dictionary of DataFrame.

        Args:
            model_engine: Model engine object.
            df_dict (dict[str, pd.DataFrame]): Dictionary of DataFrame containing asset data.
            econs (dict[str, ...] | EsgMaster): Economic variables.
            fund_id (str): Fund id to filter the df.

        Returns:
            dict[str, list[Asset]]: Dictionary of list of asset objects.
        """
        if isinstance(econs, EsgMaster):
            currencies = [item.econ_obj for item in econs.currencies]
            equity_indices = [item.econ_obj for item in econs.equity_indices]
            yield_curves = [item.econ_obj for item in econs.yield_curves]
            credit_bands = [item.econ_obj for item in econs.credit_bands]
        elif isinstance(econs, dict):
            currencies = econs['currencies']
            equity_indices = econs['equity_indices']
            yield_curves = econs['yield_curves']
            credit_bands = econs['credit_bands']
        else:
            raise TypeError(f"Invalid type of 'market_dict': {econs}, expected 'dict' or 'EsgMaster'.")

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
                currencies=currencies, yield_curves=yield_curves, credit_bands=credit_bands
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
        market_info: MarketInfo
    ) -> list:
        """
        Build cash asset objects from a DataFrame and add them to a fund if provided.

        Args:
            model_engine: Model engine object.
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

        for asset_id, row in df_flt.iterrows():
            currency_id = row["currency_id"]
            currency = next((x for x in currencies if x.currency_id == currency_id), None)
            ifrs_classification = "FVTPL"
            local_classification = "FVTPL"
            stat_classification = "FVTPL"

            # create instance
            cash = create_asset(
                asset_cls="cash",
                model_engine=model_engine,
                asset_id=asset_id,
                flex_attr_map=cls.get_report_basis_to_attr(
                    asset_cls="cash",
                    ifrs_classification=ifrs_classification,
                    local_classification=local_classification,
                    stat_classification=stat_classification,
                ),
                currency=currency,
                nominal=row["nominal"],
                market_info=market_info,
                ret_id=row["positive_cash_balance_ret_id"],
                ret_id_short_pos=row["negative_cash_balance_ret_id"],
                dyn_attrs={
                    'ifrs_classification': ifrs_classification,
                    'local_classification': local_classification,
                    'stat_classification': stat_classification,
                    "category": ASSET_CATEGORY_MAPPING["cash"],
                    "allocation_group": row["allocation_group"],
                    "fund_id": row["fund_id"],
                },
            )
            # append to list
            cash_list.append(cash)

        return cash_list

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
        credit_bands: list['CreditBand'],
    ) -> list:
        """
        Build bond asset objects from a DataFrame and add them to a fund if provided.

        Args:
            model_engine: Model object.
            df (pd.DataFrame): DataFrame containing bond asset dataF.
            fund_id (str | None): Fund id to filter the df, or None.
            provided_cash_flow_df (pd.DataFrame): DataFrame of provided cash flow.
            currencies (list): List of Currency objects.
            yield_curves (list): List of YieldCurve objects.
            credit_bands (list): List of CreditBand objects.

        Returns:
            list: List of Bond asset objects.
        """
        fixed_bond_list = []

        df_flt: pd.DataFrame = df.copy()
        if fund_id is not None: df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

        for asset_id, row in df_flt.iterrows():
            currency_id = row["currency_id"]
            currency = next((x for x in currencies if x.currency_id == currency_id), None)
            rf_curve_id = row["rf_curve_id"]
            rf_curve = next((x for x in yield_curves if x.curve_id == rf_curve_id), None)
            credit_band_id = row["credit_band_id"]
            credit_band = next((x for x in credit_bands if x.band_id == credit_band_id), None)
            provided_cash_flow_id = row["provided_cash_flow_id"]
            is_cash_flow_provided = provided_cash_flow_id.lower() != 'none'
            if is_cash_flow_provided:
                provided_cash_flow_dict = cls.get_provided_bond_cash_flow_from_df(
                    df=provided_cash_flow_df,
                    provided_cash_flow_id=provided_cash_flow_id,
                    issue_date=pd.Period(row["issue_date"], freq='M'),
                    maturity_date=pd.Period(row["maturity_date"], freq='M'),
                )
            else:
                provided_cash_flow_dict = None

            ifrs_classification = row["ifrs_classification"]
            local_classification = row["local_classification"]
            stat_classification = row["stat_classification"]

            # create instance
            fixed_bond = create_asset(
                model_engine=model_engine,
                asset_cls="fixed_bond",
                build_pipeline=row["build_pipeline"] if "build_pipeline" in df.columns else None,
                asset_id=asset_id,
                currency=currency,
                issue_date=pd.Period(row["issue_date"], freq='M'),
                maturity_date=pd.Period(row["maturity_date"], freq='M'),
                coupon_rate=row["coupon_rate"],
                coupon_freq=row["coupon_freq"],
                face_value=row["face_value"],
                provided_cash_flow_dict=provided_cash_flow_dict,
                units=row["units"],
                flex_attr_map=cls.get_report_basis_to_attr(
                    asset_cls="fixed_bond",
                    ifrs_classification=ifrs_classification,
                    local_classification=local_classification,
                    stat_classification=stat_classification,
                ),
                rf_curve=rf_curve,
                credit_band=credit_band,
                abv_price=row["abv_price_dirty"],
                mv_price=row["mv_price_dirty"],
                market_spread=row["market_spread"],
                is_profile=False,
                dyn_attrs={
                    'ifrs_classification': ifrs_classification,
                    'local_classification': local_classification,
                    'stat_classification': stat_classification,
                    "category": ASSET_CATEGORY_MAPPING["fixed_bond"],
                    "allocation_group": row["allocation_group"],
                    "fund_id": row["fund_id"],
                    'provided_cash_flow_id': provided_cash_flow_id,
                },
            )
            # append to list
            fixed_bond_list.append(fixed_bond)

        return fixed_bond_list

    @classmethod
    def get_provided_bond_cash_flow_from_df(
        cls,
        df: pd.DataFrame,
        provided_cash_flow_id: str,
        *,
        issue_date: pd.Period,
        maturity_date: pd.Period
    ) -> dict[str, np.ndarray]:
        n_months = (maturity_date - issue_date).n
        arr_dict = {"principal": np.zeros(n_months), "interest": np.zeros(n_months)}

        df = df.loc[(df["provided_cash_flow_id"] == provided_cash_flow_id)].astype(
            {"month": int, "principal": float, "interest": float}).set_index("month")

        for i in range(n_months):
            month = i + 1
            if month in df.index:
                arr_dict["principal"][i] = df.at[month, "principal"]
                arr_dict["interest"][i] = df.at[month, "interest"]

        return arr_dict

    @classmethod
    def build_profile_fixed_bond_from_df(
        cls,
        df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        fund_id: str | None,
        currencies: list['Currency'],
        yield_curves: list['YieldCurve'],
        credit_bands: list['CreditBand']
    ) -> list:
        """
        Build a profile of bond assets for a fund from a DataFrame.

        Args:
            model_engine: Model object.
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
        p: pd.Period = model_engine.period
        str_cal_ym = str(p.year * 100 + p.month)

        # initialize assets profile - bond
        for _asset_id, row in df_flt.iterrows():
            # read profile information
            currency_id = row["currency_id"]
            currency = next((x for x in currencies if x.currency_id == currency_id), None)
            rf_curve_id = row["rf_curve_id"]
            rf_curve = next((x for x in yield_curves if x.curve_id == rf_curve_id), None)
            credit_band_id = row["credit_band_id"]
            credit_band = next((x for x in credit_bands if x.band_id == credit_band_id), None)
            ifrs_classification = row["ifrs_classification"]
            local_classification = row["local_classification"]
            stat_classification = row["stat_classification"]

            fixed_bond = create_asset(
                model_engine=model_engine,
                asset_cls="fixed_bond",
                build_pipeline='coupon_rate',
                asset_id=f"{str_cal_ym}{_asset_id}",
                currency=currency,
                issue_date=p,
                maturity_date=p + row["maturity_term_y"] * 12,
                coupon_freq=row["coupon_freq"],
                face_value=row["face_value"],
                redemp_sched=None,
                units=row["units"],
                flex_attr_map=cls.get_report_basis_to_attr(
                    asset_cls="fixed_bond",
                    ifrs_classification=ifrs_classification,
                    local_classification=local_classification,
                    stat_classification=stat_classification,
                ),
                rf_curve=rf_curve,
                credit_band=credit_band,
                abv_price=row["face_value"],
                mv_price=row["face_value"],
                market_spread=row["market_spread"],
                is_profile=True,
                dyn_attrs={
                    'ifrs_classification': ifrs_classification,
                    'local_classification': local_classification,
                    'stat_classification': stat_classification,
                    "category": ASSET_CATEGORY_MAPPING["fixed_bond"],
                    "allocation_group": row["allocation_group"],
                    "fund_id": row["fund_id"],
                },
            )
            # append to list
            fixed_bond_list.append(fixed_bond)

        return fixed_bond_list

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
        equity_list = []

        df_flt: pd.DataFrame = df.copy()
        if fund_id is not None: df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

        for asset_id, row in df_flt.iterrows():
            currency_id = row["currency_id"]
            currency = next((x for x in currencies if x.currency_id == currency_id), None)
            equity_index_id = row["equity_index_id"]
            equity_index = next((x for x in equity_indices if x.index_id == equity_index_id), None)
            ifrs_classification = row["ifrs_classification"]
            local_classification = row["local_classification"]
            stat_classification = row["stat_classification"]

            # create instance
            equity = create_asset(
                asset_cls="equity",
                model_engine=model_engine,
                asset_id=asset_id,
                flex_attr_map=cls.get_report_basis_to_attr(
                    asset_cls="equity",
                    ifrs_classification=ifrs_classification,
                    local_classification=local_classification,
                    stat_classification=stat_classification,
                ),
                currency=currency,
                market_value=row["market_value"],
                purchase_cost=row["purchase_cost"],
                equity_index=equity_index,
                is_profile=False,
                dyn_attrs={
                    'ifrs_classification': ifrs_classification,
                    'local_classification': local_classification,
                    'stat_classification': stat_classification,
                    "category": ASSET_CATEGORY_MAPPING["equity"],
                    "allocation_group": row["allocation_group"],
                    "fund_id": row["fund_id"],
                },
            )
            # append to list
            equity_list.append(equity)

        return equity_list

    @classmethod
    def build_profile_equity_from_df(
        cls,
        df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        fund_id: str,
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
        equity_list = []

        df_flt: pd.DataFrame = df.copy()
        df_flt = df_flt.loc[(df["fund_id"] == fund_id)]
        p: pd.Period = model_engine.period
        str_cal_ym = str(p.year * 100 + p.month)

        for _asset_id, row in df_flt.iterrows():
            currency_id = row["currency_id"]
            currency = next((x for x in currencies if x.currency_id == currency_id), None)
            equity_index_id = row["equity_index_id"]
            equity_index = next((x for x in equity_indices if x.index_id == equity_index_id), None)
            ifrs_classification = row["ifrs_classification"]
            local_classification = row["local_classification"]
            stat_classification = row["stat_classification"]

            # create instance
            equity = create_asset(
                asset_cls="equity",
                model_engine=model_engine,
                asset_id=f"{str_cal_ym}{_asset_id}",
                flex_attr_map=cls.get_report_basis_to_attr(
                    asset_cls="equity",
                    ifrs_classification=ifrs_classification,
                    local_classification=local_classification,
                    stat_classification=stat_classification,
                ),
                currency=currency,
                market_value=row["amount"],
                equity_index=equity_index,
                is_profile=True,
                dyn_attrs={
                    'ifrs_classification': ifrs_classification,
                    'local_classification': local_classification,
                    'stat_classification': stat_classification,
                    "category": ASSET_CATEGORY_MAPPING["equity"],
                    "allocation_group": row["allocation_group"],
                    "fund_id": row["fund_id"],
                },
            )
            # append to list
            equity_list.append(equity)

        return equity_list

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
        equity_option_list = []

        df_flt: pd.DataFrame = df.copy()
        if fund_id is not None: df_flt = df_flt.loc[(df["fund_id"] == fund_id)]

        for asset_id, row in df_flt.iterrows():
            currency_id = row["currency_id"]
            currency = next((x for x in currencies if x.currency_id == currency_id), None)
            equity_index_id = row["equity_index_id"]
            equity_index = next((x for x in equity_indices if x.index_id == equity_index_id), None)
            rf_curve_id = row["rf_curve_id"]
            rf_curve = next((x for x in yield_curves if x.curve_id == rf_curve_id), None)
            ifrs_classification = row["ifrs_classification"]
            local_classification = row["local_classification"]
            stat_classification = row["stat_classification"]

            # create instance
            equity_option = create_asset(
                asset_cls="equity_option",
                model_engine=model_engine,
                build_pipeline=row["build_pipeline"] if "build_pipeline" in df.columns else None,
                asset_id=asset_id,
                flex_attr_map=cls.get_report_basis_to_attr(
                    asset_cls="equity_option",
                    ifrs_classification=ifrs_classification,
                    local_classification=local_classification,
                    stat_classification=stat_classification,
                ),
                currency=currency,
                call_or_put=row["call_or_put"],
                exercise_date=pd.Period(row["exercise_date"], freq='M'),
                units=row["units"],
                price=row["price"],
                stock_price=row["stock_price"],
                strike_price=row["strike_price"],
                equity_index=equity_index,
                rf_curve=rf_curve,
                std_dev=row["std_dev"],
                is_pay_dividend=(row["is_pay_dividend"].lower() in ('y', 'yes')),
                is_profile=False,
                dyn_attrs={
                    'ifrs_classification': ifrs_classification,
                    'local_classification': local_classification,
                    'stat_classification': stat_classification,
                    "category": ASSET_CATEGORY_MAPPING["equity_option"],
                    "allocation_group": row["allocation_group"],
                    "fund_id": row["fund_id"],
                },
            )
            # append to list
            equity_option_list.append(equity_option)

        return equity_option_list

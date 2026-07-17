import math
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from typing import Self

from vates import ProjModelEngine, KeyedArray
from vates.utils import curve_interp, parse_str_to_int_list
from vates.alm import YieldCurve, CreditBand, EquityIndex, Currency, MarketInfo


@dataclass(slots=True)
class EsgVariable:
    """
    Dataclass representing an ESG variable.

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
    measure: str
    divby: float = 1
    term: int | list[int] = 0
    term_type: str | None = None
    interp_method: str | None = None
    compound_freq: str | None = None
    ipos_economy: int = 0
    ipos_class: int = 0
    ipos_measure: int = 0
    ipos_term: int | list[int] = 0
    descr: str = ""

    def __post_init__(self):
        if isinstance(self.term, str):
            self.term = parse_str_to_int_list(self.term, separator=';', joiner='-', sort_list='asc')
            if (entry := self.term[0]) <= 0:
                raise ValueError(f"Esg term entry {entry} <= 0, expected positive integer.")
            if len(self.term) > 1200:
                warnings.warn(f"Number of esg term nodes exceeds 1200.")


@dataclass(slots=True)
class EsgItem:
    econ_obj: YieldCurve | CreditBand | EquityIndex | Currency | MarketInfo
    descr: str = ""
    esg_vars: dict[str, EsgVariable] | None = None
    esg_data: KeyedArray | pd.DataFrame | None = None

    def include_esg_data(self, esg_df: pd.DataFrame) -> None:
        class_list = [v.class_ for k, v in self.esg_vars.items()]
        df = esg_df[esg_df["CLASS"].isin(class_list)].copy()
        df.set_index(['ECONOMY', 'CLASS', 'MEASURE', 'TERM'], inplace=True)
        self.esg_data = KeyedArray.from_df(df=df, unpack_multi_index=True, col_index_name='date_col')
        self._set_esg_vars_intpos()

    def _set_esg_vars_intpos(self) -> None:
        for key, var in self.esg_vars.items():
            var.ipos_economy = self.esg_data.key_to_pos(dim='ECONOMY', key=var.economy)
            var.ipos_class = self.esg_data.key_to_pos(dim='CLASS', key=var.class_)
            var.ipos_measure = self.esg_data.key_to_pos(dim='MEASURE', key=var.measure)

            if isinstance(var.term, int):
                var.ipos_term = self.esg_data.key_to_pos(dim='TERM', key=var.term)
            elif isinstance(var.term, list):
                var.ipos_term = [self.esg_data.key_to_pos(dim='TERM', key=x) for x in var.term]
            else:
                raise TypeError(f"{var.descr}: invalid term type {type(var.term)}, expected 'int' or 'list'.")


class EsgMaster:

    def __init__(
        self,
        yield_curves: list[EsgItem] | None = None,
        credit_bands: list[EsgItem] | None = None,
        equity_indices: list[EsgItem] | None = None,
        currencies: list[EsgItem] | None = None,
        market_info: EsgItem | None = None,
    ) -> None:
        self.yield_curves: list[EsgItem] = yield_curves or []
        self.credit_bands: list[EsgItem] = credit_bands or []
        self.equity_indices: list[EsgItem] = equity_indices or []
        self.currencies: list[EsgItem] = currencies or []
        self.market_info: EsgItem | None = market_info

    def update_econ_data(self, period: pd.Period, *, esg_step: int = 1) -> None:
        date_col = self._get_esg_date_col(period, esg_step)
        # update yield curves
        for esg_item in self.yield_curves:
            if date_col:
                self.update_yield_curve(esg_item, date_col)
            else:
                esg_item.econ_obj.no_change_on_update()
        # update credit bands
        for esg_item in self.credit_bands:
            if date_col:
                self.update_credit_band(esg_item, date_col)
            else:
                esg_item.econ_obj.no_change_on_update()
        # update equity indices
        for esg_item in self.equity_indices:
            if date_col:
                self.update_equity_index(esg_item, date_col, esg_step)
            else:
                esg_item.econ_obj.compound_growth_on_update()
        # update currencies
        for esg_item in self.currencies:
            if date_col:
                self.update_currency(esg_item, date_col)
            else:
                esg_item.econ_obj.update(fx_rate=1.0)  # revisit the formula once there are foreign currencies
        # update market info
        self.update_market_info(date_col)

    @staticmethod
    def _get_esg_date_col(period: pd.Period, esg_step: int) -> str | None:
        if esg_step not in (1, 3, 6, 12):
            raise ValueError(f'Invalid esg_step: {esg_step}')
        if esg_step == 1:
            return str(period.year * 100 + period.month)
        elif period.month % esg_step == 1:  # read esg at the first month of the period, using period end date
            return str(period.year * 100 + period.month + esg_step - 1)
        else:
            return None

    @staticmethod
    def update_yield_curve(esg_item: EsgItem, date_col: str) -> None:
        """
        Update risk-free yield curve with ESG data.

        Args:
            esg_item (YieldCurve): Curve to update.
            date_col (str): Date column.
        """
        yield_curve_obj = esg_item.econ_obj
        esg_vars = esg_item.esg_vars
        esg_data = esg_item.esg_data
        ipos_date = esg_data.key_to_pos("date_col", date_col)

        var = esg_vars["rates"]
        rates = np.array(esg_data[var.ipos_economy, var.ipos_class, var.ipos_measure, var.ipos_term, ipos_date])

        # convert term to months
        if var.term_type == "Y":
            term = np.insert(arr=np.array(var.term) * 12, obj=0, values=0)
        elif var.term_type == "M":
            term = np.insert(arr=np.array(var.term), obj=0, values=0)
        else:
            raise ValueError(f"Invalid {var.term_type=}, expected 'Y' or 'M'.")

        # determine rate type and process curve data
        if var.measure == "PRICE":
            rates = np.insert(arr=rates, obj=0, values=1)  # let discount_factor = 1 at term 0
            yield_curve_obj.disc_factors = curve_interp(term, rates, var.interp_method)
        elif var.measure == "SPOT":
            rates = np.insert(arr=rates, obj=0, values=0)  # let spot_rate = 0 at term 0
            yield_curve_obj.spot_rates = curve_interp(term, rates, var.interp_method)
        else:
            raise ValueError(f"{var.descr}: invalid {var.measure=}")

    @staticmethod
    def update_credit_band(esg_item: EsgItem, date_col: str) -> None:
        """
        Update credit band(s) with ESG data.

        Args:
            esg_item (CreditBand): credit band(s) to update.
            date_col (str): Date column.
        """
        esg_vars = esg_item.esg_vars
        esg_data = esg_item.esg_data
        ipos_date = esg_data.key_to_pos("date_col", date_col)

        # 1. prop_of_default
        var = esg_vars["prop_of_default"]
        pod = esg_data[var.ipos_economy, var.ipos_class, var.ipos_measure, var.ipos_term, ipos_date] / var.divby
        if var.compound_freq == 'annual':
            prop_of_default_ac = pod
        elif var.compound_freq == 'monthly':
            prop_of_default_ac = 1 - (1 - pod) ** 12
        else:
            raise ValueError(f"{var.descr}: invalid {var.compound_freq=}.")

        # 2. recovery_rate
        var = esg_vars["recovery_rate"]
        recovery_rate = esg_data[var.ipos_economy, var.ipos_class, var.ipos_measure, var.ipos_term, ipos_date] / var.divby

        # 3. credit_spotmult
        var = esg_vars["credit_spotmult"]
        credit_spotmult = esg_data[var.ipos_economy, var.ipos_class, var.ipos_measure, var.ipos_term, ipos_date] / var.divby

        # 4. credit_spread
        var = esg_vars["credit_spread"]
        credit_spread = esg_data[var.ipos_economy, var.ipos_class, var.ipos_measure, var.ipos_term, ipos_date] / var.divby

        esg_item.econ_obj.update(
            prop_of_default_ac=float(prop_of_default_ac),
            recovery_rate=float(recovery_rate),
            credit_spotmult=np.full(shape=13, fill_value=credit_spotmult),
            credit_spread=np.full(shape=13, fill_value=credit_spread),
        )

    @staticmethod
    def update_equity_index(esg_item: EsgItem, date_col: str, esg_step: int) -> None:
        """
        Update equity index(es) with ESG data.

        Args:
            esg_item (EsgItem): Equity index(es) to update.
            date_col (str): Date column.
            esg_step (int): esg step.
        """
        esg_vars = esg_item.esg_vars
        esg_data = esg_item.esg_data
        ipos_date = esg_data.key_to_pos("date_col", date_col)

        # 1. total_return_index
        var = esg_vars["total_return_index"]
        total_return_index = esg_data[var.ipos_economy, var.ipos_class, var.ipos_measure, var.ipos_term, ipos_date] / var.divby
        if esg_step != 1:  # convert 'index as at period end' to 'index as at first month of the period'
            equity_index = esg_item.econ_obj
            p = equity_index.period
            mon_tot_ret = (total_return_index / equity_index.tdv_tot_return_index[p - 1]) ** (1 / esg_step)
            total_return_index = equity_index.tdv_tot_return_index[p - 1] * mon_tot_ret

        # 2. dividend_yield
        var = esg_vars["dividend_yield"]
        dvy = esg_data[var.ipos_economy, var.ipos_class, var.ipos_measure, var.ipos_term, ipos_date] / var.divby
        if var.compound_freq == 'annual':
            dividend_yield_ac = dvy
        elif var.compound_freq == 'monthly':
            dividend_yield_ac = (1 + dvy) ** 12 - 1
        elif var.compound_freq == 'continuous':
            dividend_yield_ac = math.exp(dvy * 12) - 1
        else:
            raise ValueError(f"{var.descr}: {var.compound_freq=} not defined.")

        esg_item.econ_obj.update(total_return_index, dividend_yield_ac)

    @staticmethod
    def update_currency(esg_item: EsgItem, date_col: str) -> None:
        """
        Update currency(ies) with ESG data.

        Args:
            esg_item (Currency or list of Currency): Currency(ies) to update.
            date_col (str): Date column.
        """
        esg_item.econ_obj.update(fx_rate=1.0)  # read from esg once there are foreign currencies


    def update_market_info(self, date_col: str | None) -> None:
        if self.market_info is None:
            return
        market_info_obj = self.market_info.econ_obj
        data_df = self.market_info.esg_data

        if data_df is not None:
            data = {item: data_df.loc[item, date_col] for item in data_df.index}
        else:
            data = {}

        # update short rate
        for curve_esg_item in self.yield_curves:
            curve_obj = curve_esg_item.econ_obj
            curve_id = curve_obj.curve_id
            short_rate_id = f'{curve_id}:short_rate'
            if short_rate_id not in data:
                data[short_rate_id] = market_info_obj.get(f'{short_rate_id}:next', 0)
                if curve_obj.last_update != market_info_obj.time:
                    warnings.warn(f'{curve_id} is not updated on {market_info_obj.time} ({market_info_obj.period}), '
                                  f'skip update the corresponding short rate.')
                    continue
                data[f'{short_rate_id}:next'] = (1 / curve_obj.disc_factors[1]) ** 12 - 1  # convert to annual effective rates

        for key, value in data.items():
            market_info_obj[key] = value

    @classmethod
    def from_df(
        cls,
        esg_df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        esg_params: dict,
        market_info_df: pd.DataFrame | None = None
    ) -> Self:
        yield_curves = cls.build_esg_items(model_engine=model_engine, econ_cls=YieldCurve, econ_id_attr="curve_id",
                                           esg_params=esg_params.get("yield_curves"), esg_df=esg_df,
                                           variable_spec={"rates": ["measure", "term", "term_type", "interp_method"]})
        credit_bands = cls.build_esg_items(model_engine=model_engine, econ_cls=CreditBand, econ_id_attr="band_id",
                                           esg_params=esg_params.get("credit_bands"), esg_df=esg_df,
                                           variable_spec={
                                               "prop_of_default": ["measure", "divby", "compound_freq"],
                                               "recovery_rate": ["measure", "divby"],
                                               "credit_spotmult": ["measure", "divby"],
                                               "credit_spread": ["measure", "divby"],
                                           })
        equity_indices = cls.build_esg_items(model_engine=model_engine, econ_cls=EquityIndex, econ_id_attr="index_id",
                                             esg_params=esg_params.get("equity_indices"), esg_df=esg_df,
                                             variable_spec={
                                                 "total_return_index": ["measure"],
                                                 "dividend_yield": ["measure", "divby", "compound_freq"],
                                             })
        currencies = cls.build_esg_items(model_engine=model_engine, econ_cls=Currency, econ_id_attr="currency_id",
                                         esg_params=esg_params.get("currencies"), esg_df=esg_df,
                                         variable_spec={})
        market_info = EsgItem(descr="general", econ_obj=MarketInfo(model_engine=model_engine), esg_data=market_info_df)
        return EsgMaster(
            yield_curves=yield_curves,
            credit_bands=credit_bands,
            equity_indices=equity_indices,
            currencies=currencies,
            market_info=market_info,
        )

    @classmethod
    def build_esg_items(
        cls,
        *,
        model_engine: ProjModelEngine,
        econ_cls,
        econ_id_attr: str,
        esg_params: dict,
        esg_df: pd.DataFrame,
        variable_spec: dict[str, list]
    ) -> list[EsgItem]:
        """
        Build Econ objects and their ESG variables.

        Args:
            model_engine: Model engine object.
            econ_cls: Econ class (YieldCurve | CreditBand | EquityIndex | Currency).
            econ_id_attr (str): Attribute name for the id.
            esg_params (dict): Dict containing esg params.
            esg_df (pd.DataFrame): DataFrame containing esg data.
            variable_spec (dict): Dict containing variable spec.

        Returns:
            list: List of EsgItem objects
        """
        if esg_params is None:
            return []

        esg_item_list = []
        default_var_attr_dict = esg_params.get("default_variable_attribute") or {}
        for item in esg_params["item_list"]:
            econ_id = item[econ_id_attr]
            kwargs = {"model_engine": model_engine, econ_id_attr: econ_id}
            obj = econ_cls(**kwargs)
            economy = item["economy"]
            class_ = item["class"]

            esg_vars = {}
            for var_name, attr_list in variable_spec.items():
                var_attr = item.get(var_name) or {}
                default_var_attr = default_var_attr_dict.get(var_name) or {}
                kwargs = {attr: var_attr.get(attr) or default_var_attr[attr] for attr in attr_list}
                esg_vars[var_name] = EsgVariable(economy, class_, **kwargs)

            esg_item = EsgItem(descr=econ_id, econ_obj=obj, esg_vars=esg_vars)
            esg_item.include_esg_data(esg_df)
            esg_item_list.append(esg_item)

        return esg_item_list

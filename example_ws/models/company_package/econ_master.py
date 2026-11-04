import math
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

from vates import ProjModelEngine
from vates._core import time_synchronized
from vates.alm import YieldCurve, Spread, DefaultRisk, EquityIndex, Currency
from vates.finmath import interpolate_interest_rates
from vates.utils import KeyedArray, parse_int_list_from_str


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
    ipos_term: int | list[int] = 0
    descr: str = ""
    fixed_value: Any = None
    data: KeyedArray | None = None

    def __post_init__(self):
        if isinstance(self.term, str):
            self.term = parse_int_list_from_str(self.term, separator=';', joiner='-', sort_list='asc')
            if (entry := self.term[0]) <= 0:
                raise ValueError(f"Esg term entry {entry} <= 0, expected positive integer.")
            if len(self.term) > 1200:
                warnings.warn(f"Number of esg term nodes exceeds 1200.")


@dataclass(slots=True)
class EconContext[T]:
    econ_obj: T
    data: KeyedArray | pd.DataFrame | None = None
    esg_vars: dict[str, EsgVariable] = field(default_factory=dict)
    descr: str = ""

    def set_esg_var_data(self, df: pd.DataFrame) -> None:
        for v in self.esg_vars.values():
            filtered_df = (df[(df['ECONOMY'] == v.economy) & (df['CLASS'] == v.class_) & (df['MEASURE'] == v.measure)]
                           .drop(['ECONOMY', 'CLASS', 'MEASURE'], axis=1)
                           .set_index('TERM'))
            if len(filtered_df) == 0:
                continue
            data: KeyedArray = KeyedArray.from_df(df=filtered_df, col_index_name='date_col')
            v.data = data
            if isinstance(v.term, int):
                v.ipos_term = data.key_to_pos(dim='TERM', key=v.term)
            elif isinstance(v.term, list):
                v.ipos_term = [data.key_to_pos(dim='TERM', key=x) for x in v.term]
            else:
                raise TypeError(f"{v.descr}: invalid term type {type(v.term)}, expected 'int' or 'list'.")


@time_synchronized("model_engine.time_synchronizer")
class EconMaster:

    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`

    def __init__(
        self,
        *,
        yield_curves: list[EconContext[YieldCurve]] | None = None,
        spreads: list[EconContext[Spread]] | None = None,
        default_risks: list[EconContext[DefaultRisk]] | None = None,
        equity_indices: list[EconContext[EquityIndex]] | None = None,
        currencies: list[EconContext[Currency]] | None = None,
        econ_asm_df: pd.DataFrame | None = None,
        model_engine: ProjModelEngine | None = None,
        esg_step_map: dict | Callable | None = None,
    ) -> None:
        self.yield_curves: list[EconContext[YieldCurve]] = yield_curves or []
        self.spreads: list[EconContext[Spread]] = spreads or []
        self.default_risks: list[EconContext[DefaultRisk]] = default_risks or []
        self.equity_indices: list[EconContext[EquityIndex]] = equity_indices or []
        self.currencies: list[EconContext[Currency]] = currencies or []
        self.econ_asm_df: pd.DataFrame | None = econ_asm_df
        self.econ_asm: dict[str, float] = {}
        self.esg_step_map: dict | Callable | None = esg_step_map  # take one argument of `period`

    def update(self) -> None:
        period = self.period
        if self.esg_step_map is None:
            esg_step = 1  # defaults to 1
        elif isinstance(self.esg_step_map, dict):
            esg_step = self.esg_step_map.get(period, 1)  # defaults to 1
        elif callable(self.esg_step_map):
            esg_step = self.esg_step_map(period)
        else:
            raise TypeError(f"Invalid {type(period)=}, expected 'dict' or 'callable'.")
        date_col = self._get_esg_date_col(period, esg_step)
        for ctx in self.yield_curves:
            self.update_yield_curve(ctx, date_col)
        for ctx in self.spreads:
            self.update_spread(ctx, date_col)
        for ctx in self.default_risks:
            self.update_default_risk(ctx, date_col)
        for ctx in self.equity_indices:
            self.update_equity_index(ctx, date_col, esg_step)
        for ctx in self.currencies:
            self.update_currency(ctx, date_col)
        self.update_econ_asm(date_col)

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
    def update_yield_curve(ctx: EconContext, date_col: str | None) -> None:
        """
        Update risk-free yield curve with ESG data.

        Args:
            ctx (EconContext[YieldCurve]): Curve to update.
            date_col (str): Date column.
        """
        if date_col is None:
            ctx.econ_obj.update()
            return

        esg_vars = ctx.esg_vars

        var = esg_vars["rates"]
        ipos_date = var.data.key_to_pos("date_col", date_col)
        rates = np.array(var.data[var.ipos_term, ipos_date])

        # convert term to months
        if var.term_type == "Y":
            term = np.insert(arr=np.array(var.term) * 12, obj=0, values=0)
        elif var.term_type == "M":
            term = np.insert(arr=np.array(var.term), obj=0, values=0)
        else:
            raise ValueError(f"Invalid {var.term_type=}, expected 'Y' or 'M'.")

        # determine rate type and process curve data
        if var.measure == "PRICE":
            from_kind = "discount_factors"
            rates = np.insert(arr=rates, obj=0, values=1)  # let discount_factor = 1 at term 0
        elif var.measure == "SPOT":
            from_kind = "spot_rates"
            rates = np.insert(arr=rates, obj=0, values=0)  # let spot_rate = 0 at term 0
        else:
            raise ValueError(f"{var.descr}: invalid {var.measure=}")

        ctx.econ_obj.update(from_kind=from_kind, value=interpolate_interest_rates(term, rates, interp_method=var.interp_method))

    @staticmethod
    def update_spread(ctx: EconContext[Spread], date_col: str | None) -> None:
        """
        Update spread(s) with ESG data.

        Args:
            ctx (EconContext[Spread]): spread(s) to update.
            date_col (str): Date column.
        """
        econ_obj: Spread = ctx.econ_obj

        if date_col is None:
            econ_obj.update()
            return

        esg_vars = ctx.esg_vars
        ipos_date = None

        # 1. spread_add
        var = esg_vars["spread_add"]
        if (fixed_value := var.fixed_value) is None:
            ipos_date = var.data.key_to_pos("date_col", date_col)
            spread_add = var.data[var.ipos_term, ipos_date] / var.divby
        else:
            spread_add = fixed_value / var.divby

        # 2. spread_propn
        var = esg_vars["spread_propn"]
        if (fixed_value := var.fixed_value) is None:
            ipos_date = ipos_date if ipos_date is not None else var.data.key_to_pos("date_col", date_col)
            spread_propn = var.data[var.ipos_term, ipos_date] / var.divby
        else:
            spread_propn = fixed_value / var.divby

        econ_obj.update(
            additive=np.full(shape=13, fill_value=spread_add),
            proportional=np.full(shape=13, fill_value=spread_propn),
        )

    @staticmethod
    def update_default_risk(ctx: EconContext[DefaultRisk], date_col: str | None) -> None:
        """
        Update default risk(s) with ESG data.

        Args:
            ctx (EconContext[DefaultRisk]): default risk(s) to update.
            date_col (str): Date column.
        """
        econ_obj: DefaultRisk = ctx.econ_obj

        if date_col is None:
            econ_obj.update()
            return

        esg_vars = ctx.esg_vars

        # 1. MPD (Marginal Probability of Default)
        var = esg_vars["MPD"]
        ipos_date = var.data.key_to_pos("date_col", date_col)
        mpd = var.data[var.ipos_term, ipos_date] / var.divby
        if var.compound_freq == 'annual':
            mpd = 1 - (1 - mpd) ** (1 / 12)
        elif var.compound_freq == 'monthly':
            pass
        else:
            raise ValueError(f"{var.descr}: invalid {var.compound_freq=}.")

        # 2. LGD (Loss Given Default)
        var = esg_vars["LGD"]
        lgd = var.data[var.ipos_term, ipos_date] / var.divby

        # 3. pd_affects_decrement
        var = esg_vars["pd_affects_decrement"]
        pd_affects_decrement = var.fixed_value

        econ_obj.update(
            marginal_prob_of_default=float(mpd),
            loss_given_default=float(lgd),
            pd_affects_decrement=pd_affects_decrement,
        )

    @staticmethod
    def update_equity_index(ctx: EconContext, date_col: str | None, esg_step: int) -> None:
        """
        Update equity index(es) with ESG data.

        Args:
            ctx (EconContext[EquityIndex]): Equity index(es) to update.
            date_col (str): Date column.
            esg_step (int): esg step.
        """
        if date_col is None:
            ctx.econ_obj.update(apply_compound_growth=True)
            return

        esg_vars = ctx.esg_vars

        # 1. total_return_index
        var = esg_vars["total_return_index"]
        ipos_date = var.data.key_to_pos("date_col", date_col)
        total_return_index = var.data[var.ipos_term, ipos_date] / var.divby
        if esg_step != 1:  # convert 'index as at period end' to 'index as at first month of the period'
            equity_index = ctx.econ_obj
            p = equity_index.period
            mon_tot_ret = (total_return_index / equity_index.tdv_tot_return_index[p - 1]) ** (1 / esg_step)
            total_return_index = equity_index.tdv_tot_return_index[p - 1] * mon_tot_ret

        # 2. dividend_yield
        var = esg_vars["dividend_yield"]
        dvy = var.data[var.ipos_term, ipos_date] / var.divby
        if var.compound_freq == 'annual':
            dividend_yield_ac = dvy
        elif var.compound_freq == 'monthly':
            dividend_yield_ac = (1 + dvy) ** 12 - 1
        elif var.compound_freq == 'continuous':
            dividend_yield_ac = math.exp(dvy * 12) - 1
        else:
            raise ValueError(f"{var.descr}: {var.compound_freq=} not defined.")

        ctx.econ_obj.update(total_return_index=total_return_index, dividend_yield_ac=dividend_yield_ac)

    @staticmethod
    def update_currency(ctx: EconContext, date_col: str | None) -> None:
        """
        Update currency(ies) with ESG data.

        Args:
            ctx (EconContext[Currency]): Currency(ies) to update.
            date_col (str): Date column.
        """
        if date_col is None:
            ctx.econ_obj.update(apply_compound_growth=True)
            return
        ctx.econ_obj.update(fx_rate=1.0)  # read from esg once there are foreign currencies

    def update_econ_asm(self, date_col: str | None) -> None:
        if (df := self.econ_asm_df) is not None:
            econ_asm: dict[str, float] = {item: df.loc[item, date_col] for item in df.index}
        else:
            econ_asm = {}

        # update short rate
        for curve_ctx in self.yield_curves:
            curve_obj = curve_ctx.econ_obj
            curve_id = curve_obj.curve_id
            short_rate_id = f'{curve_id}:short_rate'
            if short_rate_id not in econ_asm:
                econ_asm[short_rate_id] = self.econ_asm.get(f'{short_rate_id}:next', 0)
                econ_asm[f'{short_rate_id}:next'] = (1 / curve_obj.discount_factors[1]) ** 12 - 1  # convert to annual effective rates

        self.econ_asm |= econ_asm

    @classmethod
    def from_df(
        cls,
        esg_df: pd.DataFrame,
        *,
        model_engine: ProjModelEngine,
        esg_params: dict,
        econ_asm_df: pd.DataFrame | None = None,
        esg_step_map: dict | Callable | None = None
    ) -> 'EconMaster':
        yield_curves = cls.build_esg_context(
            model_engine=model_engine,
            econ_cls=YieldCurve,
            econ_id_attr="curve_id",
            esg_params=esg_params.get("yield_curves"),
            esg_df=esg_df,
            variable_spec={"rates": ["measure", "term", "term_type", "interp_method"]}
        )
        spreads = cls.build_esg_context(
            model_engine=model_engine,
            econ_cls=Spread,
            econ_id_attr="spread_id",
            esg_params=esg_params.get("spreads"),
            esg_df=esg_df,
            variable_spec={"spread_add": ["measure", "divby"],
                           "spread_propn": ["measure", "divby", "fixed_value"]}
        )
        default_risks = cls.build_esg_context(
            model_engine=model_engine,
            econ_cls=DefaultRisk,
            econ_id_attr="default_risk_id",
            esg_params=esg_params.get("default_risks"),
            esg_df=esg_df,
            variable_spec={"MPD": ["measure", "divby", "compound_freq"],
                           "LGD": ["measure", "divby"],
                           "pd_affects_decrement": ["measure", "fixed_value"]}
        )
        equity_indices = cls.build_esg_context(
            model_engine=model_engine,
            econ_cls=EquityIndex,
            econ_id_attr="index_id",
            esg_params=esg_params.get("equity_indices"),
            esg_df=esg_df,
            variable_spec={"total_return_index": ["measure"],
                           "dividend_yield": ["measure", "divby", "compound_freq"]}
        )
        currencies = cls.build_esg_context(
            model_engine=model_engine,
            econ_cls=Currency,
            econ_id_attr="currency_id",
            esg_params=esg_params.get("currencies"),
            esg_df=esg_df,
            variable_spec={}
        )
        obj = EconMaster(
            yield_curves=yield_curves,
            spreads=spreads,
            default_risks=default_risks,
            equity_indices=equity_indices,
            currencies=currencies,
            econ_asm_df=econ_asm_df,
            model_engine=model_engine,
            esg_step_map=esg_step_map
        )
        if obj.period is not None:
            obj.update()
        return obj

    @classmethod
    def build_esg_context(
        cls,
        *,
        model_engine: ProjModelEngine,
        econ_cls,
        econ_id_attr: str,
        esg_params: dict | None,
        esg_df: pd.DataFrame,
        variable_spec: dict[str, list]
    ) -> list[EconContext]:
        """
        Build Econ objects and their ESG variables.

        Args:
            model_engine: Model engine object.
            econ_cls: Econ class (YieldCurve | Spread | DefaultRisk | EquityIndex | Currency).
            econ_id_attr (str): Attribute name for the id.
            esg_params (dict): Dict containing esg params.
            esg_df (pd.DataFrame): DataFrame containing esg data.
            variable_spec (dict): Dict containing variable spec.

        Returns:
            list: List of EsgContext objects
        """
        if esg_params is None:
            return []

        ctx_list = []
        var_default_attr_dict = esg_params.get("variable_default_attribute") or {}
        for instrument in esg_params["instruments"]:
            econ_id = instrument[econ_id_attr]
            kwargs = {"model_engine": model_engine, econ_id_attr: econ_id}
            obj = econ_cls(**kwargs)
            economy = instrument["economy"]
            class_ = instrument["class"]

            esg_vars = {}
            for var_name, attr_list in variable_spec.items():
                var_attr = instrument.get(var_name) or {}
                var_default_attr = var_default_attr_dict.get(var_name) or {}
                kwargs = {key: val if (val := var_attr.get(key, None)) is not None else var_default_attr[key]
                          for key in attr_list}
                esg_vars[var_name] = EsgVariable(economy, class_, **kwargs)

            ctx = EconContext(descr=econ_id, econ_obj=obj, esg_vars=esg_vars)
            ctx.set_esg_var_data(esg_df)
            ctx_list.append(ctx)

        return ctx_list

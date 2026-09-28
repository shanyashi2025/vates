import pandas as pd
from dataclasses import dataclass

from vates._core import ProjModelEngine, time_synchronized, TDimVariable
from vates.utils import maybe_raise_if_ne
from vates.solvency.risk_node import RiskNode, risk_aggregation
from vates.solvency.cn_cross2.params import (
    MC_CORR_MATRIX,
    MORB_MC_CORR_MATRIX,
    LOSS_MC_CORR_MATRIX,
    LIFE_MC_CORR_MATRIX,
    MARKET_MC_CORR_MATRIX,
    CREDIT_MC_CORR_MATRIX,
    calculate_loss_absorbency,
)


def _overall_risk_agg(life: float, non_life: float, market: float, credit: float) -> float:
    return risk_aggregation(life, non_life, market, credit, corr_matrix=MC_CORR_MATRIX)

def _life_risk_agg(loss: float, expense: float, lapse: float) -> float:
    return risk_aggregation(loss, expense, lapse, corr_matrix=LIFE_MC_CORR_MATRIX)

def _loss_risk_agg(mortality: float, catastrophe: float, longevity: float, morbidity: float, health: float,
                   other_loss: float) -> float:
    return risk_aggregation(mortality, catastrophe, longevity, morbidity, health, other_loss,
                            corr_matrix=LOSS_MC_CORR_MATRIX)

def _morb_risk_agg(morb_incidence: float, morb_trend: float) -> float:
    return risk_aggregation(morb_incidence, morb_trend, corr_matrix=MORB_MC_CORR_MATRIX)

def _market_risk_agg(interest_rate: float, equity: float, real_estate: float, overseas_fixed_income: float,
                     overseas_equity: float, exchange_rate: float) -> float:
    return risk_aggregation(interest_rate, equity, real_estate, overseas_fixed_income, overseas_equity, exchange_rate,
                         corr_matrix=MARKET_MC_CORR_MATRIX)

def _credit_risk_agg(spread: float, counterparty_default: float) -> float:
    return risk_aggregation(spread, counterparty_default, corr_matrix=CREDIT_MC_CORR_MATRIX)

def make_cross2_risk_module(name: str, /) -> RiskNode:
    root = RiskNode(name)
    # root
    root.attach_sub_risk(RiskNode("Life"), RiskNode("Non-life"), RiskNode("Market"), RiskNode("Credit"))
    root.set_agg_func(_overall_risk_agg)

    # ./Life
    node = root.select("Life")
    node.attach_sub_risk(RiskNode("Loss"), RiskNode("Expense"), RiskNode("Lapse"))
    node.set_agg_func(_life_risk_agg)

    # ./Life/Loss
    node = root.select("Life/Loss")
    node.attach_sub_risk(RiskNode("Mortality"), RiskNode("Catastrophe"), RiskNode("Longevity"),
                         RiskNode("Morbidity"), RiskNode("Health & Medical", slug="health"),
                         RiskNode("Other", slug="other_loss"))
    node.set_agg_func(_loss_risk_agg)

    # ./Life/Loss/Morbidity
    node = root.select("Life/Loss/Morbidity")
    node.attach_sub_risk(RiskNode("Incidence", slug="morb_incidence"), RiskNode("Trend", slug="morb_trend"))
    node.set_agg_func(_morb_risk_agg)

    # ./Life/Lapse
    node = root.select("Life/Lapse")
    node.attach_sub_risk(RiskNode("Lapse Rate"), RiskNode("Mass Lapse"))
    node.set_agg_func(lambda lapse_rate, mass_lapse: max(lapse_rate, mass_lapse, 0))

    # ./Life/Lapse/Lapse Rate
    node = root.select("Life/Lapse/Lapse Rate")
    node.attach_sub_risk(RiskNode("Lapse_Up"), RiskNode("Lapse_Down"))
    node.set_agg_func(lambda lapse_up, lapse_down: max(lapse_up, lapse_down, 0))

    # ./Market
    node = root.select("Market")
    node.attach_sub_risk(RiskNode("Interest Rate"), RiskNode("Equity"), RiskNode("Real Estate"),
                         RiskNode("Overseas Fixed-income"), RiskNode("Overseas Equity"), RiskNode("Exchange Rate"))
    node.set_agg_func(_market_risk_agg)

    # ./Market/Interest Rate
    node = root.select("Market/Interest Rate")
    node.attach_sub_risk(RiskNode("Interest Rate Up"), RiskNode("Interest Rate Down"))
    node.set_agg_func(lambda interest_rate_up, interest_rate_down: max(interest_rate_up, interest_rate_down, 0))

    # ./Credit
    node = root.select("Credit")
    node.attach_sub_risk(RiskNode("Spread"), RiskNode("Counterparty Default"))
    node.set_agg_func(_credit_risk_agg)

    return root

@dataclass(slots=True)
class AdditiveRiskCharge:
    # Life
    mortality: float
    catastrophe: float
    longevity: float
    morb_incidence: float
    morb_trend: float
    health: float
    other_loss: float
    expense: float
    lapse_up: float
    lapse_down: float
    mass_lapse: float
    # Non-life
    non_life: float
    # Market
    interest_rate_up: float
    interest_rate_down: float
    equity: float
    real_estate: float
    overseas_fixed_income: float
    overseas_equity: float
    exchange_rate: float
    # Credit
    spread: float
    counterparty_default: float


@time_synchronized
class MinCapUnit:
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`
    
    def __init__(
        self,
        name: str,
        *,
        model_engine: ProjModelEngine = None,
        require_loss_absorbency: bool,
    ):
        self.name: str = name
        self.require_loss_absorbency: bool = require_loss_absorbency
        self._risk_module: RiskNode = make_cross2_risk_module(name)
        self._la_pv_base: float = 0.0
        self._la_pv_lower_limit = 0.0
        self._loss_absorbency: float = 0.0
        self._last_calculate: int | None = None

        create_tdv = lambda varname: TDimVariable(varname, model_engine=model_engine, owner=name, group='CROSS_MC')
        self.tdv_min_cap: TDimVariable = create_tdv("minimum_capital")
        self.tdv_life_mc: TDimVariable = create_tdv("life_mc")
        self.tdv_nonlife_mc: TDimVariable = create_tdv("nonlife_mc")
        self.tdv_market_mc: TDimVariable = create_tdv("market_mc")
        self.tdv_credit_mc: TDimVariable = create_tdv("credit_mc")
        self.tdv_divers: TDimVariable = create_tdv("diversification")
        self.tdv_loss_absorb: TDimVariable = create_tdv("loss_absorbency")

    def calculate(self, *, additive_risk_charge: AdditiveRiskCharge | None = None,
                  la_pv_base: float = 0.0, la_pv_lower_limit = 0.0, **kwargs) -> None:
        for node in self._risk_module.list_leaves():
            key = node.slug
            if additive_risk_charge is not None:
                node.update_risk_charge(getattr(additive_risk_charge, key))
            elif key in kwargs:
                node.update_risk_charge(kwargs[key])

        if self.require_loss_absorbency:
            self._la_pv_base = la_pv_base
            self._la_pv_lower_limit = la_pv_lower_limit
            self._loss_absorbency = calculate_loss_absorbency(
                mc_market=self._risk_module.select("Market").risk_charge,
                mc_credit=self._risk_module.select("Credit").risk_charge,
                pv_base=self._la_pv_base,
                pv_lower_limit=self._la_pv_lower_limit
            )
        else:
            self._loss_absorbency = 0.0

        t = self.time
        self.tdv_life_mc[t] = self._risk_module.select("Life").risk_charge
        self.tdv_nonlife_mc[t] = self._risk_module.select("Non-life").risk_charge
        self.tdv_market_mc[t] = self._risk_module.select("Market").risk_charge
        self.tdv_credit_mc[t] = self._risk_module.select("Credit").risk_charge
        self.tdv_divers[t] = self._risk_module.risk_diversification
        self.tdv_loss_absorb[t] = self._loss_absorbency
        self.tdv_min_cap[t] = self._risk_module.risk_charge - self._loss_absorbency
        self._last_calculate = t

    @property
    def risk_module(self) -> RiskNode:
        maybe_raise_if_ne(self._last_calculate, self.time)
        return self._risk_module

    @property
    def loss_absorbency(self) -> float:
        if not self.require_loss_absorbency:
            return 0.0
        maybe_raise_if_ne(self._last_calculate, self.time)
        return self._loss_absorbency

    @property
    def la_pv_base(self) -> float:
        if not self.require_loss_absorbency:
            return 0.0
        maybe_raise_if_ne(self._last_calculate, self.time)
        return self._la_pv_base

    @property
    def la_pv_lower_limit(self) -> float:
        if not self.require_loss_absorbency:
            return 0.0
        maybe_raise_if_ne(self._last_calculate, self.time)
        return self._la_pv_lower_limit


@time_synchronized
class MinCapConsolidator:
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`
    
    def __init__(
        self,
        name: str,
        *,
        model_engine: ProjModelEngine = None,
        min_cap_units: list[MinCapUnit] | tuple[MinCapUnit]
    ):
        self.name: str = name
        self._units: tuple[MinCapUnit] = tuple(min_cap_units)
        self._risk_module: RiskNode = self._make_consolidate_risk_module(
            *[item.risk_module for item in self._units], name=self.name)
        self._loss_absorbency: float = 0.0
        self._min_cap: float = 0.0
        self._last_calculate: int | None = None

        create_tdv = lambda varname: TDimVariable(varname, model_engine=model_engine, owner=name, group='CROSS_MC')
        self.tdv_min_cap: TDimVariable = create_tdv("minimum_capital")
        self.tdv_life_mc: TDimVariable = create_tdv("life_mc")
        self.tdv_nonlife_mc: TDimVariable = create_tdv("nonlife_mc")
        self.tdv_market_mc: TDimVariable = create_tdv("market_mc")
        self.tdv_credit_mc: TDimVariable = create_tdv("credit_mc")
        self.tdv_divers: TDimVariable = create_tdv("diversification")
        self.tdv_loss_absorb: TDimVariable = create_tdv("loss_absorbency")

    @classmethod
    def _make_consolidate_risk_module(cls, *args, name: str, slug: str = None) -> RiskNode:
        if len(args) == 0:
            raise ValueError(f"Nothing to consolidate.")
        ref_path_set: set[str] | None = None
        for obj in args:
            path_set = set([node.path for node in obj.preorder_traversal()])
            if ref_path_set is None:
                ref_path_set = path_set
            elif len(path_set - ref_path_set) > 0:
                raise ValueError(f"Can't consolidate '{obj.name}' and '{args[0].name}': structures are differenct.")
        return args[0].copy_structure(new_name=name, new_slug=slug)

    def consolidate(self) -> None:
        for node in self._risk_module.list_leaves():
            path = node.path
            risk_charge = sum(unit.risk_module.select(path).risk_charge for unit in self._units)
            node.update_risk_charge(risk_charge)

        self._loss_absorbency = self._calculate_loss_absorbency()

        t = self.time
        self.tdv_life_mc[t] = self._risk_module.select("Life").risk_charge
        self.tdv_nonlife_mc[t] = self._risk_module.select("Non-life").risk_charge
        self.tdv_market_mc[t] = self._risk_module.select("Market").risk_charge
        self.tdv_credit_mc[t] = self._risk_module.select("Credit").risk_charge
        self.tdv_divers[t] = self._risk_module.risk_diversification
        self.tdv_loss_absorb[t] = self._loss_absorbency
        self.tdv_min_cap[t] = self._risk_module.risk_charge -  self._loss_absorbency
        self._last_calculate = t

    def _calculate_loss_absorbency(self) -> float:
        la_risk_module = make_cross2_risk_module("loss_absorbency")
        la_leaves = la_risk_module.select("Market").list_leaves() + la_risk_module.select("Credit").list_leaves()
        la_pv_base = 0.0
        la_pv_lower_limit = 0.0

        for unit in self._units:
            if unit.require_loss_absorbency:
                unit_risk_module = unit.risk_module
                for node in la_leaves:
                    unit_risk_charge = unit_risk_module.select(node.path).risk_charge
                    if node._risk_charge is None:
                        node.update_risk_charge(unit_risk_charge)
                    else:
                        node.update_risk_charge(node._risk_charge + unit_risk_charge)
                la_pv_base += unit.la_pv_base
                la_pv_lower_limit += unit.la_pv_lower_limit

        return calculate_loss_absorbency(
            mc_market=la_risk_module.select("Market").risk_charge,
            mc_credit=la_risk_module.select("Credit").risk_charge,
            pv_base=la_pv_base,
            pv_lower_limit=la_pv_lower_limit
        )

    @property
    def minimum_capital(self) -> float:
        maybe_raise_if_ne(self._last_calculate, self.time)
        return self._risk_module.risk_charge -  self._loss_absorbency

    @property
    def loss_absorbency(self) -> float:
        maybe_raise_if_ne(self._last_calculate, self.time)
        return self._loss_absorbency

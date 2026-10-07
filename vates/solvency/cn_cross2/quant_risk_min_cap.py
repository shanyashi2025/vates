import math
import pandas as pd
from functools import partial

from vates._core import ProjModelEngine, time_synchronized, TDimVariable
from vates.utils import maybe_raise_if_ne
from vates.solvency.risk_tree import  RiskTree, RiskNode, risk_aggregation
from vates.solvency.cn_cross2.rules import (
    MC_CORR_MATRIX,
    MORB_MC_CORR_MATRIX,
    LOSS_MC_CORR_MATRIX,
    LIFE_MC_CORR_MATRIX,
    MARKET_MC_CORR_MATRIX,
    CREDIT_MC_CORR_MATRIX,
    calculate_loss_absorbency,
)


def _max_at_zero(**capitals: float) -> float:
    return max(max(capitals.values()), 0.0)

def _floor_at_zero(*args: float) -> tuple[float, ...]:
    return tuple([max(x, 0.0) for x in args])

def _overall_risk_agg(life: float, non_life: float, market: float, credit: float) -> float:
    return risk_aggregation(life, non_life, market, credit, corr_matrix=MC_CORR_MATRIX)

def _life_risk_agg(loss: float, expense: float, lapse: float) -> float:
    return risk_aggregation(loss, expense, lapse, corr_matrix=LIFE_MC_CORR_MATRIX)

def _loss_risk_agg(mortality: float, catastrophe: float, longevity: float, morbidity: float, health: float, other: float
                   ) -> float:
    return risk_aggregation(*_floor_at_zero(mortality, catastrophe, longevity, morbidity, health, other),
                            corr_matrix=LOSS_MC_CORR_MATRIX)

def _nonlife_risk_agg(premium_reserve: float, catastrophe: float, k: float):
    return math.sqrt(premium_reserve ** 2 + 2 * 0.25 * premium_reserve * catastrophe + catastrophe ** 2) * k

def _morb_risk_agg(incidence: float, trend: float) -> float:
    return risk_aggregation(incidence, trend, corr_matrix=MORB_MC_CORR_MATRIX)

def _market_risk_agg(interest_rate: float, equity: float, real_estate: float, overseas_fixed_income: float,
                     overseas_equity: float, exchange_rate: float) -> float:
    return risk_aggregation(interest_rate, equity, real_estate, overseas_fixed_income, overseas_equity, exchange_rate,
                         corr_matrix=MARKET_MC_CORR_MATRIX)

def _credit_risk_agg(spread: float, counterparty_default: float) -> float:
    return risk_aggregation(spread, counterparty_default, corr_matrix=CREDIT_MC_CORR_MATRIX)

def make_cross2_mc_module(*, submodule: str | None = None, is_zeroize: bool = True,
                          nonlife_mc_k: float = 1.0) -> RiskTree:
    tree = RiskTree(root="C-ROSS MC")

    # (root)
    tree.grow("", children=("Life", "Non-life", "Market", "Credit"), agg_func=_overall_risk_agg)

    # Life
    tree.grow("Life", children=("Loss", "Expense", "Lapse"), agg_func=_life_risk_agg)

    # Life/Loss
    tree.grow("Life/Loss", children=("Mortality", "Catastrophe", "Longevity", "Morbidity", "Health", "Other"),
              agg_func=_loss_risk_agg)

    # Life/Loss/Morbidity
    tree.grow("Life/Loss/Morbidity", children=("Incidence", "Trend"), agg_func=_morb_risk_agg)

    # Life/Lapse
    tree.grow("Life/Lapse", children=("Lapse Rate", "Mass Lapse"), agg_func=_max_at_zero)

    # Life/Lapse/Lapse Rate
    tree.grow("Life/Lapse/Lapse Rate", children=("Lapse Up", "Lapse Down"), agg_func=_max_at_zero)

    # Non-Life
    tree.grow("Non-life", children=("Premium Reserve", "Catastrophe"),
              agg_func=partial(_nonlife_risk_agg, k=nonlife_mc_k))

    # Market
    tree.grow("Market",
              children=("Interest Rate", "Equity", "Real Estate", "Overseas Fixed-income", "Overseas Equity", "Exchange Rate"),
              agg_func=_market_risk_agg)

    # Market/Interest Rate
    tree.grow("Market/Interest Rate", children=("Interest Rate Up", "Interest Rate Down"), agg_func=_max_at_zero)

    # Credit
    tree.grow("Credit", children=("Spread", "Counterparty Default"), agg_func=_credit_risk_agg)

    tree.lock_structure()
    if is_zeroize:
        tree.zeroize()

    if submodule is None:
        return tree
    else:
        return tree.get_subtree(submodule).deepcopy()


@time_synchronized
class MinCapUnit:
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`
    
    def __init__(
        self,
        *,
        model_engine: ProjModelEngine = None,
        risk_module: RiskTree | None = None,
        require_loss_absorbency: bool,
        tdv_owner: str,
    ):
        self.require_loss_absorbency: bool = require_loss_absorbency
        self._risk_module: RiskTree = risk_module or make_cross2_mc_module()
        self._la_pv_base: float = 0.0
        self._la_pv_lower_limit = 0.0
        self._loss_absorbency: float = 0.0
        self._last_calculate: int | None = None

        create_tdv = lambda varname: TDimVariable(varname, model_engine=model_engine, owner=tdv_owner, group='CROSS_MC')
        self.tdv_min_cap: TDimVariable = create_tdv("minimum_capital")
        self.tdv_life_mc: TDimVariable = create_tdv("life_mc")
        self.tdv_nonlife_mc: TDimVariable = create_tdv("nonlife_mc")
        self.tdv_market_mc: TDimVariable = create_tdv("market_mc")
        self.tdv_credit_mc: TDimVariable = create_tdv("credit_mc")
        self.tdv_divers: TDimVariable = create_tdv("diversification")
        self.tdv_loss_absorb: TDimVariable = create_tdv("loss_absorbency")

    def calculate(self, *, risk_capital_dict: dict[str, float], la_pv_base: float = 0.0, la_pv_lower_limit = 0.0) -> None:
        self._risk_module.batch_set_risk_capital(risk_capital_dict)

        if self.require_loss_absorbency:
            self._la_pv_base = la_pv_base
            self._la_pv_lower_limit = la_pv_lower_limit
            self._loss_absorbency = calculate_loss_absorbency(
                mc_market=self._risk_module.get_risk_capital("Market"),
                mc_credit=self._risk_module.get_risk_capital("Credit"),
                pv_base=self._la_pv_base,
                pv_lower_limit=self._la_pv_lower_limit
            )
        else:
            self._loss_absorbency = 0.0

        t = self.time
        self.tdv_life_mc[t] = self._risk_module.get_risk_capital("Life")
        self.tdv_nonlife_mc[t] = self._risk_module.get_risk_capital("Non-life")
        self.tdv_market_mc[t] = self._risk_module.get_risk_capital("Market")
        self.tdv_credit_mc[t] = self._risk_module.get_risk_capital("Credit")
        self.tdv_divers[t] = self._risk_module.get_risk_diversification()
        self.tdv_loss_absorb[t] = self._loss_absorbency
        self.tdv_min_cap[t] = self._risk_module.get_risk_capital() - self._loss_absorbency
        self._last_calculate = t

    @property
    def risk_module(self) -> RiskTree:
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

    def get_risk_capital(self, path: str | None = None, /) -> float:
        return self._risk_module.get_risk_capital(path)

    def get_risk_diversification(self, path: str | None = None, /) -> float:
        return self._risk_module.get_risk_diversification(path)


@time_synchronized
class MinCapConsolidator:
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`
    
    def __init__(
        self,
        *,
        model_engine: ProjModelEngine = None,
        min_cap_units: list[MinCapUnit] | tuple[MinCapUnit],
        consolidated_risk_module: RiskTree | None = None,
        tdv_owner: str,
    ):
        self._units: tuple[MinCapUnit] = tuple(min_cap_units)
        self._risk_module: RiskTree = consolidated_risk_module or make_cross2_mc_module()
        self._loss_absorbency: float = 0.0
        self._min_cap: float = 0.0
        self._last_calculate: int | None = None

        create_tdv = lambda varname: TDimVariable(varname, model_engine=model_engine, owner=tdv_owner, group='CROSS_MC')
        self.tdv_min_cap: TDimVariable = create_tdv("minimum_capital")
        self.tdv_life_mc: TDimVariable = create_tdv("life_mc")
        self.tdv_nonlife_mc: TDimVariable = create_tdv("nonlife_mc")
        self.tdv_market_mc: TDimVariable = create_tdv("market_mc")
        self.tdv_credit_mc: TDimVariable = create_tdv("credit_mc")
        self.tdv_divers: TDimVariable = create_tdv("diversification")
        self.tdv_loss_absorb: TDimVariable = create_tdv("loss_absorbency")

    def consolidate(self) -> None:
        for node in self._risk_module.get_leaf_nodes():
            node.set_risk_capital(sum(unit.get_risk_capital(node.path) for unit in self._units))

        self._loss_absorbency = self._calculate_loss_absorbency()

        t = self.time
        self.tdv_life_mc[t] = self._risk_module.get_risk_capital("Life")
        self.tdv_nonlife_mc[t] = self._risk_module.get_risk_capital("Non-life")
        self.tdv_market_mc[t] = self._risk_module.get_risk_capital("Market")
        self.tdv_credit_mc[t] = self._risk_module.get_risk_capital("Credit")
        self.tdv_divers[t] = self._risk_module.get_risk_diversification()
        self.tdv_loss_absorb[t] = self._loss_absorbency
        self.tdv_min_cap[t] = self._risk_module.get_risk_capital() - self._loss_absorbency
        self._last_calculate = t

    def _calculate_loss_absorbency(self) -> float:
        units = [unit for unit in self._units if unit.require_loss_absorbency]
        if len(units) == 0:
            return 0.0

        la_risk_module = make_cross2_mc_module()
        for node in (la_risk_module.get_subtree("Market").get_leaf_nodes() +
                     la_risk_module.get_subtree("Credit").get_leaf_nodes()):
            node.set_risk_capital(sum(unit.get_risk_capital(node.path) for unit in units))

        return calculate_loss_absorbency(
            mc_market=la_risk_module.get_risk_capital("Market"),
            mc_credit=la_risk_module.get_risk_capital("Credit"),
            pv_base=sum(unit.la_pv_base for unit in units),
            pv_lower_limit=sum(unit.la_pv_lower_limit for unit in units)
        )

    @property
    def minimum_capital(self) -> float:
        maybe_raise_if_ne(self._last_calculate, self.time)
        return self._risk_module.get_risk_capital() -  self._loss_absorbency

    @property
    def loss_absorbency(self) -> float:
        maybe_raise_if_ne(self._last_calculate, self.time)
        return self._loss_absorbency

    def get_risk_capital(self, path: str | None = None, /) -> float:
        return self._risk_module.get_risk_capital(path)

    def get_risk_diversification(self, path: str | None = None, /) -> float:
        return self._risk_module.get_risk_diversification(path)

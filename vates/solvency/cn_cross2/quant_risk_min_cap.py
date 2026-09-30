import pandas as pd

from vates._core import ProjModelEngine, time_synchronized, TDimVariable
from vates.utils import maybe_raise_if_ne
from vates.solvency.risk_tree import RiskNode, RiskTree, risk_aggregation
from vates.solvency.cn_cross2.rules import (
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

def make_cross2_risk_module(name: str = "C-ROSS", /, is_zeroize: bool = True) -> RiskTree:
    tree = RiskTree(name=name)

    # (root)
    node = tree.root
    node.attach_sub_risk(RiskNode("Life"), RiskNode("Non-life"), RiskNode("Market"), RiskNode("Credit"))
    node.set_agg_func(_overall_risk_agg)

    # Life
    node = tree.select_node("Life")
    node.attach_sub_risk(RiskNode("Loss"), RiskNode("Expense"), RiskNode("Lapse"))
    node.set_agg_func(_life_risk_agg)

    # Life/Loss
    node = tree.select_node("Life/Loss")
    node.attach_sub_risk(RiskNode("Mortality"), RiskNode("Catastrophe"), RiskNode("Longevity"),
                         RiskNode("Morbidity"), RiskNode("Health & Medical", identifier="health"),
                         RiskNode("Other", identifier="other_loss"))
    node.set_agg_func(_loss_risk_agg)

    # Life/Loss/Morbidity
    node = tree.select_node("Life/Loss/Morbidity")
    node.attach_sub_risk(RiskNode("Incidence", identifier="morb_incidence"), RiskNode("Trend", identifier="morb_trend"))
    node.set_agg_func(_morb_risk_agg)

    # Life/Lapse
    node = tree.select_node("Life/Lapse")
    node.attach_sub_risk(RiskNode("Lapse Rate"), RiskNode("Mass Lapse"))
    node.set_agg_func(lambda lapse_rate, mass_lapse: max(lapse_rate, mass_lapse, 0))

    # Life/Lapse/Lapse Rate
    node = tree.select_node("Life/Lapse/Lapse Rate")
    node.attach_sub_risk(RiskNode("Lapse_Up"), RiskNode("Lapse_Down"))
    node.set_agg_func(lambda lapse_up, lapse_down: max(lapse_up, lapse_down, 0))

    # Market
    node = tree.select_node("Market")
    node.attach_sub_risk(RiskNode("Interest Rate"), RiskNode("Equity"), RiskNode("Real Estate"),
                         RiskNode("Overseas Fixed-income"), RiskNode("Overseas Equity"), RiskNode("Exchange Rate"))
    node.set_agg_func(_market_risk_agg)

    # Market/Interest Rate
    node = tree.select_node("Market/Interest Rate")
    node.attach_sub_risk(RiskNode("Interest Rate Up"), RiskNode("Interest Rate Down"))
    node.set_agg_func(lambda interest_rate_up, interest_rate_down: max(interest_rate_up, interest_rate_down, 0))

    # Credit
    node = tree.select_node("Credit")
    node.attach_sub_risk(RiskNode("Spread"), RiskNode("Counterparty Default"))
    node.set_agg_func(_credit_risk_agg)

    if is_zeroize:
        tree.zeroize()

    return tree


@time_synchronized
class MinCapUnit:
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`
    
    def __init__(
        self,
        name: str,
        *,
        model_engine: ProjModelEngine = None,
        risk_module: RiskTree | None = None,
        require_loss_absorbency: bool,
    ):
        self.name: str = name
        self.require_loss_absorbency: bool = require_loss_absorbency
        self._risk_module: RiskTree = risk_module or make_cross2_risk_module(name)
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

    def calculate(self, *, risk_capital_dict: dict[str, float], la_pv_base: float = 0.0, la_pv_lower_limit = 0.0) -> None:
        for key, val in risk_capital_dict.items():
            self._risk_module.set_risk_capital(key, val)

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
        name: str,
        *,
        model_engine: ProjModelEngine = None,
        min_cap_units: list[MinCapUnit] | tuple[MinCapUnit],
        consolidated_risk_module: RiskTree | None = None,
    ):
        self.name: str = name
        self._units: tuple[MinCapUnit] = tuple(min_cap_units)
        self._risk_module: RiskTree = consolidated_risk_module or make_cross2_risk_module(self.name)
        self._validate_risk_module_structure()
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

    def _validate_risk_module_structure(self) -> None:
        if len(self._units) == 0:
            raise ValueError(f"Nothing to consolidate.")
        ref_path_set: set[str] = set([node.path for node in self._risk_module.preorder_traversal()])
        for item in self._units:
            path_set = set([node.path for node in item.risk_module.preorder_traversal()])
            if len(path_set - ref_path_set) > 0:
                raise ValueError(f"Can't consolidate '{item.risk_module.name}': structures are differenct.")

    def consolidate(self) -> None:
        for node in self._risk_module.list_leaf_nodes():
            path = node.path
            risk_capital = sum(unit.get_risk_capital(path) for unit in self._units)
            node.set_risk_capital(risk_capital)

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
        la_risk_module = make_cross2_risk_module()
        market_credit_leaf_nodes = (la_risk_module.select_subtree("Market").list_leaf_nodes() +
                                    la_risk_module.select_subtree("Credit").list_leaf_nodes())

        la_pv_base = 0.0
        la_pv_lower_limit = 0.0

        for unit in self._units:
            if unit.require_loss_absorbency:
                for node in market_credit_leaf_nodes:
                    node.set_risk_capital(node.risk_capital + unit.get_risk_capital(node.path))
                la_pv_base += unit.la_pv_base
                la_pv_lower_limit += unit.la_pv_lower_limit

        return calculate_loss_absorbency(
            mc_market=la_risk_module.get_risk_capital("Market"),
            mc_credit=la_risk_module.get_risk_capital("Credit"),
            pv_base=la_pv_base,
            pv_lower_limit=la_pv_lower_limit
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

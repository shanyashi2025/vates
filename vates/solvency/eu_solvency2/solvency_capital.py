import math

from vates.solvency.risk_tree import RiskNode, RiskTree, risk_aggregation
from vates.solvency.eu_solvency2.rules import (
    CORR_MATRIX_SCR,
    CORR_MATRIX_NONLIFE,
    CORR_MATRIX_LIFE,
    CORR_MATRIX_HEALTH,
    CORR_MATRIX_SLTH,
    CORR_MATRIX_MARKET,
)


def _max_at_zero(**capitals: float) -> float:
    return max(max(capitals.values()), 0.0)

def _floor_at_zero(*args: float) -> tuple[float, ...]:
    return tuple([max(x, 0.0) for x in args])

def _scr_agg(market: float, counterparty_default: float, life: float, health: float, non_life: float,
             intangibles: float) -> float:
    return risk_aggregation(market, counterparty_default, life, health, non_life,
                            corr_matrix=CORR_MATRIX_SCR) + intangibles

def _nonlife_risk_agg(premium_reserve: float, catastrophe: float, lapse: float) -> float:
    return risk_aggregation(premium_reserve, catastrophe, lapse, corr_matrix=CORR_MATRIX_NONLIFE)

def _life_risk_agg(mortality: float, longevity: float, disability: float, expense: float, revision: float, lapse: float,
                   catastrophe: float) -> float:
    return risk_aggregation(*_floor_at_zero(mortality, longevity, disability, expense, revision, lapse, catastrophe),
                            corr_matrix=CORR_MATRIX_LIFE)

def _health_risk_agg(nslt: float, slt: float, catastrophe: float) -> float:
    return risk_aggregation(nslt, slt, catastrophe, corr_matrix=CORR_MATRIX_HEALTH)

def _slth_risk_agg(mortality: float, longevity: float, disability_morbidity: float, expense: float, revision: float,
                   lapse: float) -> float:
    return risk_aggregation(*_floor_at_zero(mortality, longevity, disability_morbidity, expense, revision, lapse),
                            corr_matrix=CORR_MATRIX_SLTH)

def _market_risk_agg(interest_rate_increase: float, interest_rate_decrease: float, equity: float, property_: float,
                     spread: float, concentration: float, currency: float, **kwargs) -> float:
    if interest_rate_increase > interest_rate_decrease:
        interest_rate = interest_rate_increase
        corr_matrix = CORR_MATRIX_MARKET("increase")
    else:
        interest_rate = interest_rate_decrease
        corr_matrix = CORR_MATRIX_MARKET("decrease")
    return risk_aggregation(max(interest_rate, 0.0), equity, property_, spread, concentration, currency,
                            corr_matrix=corr_matrix)

def _counterparty_default_risk_agg(type_1: float, type_2: float) -> float:
    return math.sqrt(type_1 ** 2 + 1.5 * type_1 * type_2 + type_2 ** 2)


def make_solvency2_scr_module(*, submodule: str | None = None, is_zeroize: bool = True
                              ) -> RiskTree:
    tree = RiskTree(root="Solvency II SCR")

    # (root)
    tree.grow("", children=("Market", "Counterparty Default", "Life", "Health", "Non-life", "Intangibles"),
              agg_func=_scr_agg)

    # Non-life
    tree.grow("Non-life", children=("Premium Reserve", "Catastrophe", "Lapse"), agg_func=_nonlife_risk_agg)

    # Life
    tree.grow("Life",
              children=("Mortality", "Longevity", "Disability", "Expense", "Revision", "Lapse", "Catastrophe"),
              agg_func=_life_risk_agg)

    # Life/Lapse
    tree.grow("Life/Lapse", children=("Increase", "Decrease", "Mass"), agg_func=_max_at_zero)

    # Health
    tree.grow("Health", children=("NSLT", "SLT", "Catastrophe"), agg_func=_health_risk_agg)

    # Health/SLT
    tree.grow("Health/SLT",
              children=("Mortality", "Longevity", "Disability-Morbidity", "Expense", "Revision", "Lapse"),
              agg_func=_slth_risk_agg)

    # Health/SLT/Disability-Morbidity
    tree.grow("Health/SLT/Disability-Morbidity",
              children=("Medical Payment Increase", "Medical Payment Decrease", "Income Protection"),
              agg_func=lambda medical_payment_increase, medical_payment_decrease, income_protection: max(
                  medical_payment_increase, medical_payment_decrease) + income_protection)

    # Health/SLT/Lapse
    tree.grow("Health/SLT/Lapse", children=("Increase", "Decrease", "Mass"), agg_func=_max_at_zero)

    # Market
    tree.grow("Market",
              children=("Interest Rate", "Equity", RiskNode("Property", identifier="property_"), "Spread",
                        "Concentration", "Currency"),
              agg_func=_market_risk_agg, agg_scope="descendants")

    tree.grow("Market/Interest Rate", children=("Interest Rate Increase", "Interest Rate Decrease"),
              agg_func=_max_at_zero)

    # Counterparty Default
    tree.grow("Counterparty Default", children=("Type 1", "Type 2"), agg_func=_counterparty_default_risk_agg)

    if is_zeroize:
        tree.zeroize()

    if submodule is None:
        return tree
    else:
        return tree.get_subtree(submodule).deepcopy()

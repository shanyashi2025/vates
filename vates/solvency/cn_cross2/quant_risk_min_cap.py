from dataclasses import dataclass
import pandas as pd
import warnings

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDimVariable
from vates.utils import RiskModule, SubRisk, NumVarGroup
from vates.solvency.cn_cross2.params import (
    AccountType,
    MC_CORR_MATRIX,
    MORB_MC_CORR_MATRIX,
    LOSS_MC_CORR_MATRIX,
    LIFE_MC_CORR_MATRIX,
    MARKET_MC_CORR_MATRIX,
    CREDIT_MC_CORR_MATRIX,
    calculate_loss_absorbency,
)


@dataclass
class MinCapInputer(NumVarGroup):
    pv_base: float = 0.0
    pv_mortality: float = 0.0
    pv_catastrophe: float = 0.0
    pv_longevity: float = 0.0
    pv_morb_incidence: float = 0.0
    pv_morb_trend: float = 0.0
    pv_health: float = 0.0
    pv_other_loss: float = 0.0
    pv_expense: float = 0.0
    pv_lapse_up: float = 0.0
    pv_lapse_dn: float = 0.0
    pv_lapse_mass: float = 0.0
    pv_int_base: float = 0.0
    pv_int_up: float = 0.0
    pv_int_dn: float = 0.0
    pv_la_lower_limit: float = 0.0
    aa_int_base: float = 0.0
    aa_int_up: float = 0.0
    aa_int_dn: float = 0.0
    mc_non_life: float = 0.0
    mc_equity: float = 0.0
    mc_real_estate: float = 0.0
    mc_overseas_fixed_income: float = 0.0
    mc_overseas_equity: float = 0.0
    mc_exchange_rate: float = 0.0
    mc_spread: float = 0.0
    mc_counterparty_default: float = 0.0


class MinCapCalculator:
    def __init__(self, name: str):
        self.name: str = name
        # Set up C-ROSS risk hierarchy
        self._mort: SubRisk = SubRisk(f'{name}:mortality')  # mortality
        self._cata: SubRisk = SubRisk(f'{name}:catastrophe')  # catastrophe
        self._lgvt: SubRisk = SubRisk(f'{name}:longevity')  # longevity
        self._morb_inc: SubRisk = SubRisk(f'{name}:morbidity_incidence')  # morbidity incidence
        self._morb_trd: SubRisk = SubRisk(f'{name}:morbidity_trend')  # morbidity trend
        self._morb: RiskModule = RiskModule(
            name=f'{name}:morbidity',
            sub_risk_list=[self._morb_inc, self._morb_trd],
            corr_matrix=MORB_MC_CORR_MATRIX
        )
        self._hlth: SubRisk = SubRisk(f'{name}:health_medical')  # health and medical
        self._othl: SubRisk = SubRisk(f'{name}:other_loss')  # other loss
        self._loss: RiskModule = RiskModule(
            name=f'{name}:loss',
            sub_risk_list=[self._mort, self._cata, self._lgvt, self._morb, self._hlth, self._othl],
            corr_matrix=LOSS_MC_CORR_MATRIX
        )
        self._expn: SubRisk = SubRisk(f'{name}:expense')  # expense
        self._laps: SubRisk = SubRisk(f'{name}:lapse')  # lapse
        self._life: RiskModule = RiskModule(
            name=f'{name}:life',
            sub_risk_list=[self._loss, self._expn, self._laps],
            corr_matrix=LIFE_MC_CORR_MATRIX
        )
        self._nolf: SubRisk = SubRisk(f'{name}:non_life')  # non-life
        self._intr: SubRisk = SubRisk(f'{name}:interest_rate')  # interest rate
        self._eqty: SubRisk = SubRisk(f'{name}:equity')  # equity price
        self._rles: SubRisk = SubRisk(f'{name}:real_estate')  # real estate
        self._osfi: SubRisk = SubRisk(f'{name}:overseas_fixed_income')  # overseas fixed-income
        self._oseq: SubRisk = SubRisk(f'{name}:overseas_equity')  # overseas equity
        self._frex: SubRisk = SubRisk(f'{name}:exchange_rate')  # exchange rate
        self._mrkt: RiskModule = RiskModule(
            name=f'{name}:market',
            sub_risk_list=[self._intr, self._eqty, self._rles, self._osfi, self._oseq, self._frex],
            corr_matrix=MARKET_MC_CORR_MATRIX
        )
        self._sprd: SubRisk = SubRisk(f'{name}:spread')  # spread
        self._cpdf: SubRisk = SubRisk(f'{name}:counterparty_default')  # counterparty default
        self._cred: RiskModule = RiskModule(
            name=f'{name}:credit',
            sub_risk_list=[self._sprd, self._cpdf],
            corr_matrix=CREDIT_MC_CORR_MATRIX
        )
        self._overall: RiskModule = RiskModule(
            name=f'{name}:mincap',
            sub_risk_list=[self._life, self._nolf, self._mrkt, self._cred],
            corr_matrix=MC_CORR_MATRIX
        )

        self._min_cap_calc_input: MinCapInputer = MinCapInputer()

    def calculate_minimum_capital(self, mc_in: MinCapInputer) -> None:
        # --- (1) calculate risk charge for each sub-risk ---
        self._mort.risk_charge = max(mc_in.pv_mortality - mc_in.pv_base, 0)
        self._cata.risk_charge = max(mc_in.pv_catastrophe - mc_in.pv_base, 0)
        self._lgvt.risk_charge = max(mc_in.pv_longevity - mc_in.pv_base, 0)
        self._morb_inc.risk_charge = max(mc_in.pv_morb_incidence - mc_in.pv_base, 0)
        self._morb_trd.risk_charge = max(mc_in.pv_morb_trend - mc_in.pv_base, 0)
        self._hlth.risk_charge = max(mc_in.pv_health - mc_in.pv_base, 0)
        self._othl.risk_charge = max(mc_in.pv_other_loss - mc_in.pv_base, 0)
        self._expn.risk_charge = max(mc_in.pv_expense - mc_in.pv_base, 0)
        self._laps.risk_charge = max(max(mc_in.pv_lapse_up, mc_in.pv_lapse_dn, mc_in.pv_lapse_mass)
                                     - mc_in.pv_base, 0)
        self._nolf.risk_charge = mc_in.mc_non_life
        self._intr.risk_charge = max(max(mc_in.pv_int_up - mc_in.aa_int_up,
                                         mc_in.pv_int_dn - mc_in.aa_int_dn)
                                     - (mc_in.pv_int_base - mc_in.aa_int_base), 0)
        self._eqty.risk_charge = mc_in.mc_equity
        self._rles.risk_charge = mc_in.mc_real_estate
        self._osfi.risk_charge = mc_in.mc_overseas_fixed_income
        self._oseq.risk_charge = mc_in.mc_overseas_equity
        self._frex.risk_charge = mc_in.mc_exchange_rate
        self._sprd.risk_charge = mc_in.mc_spread
        self._cpdf.risk_charge = mc_in.mc_counterparty_default
        # --- (2) calculate risk charge (risk aggregation) for risk module and sub-risk modules ---
        self._overall.calculate_risk_charge(recalculate_sub_risk=True)
        # --- (3) store input to be retrieved by top level ---
        self._min_cap_calc_input = mc_in

    @property
    def minimum_capital(self) -> float:
        return self._overall.risk_charge

    @property
    def min_cap_calc_input(self) -> MinCapInputer:
        return self._min_cap_calc_input

    @property
    def overall_risk_module(self) -> dict[str, float]:
        return {
            'min_cap': self._overall.risk_charge,
            'life': self._life.risk_charge,
            'non_life': self._nolf.risk_charge,
            'market': self._mrkt.risk_charge,
            'credit': self._cred.risk_charge,
            'diversification': self._overall.diversification
        }

    @property
    def life_risk_module(self) -> dict[str, float]:
        return {
            'min_cap': self._life.risk_charge,
            'loss': self._loss.risk_charge,
            'expense': self._expn.risk_charge,
            'lapse': self._laps.risk_charge,
            'diversification': self._life.diversification
        }

    @property
    def loss_risk_module(self) -> dict[str, float]:
        return {
            'min_cap': self._loss.risk_charge,
            'mortality': self._mort.risk_charge,
            'catastrophe': self._cata.risk_charge,
            'longevity': self._lgvt.risk_charge,
            'morbidity': self._morb.risk_charge,
            'health': self._hlth.risk_charge,
            'other_loss': self._othl.risk_charge,
            'diversification': self._loss.diversification
        }

    @property
    def loss_morb_module(self) -> dict[str, float]:
        return {
            'min_cap': self._morb.risk_charge,
            'morbidity_incidence': self._morb_inc.risk_charge,
            'morbidity_trend': self._morb_trd.risk_charge,
            'diversification': self._morb.diversification
        }

    @property
    def market_risk_module(self) -> dict[str, float]:
        return {
            'min_cap': self._mrkt.risk_charge,
            'interest_rate': self._intr.risk_charge,
            'equity': self._eqty.risk_charge,
            'real_estate': self._rles.risk_charge,
            'overseas_fixed_income': self._osfi.risk_charge,
            'overseas_equity': self._oseq.risk_charge,
            'exchange_rate': self._frex.risk_charge,
            'diversification': self._mrkt.diversification
        }

    @property
    def credit_risk_module(self) -> dict[str, float]:
        return {
            'min_cap': self._cred.risk_charge,
            'spread': self._sprd.risk_charge,
            'counterparty_default': self._cpdf.risk_charge,
            'diversification': self._cred.diversification
        }

    def __call__(self, *args, **kwargs):
        self.calculate_minimum_capital(*args, **kwargs)

@add_projection_time_synchronizer
class MinCapUnit:
    time: int           # for type hint only, will be injected by decorator `has_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `has_time_synchronizer`
    
    def __init__(
        self,
        *,
        name: str = "untitled",
        model_engine: ProjModelEngine | None = None,
        account_type: AccountType,
    ):
        self.name: str = name
        self._account_type: AccountType = account_type

        self._loss_absorbency: float = 0.0
        self._min_cap: float = 0.0
        self._min_cap_calculator: MinCapCalculator = MinCapCalculator(name)
        self._last_mc_calc: pd.Period | None = None

        create_tdv = lambda varname: TDimVariable(varname, model_engine=model_engine, owner=name, group='CROSS_MC')
        self.tdv_min_cap: TDimVariable = create_tdv("minimum_capital")
        self.tdv_life_mc: TDimVariable = create_tdv("life_mc")
        self.tdv_nonlife_mc: TDimVariable = create_tdv("nonlife_mc")
        self.tdv_market_mc: TDimVariable = create_tdv("market_mc")
        self.tdv_credit_mc: TDimVariable = create_tdv("credit_mc")
        self.tdv_divers: TDimVariable = create_tdv("diversification")
        self.tdv_loss_absorb: TDimVariable = create_tdv("loss_absorbency")

    def calculate_minimum_capital(self, mc_in: MinCapInputer) -> None:
        t = self.time
        self._min_cap_calculator(mc_in)
        if self._account_type in (AccountType.PAR, AccountType.UNIV):
            self._loss_absorbency = calculate_loss_absorbency(
                mc_market=self._min_cap_calculator.overall_risk_module['market'],
                mc_credit=self._min_cap_calculator.overall_risk_module['credit'],
                pv_base=mc_in.pv_base,
                pv_lower_limit=mc_in.pv_la_lower_limit
            )
        self._min_cap = self._min_cap_calculator.overall_risk_module['min_cap'] - self._loss_absorbency

        self.tdv_min_cap[t] = self._min_cap
        self.tdv_life_mc[t] = self._min_cap_calculator.overall_risk_module['life']
        self.tdv_nonlife_mc[t] = self._min_cap_calculator.overall_risk_module['non_life']
        self.tdv_market_mc[t] = self._min_cap_calculator.overall_risk_module['market']
        self.tdv_credit_mc[t] = self._min_cap_calculator.overall_risk_module['credit']
        self.tdv_divers[t] = self._min_cap_calculator.overall_risk_module['diversification']
        self.tdv_loss_absorb[t] = self._loss_absorbency

        self._last_mc_calc = self.period

    @property
    def account_tpye(self) -> AccountType:
        return self._account_type

    @property
    def minimum_capital(self) -> float:
        return self._min_cap

    @property
    def loss_absorbency(self) -> float:
        return self._loss_absorbency

    @property
    def min_cap_calculator(self) -> MinCapCalculator:
        return self._min_cap_calculator

    @property
    def min_cap_dict(self) -> dict[str, float]:
        return {
            'min_cap': self._min_cap,
            'life': self._min_cap_calculator.overall_risk_module['life'],
            'non_life': self._min_cap_calculator.overall_risk_module['non-life'],
            'market': self._min_cap_calculator.overall_risk_module['market'],
            'credit': self._min_cap_calculator.overall_risk_module['credit'],
            'diversification': self._min_cap_calculator.overall_risk_module['diversification'],
            'loss_absorbency': self._loss_absorbency,
        }

    @property
    def last_min_cap_calc(self) -> pd.Period | None:
        return self._last_mc_calc


@add_projection_time_synchronizer
class MinCapConsolidator:
    time: int           # for type hint only, will be injected by decorator `has_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `has_time_synchronizer`
    
    def __init__(
        self,
        *,
        name: str = 'untitled',
        model_engine: ProjModelEngine | None = None,
        bus_unit_list: list[MinCapUnit]
    ):
        self.name: str = name
        self._bus_unit_list: list[MinCapUnit] = bus_unit_list

        self._loss_absorbency: float = 0.0
        self._min_cap: float = 0.0
        self._min_cap_calculator: MinCapCalculator = MinCapCalculator(name)
        self._min_cap_calculator_la: MinCapCalculator = MinCapCalculator(f'{name}:loss_absorb')
        self._last_mc_calc: pd.Period | None = None

        create_tdv = lambda varname: TDimVariable(varname, model_engine=model_engine, owner=name, group='CROSS_MC')
        self.tdv_min_cap: TDimVariable = create_tdv("minimum_capital")
        self.tdv_life_mc: TDimVariable = create_tdv("life_mc")
        self.tdv_nonlife_mc: TDimVariable = create_tdv("nonlife_mc")
        self.tdv_market_mc: TDimVariable = create_tdv("market_mc")
        self.tdv_credit_mc: TDimVariable = create_tdv("credit_mc")
        self.tdv_divers: TDimVariable = create_tdv("diversification")
        self.tdv_loss_absorb: TDimVariable = create_tdv("loss_absorbency")

    def calculate_minimum_capital(self) -> None:
        t, p = self.time, self.period
        mc_in, mc_in_la = MinCapInputer(), MinCapInputer()

        for unit in self._bus_unit_list:
            if unit.last_min_cap_calc != p:
                warnings.warn(f'{p} {self.name}: min cap result might be incorrect, because min cap is last calculated '
                              f'on {unit.last_min_cap_calc} for {unit.name}.')
            mc_in += unit.min_cap_calculator.min_cap_calc_input
            if unit.account_tpye in (AccountType.PAR, AccountType.UNIV):
                mc_in_la += unit.min_cap_calculator.min_cap_calc_input

        self._min_cap_calculator(mc_in)
        self._min_cap_calculator_la(mc_in_la)

        self._loss_absorbency = calculate_loss_absorbency(
            mc_market=self._min_cap_calculator_la.overall_risk_module['market'],
            mc_credit=self._min_cap_calculator_la.overall_risk_module['credit'],
            pv_base=mc_in_la.pv_base,
            pv_lower_limit=mc_in_la.pv_la_lower_limit
        )

        self._min_cap = self._min_cap_calculator.overall_risk_module['min_cap'] - self._loss_absorbency

        self.tdv_min_cap[t] = self._min_cap
        self.tdv_life_mc[t] = self._min_cap_calculator.overall_risk_module['life']
        self.tdv_nonlife_mc[t] = self._min_cap_calculator.overall_risk_module['non_life']
        self.tdv_market_mc[t] = self._min_cap_calculator.overall_risk_module['market']
        self.tdv_credit_mc[t] = self._min_cap_calculator.overall_risk_module['credit']
        self.tdv_divers[t] = self._min_cap_calculator.overall_risk_module['diversification']
        self.tdv_loss_absorb[t] = self._loss_absorbency

        self._last_mc_calc = self.period

    @property
    def minimum_capital(self) -> float:
        return self._min_cap

    @property
    def loss_absorbency(self) -> float:
        return self._loss_absorbency

    @property
    def min_cap_calculator(self) -> MinCapCalculator:
        return self._min_cap_calculator

    @property
    def min_cap_dict(self) -> dict[str, float]:
        return {
            'min_cap': self._min_cap,
            'life': self._min_cap_calculator.overall_risk_module['life'],
            'non_life': self._min_cap_calculator.overall_risk_module['non-life'],
            'market': self._min_cap_calculator.overall_risk_module['market'],
            'credit': self._min_cap_calculator.overall_risk_module['credit'],
            'diversification': self._min_cap_calculator.overall_risk_module['diversification'],
            'loss_absorbency': self._loss_absorbency,
        }

    @property
    def last_min_cap_calc(self) -> pd.Period | None:
        return self._last_mc_calc

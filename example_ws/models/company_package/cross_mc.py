import pandas as pd
from vates import ProjModelEngine, time_synchronized, make_proj_variable, TDimVariable
from vates.solvency import RiskTree, cn_cross2


@time_synchronized("model_engine")
class MinCapUnit:
    time: int  # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period  # for type hint only, will be injected by decorator `time_synchronized`

    def __init__(
            self,
            *,
            model_engine: ProjModelEngine | None = None,
            risk_module: RiskTree | None = None,
            require_loss_absorbency: bool,
            tdv_owner: str,
    ):
        self.require_loss_absorbency: bool = require_loss_absorbency
        self._risk_module: RiskTree = risk_module or cn_cross2.make_mc_module()
        self._la_pv_base: float = 0.0
        self._la_pv_lower_limit = 0.0
        self._loss_absorbency: float = 0.0
        self._last_calculate: int | None = None

        create_tdv = lambda varname: make_proj_variable(varname, model_engine=model_engine, owner=tdv_owner,
                                                        group='CROSS_MC')
        self.tdv_min_cap: TDimVariable = create_tdv("minimum_capital")
        self.tdv_life_mc: TDimVariable = create_tdv("life_mc")
        self.tdv_nonlife_mc: TDimVariable = create_tdv("nonlife_mc")
        self.tdv_market_mc: TDimVariable = create_tdv("market_mc")
        self.tdv_credit_mc: TDimVariable = create_tdv("credit_mc")
        self.tdv_divers: TDimVariable = create_tdv("diversification")
        self.tdv_loss_absorb: TDimVariable = create_tdv("loss_absorbency")

    def calculate(self, *, leaf_values: dict[str, float], la_pv_base: float = 0.0, la_pv_lower_limit=0.0) -> None:
        self._risk_module.set_capitals(leaf_values)

        if self.require_loss_absorbency:
            self._la_pv_base = la_pv_base
            self._la_pv_lower_limit = la_pv_lower_limit
            self._loss_absorbency = cn_cross2.calculate_loss_absorbency(
                mc_market=self._risk_module.get_capital("Market"),
                mc_credit=self._risk_module.get_capital("Credit"),
                pv_base=self._la_pv_base,
                pv_lower_limit=self._la_pv_lower_limit
            )
        else:
            self._loss_absorbency = 0.0

        t = self.time
        self.tdv_life_mc[t] = self._risk_module.get_capital("Life")
        self.tdv_nonlife_mc[t] = self._risk_module.get_capital("Non-life")
        self.tdv_market_mc[t] = self._risk_module.get_capital("Market")
        self.tdv_credit_mc[t] = self._risk_module.get_capital("Credit")
        self.tdv_divers[t] = self._risk_module.get_diversification(sign="+")
        self.tdv_loss_absorb[t] = self._loss_absorbency
        self.tdv_min_cap[t] = self._risk_module.get_capital() - self._loss_absorbency
        self._last_calculate = t

    @property
    def risk_module(self) -> RiskTree:
        return self._risk_module

    @property
    def loss_absorbency(self) -> float:
        if not self.require_loss_absorbency:
            return 0.0
        return self._loss_absorbency

    @property
    def la_pv_base(self) -> float:
        if not self.require_loss_absorbency:
            return 0.0
        return self._la_pv_base

    @property
    def la_pv_lower_limit(self) -> float:
        if not self.require_loss_absorbency:
            return 0.0
        return self._la_pv_lower_limit

    def get_risk_capital(self, path: str = "", /) -> float:
        return self._risk_module.get_capital(path)

    def get_risk_diversification(self, path: str = "", /) -> float:
        return self._risk_module.get_diversification(path, sign="+")


@time_synchronized("model_engine")
class MinCapConsolidator:
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    def __init__(
            self,
            *,
            model_engine: ProjModelEngine | None = None,
            min_cap_units: list[MinCapUnit] | tuple[MinCapUnit],
            consolidated_risk_module: RiskTree | None = None,
            tdv_owner: str,
    ):
        self._units: tuple[MinCapUnit] = tuple(min_cap_units)
        self._risk_module: RiskTree = consolidated_risk_module or cn_cross2.make_mc_module()
        self._loss_absorbency: float = 0.0
        self._min_cap: float = 0.0
        self._last_calculate: int | None = None

        create_tdv = lambda varname: make_proj_variable(varname, model_engine=model_engine, owner=tdv_owner, group='CROSS_MC')
        self.tdv_min_cap: TDimVariable = create_tdv("minimum_capital")
        self.tdv_life_mc: TDimVariable = create_tdv("life_mc")
        self.tdv_nonlife_mc: TDimVariable = create_tdv("nonlife_mc")
        self.tdv_market_mc: TDimVariable = create_tdv("market_mc")
        self.tdv_credit_mc: TDimVariable = create_tdv("credit_mc")
        self.tdv_divers: TDimVariable = create_tdv("diversification")
        self.tdv_loss_absorb: TDimVariable = create_tdv("loss_absorbency")

    def consolidate(self) -> None:
        for node in self._risk_module.get_leaf_nodes():
            node.capital = sum(unit.get_risk_capital(node.path) for unit in self._units)

        self._loss_absorbency = self._calculate_loss_absorbency()

        t = self.time
        self.tdv_life_mc[t] = self._risk_module.get_capital("Life")
        self.tdv_nonlife_mc[t] = self._risk_module.get_capital("Non-life")
        self.tdv_market_mc[t] = self._risk_module.get_capital("Market")
        self.tdv_credit_mc[t] = self._risk_module.get_capital("Credit")
        self.tdv_divers[t] = self._risk_module.get_diversification(sign="+")
        self.tdv_loss_absorb[t] = self._loss_absorbency
        self.tdv_min_cap[t] = self._risk_module.get_capital() - self._loss_absorbency
        self._last_calculate = t

    def _calculate_loss_absorbency(self) -> float:
        units = [unit for unit in self._units if unit.require_loss_absorbency]
        if len(units) == 0:
            return 0.0

        la_risk_module = cn_cross2.make_mc_module()
        for node in (la_risk_module.get_subtree("Market").get_leaf_nodes() +
                     la_risk_module.get_subtree("Credit").get_leaf_nodes()):
            node.capital = sum(unit.get_risk_capital(node.path) for unit in units)

        return cn_cross2.calculate_loss_absorbency(
            mc_market=la_risk_module.get_capital("Market"),
            mc_credit=la_risk_module.get_capital("Credit"),
            pv_base=sum(unit.la_pv_base for unit in units),
            pv_lower_limit=sum(unit.la_pv_lower_limit for unit in units)
        )

    @property
    def minimum_capital(self) -> float:
        return self._risk_module.get_capital() -  self._loss_absorbency

    @property
    def loss_absorbency(self) -> float:
        return self._loss_absorbency

    def get_risk_capital(self, path: str = "", /) -> float:
        return self._risk_module.get_capital(path)

    def get_risk_diversification(self, path: str = "", /) -> float:
        return self._risk_module.get_diversification(path, sign="+")


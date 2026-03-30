import numpy as np
import math
from dataclasses import dataclass

def risk_aggregation(sub_risk_vector: np.ndarray, corr_matrix: np.ndarray, is_validate_args: bool=False) -> float:
    if is_validate_args:
        if type(sub_risk_vector) != np.ndarray:
            raise TypeError(f'Invalid {type(sub_risk_vector)=}, expected np.ndarray.')
        if type(corr_matrix) != np.ndarray:
            raise TypeError(f'Invalid {type(corr_matrix)=}, expected np.ndarray.')
        if corr_matrix.ndim != 2:
            raise ValueError(f'{corr_matrix.ndim=}, expected 2.')
        if corr_matrix.shape[0] != len(sub_risk_vector):
            raise ValueError(f'corr matrix shape {corr_matrix.shape} not consistent with sub risk {len(sub_risk_vector)}.')
    return math.sqrt(sub_risk_vector @ corr_matrix @ sub_risk_vector.T)

@dataclass
class SubRisk:
    name: str
    risk_charge: float = 0.0

class RiskModule:
    def __init__(self, name: str, sub_risk_list: list, corr_matrix: np.ndarray) -> None:
        if corr_matrix.ndim != 2:
            raise ValueError(f'{corr_matrix.ndim=}, expected 2.')
        if corr_matrix.shape[0] != len(sub_risk_list):
            raise ValueError(f'{corr_matrix.shape=} inconsistent with {len(sub_risk_list)=}.')
        self._name: str = name
        self._sub_risk_list: list[SubRisk | RiskModule] = sub_risk_list
        self._corr_matrix: np.ndarray = corr_matrix
        self._risk_charge: float = 0.0
        self._diversification: float = 0.0

    @property
    def risk_charge(self) -> float:
        return self._risk_charge

    @property
    def diversification(self) -> float:
        return self._diversification

    def calculate_risk_charge(self, recalculate_sub_risk: bool):
        sub_risk_vector = np.zeros(len(self._sub_risk_list))
        for i, risk in enumerate(self._sub_risk_list):
            if type(risk) == SubRisk:
                sub_risk_vector[i] = risk.risk_charge
            elif type(risk) == RiskModule:
                if recalculate_sub_risk:
                    risk.calculate_risk_charge(recalculate_sub_risk)
                sub_risk_vector[i] = risk.risk_charge
            else:
                raise ValueError(f"Risk module {self._name} sub-risk #{i + 1}: "
                                 f"invalid type {type(risk)}, expected 'SubRisk' or 'RiskModule'.")
        self._risk_charge = risk_aggregation(sub_risk_vector, self._corr_matrix)
        self._diversification = np.sum(sub_risk_vector) - self._risk_charge

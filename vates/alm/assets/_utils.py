import numpy as np
import numpy.typing as npt


def make_risk_adjusted_rates(rf_rates: npt.NDArray[np.float64],
                             additive_spread: float | npt.NDArray[np.float64],
                             proportional_spread: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Make risk-adjusted rates: `ra = rf * (1 + proportional) + additive`, where the
    total spread is `rf * proportional + additive`.

    Args:
        rf_rates (npt.NDArray[np.float64]): Risk-free rates.
        additive_spread (float, npt.NDArray[np.float64]): Additive spread.
        proportional_spread (float, npt.NDArray[np.float64]): Proportional spread, multiplied by risk-free rates.

    Returns:
        npt.NDArray[np.float64]: Risk-adjusted rates.
    """
    len_rate = len(rf_rates)

    def _align(val: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        if isinstance(val, float):
            return val
        len_arr = len(val)
        if len_arr == len_rate:
            return val
        elif len_arr > len_rate:
            return val[:len_rate]  # slicing
        else:  # len_arr < len_spot
            return np.pad(val, (0, len_rate - len_arr), mode='constant', constant_values=val[-1])  # padding

    return rf_rates * (1 + _align(proportional_spread)) + _align(additive_spread)

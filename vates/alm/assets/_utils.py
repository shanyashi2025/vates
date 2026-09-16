import numpy as np
import numpy.typing as npt

from vates.utils import maybe_check_state

def calculate_risk_adj_spot(rf_spots: npt.NDArray[np.float64], mult: float | npt.NDArray[np.float64],
                            add: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
    """
    Calculate risk-adjusted spot rates: `ra = rf * (1 + mult) + add`.

    Args:
        rf_spots (npt.NDArray[np.float64]): Risk-free spot rates.
        mult (float, npt.NDArray[np.float64]): Multiplicative adjustment factors.
        add (float, npt.NDArray[np.float64]): Additive spread adjustments.

    Returns:
        npt.NDArray[np.float64]: Risk-adjusted spot rates.
    """
    len_spot = len(rf_spots)

    def _align(val: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        if isinstance(val, float):
            return val
        len_arr = len(val)
        if len_arr == len_spot:
            return val
        elif len_arr > len_spot:
            return val[len_spot]  # slicing
        else:  # len_arr < len_spot
            return np.pad(val, (0, len_spot - len(val)), mode='constant', constant_values=val[-1])  # padding

    return rf_spots * (1 + _align(mult)) + _align(add)


def maybe_check_asset_state_roll(func):
    def wrapper(obj, *args, **kwargs):
        if obj._state != ("initialized", obj.time - 1):
            maybe_check_state(obj, ("closed", obj.time - 1))
        result = func(obj, *args, **kwargs)
        obj._state = ("rolled", obj.time)
        return result
    return wrapper

def maybe_check_asset_state_close(func):
    def wrapper(obj, *args, **kwargs):
        if obj._state != ("initialized", obj.time):
            maybe_check_state(obj, ("rolled", obj.time))
        result = func(obj, *args, **kwargs)
        obj._state = ("closed", obj.time)
        return result
    return wrapper

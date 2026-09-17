import numpy as np
import warnings

from vates.global_conf import STRICTNESS_LEVEL,StrictnessLevel

def new_business_convolve(nb_profile: np.ndarray, nb_premium: float, sales_file: np.ndarray,
                          output_len: int | None = None) -> np.ndarray:
    v = sales_file / nb_premium
    conv = np.array([np.convolve(a, v, mode='full') for a in nb_profile])
    return conv if output_len is None else conv[:, :output_len]

def maybe_check_state(obj, expectation, /, *, state_attr: str = "state",
                      strictness: StrictnessLevel = STRICTNESS_LEVEL) -> bool:
    if strictness == StrictnessLevel.BYPASS:
        return True

    state = getattr(obj, state_attr)
    if state == expectation:
        return True
    elif strictness == StrictnessLevel.WARN:
        warnings.warn(f"{obj} state {state} != {expectation}.")
        return False
    else:  # strictness == StrictnessLevel.ERROR
        raise ValueError(f"{obj} state {state} != {expectation}.")


class class_lazy_property:
    """A descriptor that caches a property at the Class level."""
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, owner):
        value = self.func(owner)
        setattr(owner, self.name, value)
        return value

import numpy as np
import warnings

from vates.global_conf import CHECK_LEVEL,CheckLevel

def new_business_convolve(nb_profile: np.ndarray, nb_premium: float, sales_file: np.ndarray,
                          output_len: int | None = None) -> np.ndarray:
    v = sales_file / nb_premium
    conv = np.array([np.convolve(a, v, mode='full') for a in nb_profile])
    return conv if output_len is None else conv[:, :output_len]

def maybe_raise_if_ne(a, b, /, *, check_level: CheckLevel = CHECK_LEVEL) -> None:
    if check_level == CheckLevel.BYPASS:
        return

    condition = a == b

    if condition:
        return

    if check_level == CheckLevel.WARN:
        warnings.warn(f"{a} != {b}.")
        return

    raise ValueError(f"{a} != {b}.")


class class_lazy_property:
    """A descriptor that caches a property at the Class level."""
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, owner):
        value = self.func(owner)
        setattr(owner, self.name, value)
        return value

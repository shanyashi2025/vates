import numpy as np
import warnings

from vates.global_conf import GLOBAL_CHECK_LEVEL, CheckLevel

def new_business_convolve(nb_profile: np.ndarray, nb_premium: float, sales_file: np.ndarray,
                          output_len: int | None = None) -> np.ndarray:
    v = sales_file / nb_premium
    conv = np.array([np.convolve(a, v, mode='full') for a in nb_profile])
    return conv if output_len is None else conv[:, :output_len]

def maybe_raise_on_compare(v, /, must_eq = None, *, must_ne = None, must_lt = None, must_le = None,
                           must_gt = None, must_ge = None, check_level: CheckLevel = GLOBAL_CHECK_LEVEL) -> None:
    if check_level == CheckLevel.BYPASS:
        return

    errors = []

    if must_eq is not None and v != must_eq:
        errors.append(f"{v} != {must_eq}")
    if must_ne is not None and v == must_ne:
        errors.append(f"{v} == {must_ne}")
    if must_lt is not None and v >= must_lt:
        errors.append(f"{v} >= {must_lt}")
    if must_le is not None and v > must_le:
        errors.append(f"{v} > {must_le}")
    if must_gt is not None and v <= must_gt:
        errors.append(f"{v} <= {must_gt}")
    if must_ge is not None and v < must_ge:
        errors.append(f"{v} < {must_ge}")

    if not errors:
        return

    msg = "; ".join(errors)

    if check_level == CheckLevel.WARN:
        warnings.warn(msg)
        return

    raise ValueError(msg)  # CheckLevel.ERROR


class class_lazy_property:
    """A descriptor that caches a property at the Class level."""
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, owner):
        value = self.func(owner)
        setattr(owner, self.name, value)
        return value

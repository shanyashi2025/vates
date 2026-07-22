import numpy as np
import warnings

def new_business_convolve(nb_profile: np.ndarray, nb_premium: float, sales_file: np.ndarray,
                          output_len: int | None = None) -> np.ndarray:
    v = sales_file / nb_premium
    conv = np.array([np.convolve(a, v, mode='full') for a in nb_profile])
    return conv if output_len is None else conv[:, :output_len]

T_CHECKER_STRICT_LEVEL = "WARNING"  # "ERROR"

def t_checker(checklist: dict[str, int] | None, tracker_key: str | None = None, tt_dict_name: str = '_tt_dict'):
    """Check if `time` is appropriate for the calculation

    Args:
        checklist (dict[str, int]): A dictionary containing items to be verified, value (int) represents the time offset.
        tracker_key (str): Update the time tracker dict once the function called. Defaults to None.
        tt_dict_name (str): The name of the time tracker dict. Defaults to '_tt_dict'.

    Returns:
        A decorator.
    """
    def decorator(func):
        def wrapper(obj, *args, **kwargs):
            if not hasattr(obj, tt_dict_name):
                setattr(obj, tt_dict_name, {})
            tt_dict = getattr(obj, tt_dict_name)
            t = obj.time
            if checklist is not None:
                for key, offset in checklist.items():
                    _t = tt_dict.get(key, None)
                    if _t is not None and _t != t + offset:
                        msg = f"'{obj}' on {t}: '{key}' last calculated on {_t}, expected {t + offset}."
                        if T_CHECKER_STRICT_LEVEL.upper() == "ERROR":
                            raise ValueError(msg)
                        else:
                            warnings.warn(msg)
            result = func(obj, *args, **kwargs)
            if tracker_key is not None:
                tt_dict[tracker_key] = t
            return result
        return wrapper
    return decorator

class class_lazy_property:
    """A descriptor that caches a property at the Class level."""
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, owner):
        value = self.func(owner)
        setattr(owner, self.name, value)
        return value

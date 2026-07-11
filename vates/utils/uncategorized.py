import numpy as np
import warnings

def new_business_convolve(nb_profile: np.ndarray, nb_premium: float, sales_file: np.ndarray,
                          output_len: int | None = None) -> np.ndarray:
    v = sales_file / nb_premium
    conv = np.array([np.convolve(a, v, mode='full') for a in nb_profile])
    return conv if output_len is None else conv[:, :output_len]

T_CHECKER_STRICT_LEVEL = "WARNING"  # "ERROR"

def t_checker(checklist: dict[str, int] | None, store_key: str | None = None, lct_dict_name: str = '_tc_dict'):
    """Check calculation period"""
    def decorator(func):
        def wrapper(obj, *args, **kwargs):
            if not hasattr(obj, lct_dict_name):
                setattr(obj, lct_dict_name, {})
            lct_dict = getattr(obj, lct_dict_name)
            t = obj.time
            if checklist is not None:
                for key, offset in checklist.items():
                    lct = lct_dict.get(key, None)
                    if lct is not None and lct != t + offset:
                        p = obj.period
                        lcp = p + (lct - t)
                        msg = f"'{obj}' on {t} ({p}): '{key}' last calculated on {lct} ({lcp}), expected {t + offset} ({p + offset})."
                        if T_CHECKER_STRICT_LEVEL.upper() == "ERROR":
                            raise ValueError(msg)
                        else:
                            warnings.warn(msg)
            result = func(obj, *args, **kwargs)
            if store_key is not None:
                lct_dict[store_key] = t
            return result
        return wrapper
    return decorator

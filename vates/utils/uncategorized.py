import numpy as np

def new_business_convolve(nb_profile: np.ndarray, nb_premium: float, sales_file: np.ndarray,
                          output_len: int | None = None) -> np.ndarray:
    v = sales_file / nb_premium
    conv = np.array([np.convolve(a, v, mode='full') for a in nb_profile])
    return conv if output_len is None else conv[:, :output_len]

def check_calc_time(checklist: dict[str, int] | None, lct_key: str | None=None, lct_dict_name: str= '_lct_dict'):
    """Check calculation period"""
    def decorator(func):
        def wrapper(obj, *args, **kwargs):
            t = obj.time
            lct_dict = getattr(obj, lct_dict_name)
            if checklist is not None:
                for key, offset in checklist.items():
                    lct = lct_dict.get(key, None)
                    if lct is not None and lct != t + offset:
                        p = obj.period
                        lcp = p + (lct - t)
                        raise ValueError(f"'{str(obj)}' on {t} ({p}): '{key}' last calculated on {lct} ({lcp}), "
                                         f"expected {t + offset} ({p + offset}).")
            result = func(obj, *args, **kwargs)
            if lct_key is not None:
                lct_dict[lct_key] = t
            return result
        return wrapper
    return decorator

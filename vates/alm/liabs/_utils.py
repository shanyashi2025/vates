from vates.utils import maybe_raise_if_ne

def maybe_check_liab_state_roll(func):
    def wrapper(obj, *args, **kwargs):
        if obj._state != ("initialized", obj.time - 1):
            maybe_raise_if_ne(obj._state, ("closed", obj.time - 1))
        result = func(obj, *args, **kwargs)
        obj._state = ("rolled", obj.time)
        return result
    return wrapper

def maybe_check_liab_state_close(func):
    def wrapper(obj, *args, **kwargs):
        if obj._state != ("initialized", obj.time):
            maybe_raise_if_ne(obj._state, ("rolled", obj.time))
        result = func(obj, *args, **kwargs)
        obj._state = ("closed", obj.time)
        return result
    return wrapper
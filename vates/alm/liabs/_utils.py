from vates.utils import maybe_check_state

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
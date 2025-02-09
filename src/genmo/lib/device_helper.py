import torch
from contextlib import contextmanager
from functools import wraps

def get_default_device():
    """
    Return the default device string.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

@contextmanager
def autocast_device(**kwargs):
    """
    Context manager that wraps torch.autocast with a default device.
    """
    device = get_default_device()
    with torch.autocast(device, **kwargs):
        yield

def with_autocast(func=None, **ac_kwargs):
    """
    Decorator that wraps a function call inside a torch.autocast context,
    using the default device
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            device = get_default_device()
            with torch.autocast(device, **ac_kwargs):
                return fn(*args, **kwargs)
        return wrapper
    if func is None:
        return decorator
    else:
        return decorator(func)
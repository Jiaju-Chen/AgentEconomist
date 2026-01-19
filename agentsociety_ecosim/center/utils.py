from agentsociety_ecosim.logger import get_logger

logger = get_logger(__name__)

import functools

def safe_call(log_msg: str = "Exception catcher", level: str = "error"):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                getattr(logger, level)(f"[{log_msg}] {fn.__name__} failed: {e}")
                return None
        return wrapper
    return decorator

import importlib.util

HAS_TRITON = importlib.util.find_spec("triton") is not None

import importlib.util

HAS_TRITON = importlib.util.find_spec("triton") is not None
HAS_FLA = importlib.util.find_spec("fla") is not None

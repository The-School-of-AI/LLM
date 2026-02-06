import platform
import sys

def capture_env():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }

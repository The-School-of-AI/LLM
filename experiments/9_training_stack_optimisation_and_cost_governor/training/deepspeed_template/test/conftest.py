"""
Test configuration for data pipeline tests.

Adds the experiment root directory to sys.path so that data_loader can be
imported directly.
"""

import os
import sys

# Allow importing data_loader directly: `from data_loader.xxx import Yyy`
# test/ → deepspeed_template/ → training/ → experiment_root/
_experiment_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _experiment_root not in sys.path:
    sys.path.insert(0, _experiment_root)

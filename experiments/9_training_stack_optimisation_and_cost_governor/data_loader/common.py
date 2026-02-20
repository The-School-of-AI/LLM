import os

import yaml


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get("SPDL_CONFIG", "configuration_P4.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


_config = load_config()

# SPDL Pipeline Constants (parameterized)
BATCH_SIZE = _config.get("batch_size", 1024)
NUM_THREADS = _config.get("num_threads", os.cpu_count())
PREFETCH_BUFFER = _config.get("prefetch_buffer", 8)
SEQUENCE_LENGTH = _config.get("sequence_length", 4096)
DTYPE = _config.get("dtype", "uint32")

LOG_BACKUP_COUNT = 20

LOG_FILE_SIZE = 100 * 1024 * 1024

import os

# SPDL Pipeline Constants
BATCH_SIZE = 1024
NUM_THREADS = os.cpu_count()
PREFETCH_BUFFER = 8

# Data Column Names
TOKENS_COLUMN = "tokens"
"""
AWS Configuration Package.

This package contains AWS-related configuration for S3 checkpoint management.
"""

from .config import S3Config, get_default_config, DEFAULT_CONFIGS

__all__ = [
    'S3Config',
    'get_default_config',
    'DEFAULT_CONFIGS',
]

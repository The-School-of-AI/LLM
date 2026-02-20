"""
Error handling and recovery system for production coreset selection.
Provides retry logic, graceful degradation, and detailed error reporting.
"""

import functools
import logging
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors"""

    FATAL = "fatal"  # Pipeline must stop
    ERROR = "error"  # Stage failed, continue to next stage
    WARNING = "warning"  # Recoverable, log and continue
    INFO = "info"  # Informational, continue


@dataclass
class ErrorContext:
    """Context for error reporting"""

    error_type: str
    message: str
    severity: ErrorSeverity
    stage_name: Optional[str] = None
    batch_num: Optional[int] = None
    traceback_str: Optional[str] = None
    timestamp: str = ""
    retries_attempted: int = 0
    is_retriable: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "stage_name": self.stage_name,
            "batch_num": self.batch_num,
            "retries_attempted": self.retries_attempted,
            "is_retriable": self.is_retriable,
            "timestamp": self.timestamp,
        }


class RetryableError(Exception):
    """Base class for retryable errors"""

    pass


class MemoryError(RetryableError):
    """Memory-related errors (transient)"""

    pass


class IOError(RetryableError):
    """I/O errors (transient)"""

    pass


class CheckpointError(Exception):
    """Checkpoint-related errors"""

    pass


class ValidationError(Exception):
    """Validation failures (non-retriable)"""

    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retriable_exceptions: Optional[List[Type[Exception]]] = None,
):
    """
    Decorator to retry function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
        retriable_exceptions: List of exception types to retry on (defaults to RetryableError)
    """
    if retriable_exceptions is None:
        retriable_exceptions = [RetryableError]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except tuple(retriable_exceptions) as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )

                except Exception as e:
                    logger.error(
                        f"{func.__name__} failed with non-retriable error: {e}",
                        exc_info=True,
                    )
                    raise

            # All retries exhausted
            if last_exception:
                raise last_exception

            return None

        return wrapper

    return decorator


class ErrorRecoveryManager:
    """
    Centralized error handling and recovery.

    Features:
    - Categorizes errors by severity and retriability
    - Logs detailed error context
    - Suggests recovery actions
    - Maintains error statistics
    """

    def __init__(self, error_log_path: str = "coreset_errors.log"):
        self.error_log_path = error_log_path
        self.errors: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        self._setup_logging()

    def _setup_logging(self):
        """Setup error-specific logging"""
        error_logger = logging.getLogger("coreset_errors")
        handler = logging.FileHandler(self.error_log_path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        error_logger.addHandler(handler)
        error_logger.setLevel(logging.WARNING)

    def handle_error(
        self,
        exception: Exception,
        error_type: str,
        stage_name: Optional[str] = None,
        batch_num: Optional[int] = None,
        severity: Optional[ErrorSeverity] = None,
    ) -> ErrorContext:
        """
        Handle an exception and return error context.

        Args:
            exception: The exception that occurred
            error_type: Type/category of error
            stage_name: Name of stage where error occurred
            batch_num: Batch number (if applicable)
            severity: Error severity (defaults to based on exception type)

        Returns:
            ErrorContext with handling recommendations
        """
        import datetime

        # Determine severity if not specified
        if severity is None:
            severity = self._infer_severity(exception)

        # Determine if retriable
        is_retriable = isinstance(exception, RetryableError)

        # Create error context
        error_context = ErrorContext(
            error_type=error_type,
            message=str(exception),
            severity=severity,
            stage_name=stage_name,
            batch_num=batch_num,
            traceback_str=traceback.format_exc(),
            timestamp=datetime.datetime.now().isoformat(),
            is_retriable=is_retriable,
        )

        self.errors.append(error_context)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log based on severity
        self._log_error(error_context)

        return error_context

    def _infer_severity(self, exception: Exception) -> ErrorSeverity:
        """Infer error severity from exception type"""
        if isinstance(exception, ValidationError):
            return ErrorSeverity.FATAL
        elif isinstance(exception, RetryableError):
            return ErrorSeverity.WARNING
        else:
            return ErrorSeverity.ERROR

    def _log_error(self, context: ErrorContext):
        """Log error based on severity"""
        # Don't use 'message' as extra key - it's reserved by logging
        error_dict = {
            "error_type": context.error_type,
            "severity": context.severity.value,
            "stage_name": context.stage_name,
            "batch_num": context.batch_num,
            "retries_attempted": context.retries_attempted,
            "is_retriable": context.is_retriable,
            "timestamp": context.timestamp,
        }

        if context.severity == ErrorSeverity.FATAL:
            logger.critical(f"FATAL: {context.message}", extra=error_dict)
        elif context.severity == ErrorSeverity.ERROR:
            logger.error(
                f"ERROR in {context.stage_name} batch {context.batch_num}: {context.message}",
                extra=error_dict,
            )
        elif context.severity == ErrorSeverity.WARNING:
            logger.warning(f"WARNING: {context.message}", extra=error_dict)
        else:
            logger.info(f"INFO: {context.message}", extra=error_dict)

    def get_recovery_action(self, context: ErrorContext) -> str:
        """Suggest recovery action for an error"""
        actions = {
            "MemoryError": "Reduce batch_size or enable streaming mode. Check available RAM.",
            "IOError": "Verify input data accessibility. Check file permissions and disk space.",
            "CheckpointError": "Delete corrupted checkpoint and restart pipeline.",
            "ValidationError": "Fix configuration or data format. Review pipeline.yaml and curriculum.yaml.",
            "TokenFrequencyAnalyzerError": "Increase vocab_size or disable token frequency analysis.",
            "DeduplicationError": "Disable deduplication (enable_exact_dedup=false) and retry.",
            "DiversityError": "Reduce diversity weight or disable diversity scoring.",
        }

        return actions.get(
            context.error_type, "Check logs and retry with different configuration."
        )

    def print_error_summary(self):
        """Print summary of all errors encountered"""
        if not self.errors:
            logger.info("No errors encountered")
            return

        logger.info(f"\nError Summary ({len(self.errors)} total errors):")
        logger.info("-" * 70)

        # Group by error type
        by_type = defaultdict(list)
        for err in self.errors:
            by_type[err.error_type].append(err)

        for error_type, errors in sorted(by_type.items()):
            logger.info(f"\n{error_type}: {len(errors)} occurrences")

            # Show first error as example
            first = errors[0]
            logger.info(f"  Example: {first.message}")
            logger.info(f"  Severity: {first.severity.value}")
            logger.info(f"  Retriable: {first.is_retriable}")

            recovery = self.get_recovery_action(first)
            logger.info(f"  Recovery: {recovery}")

    def should_fail_pipeline(self) -> bool:
        """Check if pipeline should halt due to fatal errors"""
        fatal_errors = [e for e in self.errors if e.severity == ErrorSeverity.FATAL]
        return len(fatal_errors) > 0


def handle_batch_error(
    recovery_manager: ErrorRecoveryManager,
    stage_name: str,
    batch_num: int,
    skip_failed_batch: bool = True,
):
    """
    Decorator to handle errors in batch processing.

    Args:
        recovery_manager: ErrorRecoveryManager instance
        stage_name: Name of stage
        batch_num: Batch number
        skip_failed_batch: If True, skip failed batch instead of failing
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)

            except Exception as e:
                error_context = recovery_manager.handle_error(
                    e,
                    error_type=type(e).__name__,
                    stage_name=stage_name,
                    batch_num=batch_num,
                )

                if skip_failed_batch:
                    logger.warning(
                        f"Skipping batch {batch_num} due to {error_context.error_type}"
                    )
                    return None
                else:
                    raise

        return wrapper

    return decorator

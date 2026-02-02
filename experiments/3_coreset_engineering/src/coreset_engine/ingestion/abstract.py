from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional

class DataSource(ABC):
    """
    Abstract base class for data sources.
    Adapters (Parquet, JSONL, S3) must implement this interface.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Yields data items one by one.
        Each item must be a dictionary with at least:
        - 'text': str
        - 'token_count': int
        - 'domain': str
        - 'file_path': str
        - 'file_line': int
        """
        pass

    @abstractmethod
    def count(self) -> Optional[int]:
        """
        Returns total number of items if known/cheap to compute.
        Otherwise Returns None.
        """
        pass

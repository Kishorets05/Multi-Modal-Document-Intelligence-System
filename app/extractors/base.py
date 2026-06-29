from abc import ABC, abstractmethod
from pathlib import Path


class BaseTextExtractor(ABC):
    """Abstract base for all document text extractors."""

    @abstractmethod
    def extract(self, file_path: Path) -> str:
        """Extract and return the full plain text from the given file.

        Args:
            file_path: Absolute path to the document on disk.

        Returns:
            The extracted plain text as a single string.

        Raises:
            TextExtractionError: If the document contains no extractable text
                or is structurally invalid.
        """

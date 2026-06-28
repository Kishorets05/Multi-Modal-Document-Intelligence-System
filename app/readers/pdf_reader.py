from pathlib import Path
from typing import Any

import fitz


class PDFReader:
    """Extract metadata from a PDF file without performing OCR."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)

    def read(self) -> dict[str, Any]:
        """Return extracted PDF metadata."""
        with fitz.open(self.file_path) as document:
            metadata = document.metadata or {}
            page_count = document.page_count
            has_extractable_text = any(
                page.get_text().strip() for page in document
            )

        return {
            "document_type": "pdf",
            "page_count": page_count,
            "paragraph_count": None,
            "image_width": None,
            "image_height": None,
            "document_size": self.file_path.stat().st_size,
            "document_metadata": metadata,
            "has_extractable_text": has_extractable_text,
            "reader_used": self.__class__.__name__,
        }

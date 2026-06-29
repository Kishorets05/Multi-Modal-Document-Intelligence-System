import logging
from pathlib import Path

import fitz

from app.extractors.base import BaseTextExtractor
from app.services.text_extraction_service import TextExtractionError


class PdfTextExtractor(BaseTextExtractor):
    """Extract embedded text from PDF documents using PyMuPDF.

    Does NOT perform OCR. If no embedded text is found across all pages,
    a TextExtractionError is raised immediately.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("app")

    def extract(self, file_path: Path) -> str:
        """Return the concatenated text of all PDF pages.

        Args:
            file_path: Path to the PDF document.

        Returns:
            Non-empty string of extracted text with pages separated by
            double newlines.

        Raises:
            TextExtractionError: If the PDF has no embedded text on any page.
            TextExtractionError: If the PDF file is corrupted or unreadable.
        """
        file_path = Path(file_path)
        self.logger.info("PDF text extraction started: %s", file_path.name)

        try:
            text_parts: list[str] = []
            with fitz.open(file_path) as document:
                for page_num, page in enumerate(document, start=1):
                    try:
                        page_text = page.get_text("text").strip()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as exc:
                        self.logger.warning(
                            "Could not read PDF page %d in '%s': %s",
                            page_num,
                            file_path.name,
                            exc,
                        )
        except fitz.FileDataError as exc:
            self.logger.error("Corrupted PDF file '%s': %s", file_path.name, exc)
            raise TextExtractionError(
                f"PDF file '{file_path.name}' is corrupted or cannot be opened."
            ) from exc

        if not text_parts:
            self.logger.warning(
                "No embedded text found in PDF '%s'.", file_path.name
            )
            raise TextExtractionError("No embedded text found in PDF.")

        result = "\n\n".join(text_parts)
        self.logger.info(
            "PDF text extraction completed: '%s' — %d characters extracted.",
            file_path.name,
            len(result),
        )
        return result

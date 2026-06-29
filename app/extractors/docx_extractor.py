import logging
from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from app.extractors.base import BaseTextExtractor
from app.services.text_extraction_service import TextExtractionError


class DocxTextExtractor(BaseTextExtractor):
    """Extract text from DOCX documents using python-docx.

    Reads every paragraph in document order and joins them with newline
    characters. Empty paragraphs (whitespace-only) are skipped.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("app")

    def extract(self, file_path: Path) -> str:
        """Return the concatenated paragraph text of the DOCX document.

        Args:
            file_path: Path to the DOCX document.

        Returns:
            All non-empty paragraphs joined by newline characters.

        Raises:
            TextExtractionError: If the DOCX file is corrupted or unreadable.
        """
        file_path = Path(file_path)
        self.logger.info("DOCX text extraction started: %s", file_path.name)

        try:
            document = Document(file_path)
        except PackageNotFoundError as exc:
            self.logger.error(
                "Corrupted or invalid DOCX file '%s': %s", file_path.name, exc
            )
            raise TextExtractionError(
                f"DOCX file '{file_path.name}' is corrupted or cannot be opened."
            ) from exc

        text_parts: list[str] = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                text_parts.append(text)

        result = "\n".join(text_parts)
        self.logger.info(
            "DOCX text extraction completed: '%s' — %d characters extracted.",
            file_path.name,
            len(result),
        )
        return result

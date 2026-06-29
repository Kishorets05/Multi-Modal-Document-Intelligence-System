import logging
from pathlib import Path


class TextExtractionError(Exception):
    """Raised when text cannot be extracted from a document.

    This covers:
    - PDFs with no embedded text.
    - Corrupted or unreadable PDF / DOCX files.
    - Unsupported file types passed to this service.
    """


class TextExtractionService:
    """Orchestrate text extraction for supported document types.

    Dispatches to the appropriate concrete extractor based on file extension.
    Supported types: .pdf, .docx
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def __init__(self) -> None:
        self.logger = logging.getLogger("app")

    def extract(self, file_path: Path) -> str:
        """Extract and return plain text from the given document.

        Args:
            file_path: Absolute path to the document on disk.

        Returns:
            Extracted plain text as a single string.

        Raises:
            TextExtractionError: If the file type is unsupported, the document
                contains no extractable text, or the file is corrupted.
        """
        # Imports are deferred here to avoid a circular import at module load
        # time (the extractor modules import TextExtractionError from this
        # module, so importing them at the top of this file would create a
        # cycle).
        from app.extractors.pdf_extractor import PdfTextExtractor
        from app.extractors.docx_extractor import DocxTextExtractor

        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        self.logger.info(
            "Text extraction started — document: '%s'", file_path.name
        )

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise TextExtractionError(
                f"Unsupported file type '{extension}' for text extraction. "
                f"Supported types: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        if extension == ".pdf":
            extractor = PdfTextExtractor()
        else:
            extractor = DocxTextExtractor()

        text = extractor.extract(file_path)

        self.logger.info(
            "Text extraction completed — document: '%s', characters: %d",
            file_path.name,
            len(text),
        )
        return text

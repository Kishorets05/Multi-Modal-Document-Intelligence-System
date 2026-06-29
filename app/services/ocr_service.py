import logging
from pathlib import Path
from typing import Any

import fitz
from docx import Document


class OCRService:
    """Extract plain text from uploaded documents using direct text extraction only."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.logger = logging.getLogger("app")
        self.ocr_used = False
        self.ocr_engine = None

    def extract_text(self, file_path: Path, metadata: dict[str, Any]) -> str:
        """Return plain text for the given document."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        try:
            if extension == ".pdf":
                return self._extract_from_pdf(file_path)
            if extension == ".docx":
                return self._extract_from_docx(file_path)
            if extension in {".png", ".jpg", ".jpeg"}:
                return self._extract_from_image(file_path)
        except Exception as exc:
            self.logger.exception("Text extraction failed for %s", file_path.name)
            raise RuntimeError("Text extraction failed") from exc

        return ""

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract all embedded text from PDF pages using PyMuPDF."""
        self.logger.info("Extracting text from PDF: %s", file_path.name)
        text_parts: list[str] = []

        with fitz.open(file_path) as document:
            for page_num, page in enumerate(document, start=1):
                try:
                    page_text = page.get_text("text").strip()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as exc:
                    self.logger.warning("Failed to extract text from PDF page %d: %s", page_num, exc)

        result = "\n\n".join(text_parts)
        self.logger.info("Text extraction completed for PDF: %d characters", len(result))
        return result

    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract all paragraphs from a DOCX document."""
        self.logger.info("Extracting text from DOCX: %s", file_path.name)
        text_parts: list[str] = []

        try:
            document = Document(file_path)
            for paragraph in document.paragraphs:
                text = paragraph.text.strip()
                if text:
                    text_parts.append(text)
        except Exception as exc:
            self.logger.exception("Failed to extract text from DOCX")
            raise

        result = "\n".join(text_parts)
        self.logger.info("Text extraction completed for DOCX: %d characters", len(result))
        return result

    def _extract_from_image(self, file_path: Path) -> str:
        """Image OCR is not implemented. Return empty string."""
        self.logger.info(
            "Image file detected; OCR is not implemented for this module: %s",
            file_path.name,
        )
        return ""

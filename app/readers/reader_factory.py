from pathlib import Path
from typing import Any

from app.readers.docx_reader import DocxReader
from app.readers.image_reader import ImageReader
from app.readers.pdf_reader import PDFReader
from app.utils.file_validator import get_file_extension


class ReaderFactory:
    """Create the appropriate reader for a document based on MIME type or extension."""

    MIME_MAP = {
        "application/pdf": PDFReader,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxReader,
        "image/png": ImageReader,
        "image/jpeg": ImageReader,
    }

    @staticmethod
    def get_reader(file_path: Path, mime_type: str | None = None) -> Any:
        if mime_type:
            reader_cls = ReaderFactory.MIME_MAP.get(mime_type.lower())
            if reader_cls:
                return reader_cls(file_path)

        extension = get_file_extension(file_path.name)
        if extension == ".pdf":
            return PDFReader(file_path)
        if extension == ".docx":
            return DocxReader(file_path)
        if extension in {".png", ".jpg", ".jpeg"}:
            return ImageReader(file_path)

        raise ValueError(f"No reader available for extension '{extension}'")

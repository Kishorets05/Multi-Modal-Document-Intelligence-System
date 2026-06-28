import io
from pathlib import Path

from fastapi import UploadFile

from app.config.settings import settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}


class FileValidationError(ValueError):
    """Raised when an uploaded file does not meet validation rules."""


def get_file_extension(filename: str) -> str:
    """Return the normalized extension for the uploaded filename."""
    extension = Path(filename).suffix.lower()
    return extension


def validate_file_extension(filename: str) -> str:
    """Validate the uploaded file extension against allowed types."""
    extension = get_file_extension(filename)
    if not extension or extension not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type '{extension}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return extension


def validate_file_size(upload_file: UploadFile) -> int:
    """Validate the size of the uploaded file without reading its contents."""
    file_object = upload_file.file
    file_object.seek(0, io.SEEK_END)
    size = file_object.tell()
    file_object.seek(0)

    if size == 0:
        raise FileValidationError("Empty files are not allowed.")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size > max_bytes:
        raise FileValidationError(
            f"File size exceeds the maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    return size

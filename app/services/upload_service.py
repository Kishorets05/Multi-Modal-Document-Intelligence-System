import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config.settings import settings
from app.services.document_reader_service import DocumentReaderError, DocumentReaderService
from app.utils.file_validator import FileValidationError, validate_file_extension, validate_file_size


class UploadProcessingError(Exception):
    """Raised when an upload cannot be saved to disk."""


class UploadService:
    def __init__(self, upload_dir: Path = settings.UPLOAD_DIR) -> None:
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("app")

    def save_upload(self, upload_file: UploadFile) -> dict:
        """Validate and persist the uploaded file to disk."""
        try:
            original_name = upload_file.filename
            extension = validate_file_extension(original_name)
            file_size = validate_file_size(upload_file)

            file_id = uuid4().hex
            stored_name = f"{file_id}{extension}"
            destination = self.upload_dir / stored_name

            upload_file.file.seek(0)
            with destination.open("wb") as destination_file:
                shutil.copyfileobj(upload_file.file, destination_file)

            upload_time = datetime.now(timezone.utc).isoformat()
            file_type = upload_file.content_type or "application/octet-stream"

            self.logger.info(
                "Upload successful: %s %s %d bytes",
                original_name,
                stored_name,
                file_size,
            )

            upload_response = {
                "success": True,
                "file_id": file_id,
                "original_name": original_name,
                "stored_name": stored_name,
                "file_size": file_size,
                "file_type": file_type,
                "upload_time": upload_time,
            }

            # Process the uploaded file to extract metadata and create a workspace.
            DocumentReaderService(settings.UPLOAD_DIR).process_document(upload_response)

            return upload_response
        except FileValidationError:
            raise
        except DocumentReaderError as exc:
            self.logger.error("Document reading failed after upload: %s", exc)
            raise UploadProcessingError("Uploaded file could not be read for metadata extraction.") from exc
        except Exception as exc:
            self.logger.exception("Unexpected error while saving uploaded file.")
            raise UploadProcessingError("Failed to process upload.") from exc

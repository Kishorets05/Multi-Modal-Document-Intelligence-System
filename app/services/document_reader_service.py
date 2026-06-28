import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.readers.reader_factory import ReaderFactory
from app.config.settings import settings


class DocumentReaderError(Exception):
    """Raised when the document reader cannot extract metadata."""


class DocumentReaderService:
    def __init__(self, upload_dir: Path = settings.UPLOAD_DIR) -> None:
        self.upload_dir = Path(upload_dir)
        self.logger = logging.getLogger("app")

    def process_document(self, upload_response: dict) -> dict:
        """Move uploaded file into a workspace, read metadata, and persist metadata.json."""
        document_id = upload_response["file_id"]
        original_name = upload_response["original_name"]
        stored_name = upload_response["stored_name"]
        mime_type = upload_response["file_type"]

        workspace_dir = self.upload_dir / document_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Workspace created: %s", workspace_dir)

        source_file = self.upload_dir / stored_name
        destination_file = workspace_dir / f"original{Path(original_name).suffix.lower()}"

        if not source_file.exists():
            raise DocumentReaderError("Uploaded file not found for document processing.")

        source_file.rename(destination_file)
        self.logger.info("Moved uploaded file to workspace: %s", destination_file)

        try:
            reader = ReaderFactory.get_reader(destination_file, mime_type=mime_type)
            self.logger.info("Reader selected: %s", reader.__class__.__name__)
            metadata = reader.read()
        except Exception as exc:
            self.logger.exception("Failed to extract metadata from document.")
            raise DocumentReaderError("Document metadata extraction failed.") from exc

        document_metadata = {
            "document_id": document_id,
            "original_filename": original_name,
            "stored_filename": destination_file.name,
            "mime_type": mime_type,
            "document_type": metadata.get("document_type"),
            "file_size": metadata.get("document_size"),
            "page_count": metadata.get("page_count"),
            "paragraph_count": metadata.get("paragraph_count"),
            "heading_count": metadata.get("heading_count"),
            "image_width": metadata.get("image_width"),
            "image_height": metadata.get("image_height"),
            "uploaded_at": upload_response["upload_time"],
            "reader_used": metadata.get("reader_used"),
            "document_metadata": metadata.get("document_metadata"),
        }

        metadata_file = workspace_dir / "metadata.json"
        metadata_file.write_text(
            json.dumps(self._serialize(document_metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.logger.info("Metadata stored: %s", metadata_file)

        return upload_response

    def _serialize(self, value: Any) -> Any:
        """Convert metadata values into JSON-serializable representations."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize(item) for item in value]
        if isinstance(value, tuple):
            return [self._serialize(item) for item in value]
        return str(value)

        return upload_response

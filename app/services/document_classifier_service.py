import json
import logging
from pathlib import Path

from app.classifiers.classifier_factory import ClassifierFactory
from app.config.settings import settings


class DocumentClassificationError(Exception):
    """Raised when classification cannot proceed for a document.

    This covers:
    - Missing extracted_text.txt.
    - Empty extracted text (nothing to classify).
    - Unreadable or malformed metadata.json.
    """


class DocumentClassifierService:
    """Classify an uploaded document and persist the result to metadata.json.

    Reads the pre-computed extracted_text.txt produced by Module 3.
    Never reads the original document file directly.
    """

    def __init__(self, upload_dir: Path = settings.UPLOAD_DIR) -> None:
        self.upload_dir = Path(upload_dir)
        self.logger = logging.getLogger("app")

    def classify_document(self, document_id: str) -> str:
        """Classify the document and update metadata.json with the result.

        Args:
            document_id: The unique document identifier (workspace folder name).

        Returns:
            The classified category string, e.g. 'resume', 'invoice'.

        Raises:
            DocumentClassificationError: If extracted_text.txt is missing,
                empty, or metadata.json cannot be read or written.
        """
        workspace_dir = self.upload_dir / document_id
        text_file = workspace_dir / "extracted_text.txt"
        metadata_file = workspace_dir / "metadata.json"

        self.logger.info(
            "Classification started — document_id: '%s'", document_id
        )

        # --- Read extracted text -------------------------------------------------
        if not text_file.is_file():
            raise DocumentClassificationError(
                f"extracted_text.txt not found for document '{document_id}'. "
                "Text extraction must complete successfully before classification."
            )

        try:
            text = text_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise DocumentClassificationError(
                f"Failed to read extracted_text.txt for document '{document_id}'."
            ) from exc

        if not text.strip():
            raise DocumentClassificationError(
                f"extracted_text.txt is empty for document '{document_id}'. "
                "Cannot classify a document with no text."
            )

        # --- Classify ------------------------------------------------------------
        classifier = ClassifierFactory.get_classifier()
        document_type = classifier.classify(text)

        self.logger.info(
            "Detected document type: '%s' — document_id: '%s'",
            document_type,
            document_id,
        )

        # --- Persist result to metadata.json -------------------------------------
        if not metadata_file.is_file():
            raise DocumentClassificationError(
                f"metadata.json not found for document '{document_id}'."
            )

        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise DocumentClassificationError(
                f"Failed to read metadata.json for document '{document_id}'."
            ) from exc

        metadata["document_type"] = document_type

        try:
            metadata_file.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise DocumentClassificationError(
                f"Failed to write classification result to metadata.json "
                f"for document '{document_id}'."
            ) from exc

        self.logger.info(
            "Classification completed — document_id: '%s', document_type: '%s'",
            document_id,
            document_type,
        )

        return document_type

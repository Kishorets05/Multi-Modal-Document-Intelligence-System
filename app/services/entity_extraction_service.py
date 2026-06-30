import json
import logging
from pathlib import Path

from app.config.settings import settings
from app.entity_extractors.base import BaseEntityExtractor
from app.entity_extractors.contract_extractor import ContractEntityExtractor
from app.entity_extractors.general_extractor import GeneralEntityExtractor
from app.entity_extractors.invoice_extractor import InvoiceEntityExtractor
from app.entity_extractors.medical_extractor import MedicalEntityExtractor
from app.entity_extractors.research_extractor import ResearchEntityExtractor
from app.entity_extractors.resume_extractor import ResumeEntityExtractor


class EntityExtractionError(Exception):
    """Raised when entity extraction cannot proceed for a document.

    This covers:
    - Missing extracted_text.txt.
    - Empty extracted text.
    - Missing or unreadable metadata.json.
    - Unknown / unsupported document_type in metadata.
    """


# Registry mapping document_type → extractor class.
# To add support for a new document type, add one entry here and create its
# extractor.  No other file needs to change.
_EXTRACTOR_REGISTRY: dict[str, type[BaseEntityExtractor]] = {
    "resume": ResumeEntityExtractor,
    "invoice": InvoiceEntityExtractor,
    "contract": ContractEntityExtractor,
    "medical_report": MedicalEntityExtractor,
    "research_paper": ResearchEntityExtractor,
    "general_document": GeneralEntityExtractor,
}


class EntityExtractionService:
    """Orchestrate entity extraction for an uploaded, classified document.

    Workflow
    --------
    1. Read ``extracted_text.txt`` from the document workspace.
    2. Read ``metadata.json`` to determine ``document_type``.
    3. Dispatch to the appropriate ``BaseEntityExtractor`` subclass.
    4. Return the structured entity dict.

    The service deliberately does NOT persist entities to metadata.json so
    that the extraction endpoint can be called multiple times idempotently
    without ballooning the metadata file.  If persistence is required in a
    future module, add it here.
    """

    def __init__(self, upload_dir: Path = settings.UPLOAD_DIR) -> None:
        self.upload_dir = Path(upload_dir)
        self.logger = logging.getLogger("app")

    def extract_entities(self, document_id: str) -> dict:
        """Extract structured entities from a classified document.

        Args:
            document_id: The unique document identifier (workspace folder name).

        Returns:
            dict with keys:
                document_id   (str)  – echoed from the argument.
                document_type (str)  – as stored in metadata.json.
                entities      (dict) – type-specific structured fields.

        Raises:
            EntityExtractionError: On missing files, empty text, or unknown
                document type.
        """
        workspace_dir = self.upload_dir / document_id
        text_file = workspace_dir / "extracted_text.txt"
        metadata_file = workspace_dir / "metadata.json"

        self.logger.info(
            "Entity extraction started — document_id: '%s'", document_id
        )

        # --- Read extracted text -------------------------------------------------
        if not text_file.is_file():
            raise EntityExtractionError(
                f"extracted_text.txt not found for document '{document_id}'. "
                "Run text extraction before entity extraction."
            )

        try:
            text = text_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise EntityExtractionError(
                f"Failed to read extracted_text.txt for document '{document_id}'."
            ) from exc

        if not text.strip():
            raise EntityExtractionError(
                f"extracted_text.txt is empty for document '{document_id}'. "
                "Cannot extract entities from an empty document."
            )

        # --- Read metadata to get document_type ----------------------------------
        if not metadata_file.is_file():
            raise EntityExtractionError(
                f"metadata.json not found for document '{document_id}'."
            )

        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise EntityExtractionError(
                f"Failed to read metadata.json for document '{document_id}'."
            ) from exc

        document_type: str = metadata.get("document_type") or "general_document"

        # --- Dispatch to the appropriate extractor -------------------------------
        extractor_class = _EXTRACTOR_REGISTRY.get(document_type)
        if extractor_class is None:
            self.logger.warning(
                "Unknown document_type '%s' for document '%s'; "
                "falling back to general extractor.",
                document_type,
                document_id,
            )
            extractor_class = GeneralEntityExtractor
            document_type = "general_document"

        extractor: BaseEntityExtractor = extractor_class(workspace_dir=workspace_dir)

        self.logger.info(
            "Dispatching to %s — document_id: '%s'",
            extractor_class.__name__,
            document_id,
        )

        entities = extractor.extract(text)

        self.logger.info(
            "Entity extraction completed — document_id: '%s', document_type: '%s', "
            "fields extracted: %d",
            document_id,
            document_type,
            len(entities),
        )

        return {
            "document_id": document_id,
            "document_type": document_type,
            "entities": entities,
        }

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config.settings import settings
from app.schemas.entity import EntityExtractionResponse
from app.services.entity_extraction_service import (
    EntityExtractionError,
    EntityExtractionService,
)

router = APIRouter(tags=["entities"])


@router.get(
    "/documents/{document_id}/entities",
    response_model=EntityExtractionResponse,
    summary="Extract structured entities from a classified document",
)
async def get_entities(document_id: str) -> EntityExtractionResponse:
    """Extract and return structured entities from a document's plain text.

    Reads ``extracted_text.txt`` and ``metadata.json`` from the document
    workspace.  The document must have been classified first (Module 4) so
    that ``document_type`` is present in ``metadata.json``; without it the
    service falls back to the general extractor.

    Dispatches to the appropriate type-specific extractor:

    * **resume**          → name, email, phone, skills, education,
                            experience, projects, certifications
    * **invoice**         → invoice_number, invoice_date, vendor, customer,
                            subtotal, tax, total_amount, currency
    * **contract**        → party_one, party_two, effective_date,
                            termination_date, governing_law, payment_terms
    * **medical_report**  → patient_name, doctor, hospital, diagnosis,
                            medications, test_results
    * **research_paper**  → title, authors, abstract, keywords,
                            references_count
    * **general_document** → title, emails, phones, urls

    Raises:
        404: If the document workspace, ``extracted_text.txt``, or
             ``metadata.json`` does not exist.
        422: If the extracted text file is empty.
        500: If extraction fails for an unexpected reason.
    """
    logger = logging.getLogger("app")

    workspace_dir: Path = settings.UPLOAD_DIR / document_id

    if not workspace_dir.is_dir():
        logger.warning(
            "Entity extraction requested for unknown document: '%s'", document_id
        )
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found.",
        )

    try:
        service = EntityExtractionService()
        result = service.extract_entities(document_id)
    except EntityExtractionError as exc:
        error_msg = str(exc)
        logger.error(
            "Entity extraction failed — document_id: '%s': %s", document_id, error_msg
        )
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail=error_msg)
        if "empty" in error_msg:
            raise HTTPException(status_code=422, detail=error_msg)
        raise HTTPException(status_code=500, detail="Entity extraction failed.")
    except Exception:
        logger.exception(
            "Unexpected error during entity extraction — document_id: '%s'",
            document_id,
        )
        raise HTTPException(status_code=500, detail="Entity extraction failed.")

    return EntityExtractionResponse(
        document_id=result["document_id"],
        document_type=result["document_type"],
        entities=result["entities"],
    )

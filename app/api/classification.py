import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config.settings import settings
from app.schemas.classification import ClassificationResponse
from app.services.document_classifier_service import (
    DocumentClassificationError,
    DocumentClassifierService,
)

router = APIRouter(tags=["classification"])


@router.get(
    "/documents/{document_id}/classification",
    response_model=ClassificationResponse,
    summary="Classify a document and return its semantic document type",
)
async def get_classification(document_id: str) -> ClassificationResponse:
    """Run weighted keyword classification on an uploaded document.

    Reads ``extracted_text.txt`` produced by Module 3, computes a weighted
    score for every supported category, applies per-category thresholds, and
    returns the winning class together with a normalised confidence score and
    the list of matched keywords.  The result is persisted to ``metadata.json``
    as ``document_type``, ``confidence``, and ``matched_keywords``.

    Raises:
        404: If the document workspace, ``extracted_text.txt``, or
             ``metadata.json`` does not exist.
        422: If the extracted text file is empty (nothing to classify).
        500: If classification fails for an unexpected reason.
    """
    logger = logging.getLogger("app")

    workspace_dir: Path = settings.UPLOAD_DIR / document_id

    if not workspace_dir.is_dir():
        logger.warning(
            "Classification requested for unknown document: '%s'", document_id
        )
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found.",
        )

    try:
        service = DocumentClassifierService()
        result = service.classify_document(document_id)
    except DocumentClassificationError as exc:
        error_msg = str(exc)
        logger.error(
            "Classification failed — document_id: '%s': %s", document_id, error_msg
        )
        # Surface the specific failure reason with the correct HTTP status.
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail=error_msg)
        if "empty" in error_msg:
            raise HTTPException(status_code=422, detail=error_msg)
        raise HTTPException(status_code=500, detail="Classification failed.")
    except Exception:
        logger.exception(
            "Unexpected error during classification — document_id: '%s'", document_id
        )
        raise HTTPException(status_code=500, detail="Classification failed.")

    return ClassificationResponse(
        document_id=document_id,
        document_type=result["document_type"],
        confidence=result["confidence"],
        matched_keywords=result["matched_keywords"],
    )

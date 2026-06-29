import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config.settings import settings
from app.schemas.extraction import TextExtractionResponse

router = APIRouter(tags=["extraction"])


@router.get(
    "/documents/{document_id}/text",
    response_model=TextExtractionResponse,
    summary="Retrieve extracted text for a document",
)
async def get_extracted_text(document_id: str) -> TextExtractionResponse:
    """Return the extracted plain text for an uploaded document.

    The text is read directly from the pre-computed extracted_text.txt file
    that was written during the upload pipeline.

    Raises:
        404: If the document workspace or extracted_text.txt does not exist.
        422: If the document's metadata indicates extraction failed
             (text_extracted is false).
    """
    logger = logging.getLogger("app")

    workspace_dir: Path = settings.UPLOAD_DIR / document_id

    if not workspace_dir.is_dir():
        logger.warning("Document workspace not found: %s", document_id)
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found.",
        )

    metadata_file = workspace_dir / "metadata.json"
    if not metadata_file.is_file():
        logger.warning("metadata.json missing for document: %s", document_id)
        raise HTTPException(
            status_code=404,
            detail=f"Metadata for document '{document_id}' not found.",
        )

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.exception("Failed to read metadata for document: %s", document_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to read document metadata.",
        )

    text_extracted: bool = metadata.get("text_extracted", False)

    if not text_extracted:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Text extraction was not successful for document '{document_id}'. "
                "The document may contain no embedded text or extraction failed."
            ),
        )

    text_file = workspace_dir / "extracted_text.txt"
    if not text_file.is_file():
        logger.error("extracted_text.txt missing despite text_extracted=true: %s", document_id)
        raise HTTPException(
            status_code=404,
            detail=f"Extracted text file for document '{document_id}' not found.",
        )

    try:
        text = text_file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.exception("Failed to read extracted_text.txt for document: %s", document_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to read extracted text file.",
        )

    return TextExtractionResponse(
        document_id=document_id,
        text_extracted=text_extracted,
        character_count=len(text),
        text=text,
    )

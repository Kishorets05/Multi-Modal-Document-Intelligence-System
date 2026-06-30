"""Chunking API — Module 6.

Endpoint: GET /documents/{document_id}/chunks

Follows the same pattern as ``app/api/entity.py`` and
``app/api/classification.py``:
- Validate the workspace directory.
- Delegate entirely to the service layer.
- Map domain errors to HTTP status codes.
- Return a typed Pydantic response model.

Query Parameters
----------------
max_tokens : int (default 400)
    Soft upper bound on chunk size.  A paragraph that exceeds this limit is
    kept whole (never split).
overlap_tokens : int (default 50)
    Number of words from the end of one chunk that seed the start of the next,
    preserving semantic continuity at boundaries.
"""
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.config.settings import settings
from app.schemas.chunking import ChunkingResponse
from app.services.chunking_service import ChunkingError, ChunkingService

router = APIRouter(tags=["chunking"])


@router.get(
    "/documents/{document_id}/chunks",
    response_model=ChunkingResponse,
    summary="Chunk a document into semantically coherent text segments",
)
async def get_chunks(
    document_id: str,
    max_tokens: int = Query(
        default=400,
        ge=50,
        le=4096,
        description=(
            "Soft upper bound on chunk size (word tokens). "
            "An oversized paragraph is kept whole rather than split."
        ),
    ),
    overlap_tokens: int = Query(
        default=50,
        ge=0,
        le=512,
        description=(
            "Words carried forward from the end of one chunk into the next "
            "to preserve semantic continuity. Set to 0 to disable overlap."
        ),
    ),
) -> ChunkingResponse:
    """Segment a document into semantically coherent chunks.

    Reads ``extracted_text.txt`` and ``metadata.json`` from the document
    workspace.  When the workspace contains an original PDF the chunker uses
    real layout data (font sizes, page numbers, reading order) from PyMuPDF;
    otherwise it falls back to text-heuristic heading detection.

    Chunking rules
    --------------
    * Headings are detected and tracked but never included in chunk bodies.
    * Paragraph boundaries (blank lines between blocks) are always respected.
    * A paragraph is never split — it is placed in its entirety.
    * The overlap window seeds the next chunk with the last *overlap_tokens*
      words of the current chunk.
    * A single paragraph that exceeds *max_tokens* forms its own chunk.

    Raises:
        404: If the document workspace, ``extracted_text.txt``, or
             ``metadata.json`` does not exist.
        422: If the extracted text file is empty.
        500: If chunking fails for an unexpected reason.
    """
    logger = logging.getLogger("app")

    workspace_dir: Path = settings.UPLOAD_DIR / document_id

    if not workspace_dir.is_dir():
        logger.warning(
            "Chunk request for unknown document: '%s'", document_id
        )
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found.",
        )

    try:
        service = ChunkingService(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        result = service.chunk_document(document_id)
    except ChunkingError as exc:
        error_msg = str(exc)
        logger.error(
            "Chunking failed — document_id: '%s': %s", document_id, error_msg
        )
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail=error_msg)
        if "empty" in error_msg:
            raise HTTPException(status_code=422, detail=error_msg)
        raise HTTPException(status_code=500, detail="Chunking failed.")
    except Exception:
        logger.exception(
            "Unexpected error during chunking — document_id: '%s'", document_id
        )
        raise HTTPException(status_code=500, detail="Chunking failed.")

    return ChunkingResponse(
        document_id=result["document_id"],
        document_type=result["document_type"],
        chunk_count=result["chunk_count"],
        chunks=result["chunks"],
    )

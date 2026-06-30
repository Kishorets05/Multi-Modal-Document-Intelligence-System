"""Embeddings API — Module 7.

Endpoint: POST /documents/{document_id}/embed
"""
import logging
from fastapi import APIRouter, HTTPException

from app.schemas.embedding import EmbeddingResponse
from app.services.embedding_service import EmbeddingService, EmbeddingError

router = APIRouter(tags=["embeddings"])


@router.post(
    "/documents/{document_id}/embed",
    response_model=EmbeddingResponse,
    summary="Generate and store embeddings for a document's chunks",
)
async def generate_embeddings(document_id: str) -> EmbeddingResponse:
    """Generate offline embeddings for every chunk of a document and store them in ChromaDB.
    
    This endpoint:
    1. Triggers chunking (via ChunkingService).
    2. Embeds each chunk text using `all-MiniLM-L6-v2`.
    3. Prevents duplicates by overwriting existing chunks for the document.
    4. Persists the embeddings, text, and metadata to ChromaDB.

    Raises:
        404: If the document workspace or required files are not found.
        422: If the document contains empty text.
        500: If embedding generation or storage fails.
    """
    logger = logging.getLogger("app")
    
    try:
        service = EmbeddingService()
        result = service.embed_document(document_id)
    except EmbeddingError as exc:
        error_msg = str(exc)
        logger.error(
            "Embedding failed — document_id: '%s': %s", document_id, error_msg
        )
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        if "empty" in error_msg.lower():
            raise HTTPException(status_code=422, detail=error_msg)
        raise HTTPException(status_code=500, detail="Embedding failed.")
    except Exception:
        logger.exception(
            "Unexpected error during embedding — document_id: '%s'", document_id
        )
        raise HTTPException(status_code=500, detail="Embedding failed.")

    return EmbeddingResponse(**result)

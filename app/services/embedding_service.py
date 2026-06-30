import logging
from typing import Any, Dict

from app.chunkers.engine import ChunkingEngine
from app.chunkers.models import Chunk
from app.config.settings import settings
from app.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from app.services.chunking_service import ChunkingService, ChunkingError
from app.vectordb.chroma_store import ChromaStore


class EmbeddingError(Exception):
    """Raised when embedding or vector storage fails for a document."""


class EmbeddingService:
    """Orchestrate chunking, embedding generation, and vector storage (Module 7)."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("app")
        # Initialize singletons
        try:
            self.embedder = SentenceTransformerEmbedding()
            self.store = ChromaStore()
        except Exception as exc:
            self.logger.error("Failed to initialize embedding components: %s", exc)
            raise EmbeddingError("Failed to initialize embedding model or vector database.") from exc

    def embed_document(self, document_id: str) -> Dict[str, Any]:
        """Chunk the document, embed the chunks, and store in ChromaDB.
        
        Reuses the chunking logic from Module 6.
        """
        self.logger.info("Embedding process started — document_id: '%s'", document_id)
        
        # ── 1. Chunking ────────────────────────────────────────────────────────
        chunking_service = ChunkingService()
        try:
            chunking_result = chunking_service.chunk_document(document_id)
        except ChunkingError as exc:
            raise EmbeddingError(str(exc)) from exc
            
        chunks = chunking_result.get("chunks", [])
        if not chunks:
            self.logger.warning("No chunks produced for document '%s'.", document_id)
            return {
                "document_id": document_id,
                "chunks_embedded": 0,
                "embedding_dimension": self.embedder.dimension,
                "collection": "documents"
            }
            
        # ── 2. Embedding ───────────────────────────────────────────────────────
        # Extract text from each chunk
        texts = [chunk["text"] for chunk in chunks]
        
        try:
            embeddings = self.embedder.embed_texts(texts)
        except Exception as exc:
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc
            
        if len(embeddings) != len(texts):
            raise EmbeddingError("Mismatch between number of chunks and number of generated embeddings.")
            
        # ── 3. Vector Storage ──────────────────────────────────────────────────
        metadatas = []
        ids = []
        for i, chunk in enumerate(chunks):
            # Create a unique ID for each chunk
            ids.append(f"{document_id}_chunk_{i}")
            
            # Prepare metadata mapping. Values must be str, int, float or bool for Chroma.
            metadata = {
                "document_id": document_id,
                "chunk_id": chunk["chunk_id"],
                "heading": chunk["heading"] or "",
                "page_number": chunk["page_number"],
                "document_type": chunk["document_type"] or "",
                "token_count": chunk["token_count"],
                "character_count": chunk["character_count"],
            }
            metadatas.append(metadata)
            
        try:
            self.store.store_chunks(
                document_id=document_id,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts,
                ids=ids
            )
        except Exception as exc:
            raise EmbeddingError(f"Failed to store vectors in ChromaDB: {exc}") from exc
            
        self.logger.info(
            "Embedding process completed — document_id: '%s', chunks: %d", 
            document_id, len(ids)
        )
        
        return {
            "document_id": document_id,
            "chunks_embedded": len(ids),
            "embedding_dimension": self.embedder.dimension,
            "collection": "documents"
        }

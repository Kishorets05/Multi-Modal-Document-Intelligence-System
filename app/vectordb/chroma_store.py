import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config.settings import settings

logger = logging.getLogger("app")

class ChromaStore:
    """Vector storage using ChromaDB."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            logger.info("Initializing ChromaStore...")
            cls._instance = super(ChromaStore, cls).__new__(cls)
            try:
                # Store ChromaDB data inside a 'vectordb' folder in the root or next to uploads
                db_path = str(settings.UPLOAD_DIR.parent / "vectordb")
                os.makedirs(db_path, exist_ok=True)
                
                cls._instance.client = chromadb.PersistentClient(
                    path=db_path,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                # Create one Chroma collection: documents
                cls._instance.collection = cls._instance.client.get_or_create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("ChromaStore initialized successfully at %s", db_path)
            except Exception as e:
                logger.error("Failed to initialize ChromaStore: %s", e)
                raise RuntimeError(f"Failed to initialize Vector DB: {e}")
        return cls._instance
        
    def store_chunks(
        self, 
        document_id: str, 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        documents: List[str], 
        ids: List[str]
    ) -> None:
        """Store chunks, replacing any existing chunks for this document.
        
        Args:
            document_id: The ID of the document being embedded.
            embeddings: List of embedding vectors.
            metadatas: List of metadata dicts for each chunk.
            documents: List of original chunk texts.
            ids: List of unique chunk IDs (e.g., '{document_id}_chunk_{i}').
        """
        if not embeddings:
            return
            
        try:
            # Prevent duplicates by deleting existing chunks for this document_id first
            # ChromaDB supports deleting by 'where' metadata filter
            existing = self.collection.get(where={"document_id": document_id})
            if existing and existing["ids"]:
                logger.info("Deleting %d existing chunks for document %s to prevent duplicates.", len(existing["ids"]), document_id)
                self.collection.delete(where={"document_id": document_id})
                
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
            logger.info("Successfully stored %d chunks for document %s.", len(ids), document_id)
        except Exception as e:
            logger.error("Error storing chunks in ChromaDB for document %s: %s", document_id, e)
            raise RuntimeError(f"Vector storage failed: {e}")

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from app.embeddings.base import BaseEmbeddingModel

logger = logging.getLogger("app")

class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Embedding model using SentenceTransformers.
    
    Loads the model once and reuses the instance.
    """
    
    _instance = None
    _model_name = "all-MiniLM-L6-v2"
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            logger.info("Initializing SentenceTransformerEmbedding (%s)...", cls._model_name)
            cls._instance = super(SentenceTransformerEmbedding, cls).__new__(cls)
            try:
                cls._instance.model = SentenceTransformer(cls._model_name)
                cls._instance._dimension = cls._instance.model.get_sentence_embedding_dimension()
                logger.info("SentenceTransformerEmbedding initialized successfully. Dimension: %d", cls._instance._dimension)
            except Exception as e:
                logger.error("Failed to load SentenceTransformer model %s: %s", cls._model_name, e)
                raise RuntimeError(f"Failed to load embedding model: {e}")
        return cls._instance
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
            
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error("Error during embedding generation: %s", e)
            raise RuntimeError(f"Embedding generation failed: {e}")
            
    @property
    def dimension(self) -> int:
        return self._dimension

import abc

class BaseEmbeddingModel(abc.ABC):
    """Base class for all embedding models."""
    
    @abc.abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text chunks into vectors.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors (list of floats).
        """
        pass
    
    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension of the embedding model."""
        pass

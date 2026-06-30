from pydantic import BaseModel, Field

class EmbeddingResponse(BaseModel):
    """Response model for POST /documents/{document_id}/embed."""
    
    document_id: str = Field(description="Workspace identifier of the source document.")
    chunks_embedded: int = Field(description="Number of chunks successfully embedded and stored.")
    embedding_dimension: int = Field(description="Vector dimension of the embedding model.")
    collection: str = Field(description="Name of the ChromaDB collection.")

from pydantic import BaseModel, Field


class ChunkSchema(BaseModel):
    """Schema for a single document chunk.

    All fields are read-only outputs — they are never supplied by the caller.
    """

    document_id: str = Field(description="Workspace identifier of the source document.")
    chunk_id: int = Field(
        description="Zero-based ordinal position within this document's chunk list.",
    )
    page_number: int = Field(
        description="Page on which the chunk starts (0-based; 0 for text-only documents).",
    )
    heading: str = Field(
        description="Nearest section heading preceding this chunk, or empty string.",
    )
    document_type: str = Field(
        description="Classification label from Module 4.",
    )
    text: str = Field(description="Body text of the chunk.")
    token_count: int = Field(
        description="Approximate word-token count (whitespace-split).",
    )
    character_count: int = Field(description="len(text).")


class ChunkingResponse(BaseModel):
    """Response model for GET /documents/{document_id}/chunks."""

    document_id: str
    document_type: str
    chunk_count: int = Field(
        description="Total number of chunks produced for this document.",
    )
    chunks: list[ChunkSchema] = Field(
        default_factory=list,
        description="Ordered list of chunk objects in reading order.",
    )

from pydantic import BaseModel, Field


class EntityExtractionResponse(BaseModel):
    """Response model for GET /documents/{document_id}/entities.

    The ``entities`` dict is intentionally typed as ``dict`` rather than a
    fixed sub-model so that every document type can return its own field set
    without requiring a separate response schema per type.
    """

    document_id: str
    document_type: str
    entities: dict = Field(
        default_factory=dict,
        description=(
            "Type-specific structured entities extracted from the document. "
            "Field names and value types depend on document_type."
        ),
    )

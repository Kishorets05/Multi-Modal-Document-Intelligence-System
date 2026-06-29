from pydantic import BaseModel, Field


class ClassificationResponse(BaseModel):
    document_id: str
    document_type: str
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised confidence score in [0.0, 1.0].",
    )
    matched_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords from the winning category found in the document.",
    )

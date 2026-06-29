from pydantic import BaseModel


class ClassificationResponse(BaseModel):
    document_id: str
    document_type: str

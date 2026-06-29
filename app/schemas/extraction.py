from pydantic import BaseModel


class TextExtractionResponse(BaseModel):
    document_id: str
    text_extracted: bool
    character_count: int
    text: str

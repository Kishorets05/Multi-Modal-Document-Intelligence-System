from pydantic import BaseModel


class UploadResponse(BaseModel):
    success: bool
    file_id: str
    original_name: str
    stored_name: str
    file_size: int
    file_type: str
    upload_time: str

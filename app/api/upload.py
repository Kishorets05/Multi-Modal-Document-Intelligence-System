import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config.settings import settings
from app.schemas.upload import UploadResponse
from app.services.upload_service import UploadProcessingError, UploadService
from app.utils.file_validator import FileValidationError

router = APIRouter(tags=["upload"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document",
    status_code=201,
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Handle file upload requests and delegate validation/storage to the service."""
    logger = logging.getLogger("app")
    upload_service = UploadService(settings.UPLOAD_DIR)

    try:
        result = await run_in_threadpool(upload_service.save_upload, file)
        return result
    except FileValidationError as exc:
        logger.warning("Upload validation failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except UploadProcessingError as exc:
        logger.error("Upload processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to store uploaded file.")
    except Exception as exc:
        logger.exception("Unexpected upload error.")
        raise HTTPException(status_code=500, detail="Unexpected upload error.")

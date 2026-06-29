from fastapi import APIRouter

from app.api.upload import router as upload_router
from app.api.extraction import router as extraction_router
from app.api.classification import router as classification_router

router = APIRouter()
router.include_router(upload_router)
router.include_router(extraction_router)
router.include_router(classification_router)


@router.get("/", summary="Root health check")
async def root() -> dict:
    return {"message": "Multi-Modal Document Intelligence API Running"}


@router.get("/health", summary="Service health check")
async def health() -> dict:
    return {"status": "healthy"}

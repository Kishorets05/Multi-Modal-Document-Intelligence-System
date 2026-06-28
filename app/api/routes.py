from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="Root health check")
async def root() -> dict:
    return {"message": "Multi-Modal Document Intelligence API Running"}


@router.get("/health", summary="Service health check")
async def health() -> dict:
    return {"status": "healthy"}

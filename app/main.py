import logging

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config.settings import settings
from app.core.logger import setup_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    try:
        setup_logging()
        app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

        app.include_router(api_router)

        @app.on_event("startup")
        async def startup_event() -> None:
            logging.getLogger("app").info("Application startup complete")

        return app
    except Exception as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.error("Failed to initialize application", exc_info=exc)
        raise


app = create_app()

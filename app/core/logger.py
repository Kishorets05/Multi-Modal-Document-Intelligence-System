import logging
from logging.handlers import RotatingFileHandler

from app.config.settings import settings


def setup_logging() -> None:
    """Initialize application logging with a rotating file handler."""
    log_file = settings.LOG_FILE
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = [handler]

    logger.info("Logging initialized")

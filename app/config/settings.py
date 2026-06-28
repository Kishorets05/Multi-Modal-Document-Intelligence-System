from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()


class Settings(BaseSettings):
    PROJECT_NAME: str = "Multi-Modal Document Intelligence"
    PROJECT_VERSION: str = "0.1.0"
    LOG_FILE: Path = Path("logs/app.log")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

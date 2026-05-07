from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    SECRET_KEY: str = "change-this-to-a-random-32-character-string-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    DATABASE_URL: str = "sqlite:///./app/storage/autoeda.db"
    STORAGE_PATH: str = "./app/storage"
    ADMIN_EMAIL: str = "admin@jmangroup.com"
    ADMIN_PASSWORD: str = "Admin@123"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

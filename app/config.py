from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import field_validator


class Settings(BaseSettings):
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    DATABASE_URL: str
    ADMIN_EMAIL: str
    ADMIN_PASSWORD: str
    ADMIN_EMAILS: list[str] = []
    MICROSOFT_EMAILS: list[str] = []

    @field_validator("ADMIN_EMAILS", "MICROSOFT_EMAILS", mode="before")
    def _split_email_list(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    GEMINI_API_KEY: str = ""
    AZURE_TENANT_ID: str = ""
    AZURE_CLIENT_ID: str = ""
    AZURE_CLIENT_SECRET: str = ""
    SHAREPOINT_EXCEL_URL: str = ""
    SHAREPOINT_SHEET: str = "Sheet1"
    SHAREPOINT_TABLE: str = "FeedbackTable"
    # Optional — set to enable Redis-backed event bus for multi-instance deploys
    REDIS_URL: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

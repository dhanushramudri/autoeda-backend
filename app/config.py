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

    # Store raw env values
    ADMIN_EMAILS: str = ""
    MICROSOFT_EMAILS: str = ""

    GEMINI_API_KEY: str = ""

    AZURE_TENANT_ID: str = ""
    AZURE_CLIENT_ID: str = ""
    AZURE_CLIENT_SECRET: str = ""

    SHAREPOINT_EXCEL_URL: str = ""
    SHAREPOINT_SHEET: str = "Sheet1"
    SHAREPOINT_TABLE: str = "FeedbackTable"


    @property
    def admin_emails_list(self) -> list[str]:
        return [
            email.strip()
            for email in self.ADMIN_EMAILS.split(",")
            if email.strip()
        ]

    @property
    def microsoft_emails_list(self) -> list[str]:
        return [
            email.strip()
            for email in self.MICROSOFT_EMAILS.split(",")
            if email.strip()
        ]

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
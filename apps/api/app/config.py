from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str | None = None
    api_cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    raw_retrieval_base_url: str | None = None
    admin_ingest_secret: str | None = None

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.api_cors_origins.split(",")
            if origin.strip()
        ]


settings = Settings()

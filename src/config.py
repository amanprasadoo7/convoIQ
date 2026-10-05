"""Central settings. Values come from environment variables or a local .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"

    # Added later when we reach them (Day 7 and Day 6):
    # database_url: str
    # langfuse_secret_key: str


settings = Settings()

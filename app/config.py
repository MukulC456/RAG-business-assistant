
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Business Assistant"
    app_version: str = "0.1.0"

    database_url: str = (
        "postgresql://postgres:postgres@localhost:5432/rag_business_assistant"
    )

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

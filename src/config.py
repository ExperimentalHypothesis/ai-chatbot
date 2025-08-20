from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    OPENAI_API_KEY: str

    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_EMBEDDING_MODEL: str = "text-embedding-3-small"

    CHAT_MEMORY_KEY: str = "chat_memory"
    CHAT_TURNS: int = 5

    DB_DIR: str = "db"
    DB_COLLECTION: str = "OpenModelica"

    SPLITTER_CHUNK_SIZE: int = 1000
    SPLITTER_OVERLAP: int = 100

    DOCS_DIR: str = "docs"
    DOCS_FILENAME: str = "guide.pdf"

    # short doc just for quick test - delete it after
    TEST_DOC_NAME: str = "test.txt"


settings = Settings()


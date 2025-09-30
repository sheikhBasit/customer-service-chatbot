from pydantic_settings import BaseSettings
from pydantic import AnyUrl
from typing import List

# class ClipEmbeddings:
#     def embed_documents(self, texts):
#         return [embed_text(text) for text in texts]
    
#     def embed_query(self, text):
#         return embed_text(text)

class Settings(BaseSettings):

    ENVIRONMENT: str = "development"
    APP_NAME: str = "FIXIBOT"
    ALLOWED_ORIGINS: List[str] = ["*"]
    MONGODB_URL: AnyUrl
    MONGO_DB: str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_DAYS: int
    EMAIL_TOKEN_EXPIRE_MINUTES: int
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    MAIL_USERNAME: str
    MAIL_PASSWORD: str
    MAIL_FROM: str
    MAIL_SERVER: str
    MAIL_PORT: int
    MAIL_STARTTLS: bool
    MAIL_SSL_TLS: bool
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str
    DEBUG: bool = False
    PORT: int = 8000
    HUGGINGFACEHUB_API_TOKEN: str
    GROQ_API_KEY: str
    OPENAI_API_KEY: str
    HF_TOKEN: str
    USE_ML_VALIDATION: bool = True
    # EMBEDDING_MODEL = ClipEmbeddings()
    # EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    SUPPORT_EMAIL: str = "fixibot038@gmail.com"
    SUPPORT_PHONE: str = "+1-800-123-4567"
    SUPPORT_HOURS: str = "Monday-Friday, 9AM-5PM EST"
    VECTORSTORE_PATH: str = "data/vectorstore.faiss"
    KNOWLEDGE_BASE_PDF: str = "data/Vehicle_Breakdown_Queries.pdf"
    VECTOR_CACHE_DIR: str = ".vector_cache"
    MAX_IMAGE_SIZE_MB: int = 5
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]
    RATE_LIMIT: str = "100/minute"
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_URL: str
    SESSION_TTL_SECONDS: int = 3600

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()
"""
config.py - Configuration settings for Customer Chatbot Service
"""
import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # ==================== APPLICATION SETTINGS ====================
    APP_NAME: str = "Customer Chatbot API Service"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # ==================== DATABASE SETTINGS ====================
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGO_DB: str = os.getenv("MONGO_DB", "cscb")
    
    # MongoDB Collection Names
    CHATBOT_PLANS_COLLECTION: str = "chatbot_plans"
    CUSTOMERS_COLLECTION: str = "customers"
    CUSTOMER_DOCUMENTS_COLLECTION: str = "customer_documents"
    CUSTOMER_VECTORSTORES_COLLECTION: str = "customer_vectorstores"
    CUSTOMER_CHAT_SESSIONS_COLLECTION: str = "customer_chat_sessions"
    USAGE_LOGS_COLLECTION: str = "usage_logs"
    MONTHLY_USAGE_REPORTS_COLLECTION: str = "monthly_usage_reports"
    
    # ==================== REDIS SETTINGS ====================
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", None)
    REDIS_URL: str = os.getenv("REDIS_URL", None)
    
    # ==================== API KEYS ====================
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # ==================== STORAGE SETTINGS ====================
    CUSTOMER_VECTORSTORE_DIR: str = os.getenv("CUSTOMER_VECTORSTORE_DIR", ".customer_vectorstores")
    VECTOR_CACHE_DIR: str = os.getenv("VECTOR_CACHE_DIR", ".vector_cache")
    
    # ==================== UPLOAD SETTINGS ====================
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_DOCUMENT_TYPES: List[str] = [
        "application/pdf",
        "text/plain",
        "text/markdown"
    ]
    
    # ==================== CORS SETTINGS ====================
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]
    
    # Add production origins from environment
    if os.getenv("ALLOWED_ORIGINS"):
        ALLOWED_ORIGINS.extend(os.getenv("ALLOWED_ORIGINS").split(","))
    
    # ==================== SECURITY SETTINGS ====================
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ==================== SESSION SETTINGS ====================
    SESSION_TTL_SECONDS: int = 86400  # 24 hours
    
    # ==================== RATE LIMITING ====================
    DEFAULT_RATE_LIMIT_PER_MINUTE: int = 60
    
    # ==================== MODEL SETTINGS ====================
    USE_ML_VALIDATION: bool = False  # Set to True if you want image validation
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()


# Validate required settings
def validate_settings():
    """Validate that required settings are present"""
    required_settings = {
        "MONGODB_URL": settings.MONGODB_URL,
        "GROQ_API_KEY": settings.GROQ_API_KEY,
        "HF_TOKEN": settings.HF_TOKEN,
    }
    
    missing = [key for key, value in required_settings.items() if not value]
    
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please set them in your .env file or environment."
        )


# Run validation on import
try:
    validate_settings()
except ValueError as e:
    print(f"⚠️ Configuration Warning: {e}")
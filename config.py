"""
config.py - Configuration settings for Customer Chatbot Service
"""
import os
from pydantic_settings import BaseSettings
from pydantic import AnyUrl
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # ==================== APPLICATION SETTINGS ====================
    ENVIRONMENT: str = "development"
    APP_NAME: str = "Customer Chatbot API Service"
    DEBUG: bool = False
    PORT: int = 8000
    
    # ==================== DATABASE SETTINGS ====================
    MONGODB_URL: AnyUrl
    MONGO_DB: str
    
    # MongoDB Collection Names
    CHATBOT_PLANS_COLLECTION: str = "chatbot_plans"
    CUSTOMERS_COLLECTION: str = "customers"
    CUSTOMER_DOCUMENTS_COLLECTION: str = "customer_documents"
    CUSTOMER_VECTORSTORES_COLLECTION: str = "customer_vectorstores"
    CUSTOMER_CHAT_SESSIONS_COLLECTION: str = "customer_chat_sessions"
    USAGE_LOGS_COLLECTION: str = "usage_logs"
    MONTHLY_USAGE_REPORTS_COLLECTION: str = "monthly_usage_reports"
    
    # ==================== REDIS SETTINGS ====================
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_URL: str
    
    # ==================== API KEYS ====================
    GROQ_API_KEY: str
    HF_TOKEN: str
    HUGGINGFACEHUB_API_TOKEN: str
    # OPENAI_API_KEY: str = ""
    
    # ==================== STORAGE SETTINGS ====================
    CUSTOMER_VECTORSTORE_DIR: str = ".customer_vectorstores"
    VECTOR_CACHE_DIR: str = ".vector_cache"
    
    # ==================== UPLOAD SETTINGS ====================
    MAX_UPLOAD_SIZE_MB: int = 50
    MAX_IMAGE_SIZE_MB: int = 5
    ALLOWED_DOCUMENT_TYPES: List[str] = [
        "application/pdf",
        "text/plain",
        "text/markdown"
    ]
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]
    
    # ==================== CORS SETTINGS ====================
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # ==================== SECURITY SETTINGS ====================
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ==================== SESSION SETTINGS ====================
    SESSION_TTL_SECONDS: int = 86400  # 24 hours
    
    # ==================== RATE LIMITING ====================
    DEFAULT_RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT: str = "100/minute"
    
    # ==================== MODEL SETTINGS ====================
    USE_ML_VALIDATION: bool = False
    
    # ==================== SUPPORT SETTINGS ====================
    # SUPPORT_EMAIL: str = "fixibot038@gmail.com"
    # SUPPORT_PHONE: str = "+1-800-123-4567"
    # SUPPORT_HOURS: str = "Monday-Friday, 9AM-5PM EST"
    
    # Email Settings (if needed)
    # MAIL_USERNAME: str = ""
    # MAIL_PASSWORD: str = ""
    # MAIL_FROM: str = ""
    # MAIL_SERVER: str = ""
    # MAIL_PORT: int = 587
    # MAIL_STARTTLS: bool = True
    # MAIL_SSL_TLS: bool = False
    
    # Cloudinary (if needed)
    # CLOUDINARY_CLOUD_NAME: str = ""
    # CLOUDINARY_API_KEY: str = ""
    # CLOUDINARY_API_SECRET: str = ""
    
    # Google OAuth (if needed)
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()

# Set environment variables for libraries that expect them
# This happens automatically when config is imported
import os
os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
os.environ["HF_TOKEN"] = settings.HF_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.HUGGINGFACEHUB_API_TOKEN

# if settings.OPENAI_API_KEY:
#     os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


# Validate required settings
def validate_settings():
    """Validate that required settings are present"""
    required_settings = {
        "MONGODB_URL": settings.MONGODB_URL,
        "MONGO_DB": settings.MONGO_DB,
        "GROQ_API_KEY": settings.GROQ_API_KEY,
        "HF_TOKEN": settings.HF_TOKEN,
        "SECRET_KEY": settings.SECRET_KEY,
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
    print("✅ Configuration loaded successfully!")
    print(f"   Environment: {settings.ENVIRONMENT}")
    print(f"   Database: {settings.MONGO_DB}")
    print(f"   GROQ API Key: {'✓ Set' if settings.GROQ_API_KEY else '✗ Missing'}")
    print(f"   HF Token: {'✓ Set' if settings.HF_TOKEN else '✗ Missing'}")
except ValueError as e:
    print(f"⚠️ Configuration Warning: {e}")
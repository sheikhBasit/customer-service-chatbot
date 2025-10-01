"""
models/customer_chatbot.py - Data models for custom chatbot service
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from bson import ObjectId
from utils.py_object import PyObjectId

# ==================== CUSTOMER & PLAN MODELS ====================

class ChatbotPlan(BaseModel):
    """Subscription plan configuration"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str  # e.g., "Starter", "Professional", "Enterprise"
    price_monthly: float
    max_documents: int
    max_queries_per_month: int
    max_concurrent_users: int
    websocket_enabled: bool = True
    custom_branding: bool = False
    api_rate_limit: int = 100  # requests per minute
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Customer(BaseModel):
    """Customer/Company using the chatbot service"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    company_name: str
    email: str
    api_key: str = Field(default_factory=lambda: f"cbk_{PyObjectId()}")
    plan_id: PyObjectId
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Usage tracking
    current_month_queries: int = 0
    total_queries: int = 0
    documents_count: int = 0
    
    # Customization
    chatbot_name: Optional[str] = "AI Assistant"
    chatbot_greeting: Optional[str] = "Hello! How can I help you today?"
    brand_color: Optional[str] = "#0066cc"
    logo_url: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# ==================== DOCUMENT & VECTORSTORE MODELS ====================

class CustomerDocument(BaseModel):
    """Uploaded document metadata"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    customer_id: PyObjectId
    filename: str
    file_size: int
    file_type: str  # pdf, docx, txt, etc.
    upload_date: datetime = Field(default_factory=datetime.now)
    processed: bool = False
    processing_status: Literal["pending", "processing", "completed", "failed"] = "pending"
    error_message: Optional[str] = None
    
    # Vectorstore reference
    vectorstore_cache_key: Optional[str] = None
    chunks_count: int = 0
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CustomerVectorStore(BaseModel):
    """Mapping of customer to their vectorstore"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    customer_id: PyObjectId
    cache_key: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    total_embeddings: int = 0
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# ==================== CHAT SESSION MODELS ====================

class CustomerChatSession(BaseModel):
    """Chat session for end-user on customer's website"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    customer_id: PyObjectId
    session_token: str = Field(default_factory=lambda: f"sess_{PyObjectId()}")
    
    # End-user identification (optional, customer can implement their own)
    end_user_id: Optional[str] = None
    end_user_name: Optional[str] = "Guest"
    
    # Session data
    messages: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    
    # Metadata
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    referrer_url: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ChatMessage(BaseModel):
    """Individual message in a chat session"""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


# ==================== USAGE TRACKING MODELS ====================

class UsageLog(BaseModel):
    """Track API usage for billing and analytics"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    customer_id: PyObjectId
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Request details
    endpoint: str  # e.g., "/chat", "/websocket"
    method: str
    session_id: Optional[PyObjectId] = None
    
    # Usage metrics
    query_tokens: int = 0
    response_tokens: int = 0
    processing_time_ms: int = 0
    
    # Billing
    billable: bool = True
    month_year: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m"))
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class MonthlyUsageReport(BaseModel):
    """Aggregated monthly usage"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    customer_id: PyObjectId
    month_year: str
    
    total_queries: int = 0
    total_tokens: int = 0
    total_sessions: int = 0
    unique_users: int = 0
    avg_response_time_ms: float = 0.0
    
    generated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# ==================== WEBSOCKET MODELS ====================

class WebSocketConnection(BaseModel):
    """Track active WebSocket connections"""
    customer_id: str
    session_token: str
    connection_id: str
    connected_at: datetime = Field(default_factory=datetime.now)
    last_ping: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: Literal["query", "response", "error", "system", "typing"]
    content: str
    session_token: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
"""
app_customer_chatbot.py - Main application file for customer chatbot service
Integrate this with your existing app.py
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import AsyncIterator
import logging

from routes import websocket_chatbot, customer_management
from config import settings
from database import connect_to_mongo, close_mongo_connection
from services.multimodal_embeddings import initialize_clip_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan_customer_chatbot(app: FastAPI) -> AsyncIterator[None]:
    """Enhanced lifespan with customer chatbot initialization"""
    try:
        # Startup
        logger.info("🚀 Starting Customer Chatbot Service...")
        
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        app.state.mongo_client = await connect_to_mongo()
        logger.info("✓ MongoDB connected")
        
        # Initialize CLIP model for embeddings
        logger.info("Initializing CLIP model...")
        initialize_clip_model()
        logger.info("✓ CLIP model initialized")
        
        # Create necessary indexes
        from database import db
        await ensure_indexes(db)
        logger.info("✓ Database indexes created")
        
        # Schedule cleanup tasks
        import asyncio
        asyncio.create_task(schedule_cleanup_tasks())
        logger.info("✓ Cleanup tasks scheduled")
        
        logger.info("✅ Customer Chatbot Service ready!")
        
        yield
        
    except Exception as e:
        logger.critical(f"Startup failed: {e}", exc_info=True)
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Customer Chatbot Service...")
        if hasattr(app.state, "mongo_client") and app.state.mongo_client:
            await close_mongo_connection()
            logger.info("✓ MongoDB disconnected")
        logger.info("✅ Shutdown complete")


async def ensure_indexes(db):
    """Ensure all required indexes exist"""
    collections_indexes = {
        "customers": [
            ("api_key", {"unique": True}),
            ("email", {"unique": True}),
            ("is_active", {})
        ],
        "customer_documents": [
            ("customer_id", {}),
            ([("customer_id", 1), ("processed", 1)], {})
        ],
        "customer_vectorstores": [
            ("customer_id", {"unique": True})
        ],
        "customer_chat_sessions": [
            ("session_token", {"unique": True}),
            ("customer_id", {}),
            ([("customer_id", 1), ("is_active", 1)], {}),
            ("last_activity", {})
        ],
        "usage_logs": [
            ([("customer_id", 1), ("month_year", 1)], {}),
            ("timestamp", {})
        ]
    }
    
    for collection_name, indexes in collections_indexes.items():
        collection = getattr(db, f"{collection_name}_collection", None)
        if collection:
            for index_def in indexes:
                if isinstance(index_def[0], list):
                    await collection.create_index(index_def[0], **index_def[1])
                else:
                    await collection.create_index(index_def[0], **index_def[1])


async def schedule_cleanup_tasks():
    """Schedule periodic cleanup tasks"""
    import asyncio
    from datetime import datetime, timedelta
    
    while True:
        try:
            # Clean up old sessions every hour
            await asyncio.sleep(3600)  # 1 hour
            
            from database import db
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            result = await db.customer_chat_sessions_collection.delete_many({
                "last_activity": {"$lt": cutoff_time},
                "is_active": True
            })
            
            logger.info(f"Cleaned up {result.deleted_count} inactive sessions")
            
        except Exception as e:
            logger.error(f"Cleanup task error: {e}", exc_info=True)


# Create FastAPI app
app = FastAPI(
    title="Customer Chatbot API Service",
    version="1.0.0",
    description="Scalable custom chatbot API service with RAG capabilities",
    lifespan=lifespan_customer_chatbot
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(websocket_chatbot.router)
app.include_router(customer_management.router)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Customer Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/api/v1/chatbot/ws",
            "http_query": "/api/v1/chatbot/query",
            "admin": "/api/v1/admin"
        }
    }


@app.get("/health")
async def health_check(request: Request):
    """Detailed health check"""
    try:
        from database import db
        
        # Check MongoDB
        await db.customers_collection.find_one({})
        mongo_status = "healthy"
    except Exception as e:
        mongo_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok",
        "services": {
            "mongodb": mongo_status,
            "embeddings": "healthy"
        }
    }


# ==================== ADDITIONAL UTILITY ENDPOINTS ====================

@app.post("/api/v1/admin/reset-monthly-counters")
async def reset_monthly_counters():
    """
    Reset monthly query counters for all customers
    Should be called at the start of each month (can be automated with a cron job)
    """
    try:
        from database import db
        
        result = await db.customers_collection.update_many(
            {},
            {"$set": {"current_month_queries": 0}}
        )
        
        logger.info(f"Reset counters for {result.modified_count} customers")
        
        return {
            "status": "success",
            "message": f"Reset counters for {result.modified_count} customers"
        }
    except Exception as e:
        logger.error(f"Error resetting counters: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.get("/api/v1/admin/stats")
async def get_platform_stats():
    """Get overall platform statistics"""
    try:
        from database import db
        from datetime import datetime
        
        # Count active customers
        active_customers = await db.customers_collection.count_documents({"is_active": True})
        
        # Count total documents
        total_documents = await db.customer_documents_collection.count_documents({})
        
        # Count active sessions
        active_sessions = await db.customer_chat_sessions_collection.count_documents({
            "is_active": True
        })
        
        # Get current month queries
        current_month = datetime.now().strftime("%Y-%m")
        month_queries = await db.usage_logs_collection.count_documents({
            "month_year": current_month
        })
        
        return {
            "active_customers": active_customers,
            "total_documents": total_documents,
            "active_sessions": active_sessions,
            "queries_this_month": month_queries,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app_customer_chatbot:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
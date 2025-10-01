"""
database.py - MongoDB connection and collections for Customer Chatbot Service
"""
from motor.motor_asyncio import AsyncIOMotorClient
from config import settings
import logging
import certifi
from pymongo.errors import OperationFailure

logger = logging.getLogger(__name__)

class Database:
    client: AsyncIOMotorClient = None
    db = None
    
    # Customer Chatbot Collections
    chatbot_plans_collection = None
    customers_collection = None
    customer_documents_collection = None
    customer_vectorstores_collection = None
    customer_chat_sessions_collection = None
    usage_logs_collection = None
    monthly_usage_reports_collection = None

db = Database()

async def connect_to_mongo():
    try:
        db.client = AsyncIOMotorClient(
            str(settings.MONGODB_URL),
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=True,  # Changed to True to fix SSL issues
            retryWrites=True,
            w="majority",
            appName="CustomerChatbot",
            maxPoolSize=100,
            minPoolSize=10,
            connectTimeoutMS=30000,
            serverSelectionTimeoutMS=30000
        )
        
        await db.client.admin.command('ping')
        db.db = db.client[settings.MONGO_DB]
        
        # Initialize all collections
        db.chatbot_plans_collection = db.db[settings.CHATBOT_PLANS_COLLECTION]
        db.customers_collection = db.db[settings.CUSTOMERS_COLLECTION]
        db.customer_documents_collection = db.db[settings.CUSTOMER_DOCUMENTS_COLLECTION]
        db.customer_vectorstores_collection = db.db[settings.CUSTOMER_VECTORSTORES_COLLECTION]
        db.customer_chat_sessions_collection = db.db[settings.CUSTOMER_CHAT_SESSIONS_COLLECTION]
        db.usage_logs_collection = db.db[settings.USAGE_LOGS_COLLECTION]
        db.monthly_usage_reports_collection = db.db[settings.MONTHLY_USAGE_REPORTS_COLLECTION]

        # Create indexes with error handling for existing indexes
        try:
            # Chatbot Plans Indexes
            await db.chatbot_plans_collection.create_index("name", unique=True)
            
            # Customers Indexes
            await db.customers_collection.create_index("api_key", unique=True)
            await db.customers_collection.create_index("email", unique=True)
            await db.customers_collection.create_index("is_active")
            await db.customers_collection.create_index("plan_id")
            
            # Customer Documents Indexes
            await db.customer_documents_collection.create_index("customer_id")
            await db.customer_documents_collection.create_index([("customer_id", 1), ("processed", 1)])
            await db.customer_documents_collection.create_index("processing_status")
            
            # Customer Vectorstores Indexes
            await db.customer_vectorstores_collection.create_index("customer_id", unique=True)
            
            # Customer Chat Sessions Indexes
            await db.customer_chat_sessions_collection.create_index("session_token", unique=True)
            await db.customer_chat_sessions_collection.create_index("customer_id")
            await db.customer_chat_sessions_collection.create_index([("customer_id", 1), ("is_active", 1)])
            await db.customer_chat_sessions_collection.create_index([("last_activity", -1)])
            await db.customer_chat_sessions_collection.create_index("end_user_id", sparse=True)
            
            # Usage Logs Indexes
            await db.usage_logs_collection.create_index([("customer_id", 1), ("month_year", 1)])
            await db.usage_logs_collection.create_index([("timestamp", -1)])
            await db.usage_logs_collection.create_index("session_id", sparse=True)
            
            # Monthly Usage Reports Indexes
            await db.monthly_usage_reports_collection.create_index(
                [("customer_id", 1), ("month_year", 1)], 
                unique=True
            )
            
            logger.info("MongoDB indexes created successfully.")
        except OperationFailure as e:
            logger.warning(f"Failed to create some MongoDB indexes: {e}. This may be because they already exist.")
        
        logger.info("Successfully connected to MongoDB")
        
        return db.client
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    try:
        if db.client:
            db.client.close()
            logger.info("Closed MongoDB connection")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")
        raise

# Helper functions for common database operations

async def get_customer_by_api_key(api_key: str):
    """Get customer by API key"""
    return await db.customers_collection.find_one({"api_key": api_key, "is_active": True})

async def get_customer_plan(plan_id):
    """Get plan details by ID"""
    from bson import ObjectId
    return await db.chatbot_plans_collection.find_one({"_id": ObjectId(plan_id)})

async def increment_customer_usage(customer_id):
    """Increment customer's query counter"""
    from bson import ObjectId
    await db.customers_collection.update_one(
        {"_id": ObjectId(customer_id)},
        {
            "$inc": {
                "current_month_queries": 1,
                "total_queries": 1
            }
        }
    )

async def check_customer_limit(customer_id) -> bool:
    """Check if customer has exceeded their monthly limit"""
    from bson import ObjectId
    
    customer = await db.customers_collection.find_one({"_id": ObjectId(customer_id)})
    if not customer:
        return False
    
    plan = await get_customer_plan(customer["plan_id"])
    if not plan:
        return False
    
    current_usage = customer.get("current_month_queries", 0)
    max_queries = plan.get("max_queries_per_month", 0)
    
    return current_usage < max_queries
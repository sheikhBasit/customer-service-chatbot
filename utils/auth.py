"""
utils/auth.py - Authentication and authorization for customer API
"""
from typing import Optional
import time
from fastapi import HTTPException, Header
from bson import ObjectId
import redis
from datetime import datetime

from models.customer_chatbot import Customer
from database import db
from config import settings

# Redis client for rate limiting (reuse from existing setup)
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    decode_responses=True
)


async def verify_customer_api_key(api_key: str) -> Optional[Customer]:
    """
    Verify customer API key and return customer object
    
    Args:
        api_key: API key from request
        
    Returns:
        Customer object if valid, None otherwise
    """
    try:
        customer_data = await db.customers_collection.find_one({
            "api_key": api_key,
            "is_active": True
        })
        
        if not customer_data:
            return None
        
        return Customer(**customer_data)
    
    except Exception as e:
        print(f"Error verifying API key: {e}")
        return None


async def get_customer_from_header(
    x_api_key: str = Header(..., alias="X-API-Key")
) -> Customer:
    """
    FastAPI dependency for API key authentication
    
    Usage:
        @router.get("/endpoint")
        async def endpoint(customer: Customer = Depends(get_customer_from_header)):
            ...
    """
    customer = await verify_customer_api_key(x_api_key)
    if not customer:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return customer


async def check_customer_rate_limit(
    customer_id: str,
    rate_limit: int  # requests per minute
) -> bool:
    """
    Check if customer has exceeded their rate limit
    
    Args:
        customer_id: Customer ID
        rate_limit: Maximum requests per minute
        
    Returns:
        True if within limit, False if exceeded
    """
    try:
        key = f"rate_limit:{customer_id}"
        current_minute = int(time.time() / 60)
        redis_key = f"{key}:{current_minute}"
        
        # Increment counter
        current_count = redis_client.incr(redis_key)
        
        # Set expiry on first request of the minute
        if current_count == 1:
            redis_client.expire(redis_key, 60)
        
        return current_count <= rate_limit
    
    except Exception as e:
        print(f"Rate limit check error: {e}")
        # Fail open - allow request if rate limiting fails
        return True


async def check_monthly_query_limit(customer: Customer, plan_limit: int) -> bool:
    """
    Check if customer has exceeded their monthly query limit
    
    Args:
        customer: Customer object
        plan_limit: Maximum queries per month from plan
        
    Returns:
        True if within limit, False if exceeded
    """
    return customer.current_month_queries < plan_limit


async def reset_monthly_counters():
    """
    Reset monthly query counters for all customers
    Should be run as a scheduled task at the start of each month
    """
    try:
        result = await db.customers_collection.update_many(
            {},
            {"$set": {"current_month_queries": 0}}
        )
        print(f"Reset monthly counters for {result.modified_count} customers")
    except Exception as e:
        print(f"Error resetting monthly counters: {e}")


# ==================== USAGE TRACKING ====================

async def track_api_request(
    customer_id: str,
    endpoint: str,
    method: str,
    processing_time_ms: int,
    session_id: Optional[ObjectId] = None
):
    """
    Track API usage for analytics and billing
    
    Args:
        customer_id: Customer ID
        endpoint: API endpoint called
        method: HTTP method or "WS" for WebSocket
        processing_time_ms: Time taken to process request
        session_id: Optional session ID
    """
    from models.customer_chatbot import UsageLog
    
    usage_log = UsageLog(
        customer_id=ObjectId(customer_id),
        endpoint=endpoint,
        method=method,
        session_id=session_id,
        processing_time_ms=processing_time_ms
    )
    
    await db.usage_logs_collection.insert_one(usage_log.model_dump(by_alias=True))


async def get_customer_usage_stats(customer_id: str, month_year: Optional[str] = None):
    """
    Get usage statistics for a customer
    
    Args:
        customer_id: Customer ID
        month_year: Optional month in "YYYY-MM" format, defaults to current month
        
    Returns:
        Dictionary with usage statistics
    """
    if not month_year:
        month_year = datetime.now().strftime("%Y-%m")
    
    pipeline = [
        {
            "$match": {
                "customer_id": ObjectId(customer_id),
                "month_year": month_year
            }
        },
        {
            "$group": {
                "_id": "$customer_id",
                "total_queries": {"$sum": 1},
                "total_tokens": {"$sum": {"$add": ["$query_tokens", "$response_tokens"]}},
                "avg_processing_time": {"$avg": "$processing_time_ms"},
                "total_sessions": {"$addToSet": "$session_id"}
            }
        }
    ]
    
    result = await db.usage_logs_collection.aggregate(pipeline).to_list(1)
    
    if result:
        stats = result[0]
        stats["unique_sessions"] = len([s for s in stats["total_sessions"] if s])
        del stats["total_sessions"]
        return stats
    
    return {
        "total_queries": 0,
        "total_tokens": 0,
        "avg_processing_time": 0,
        "unique_sessions": 0
    }
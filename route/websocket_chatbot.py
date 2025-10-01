"""
routes/websocket_chatbot.py - WebSocket endpoint for real-time chatbot
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from typing import Dict, Set, Optional
import json
import logging
from datetime import datetime
from bson import ObjectId
import asyncio

from models.customer_chatbot import (
    Customer, CustomerChatSession, ChatMessage, 
    WebSocketConnection, WebSocketMessage, UsageLog
)
from services.customer_vectorstore import CustomerVectorStoreService
from services.customer_chatbot_engine import CustomerChatbotEngine
from database import db
from utils.auth import verify_customer_api_key, check_customer_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chatbot", tags=["Customer Chatbot"])

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        # customer_id -> {session_token -> WebSocket}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.connection_metadata: Dict[str, WebSocketConnection] = {}
    
    async def connect(
        self, 
        websocket: WebSocket, 
        customer_id: str, 
        session_token: str
    ):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        if customer_id not in self.active_connections:
            self.active_connections[customer_id] = {}
        
        self.active_connections[customer_id][session_token] = websocket
        
        # Store metadata
        connection_id = f"{customer_id}_{session_token}"
        self.connection_metadata[connection_id] = WebSocketConnection(
            customer_id=customer_id,
            session_token=session_token,
            connection_id=connection_id
        )
        
        logger.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, customer_id: str, session_token: str):
        """Remove a WebSocket connection"""
        if customer_id in self.active_connections:
            self.active_connections[customer_id].pop(session_token, None)
            
            if not self.active_connections[customer_id]:
                del self.active_connections[customer_id]
        
        connection_id = f"{customer_id}_{session_token}"
        self.connection_metadata.pop(connection_id, None)
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_message(
        self, 
        customer_id: str, 
        session_token: str, 
        message: dict
    ):
        """Send message to specific session"""
        if customer_id in self.active_connections:
            websocket = self.active_connections[customer_id].get(session_token)
            if websocket:
                await websocket.send_json(message)
    
    async def send_typing_indicator(
        self, 
        customer_id: str, 
        session_token: str, 
        is_typing: bool
    ):
        """Send typing indicator to client"""
        await self.send_message(customer_id, session_token, {
            "type": "typing",
            "is_typing": is_typing
        })
    
    def get_customer_connection_count(self, customer_id: str) -> int:
        """Get number of active connections for a customer"""
        return len(self.active_connections.get(customer_id, {}))

manager = ConnectionManager()

# Initialize services
vectorstore_service = CustomerVectorStoreService()
chatbot_engine = CustomerChatbotEngine()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    api_key: str = Query(..., description="Customer API key"),
    session_token: Optional[str] = Query(None, description="Session token (optional)"),
    end_user_id: Optional[str] = Query(None, description="End user identifier (optional)"),
    system_prompt: Optional[str] = Query(None, description="System prompt (optional)"),
    temperature: Optional[float] = Query(None, description="Temperature (optional)")
):
    """
    WebSocket endpoint for real-time chatbot communication
    
    Query Parameters:
    - api_key: Customer's API key for authentication
    - session_token: Optional session token to resume conversation
    - end_user_id: Optional identifier for the end user
    """
    customer = None
    session = None
    customer_id = None
    
    try:
        # Authenticate customer
        customer = await verify_customer_api_key(api_key)
        if not customer or not customer.is_active:
            await websocket.close(code=4001, reason="Invalid or inactive API key")
            return
        
        customer_id = str(customer.id)
        
        # Check concurrent connection limit
        plan = await db.chatbot_plans_collection.find_one({"_id": customer.plan_id})
        if not plan:
            await websocket.close(code=4002, reason="Invalid subscription plan")
            return
        
        current_connections = manager.get_customer_connection_count(customer_id)
        if current_connections >= plan["max_concurrent_users"]:
            await websocket.close(code=4003, reason="Concurrent user limit reached")
            return
        
        # Get or create session
        if session_token:
            session = await db.customer_chat_sessions_collection.find_one({
                "session_token": session_token,
                "customer_id": customer.id
            })
            if session:
                session = CustomerChatSession(**session)
        
        if not session:
            # Create new session
            session = CustomerChatSession(
                customer_id=customer.id,
                end_user_id=end_user_id
            )
            await db.customer_chat_sessions_collection.insert_one(
                session.model_dump(by_alias=True)
            )
        
        # Connect WebSocket
        await manager.connect(websocket, customer_id, session.session_token)
        
        # Send welcome message
        await websocket.send_json({
            "type": "system",
            "content": customer.chatbot_greeting,
            "session_token": session.session_token,
            "chatbot_name": customer.chatbot_name
        })
        
        # Main message loop
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_query = message_data.get("content", "").strip()
            if not user_query:
                continue
            
            # Check rate limit
            if not await check_customer_rate_limit(customer_id, plan["api_rate_limit"]):
                await websocket.send_json({
                    "type": "error",
                    "content": "Rate limit exceeded. Please try again later."
                })
                continue
            
            # Track usage start time
            start_time = datetime.now()
            
            # Send typing indicator
            await manager.send_typing_indicator(customer_id, session.session_token, True)
            
            try:
                # Process query through chatbot engine
                response = await chatbot_engine.process_query(
                    customer_id=customer_id,
                    session=session,
                    query=user_query,
                    system_prompt=system_prompt,
                    temperature=temperature
                )
                
                # Send response
                await websocket.send_json({
                    "type": "response",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update session history
                session.messages.extend([
                    ChatMessage(role="user", content=user_query).model_dump(),
                    ChatMessage(role="assistant", content=response).model_dump()
                ])
                session.last_activity = datetime.now()
                
                await db.customer_chat_sessions_collection.update_one(
                    {"_id": session.id},
                    {
                        "$set": {
                            "messages": session.messages,
                            "last_activity": session.last_activity
                        }
                    }
                )
                
                # Log usage
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                await log_usage(
                    customer_id=customer_id,
                    session_id=session.id,
                    processing_time_ms=int(processing_time),
                    query_tokens=len(user_query.split()),
                    response_tokens=len(response.split())
                )
                
                # Update customer usage counter
                await db.customers_collection.update_one(
                    {"_id": customer.id},
                    {
                        "$inc": {
                            "current_month_queries": 1,
                            "total_queries": 1
                        }
                    }
                )
                
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "content": "Sorry, I encountered an error processing your request."
                })
            finally:
                # Stop typing indicator
                await manager.send_typing_indicator(customer_id, session.session_token, False)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for customer {customer_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass
    finally:
        if customer_id and session:
            manager.disconnect(customer_id, session.session_token)


async def log_usage(
    customer_id: str,
    session_id: ObjectId,
    processing_time_ms: int,
    query_tokens: int,
    response_tokens: int
):
    """Log usage for billing and analytics"""
    usage_log = UsageLog(
        customer_id=ObjectId(customer_id),
        endpoint="/websocket",
        method="WS",
        session_id=session_id,
        query_tokens=query_tokens,
        response_tokens=response_tokens,
        processing_time_ms=processing_time_ms
    )
    
    await db.usage_logs_collection.insert_one(usage_log.model_dump(by_alias=True))


# ==================== REST API ENDPOINTS ====================

@router.post("/query")
async def http_chatbot_query(
    api_key: str = Query(...),
    session_token: Optional[str] = Query(None),
    query: str = Query(...),
    end_user_id: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    temperature: Optional[float] = Query(None)
):
    """
    HTTP endpoint for chatbot queries (alternative to WebSocket)
    """
    # Authenticate
    customer = await verify_customer_api_key(api_key)
    if not customer or not customer.is_active:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    
    customer_id = str(customer.id)
    
    # Check rate limit
    plan = await db.chatbot_plans_collection.find_one({"_id": customer.plan_id})
    if not await check_customer_rate_limit(customer_id, plan["api_rate_limit"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Get or create session
    if session_token:
        session = await db.customer_chat_sessions_collection.find_one({
            "session_token": session_token,
            "customer_id": customer.id
        })
        if session:
            session = CustomerChatSession(**session)
    
    if not session:
        session = CustomerChatSession(
            customer_id=customer.id,
            end_user_id=end_user_id
        )
        await db.customer_chat_sessions_collection.insert_one(
            session.model_dump(by_alias=True)
        )
    
    start_time = datetime.now()
    
    try:
        # Process query
        response = await chatbot_engine.process_query(
            customer_id=customer_id,
            session=session,
            query=query,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        # Update session
        session.messages.extend([
            ChatMessage(role="user", content=query).model_dump(),
            ChatMessage(role="assistant", content=response).model_dump()
        ])
        session.last_activity = datetime.now()
        
        await db.customer_chat_sessions_collection.update_one(
            {"_id": session.id},
            {"$set": {"messages": session.messages, "last_activity": session.last_activity}}
        )
        
        # Log usage
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        await log_usage(
            customer_id=customer_id,
            session_id=session.id,
            processing_time_ms=int(processing_time),
            query_tokens=len(query.split()),
            response_tokens=len(response.split())
        )
        
        # Update usage counter
        await db.customers_collection.update_one(
            {"_id": customer.id},
            {"$inc": {"current_month_queries": 1, "total_queries": 1}}
        )
        
        return {
            "response": response,
            "session_token": session.session_token,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error processing HTTP query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process query")


@router.get("/session/{session_token}/history")
async def get_session_history(
    session_token: str,
    api_key: str = Query(...)
):
    """Get chat history for a session"""
    customer = await verify_customer_api_key(api_key)
    if not customer:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    session = await db.customer_chat_sessions_collection.find_one({
        "session_token": session_token,
        "customer_id": customer.id
    })
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_token": session_token,
        "messages": session.get("messages", []),
        "created_at": session.get("created_at"),
        "last_activity": session.get("last_activity")
    }
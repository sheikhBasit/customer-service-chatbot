import json
from typing import ClassVar, Optional, List, cast

from bson import ObjectId
from models.chat import ChatState, ChatSession
from config import settings
import redis
import logging
from redis import Redis
from pydantic import ValidationError
from datetime import datetime
from database import db

logger = logging.getLogger(__name__)

class SessionManager:
    _redis_client: ClassVar[Optional[Redis]] = None
    
    # Redis-related methods
    @classmethod
    def get_redis_client(cls) -> Redis:
        """Get or create Redis client with connection test"""
        if cls._redis_client is None:
            try:
                if settings.REDIS_URL:
                    url: str = str(settings.REDIS_URL)
                    cls._redis_client = redis.Redis.from_url( # type: ignore
                        url=url,
                        decode_responses=False
                    )
                else:
                    cls._redis_client = redis.Redis(
                        host=settings.REDIS_HOST,
                        port=settings.REDIS_PORT,
                        password=settings.REDIS_PASSWORD,
                        decode_responses=False
                    )
                
                # Test connection
                if not cls._redis_client.ping(): # type: ignore
                    raise ConnectionError("Redis ping failed")
            except Exception as e:
                logger.error("Failed to connect to Redis: %s", str(e))
                raise ConnectionError(f"Redis connection failed: {str(e)}") from e
        return cls._redis_client

    @classmethod
    def save_session_state(cls, session_id: str, chat_state: ChatState) -> bool:
        """Save chat state to Redis with TTL"""
        try:
            redis_client = cls.get_redis_client()
            serialized = chat_state.model_dump_json()
            result = redis_client.setex(
                name=f"chat_session:{session_id}",
                time=settings.SESSION_TTL_SECONDS,
                value=serialized
            )
            return bool(result)
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error("Failed to save session state %s: %s", session_id, str(e), exc_info=True)
            return False

    @classmethod
    def get_session_state(cls, session_id: str) -> Optional[ChatState]:
        """Retrieve chat state from Redis"""
        try:
            redis_client = cls.get_redis_client()
            serialized = redis_client.get(f"chat_session:{session_id}")
            
            if not serialized:
                return None
                
            # Ensure we have a string for model validation
            serialized_str = serialized.decode('utf-8') if isinstance(serialized, bytes) else str(serialized)
            return ChatState.model_validate_json(serialized_str)
        except (redis.RedisError, ValidationError, json.JSONDecodeError, TypeError) as e:
            logger.error("Failed to load session state %s: %s", session_id, str(e), exc_info=True)
            return None

    @classmethod
    def delete_session_state(cls, session_id: str) -> bool:
        """Delete session state from Redis"""
        try:
            redis_client = cls.get_redis_client()
            # Explicitly cast the delete result to int
            deleted = cast(int, redis_client.delete(f"chat_session:{session_id}"))
            return deleted > 0
        except redis.RedisError as e:
            logger.error("Failed to delete session state %s: %s", session_id, str(e), exc_info=True)
            return False

    @staticmethod
    async def create_chat_session(user_id: str, initial_title: str = "New Chat") -> ChatSession:
        """Create a new chat session and return the created ChatSession object."""
        try:
            logger.debug(f"Creating chat session for user: {user_id}")
            
            chat_session = ChatSession(
                user_id=ObjectId(user_id),  # Ensure user_id is ObjectId
                chat_title=initial_title
            )
            session_data = chat_session.model_dump(by_alias=True)
            logger.debug(f"Inserting session data: {session_data}")
            
            result = await db.chat_sessions_collection.insert_one(session_data)
            logger.debug(f"Inserted new chat session with _id: {result.inserted_id}")
            
            chat_session.id = result.inserted_id
            return chat_session

        except Exception as e:
            logger.error(f"Failed to create chat session for user {user_id}: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to create chat session: {str(e)}")

    @staticmethod
    async def get_chat_session(session_id: str) -> Optional[ChatSession]:
        """Retrieve a chat session by MongoDB _id."""
        try:
            if not ObjectId.is_valid(session_id):
                logger.warning(f"Invalid session_id format: {session_id}")
                return None

            session_data = await db.chat_sessions_collection.find_one({"_id": ObjectId(session_id)})
            if session_data:
                return ChatSession(**session_data)
            logger.info(f"No chat session found for _id: {session_id}")
            return None

        except Exception as e:
            logger.error(f"Error retrieving chat session {session_id}: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to retrieve chat session: {str(e)}")

    @staticmethod
    async def update_chat_session(session_id: str, update_data: dict):
        """Update an existing chat session by MongoDB _id."""
        try:
            if not ObjectId.is_valid(session_id):
                raise ValueError(f"Invalid session_id format: {session_id}")

            update_data["updated_at"] = datetime.now()

            result = await db.chat_sessions_collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": update_data}
            )

            if result.matched_count == 0:
                logger.warning(f"No session found to update for _id: {session_id}")
            else:
                logger.debug(f"Updated chat session _id: {session_id} with data: {update_data}")

        except Exception as e:
            logger.error(f"Error updating chat session {session_id}: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to update chat session: {str(e)}")
        


    @staticmethod
    async def get_user_chats(user_id: ObjectId) -> List[ChatSession]:
        """Get all chat sessions for a user from MongoDB"""
        try:
            print(f"Fetching chat sessions for user_id: {user_id}")
            sessions = await db.chat_sessions_collection.find(
                {"user_id": user_id},  # Now using ObjectId directly
                sort=[("updated_at", -1)]
            ).to_list(None)
            
            print(f"Found {len(sessions)} sessions")
            # for session in sessions:
            #     print(f"Session: {session}")
                
            return [ChatSession(**session) for session in sessions]
        except Exception as e:
            logger.error(f"Failed to get user chats: {e}")
            raise
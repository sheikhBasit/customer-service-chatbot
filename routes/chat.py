from asyncio.log import logger
import json
from bson import ObjectId
from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional, Union
from datetime import datetime
from pydantic import BaseModel
from models.chat import ChatSession, ChatMessage
from models.user import UserInDB
from models.vehicle import VehicleModel
from services.chat_service import ChatService
from services.cloudinary import upload_image
from storage.session_manager import SessionManager
from utils.user import get_current_user
from config import settings

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(get_current_user)]
)

@router.post("/start")
async def start_chat_session(user: dict = Depends(get_current_user)):
    if not user or not user.id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    chat_session = await SessionManager.create_chat_session(user.id)
    return JSONResponse(
        content={
            "session_id": str(chat_session.id),  # Using Mongo _id as session_id
            "chat_title": chat_session.chat_title
        },
        status_code=201
    )


@router.post("/message")
async def process_message(
    request: Request,
    session_id: Optional[str] = Form(None),
    message: str = Form(...),
    user: dict = Depends(get_current_user),
    image: Optional[Union[UploadFile, str]] = File(None),
    vehicle_json: Optional[str] = Form(None)
):
    """Process a user message and maintain chat history"""
    try:
        # 1. Get or create chat session
        if not session_id:
            chat_session = await SessionManager.create_chat_session(user.id)
            session_id = str(chat_session.id)
        else:
            chat_session = await SessionManager.get_chat_session(session_id)
            if not chat_session:
                raise HTTPException(status_code=404, detail="Chat session not found")

        # 2. Parse vehicle data if provided
        vehicle = None
        if vehicle_json and vehicle_json.strip():
            try:
                vehicle_data = json.loads(vehicle_json)
                vehicle = VehicleModel(**vehicle_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid vehicle data: {str(e)}")

        # 3. Handle image upload
        image_url = None
        if isinstance(image, UploadFile) and image.filename:
            try:
                if image.content_type not in settings.ALLOWED_IMAGE_TYPES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid image type. Allowed: {settings.ALLOWED_IMAGE_TYPES}"
                    )
                image_url = await process_uploaded_image(image)
                chat_session.image_history.append(image_url)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to process image: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to process uploaded image")

        # 4. Add user message to chat history
        chat_session.chat_history.append(ChatMessage(role="user", content=message))

        # 5. Generate assistant response
        chat_service = ChatService(request)
        response_data = await chat_service.process_message(
            session=chat_session,
            user_input=message,
            image_url=image_url,
            vehicle=vehicle
        )

        # Extract response and updated session
        assistant_response = response_data.get("response", "")
        chat_session = response_data.get("updated_session", chat_session)

        # 6. Add assistant response to chat history
        chat_session.chat_history.append(ChatMessage(role="assistant", content=assistant_response))

        # 7. Update chat title if not set
        if not chat_session.chat_title and message:
            chat_session.chat_title = await ChatService.generate_chat_title(message)

        # 8. Prepare update data - CONVERT VehicleModel TO DICT
        update_data = {
            "chat_history": [
                msg.model_dump() if isinstance(msg, BaseModel) else msg
                for msg in chat_session.chat_history
            ],
            "image_history": chat_session.image_history,
            "chat_title": chat_session.chat_title,
            "updated_at": datetime.now()
        }
        
        if vehicle:
            update_data["vehicle_info"] = vehicle.model_dump()  # ✅ Convert to dict

        # 9. Save updated session to DB
        await SessionManager.update_chat_session(session_id, update_data)

        # 10. Return response
        return JSONResponse(
            content={
                "response": assistant_response,
                "session_id": session_id,
                "chat_title": chat_session.chat_title
            },
            status_code=200
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@router.get("/sessions")
async def get_chat_sessions(user: UserInDB = Depends(get_current_user)):
    """Get all chat sessions for the current user"""
    try:
        # Convert user ID to ObjectId for MongoDB query
        user_object_id = ObjectId(str(user.id))
        sessions = await SessionManager.get_user_chats(user_object_id)

        serialized_sessions = []

        for session in sessions:
            # Extract last message content for preview
            preview = ""
            if session.chat_history:
                for msg in reversed(session.chat_history):
                    content = None
                    # Handle both dict and object
                    if isinstance(msg, dict):
                        content = msg.get("content")
                    else:
                        content = getattr(msg, "content", None)
                    if content:
                        preview = content
                        break

            serialized_sessions.append({
                "_id": str(session.id),  # Use MongoDB _id
                "chat_title": session.chat_title,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "preview": preview,
                "message_count": len(session.chat_history) if session.chat_history else 0
            })

        return JSONResponse(
            content={
                "sessions": serialized_sessions,
                "count": len(serialized_sessions)
            },
            status_code=200
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get chat sessions: {str(e)}"
        )
@router.get("/sessions/{session_id}/chat")
async def get_chat_history_by_session_id(
    session_id: str,
    user: dict = Depends(get_current_user)
):
    """Fetch the chat history for a specific session ID."""
    try:
        # Ensure the session_id is a valid ObjectId for MongoDB
        session_object_id = ObjectId(session_id)
        
        # Retrieve the chat session using the SessionManager
        chat_session = await SessionManager.get_chat_session(session_object_id)

        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        
        # Verify that the session belongs to the current user
        if str(chat_session.user_id) != str(user.id):
            raise HTTPException(status_code=403, detail="Forbidden: You do not own this chat session.")

        # Prepare the response with chat history
        chat_history = [
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()}
            for msg in chat_session.chat_history
        ]

        return JSONResponse(
            content={
                "session_id": str(chat_session.id),
                "chat_title": chat_session.chat_title,
                "chat_history": chat_history,
                "updated_at": chat_session.updated_at.isoformat() if chat_session.updated_at else None,
            },
            status_code=200
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions to be handled by FastAPI
    except Exception as e:
        # Log the unexpected error for debugging
        logger.error(f"Unexpected error fetching chat history for session {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
    
    
async def process_uploaded_image(image: UploadFile) -> str:
    """Process and store uploaded image in Cloudinary"""
    try:
        # Validate file size
        max_size = 10 * 1024 * 1024  # 10MB
        contents = await image.read()
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Max size is {max_size/1024/1024}MB"
            )
        
        # Reset file pointer after reading
        await image.seek(0)
        
        # Upload to Cloudinary
        image_url = await upload_image(image, expected_type='other')
        return image_url
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )


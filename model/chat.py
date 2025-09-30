from typing import Optional, List, Dict, Any, Literal, Union, TypedDict, Protocol
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator
import json
from pathlib import Path
import logging
from models.vehicle import VehicleModel
from typing import  runtime_checkable
from datetime import datetime
from bson import ObjectId

from utils.py_object import PyObjectId
logger = logging.getLogger(__name__)


@runtime_checkable
class ChatPromptLike(Protocol):
    @property
    def messages(self) -> List[Dict[Literal["role", "content"], str]]: ...

class DiagnosisResponse(TypedDict):
    diagnosis_output: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in {"user", "assistant", "system"}:
            raise ValueError('Role must be either "user", "assistant", or "system"')
        return v

class ChatState(BaseModel):
    current_prompt: Optional[str] = None
    vehicle: Optional[VehicleModel] = None
    image_history: List[str] = []
    chat_history: List[ChatMessage] = []
    image_reference_keywords: List[str] = ["previous image", "last image", "earlier image"]

    @model_validator(mode='after')
    def validate_state(self) -> 'ChatState':
        """Validate the entire state after initialization"""
        if self.current_prompt and not any(
            msg.content == self.current_prompt and msg.role == "user" 
            for msg in self.chat_history
        ):
            logger.warning("Current prompt doesn't match last user message")
        return self

    def add_user_message(self, message: str) -> None:
        if not message:
            raise ValueError("Message cannot be empty")
        self.chat_history.append(ChatMessage(role="user", content=message))
        self.current_prompt = message

    def add_assistant_message(self, message: str) -> None:
        if not message:
            raise ValueError("Message cannot be empty")
        self.chat_history.append(ChatMessage(role="assistant", content=message))

    def add_system_message(self, message: str) -> None:
        if not message:
            raise ValueError("Message cannot be empty")
        self.chat_history.append(ChatMessage(role="system", content=message))

    def add_image(self, image_url: str) -> None:
        if not image_url:
            raise ValueError("Image URL cannot be empty")
        self.image_history.append(image_url)

    def set_vehicle(self, vehicle: VehicleModel) -> None:
        self.vehicle = vehicle

    def resolve_image_references(self, text: str) -> str:
        """Replace image references with actual image URLs from history"""        
        if not self.image_history:
            return text

        lower_text = text.lower()
        for keyword in self.image_reference_keywords:
            if keyword in lower_text:
                return text.replace(keyword, f"image ({self.image_history[-1]})")
        return text

    def prepare_chain_input(
        self,
        user_input: str,
        image_url: Optional[str] = None,
        vehicle: Optional[VehicleModel] = None
    ) -> Dict[str, Any]:
        """Prepare input for the processing chain"""
        resolved_prompt = self.resolve_image_references(user_input)
        final_image_url = image_url or (self.image_history[-1] if self.image_history else "")
        final_vehicle = vehicle or self.vehicle

        self.add_user_message(resolved_prompt)
        if final_image_url:
            self.add_image(final_image_url)
        if final_vehicle:
            self.set_vehicle(final_vehicle)

        return {
            "prompt": resolved_prompt,
            "image_url": final_image_url,
            "vehicle": final_vehicle.model_dump() if final_vehicle else None,
            "chat_history": [msg.model_dump() for msg in self.chat_history]
        }

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get chat history as a list of dictionaries"""
        return [msg.model_dump() for msg in self.chat_history]

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Serialize ChatState to JSON file"""
        file_path = Path(file_path)
        with file_path.open('w') as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'ChatState':
        """Deserialize ChatState from JSON file"""
        file_path = Path(file_path)
        with file_path.open('r') as f:
            data = json.load(f)

        if 'chat_history' in data:
            data['chat_history'] = [ChatMessage(**msg) for msg in data['chat_history']]

        return cls(**data)

    def process_chain_response(
        self, 
        response: Union[DiagnosisResponse, Dict[str, Any], str, ChatPromptLike]
    ) -> Union[DiagnosisResponse, Dict[str, Any], str, ChatPromptLike]:
        """Process and store the chain response with complete type safety."""
        
        diagnosis: str = ""
        
        if isinstance(response, str):
            diagnosis = response
        elif isinstance(response, dict):
            if "diagnosis_output" in response:
                diagnosis = str(response.get("diagnosis_output", ""))
            else:
                diagnosis = str(response)
        elif hasattr(response, "messages"):
            messages = response.messages
            diagnosis = messages[-1]["content"] if messages else ""
        else:
            diagnosis = str(response)

        if diagnosis:
            self.add_assistant_message(diagnosis)
        return response





    def reset_history(self) -> None:
        """Reset all conversation history"""
        self.chat_history.clear()
        self.image_history.clear()
        self.current_prompt = None
        self.vehicle = None




class ChatSession(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    chat_title: Optional[str] = "New Chat"
    chat_history: List[ChatMessage] = []
    vehicle_info: Optional[Union[VehicleModel, dict]] = None  
    image_history: List[str] = []

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

from groq import Groq
from typing import Literal, Optional, List
import logging
from config import settings
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Classify user intent to determine processing path"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.intent_categories = [
            "greeting", "small_talk", "technical_question", 
            "command", "vehicle_diagnosis", "other"
        ]
    
    async def classify_intent(self, message: str, chat_history: List = None) -> str:
        """Classify user intent using a fast LLM"""
        try:
            # Use chat history for context if available
            history_context = ""
            if chat_history:
                # Extract last 4 messages for context
                last_messages = []
                for msg in chat_history[-4:]:
                    if hasattr(msg, 'content'):
                        last_messages.append(f"{getattr(msg, 'role', 'user')}: {msg.content}")
                    elif isinstance(msg, dict) and 'content' in msg:
                        last_messages.append(f"{msg.get('role', 'user')}: {msg['content']}")
                
                history_context = "\n".join(last_messages)
            
            prompt = f"""
            Analyze this user message and classify its intent. Choose ONLY from these categories:
            {', '.join(self.intent_categories)}
            
            Chat History (for context):
            {history_context}
            
            User Message: "{message}"
            
            Respond with ONLY the intent category name, nothing else.
            """
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an intent classification expert. Respond with only the category name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=20
            )
            
            intent = response.choices[0].message.content.strip().lower()
            
            # Validate the response is a known category
            if intent in self.intent_categories:
                return intent
            else:
                # Default to technical_question if classification fails
                return "technical_question"
                
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "technical_question"  # Fallback to RAG

def get_intent_classifier(request):
    """Dependency injection for intent classifier"""
    classifier = getattr(request.app.state, "intent_classifier", None)
    if not classifier:
        raise HTTPException(status_code=500, detail="Intent classifier not initialized")
    return classifier
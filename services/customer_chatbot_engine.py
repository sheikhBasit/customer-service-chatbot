"""
services/customer_chatbot_engine.py - Core chatbot logic for customer queries
"""
import logging
from typing import Optional
from langchain_core.runnables import RunnableLambda
from groq import Groq

from models.customer_chatbot import CustomerChatSession
from services.customer_vectorstore import CustomerVectorStoreService
from services.multimodal_embeddings import embed_text
from config import settings

logger = logging.getLogger(__name__)


class CustomerChatbotEngine:
    """Process queries using customer's vectorstore"""
    
    def __init__(self):
        self.vectorstore_service = CustomerVectorStoreService()
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
    
    async def process_query(
        self,
        customer_id: str,
        session: CustomerChatSession,
        query: str
    ) -> str:
        """
        Process a query using RAG on customer's documents
        
        Args:
            customer_id: Customer ID
            session: Chat session
            query: User query
            
        Returns:
            Generated response
        """
        try:
            # Get customer's vectorstore
            vectorstore_data = await self.vectorstore_service.get_customer_vectorstore(customer_id)
            
            if not vectorstore_data:
                return "I don't have any documents to reference yet. Please upload documents first."
            
            vectorstore, image_data_store = vectorstore_data
            
            # Retrieve relevant context
            context = await self._retrieve_context(
                vectorstore=vectorstore,
                query=query,
                chat_history=session.messages
            )
            
            # Generate response
            response = await self._generate_response(
                query=query,
                context=context,
                chat_history=session.messages[-6:]  # Last 6 messages for context
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query for customer {customer_id}: {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your question. Please try again."
    
    async def _retrieve_context(
        self,
        vectorstore,
        query: str,
        chat_history: list
    ) -> str:
        """Retrieve relevant context from vectorstore"""
        try:
            # Build enhanced query with conversation context
            history_context = ""
            if chat_history and len(chat_history) > 0:
                recent_messages = chat_history[-4:]
                history_parts = []
                for msg in recent_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if content:
                        history_parts.append(f"{role}: {content}")
                history_context = "\n".join(history_parts)
            
            enhanced_query = f"Conversation context:\n{history_context}\n\nCurrent question: {query}"
            
            # Embed query
            query_embedding = embed_text(enhanced_query)
            
            # Search vectorstore
            docs_and_scores = vectorstore.similarity_search_by_vector(
                query_embedding,
                k=5  # Retrieve top 5 relevant chunks
            )
            
            # Combine retrieved documents
            context_parts = []
            for doc in docs_and_scores:
                if doc.metadata.get("type") == "text":
                    context_parts.append(doc.page_content)
                elif doc.metadata.get("type") == "image":
                    # Add image reference
                    context_parts.append(f"[Referenced image: {doc.metadata.get('filename')}]")
            
            return "\n---\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Context retrieval error: {e}", exc_info=True)
            return "Unable to retrieve relevant context."
    
    async def _generate_response(
        self,
        query: str,
        context: str,
        chat_history: list
    ) -> str:
        """Generate response using LLM"""
        try:
            # Build system prompt
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from documents.

Instructions:
- Answer questions accurately using ONLY the information from the context
- If the context doesn't contain relevant information, politely say so
- Be conversational and friendly
- Keep answers concise but complete
- Reference specific details from the context when relevant
- If asked about something not in the context, acknowledge the limitation"""

            # Build conversation history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent chat history
            for msg in chat_history:
                role = msg.get("role")
                content = msg.get("content")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})
            
            # Add current query with context
            user_message = f"""Context from documents:
{context}

Question: {query}"""
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # Use powerful model for customer queries
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Response generation error: {e}", exc_info=True)
            return "I apologize, but I'm having trouble generating a response right now."
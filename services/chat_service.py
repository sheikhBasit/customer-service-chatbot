from services.multimodal_embeddings import embed_text
from fastapi import Request
from langchain_core.runnables import (
    RunnableSerializable,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda,
)
from typing import Dict, Any, Optional, List, Literal
import logging
from models.chat import ChatSession
from models.vehicle import VehicleModel
from config import settings
from services.dependencies import get_diagnostic_agent, get_image_analyzer, get_vectorstore
from services.intent_classifier import get_intent_classifier
from services.simple_responses import SimpleResponseGenerator
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, request: Request):
        self.vectorstore, self.image_data_store = get_vectorstore(request)
        self.diagnostic_agent = get_diagnostic_agent(request)
        self.image_analyzer = get_image_analyzer(request)
        self.intent_classifier = get_intent_classifier(request)
        self.chain = self._create_processing_chain()
    
    async def _determine_processing_path(self, user_input: str, chat_history: list) -> Literal["simple", "rag", "command"]:
        """Determine the processing path based on user intent"""
        intent = await self.intent_classifier.classify_intent(user_input, chat_history)
        
        logger.info(f"Detected intent: {intent} for message: '{user_input}'")
        
        if intent in ["greeting", "small_talk"]:
            return "simple"
        elif intent in ["technical_question", "vehicle_diagnosis"]:
            return "rag"
        elif intent == "command":
            return "command"
        else:
            return "rag"  # Default to RAG for unknown intents
    
    def _create_simple_response_chain(self) -> RunnableSerializable:
        """Chain for simple responses (greetings, small talk)"""
        async def simple_response(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Generate simple conversational responses"""
            try:
                # First try to get a predefined response
                predefined_response = SimpleResponseGenerator.get_response(
                    await self.intent_classifier.classify_intent(inputs["prompt"], inputs.get("chat_history", [])),
                    inputs["prompt"]
                )
                
                if predefined_response:
                    return {**inputs, "diagnosis_output": predefined_response}
                
                # Fallback to LLM for more complex simple responses
                prompt = f"""
                User message: "{inputs['prompt']}"
                
                Respond naturally and conversationally. Be friendly and helpful.
                Keep your response brief and appropriate for a vehicle mechanic assistant.
                If this is small talk or greeting, keep it very short (1-2 sentences).
                """
                
                response = await self.diagnostic_agent.ainvoke({
                    "system_prompt": "You are a friendly vehicle mechanic assistant. Engage in natural conversation. Keep responses brief for greetings and small talk.",
                    "input": prompt,
                    "chat_history": inputs.get("chat_history", []),
                    "is_simple_response": True
                })
                
                return {**inputs, "diagnosis_output": response}
            except Exception as e:
                logger.error(f"Simple response failed: {e}")
                return {**inputs, "diagnosis_output": "Hello! How can I help with your vehicle today?"}
        
        return RunnableLambda(simple_response)
    
    def _create_rag_chain(self) -> RunnableSerializable:
        """Create the RAG processing chain"""
        async def image_analysis_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Process images using the image analyzer"""
            try:
                if not inputs.get("image_url"):
                    return {"context_1": "No image provided", **inputs}
                
                vehicle_info = inputs.get("vehicle", {})
                user_question = inputs.get("prompt", "Analyze this vehicle image")
                
                analysis = await self.image_analyzer.analyze(
                    inputs["image_url"],
                    prompt=user_question,
                    vehicle_info=vehicle_info
                )
                return {"context_1": analysis, **inputs}
            except Exception as e:
                logger.error(f"Image analysis failed: {e}", exc_info=True)
                return {"context_1": "Image analysis failed", **inputs}

        def retrieval_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Retrieve relevant information from vector store"""
            try:
                vehicle = inputs.get("vehicle", {})
                prompt = inputs["prompt"]
                chat_history = inputs.get("chat_history", [])

                # Get last 2 user messages for context
                history_context = "\n".join(
                    [msg.content for msg in chat_history[-4:] if hasattr(msg, 'content')]
                )

                enhanced_question = (
                 f"Conversation Context:\n{history_context}\n\n"
                    f"User Query for a specific vehicle: {vehicle.get('brand', 'Unknown')} {vehicle.get('model', 'Unknown')} {vehicle.get('fuel_type', 'Unknown')}"
                    f"({vehicle.get('transmission', 'Unknown')} {vehicle.get('year', 'Unknown')}) - {prompt}"
                )

                # Manual embedding and search
                query_embedding = embed_text(enhanced_question)
                
                # Manual similarity search
                docs_and_scores = self.vectorstore.similarity_search_by_vector(
                    query_embedding,
                    k=3,
                    filter={"vehicle_make": vehicle.get("brand")} if vehicle.get("brand") else None
                )
                
                # Combine text and image context
                text_context = "\n---\n".join([doc.page_content for doc in docs_and_scores])
                multimodal_context = []
                
                for doc in docs_and_scores:
                    if doc.metadata.get("type") == "image":
                        image_id = doc.metadata.get("image_id")
                        if image_id in self.image_data_store:
                            multimodal_context.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{self.image_data_store[image_id]}"
                                }
                            })
                
                return {
                    **inputs,
                    "context_2": text_context,
                    "multimodal_context": multimodal_context
                }
            except Exception as e:
                logger.error(f"Retrieval failed: {e}", exc_info=True)
                return {**inputs, "context_2": "Knowledge retrieval failed"}

        def diagnostic_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Generate diagnostic response using the LLM"""
            try:
                vehicle = inputs.get("vehicle", {})
                vehicle_info = {
                    "make": vehicle.get("brand", "Unknown"),
                    "model": vehicle.get("model", "Unknown"),
                    "year": vehicle.get("year", "Unknown"),
                    "fuel_type": vehicle.get("fuel_type", "Unknown"),
                    "engine_type": vehicle.get("engine_type", "Unknown")
                }
                
                # Convert ChatMessage objects to dict for the LLM
                chat_history_dicts = []
                for msg in inputs.get("chat_history", []):
                    if hasattr(msg, 'model_dump'):
                        chat_history_dicts.append(msg.model_dump())
                    else:
                        chat_history_dicts.append(msg)
                enhanced_system_prompt = f"""You are a specialized vehicle diagnostic assistant for a **{vehicle_info['year']} {vehicle_info['make']} {vehicle_info['model']}**.
                Your goal is to provide **highly specific, actionable advice** for the user's vehicle problem.
                Use the provided context and your expert knowledge to diagnose the issue and propose a solution.

                - **Vehicle Details**:
                    - Make: {vehicle_info['make']}
                    - Model: {vehicle_info['model']}
                    - Year: {vehicle_info['year']}
                    - Fuel Type: {vehicle_info['fuel_type']}
                    - Engine Type: {vehicle_info['engine_type']}

                - **User Problem**: {inputs['prompt']}

                - **Instructions**:
                1. Start by acknowledging the user's vehicle and problem.
                2. **Directly apply** any relevant information from the 'Knowledge Base Context' and 'Image Analysis' to your diagnosis.
                3. Provide a step-by-step diagnostic process tailored to the specific vehicle model.
                4. If the provided context is insufficient for a specific answer, explain what additional information is needed (e.g., "Can you provide a picture of the engine bay?").
                5. Avoid general answers unless no specific information is available.
                """

                llm_input = {
                    # "system_prompt": self._get_vehicle_system_prompt(vehicle_info),
                    "system_prompt": enhanced_system_prompt,
                    "input": inputs["prompt"],
                    "context": f"""
                    Image Analysis:
                    {inputs.get('context_1', 'No image analysis available')}
                    
                    Knowledge Base Context:
                    {inputs.get('context_2', 'No knowledge base context available')}
                    
                    Multimodal Context:
                    {inputs.get('multimodal_context', 'No additional context')}
                    """,
                    "chat_history": chat_history_dicts,
                    "is_simple_response": False
                }
                
                response = self.diagnostic_agent.invoke(llm_input)
                return {**inputs, "diagnosis_output": response}
            except Exception as e:
                logger.error(f"Diagnostic failed: {e}", exc_info=True)
                return {**inputs, "diagnosis_output": "Diagnostic service unavailable"}
        
        return (
            RunnablePassthrough()
            | RunnableLambda(image_analysis_chain)
            | RunnableLambda(retrieval_chain)
            | RunnableLambda(diagnostic_chain)
        )

    def _create_processing_chain(self) -> RunnableSerializable:
        """Create the complete processing chain with intent-based routing"""
        
        async def routing_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Route to appropriate processing chain based on intent"""
            processing_path = await self._determine_processing_path(
                inputs["prompt"], inputs.get("chat_history", [])
            )
            
            logger.info(f"Routing to: {processing_path} processing")
            
            if processing_path == "simple":
                # Use simple response chain
                simple_chain = self._create_simple_response_chain()
                return await simple_chain.ainvoke(inputs)
            else:
                # Use full RAG pipeline (for technical_question, vehicle_diagnosis, command, and fallback)
                rag_chain = self._create_rag_chain()
                return await rag_chain.ainvoke(inputs)
        
        return RunnableLambda(routing_chain)

    def _get_vehicle_system_prompt(self, vehicle_info: dict) -> str:
        """Generate system prompt for vehicle diagnosis"""
        return f"""You are an expert **vehicle mechanic assistant** trained to diagnose and resolve issues related to ground vehicles.

You are provided with:
- A user-described problem or symptoms
- Vehicle metadata:
    - Make: {vehicle_info.get("make", "Unknown")}
    - Model: {vehicle_info.get("model", "Unknown")}
    - Year: {vehicle_info.get("year", "Unknown")}
    - Fuel Type: {vehicle_info.get("fuel_type", "Unknown")}
    - Engine Type: {vehicle_info.get("engine_type", "Unknown")}
- Optional images or diagnostic documents
- Complete chat history for context

Guidelines:
1. Be professional but friendly and conversational when appropriate
2. Ask clarifying questions when needed
3. Provide step-by-step solutions when possible
4. Reference vehicle-specific information
5. Maintain conversation context
6. For complex issues, recommend professional help
7. Adapt your tone based on the conversation - be more technical for diagnosis, more conversational for greetings"""

    async def process_message(
        self,
        session: ChatSession,
        user_input: str,
        image_url: Optional[str] = None,
        vehicle: Optional[VehicleModel] = None
    ) -> Dict[str, Any]:
        """
        Process a user message through the complete chain
        
        Args:
            session: Current chat session
            user_input: User's message text
            image_url: Optional image URL/path
            vehicle: Optional vehicle information
            
        Returns:
            Dictionary containing:
            - response: Generated diagnosis/response
            - updated_session: Updated chat session
        """
        try:
            if image_url:
                session.image_history.append(image_url)
            
            if vehicle:
                session.vehicle_info = vehicle

            # Prepare chain input
            inputs = {
                "prompt": user_input,
                "image_url": image_url,
                "vehicle": vehicle.model_dump() if vehicle else {},
                "chat_history": session.chat_history
            }
            
            # Process through chain
            result = await self.chain.ainvoke(inputs)
            
            # Handle response
            diagnosis = result.get("diagnosis_output", "")
            
            # Generate title if first message
            if len(session.chat_history) <= 2 and not session.chat_title:
                session.chat_title = await self.generate_chat_title(user_input)
                
            return {
                "response": diagnosis,
                "updated_session": session
            }
        except Exception as e:
            logger.error(f"Message processing failed: {e}", exc_info=True)
            raise
    
    async def generate_chat_title(self, first_message: str) -> str:
        """Generate a summary title for the chat based on the first message"""
        try:
            prompt = f"""
            Create a very short (3-5 word) title summarizing this vehicle issue:
            "{first_message}"
            
            Respond ONLY with the title text, no quotes or other formatting.
            Example outputs:
            - Engine knocking sound
            - Brake system issue
            - Electrical problem
            - Transmission trouble
            """
            
            response = await self.diagnostic_agent.ainvoke({
                "system_prompt": "You are a vehicle expert that creates concise, descriptive chat titles.",
                "input": prompt,
                "chat_history": [],
                "is_simple_response": True
            })
            
            return response.strip('"').strip("'").strip() or "Vehicle Consultation"
        except Exception as e:
            logger.error(f"Title generation failed: {e}")
            return "Vehicle Consultation"
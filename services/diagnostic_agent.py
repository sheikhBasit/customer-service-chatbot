from groq import Groq
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def create_diagnostic_agent(api_key: str):
    try:
        client = Groq(api_key=api_key)

        def run_gemma_chat(inputs: dict) -> str:
            # Handle both dictionary inputs and message objects
            if isinstance(inputs, dict):
                system_prompt = inputs.get("system_prompt", "")
                user_input = inputs.get("input", "")
                context = inputs.get("context", "")
                chat_history = inputs.get("chat_history", [])
                is_simple_response = inputs.get("is_simple_response", False)
            else:
                try:
                    # Handle LangChain message objects
                    if hasattr(inputs, 'messages'):
                        messages = inputs.messages
                    elif isinstance(inputs, list):
                        messages = inputs
                    else:
                        messages = [inputs]

                    system_prompt = ""
                    user_input = ""
                    context = ""
                    chat_history = []
                    is_simple_response = False

                    for msg in messages:
                        if hasattr(msg, 'type'):  # LangChain message type
                            if msg.type == "system":
                                system_prompt = msg.content
                            elif msg.type == "human":
                                user_input = msg.content
                            elif msg.type == "ai":
                                chat_history.append({"role": "assistant", "content": msg.content})
                        elif hasattr(msg, 'role'):  # OpenAI-style message
                            if msg.role == "system":
                                system_prompt = msg.content
                            elif msg.role == "user":
                                user_input = msg.content
                            elif msg.role == "assistant":
                                chat_history.append({"role": "assistant", "content": msg.content})
                except Exception as e:
                    logger.error(f"Error processing non-dict input: {e}")
                    return "I'm having trouble processing your request. Please try again."

            # Model selection based on response type
            if is_simple_response:
                model = "llama-3.1-8b-instant"
                temperature = 0.7
                max_tokens = 256
            else:
                model = "gemma2-9b-it"
                temperature = 0.8
                max_tokens = 1024

            messages = [{"role": "system", "content": system_prompt}]
            
            # Add chat history (limit to last 6 messages for context)
            for msg in chat_history[-6:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

            # Add current input with context
            full_input = user_input
            if context and context.strip():
                full_input = f"{user_input}\n\nContext:\n{context}"

            messages.append({"role": "user", "content": full_input})

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=1,
                    max_completion_tokens=max_tokens,
                    stream=False
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error in diagnostic agent: {e}")
                return "I'm having trouble processing your request. Please try again."

        return RunnableLambda(run_gemma_chat)
    except Exception as e:
        logger.error(f"Failed to initialize diagnostic agent: {e}")
        return RunnableLambda(lambda x: "Diagnostic service is currently unavailable")
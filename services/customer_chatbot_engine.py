# services/customer_chatbot_engine.py

import logging
from typing import Optional, List, Dict, Any

from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables   .config import RunnableConfig  # config for invoke
from langchain_core.callbacks import AsyncCallbackHandler

# Import your existing modules
from models.customer_chatbot import CustomerChatSession
from services.customer_vectorstore import CustomerVectorStoreService
from services.multimodal_embeddings import embed_text
from config import settings

# Import the LLM interface for Groq
from langchain_groq import ChatGroq  

logger = logging.getLogger(__name__)


class CustomerChatbotEngine:
    """Process queries using customer's vectorstore, in LangChain v0.3 style"""

    def __init__(self):
        self.vectorstore_service = CustomerVectorStoreService()
        # Initialize ChatGroq as a Runnable Chat model
        # You can pass temperature, max_tokens, etc. here or override at invoke time
        self.llm: Runnable = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.5,
            max_tokens=1024,
        )

        # (Optional) If you want to use RunnableWithMessageHistory to handle history automatically:
        # You need a function get_history(session_id) -> BaseChatMessageHistory
        self.get_history_fn = None  # override externally if desired

        # (Optional) Wrap the LLM with history handling
        # We'll conditionally wrap later in `process_query`.

    async def process_query(
        self,
        customer_id: str,
        session: CustomerChatSession,
        query: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Process a query using RAG on customer's documents, using LangChain v0.3 runnables.
        """
        logger.debug(f"[process_query] customer_id={customer_id}, query={query}")

        # Fetch or initialize vectorstore for this customer
        vectorstore_data = await self.vectorstore_service.get_customer_vectorstore(customer_id)
        if not vectorstore_data:
            logger.debug("[process_query] No vectorstore found for customer, returning fallback")
            return "I don't have any documents to reference yet. Please upload documents first."
        vectorstore, image_data_store = vectorstore_data

        # Build a runnable for retrieval: embed -> search
        embed_runnable = RunnableLambda(lambda text: embed_text(text))
        search_runnable = RunnableLambda(lambda emb: vectorstore.similarity_search_by_vector(emb, k=5))

        # Optionally you might want async versions, but for simplicity we use sync runnables here

        # Debug: show chaining
        logger.debug("[process_query] chaining embed and search runnables")
        retrieval_pipeline = embed_runnable | search_runnable

        # Run the pipeline to get docs (this is sync; if you have `await` variants, you can use ainvoke)
        docs = retrieval_pipeline.invoke(query)
        logger.debug(f"[process_query] Retrieved {len(docs)} docs")

        # Format context
        context_parts: List[str] = []
        for doc in docs:
            if doc.metadata.get("type") == "text":
                context_parts.append(doc.page_content)
            elif doc.metadata.get("type") == "image":
                context_parts.append(f"[Referenced image: {doc.metadata.get('filename')}]")
        context = "\n---\n".join(context_parts)
        logger.debug(f"[process_query] Context snippet: {context[:200]}")

        # Prepare prompts / messages for LLM
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant that answers questions based "
                "on the provided context from documents.\n\n"
                "Instructions:\n"
                "- Answer questions accurately using ONLY the information from the context\n"
                "- If the context doesn't contain relevant information, politely say so\n"
                "- Be conversational and friendly\n"
                "- Keep answers concise but complete\n"
                "- Reference specific details from the context when relevant\n"
                "- If asked about something not in the context, acknowledge the limitation"
            )

        if temperature is not None:
            # override the model's default if provided
            # The ChatGroq Runnable supports passing override args via invoke config kwargs
            llm = self.llm.with_retry()  # just to illustrate you can wrap or adjust
            # or you could use `self.llm.configure(temperature=temperature)` if available
        else:
            llm = self.llm

        # Build a ChatPromptTemplate to structure message sequence
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_query}"),
        ])

        # Chain prompt + LLM
        prompt_and_llm: Runnable = prompt | llm

        # Optionally wrap with message history
        if self.get_history_fn:
            # Wrap so that prompt+LLM will manage history under a session id
            prompt_and_llm = RunnableWithMessageHistory(
                prompt_and_llm,
                get_session_history=self.get_history_fn,
                input_messages_key="user_query",
            )

        # Prepare input dict
        inputs: Dict[str, Any] = {"user_query": query}
        # Also include context in user_query or embed it into prompt
        # You might alternatively pass context via a tool or extra variable
        # Here we embed the context directly in the user query
        inputs["user_query"] = f"Context from documents:\n{context}\n\nQuestion: {query}"

        # Prepare config for invoke
        invoke_config = RunnableConfig()
        # If using RunnableWithMessageHistory, we must pass session_id
        if self.get_history_fn:
            invoke_config = RunnableConfig(
                configurable={"session_id": session.session_id}
            )

        logger.debug(f"[process_query] invoking prompt_and_llm with inputs={inputs}, config={invoke_config}")
        # Use ainvoke (async) or invoke
        ai_msg: Any
        if hasattr(prompt_and_llm, "ainvoke"):
            ai_msg = await prompt_and_llm.ainvoke(inputs, config=invoke_config)
        else:
            ai_msg = prompt_and_llm.invoke(inputs, config=invoke_config)

        # The output `ai_msg` is expected to be a `BaseMessage` or dict containing `messages`
        response: str
        if isinstance(ai_msg, BaseMessage):
            response = ai_msg.content
        elif isinstance(ai_msg, dict) and "messages" in ai_msg:
            # If a list of messages returned, pick last AI message
            msgs = ai_msg["messages"]
            for m in reversed(msgs):
                if isinstance(m, AIMessage):
                    response = m.content
                    break
            else:
                # fallback
                response = msgs[-1].content
        else:
            # fallback: try str
            response = str(ai_msg)

        logger.debug(f"[process_query] Final response: {response[:200]}")
        return response

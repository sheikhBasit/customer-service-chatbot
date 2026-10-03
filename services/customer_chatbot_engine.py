"""
services/customer_chatbot_engine.py - Agentic Chatbot Engine with Tool Calling
Transforms the chatbot from a static RAG chain into an autonomous Agentic Workflow.

Features:
1. Tool Calling (Function Calling) powered by Groq LLaMA-3.1 / 3.3
2. Built-in Tools:
   - get_order_status: Live courier tracking, status, and items breakdown
   - create_support_ticket: Support ticket generation and issue escalation
   - search_knowledge_base: Tenant multimodal RAG document search
3. Multi-turn Agent Reasoning Loop with full debugging logs and console outputs
4. Session-aware history preservation and conversation context
"""

import json
import logging
from typing import Optional, List, Dict, Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from models.customer_chatbot import CustomerChatSession
from services.agent_tools import (
    execute_get_order_status,
    execute_create_support_ticket,
    execute_search_knowledge_base
)
from config import settings

# Setup high-visibility logger
logger = logging.getLogger("agentic_engine")
logger.setLevel(logging.DEBUG)


class CustomerChatbotEngine:
    """
    Autonomous Customer Service Agent Engine
    Uses Groq's fast inference with native tool-calling capabilities to resolve
    customer queries, track orders, look up documentation, and file support tickets.
    """

    def __init__(self):
        logger.info("🤖 Initializing Agentic CustomerChatbotEngine with Groq LLaMA-3.1...")
        print("\n🚀 [AGENT INITIALIZATION] CustomerChatbotEngine loaded with Tool-Calling capabilities.")
        
        # Initialize Groq LLM with function calling support
        # llama-3.1-70b-versatile or llama-3.3-70b-versatile provides state-of-the-art tool calling
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,  # Lower temperature for accurate tool arguments
            max_tokens=1024,
        )

    def _build_agent_tools(self, customer_id: str, session: CustomerChatSession):
        """
        Dynamically constructs LangChain tools bound to the current customer/tenant context.
        """

        @tool
        async def get_order_status(order_id: str) -> str:
            """
            Look up real-time delivery status, shipping courier, tracking number, and line items for an order.
            Call this whenever a user asks 'Where is my order?', provides an order number (e.g. 'ORD-1002'),
            or inquires about package delivery.
            
            Args:
                order_id: The order identifier, e.g., 'ORD-1001', 'ORD-1002'.
            """
            print(f"🔧 [TOOL INVOKED] get_order_status for order_id='{order_id}'")
            res = await execute_get_order_status(order_id=order_id, customer_id=customer_id)
            return json.dumps(res)

        @tool
        async def create_support_ticket(
            customer_email: str,
            subject: str,
            description: str,
            priority: str = "medium",
            related_order_id: Optional[str] = None
        ) -> str:
            """
            Create an official customer support ticket and escalate the issue to the human support team.
            Call this when a customer has an unresolved issue, broken/damaged products, refund disputes,
            or explicitly requests to file a complaint or speak to a support representative.
            
            Args:
                customer_email: The customer's email address to receive updates and ticket notifications.
                subject: A brief, clear title for the ticket (e.g., 'Damaged headphone during delivery').
                description: Detailed description of the problem or customer complaint.
                priority: Urgency level ('low', 'medium', 'high', 'urgent').
                related_order_id: Order number if applicable (e.g. 'ORD-1002').
            """
            print(f"🔧 [TOOL INVOKED] create_support_ticket for email='{customer_email}', subject='{subject}'")
            res = await execute_create_support_ticket(
                customer_email=customer_email,
                subject=subject,
                description=description,
                customer_id=customer_id,
                customer_name=session.end_user_name,
                priority=priority,
                related_order_id=related_order_id,
                session_id=str(session.id) if session.id else None
            )
            return json.dumps(res)

        @tool
        async def search_company_knowledge_base(search_query: str) -> str:
            """
            Search company documents, FAQs, return policies, warranty guides, and product manuals.
            Call this to answer customer questions about return policy windows, warranty terms,
            troubleshooting steps, or company-specific documentation.
            
            Args:
                search_query: Keywords or question to look up in the company vector knowledge base.
            """
            print(f"🔧 [TOOL INVOKED] search_company_knowledge_base with query='{search_query}'")
            res = await execute_search_knowledge_base(customer_id=customer_id, query=search_query)
            return json.dumps(res)

        return [get_order_status, create_support_ticket, search_company_knowledge_base]

    async def process_query(
        self,
        customer_id: str,
        session: CustomerChatSession,
        query: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Execute an agentic problem-solving loop:
        1. Formulates agent prompt with conversational history & available tools.
        2. LLM reasons: decides whether to answer directly or call external tools.
        3. If tools are requested: executes them, logs outputs, and feeds observations back to LLM.
        4. Synthesizes a natural, helpful final response for the user.
        """
        print("\n" + "#" * 70)
        print(f"🤖 [AGENT REASONING START] Incoming User Query")
        print(f"    👤 Session ID   : {session.session_token}")
        print(f"    🏢 Customer ID  : {customer_id}")
        print(f"    💬 User Query   : \"{query}\"")
        print("#" * 70)
        
        logger.info(f"[AgenticEngine] Processing query for customer={customer_id}, session={session.session_token}")

        # 1. Define comprehensive Agent System Prompt
        default_system_prompt = (
            "You are an expert, proactive, and empathetic Customer Support AI Agent.\n\n"
            "Your Capabilities & Tools:\n"
            "1. 'get_order_status': Look up tracking, shipping carrier, delivery status, and items for any order number.\n"
            "2. 'create_support_ticket': Create an official support ticket for refunds, damages, complaints, or human escalation.\n"
            "3. 'search_company_knowledge_base': Search company policies, FAQs, warranty information, and documents.\n\n"
            "Operating Guidelines:\n"
            "- Always use tools when relevant. Do NOT guess order statuses or fake ticket reference numbers.\n"
            "- If a customer asks about an order (e.g. 'Where is ORD-1002?'), IMMEDIATELY call 'get_order_status'.\n"
            "- If a customer reports a damaged item, requests a refund, or is frustrated, offer to create a support ticket.\n"
            "  Always make sure you have or ask for their email address before or while creating the ticket.\n"
            "- If asked about company policies, return windows, or product manuals, call 'search_company_knowledge_base'.\n"
            "- Maintain a warm, courteous, professional, and solution-oriented tone at all times.\n"
            "- Keep answers concise, clear, and easy to read with bullet points when sharing details."
        )

        active_system_prompt = system_prompt or default_system_prompt

        # 2. Build Tools & Bind to LLM
        tools = self._build_agent_tools(customer_id, session)
        tool_map = {t.name: t for t in tools}
        
        llm_with_tools = self.llm.bind_tools(tools)
        if temperature is not None:
            llm_with_tools = self.llm.with_config(temperature=temperature).bind_tools(tools)

        # 3. Assemble Conversation History
        messages: List[BaseMessage] = [SystemMessage(content=active_system_prompt)]
        
        # Load up to the last 10 messages from session history for conversational context
        recent_messages = session.messages[-10:] if session.messages else []
        for msg in recent_messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # Append current user query
        messages.append(HumanMessage(content=query))

        # 4. Agent Execution Loop (Max 5 turns to prevent infinite loops)
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n🧠 [AGENT LOOP] Iteration {iteration}/{max_iterations} - Prompting LLM...")
            logger.debug(f"[AgenticEngine] Loop {iteration}: Prompting LLM with {len(messages)} messages")

            try:
                # Invoke LLM
                ai_message: AIMessage = await llm_with_tools.ainvoke(messages)
                messages.append(ai_message)

                # Check if the LLM decided to call any tools
                if not ai_message.tool_calls:
                    print(f"✨ [AGENT FINAL ANSWER REACHED] No further tool calls requested.")
                    print(f"    💬 Response Snippet: \"{ai_message.content[:150]}...\"")
                    print("#" * 70 + "\n")
                    return str(ai_message.content)

                # Execute requested tools
                print(f"⚙️  [AGENT ACTION] LLM requested {len(ai_message.tool_calls)} tool call(s)")
                
                for tool_call in ai_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    print(f"\n▶️  [EXECUTING TOOL] '{tool_name}'")
                    print(f"    📥 Arguments: {json.dumps(tool_args, indent=2)}")
                    logger.info(f"[AgenticEngine] Executing tool '{tool_name}' with args {tool_args}")

                    selected_tool = tool_map.get(tool_name)
                    if selected_tool:
                        try:
                            # Run tool (async)
                            tool_output = await selected_tool.ainvoke(tool_args)
                            print(f"    📤 Tool Raw Output: {str(tool_output)[:200]}...")
                        except Exception as tool_err:
                            logger.error(f"[AgenticEngine] Tool execution error for '{tool_name}': {tool_err}", exc_info=True)
                            tool_output = json.dumps({
                                "error": True,
                                "message": f"Error running tool '{tool_name}': {str(tool_err)}"
                            })
                            print(f"    ❌ Tool Error: {tool_err}")
                    else:
                        tool_output = json.dumps({"error": True, "message": f"Tool '{tool_name}' is not recognized."})
                        print(f"    ❌ Tool not found in registry: {tool_name}")

                    # Append Tool observation back to conversation history
                    messages.append(ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call_id
                    ))

            except Exception as e:
                logger.error(f"[AgenticEngine] Error during agent loop turn {iteration}: {e}", exc_info=True)
                print(f"❌ [AGENT RUNTIME ERROR] {e}")
                return (
                    "I encountered an unexpected issue while processing your request. "
                    "If you need immediate assistance, please let me know your email address "
                    "so I can create a support ticket for you."
                )

        # Fallback if max iterations exceeded
        print("⚠️ [AGENT WARNING] Max iterations reached without conclusive stop.")
        return "I completed the necessary checks. Please let me know if you would like me to assist you with anything else!"

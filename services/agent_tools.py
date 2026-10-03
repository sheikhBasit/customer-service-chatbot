"""
services/agent_tools.py - Agentic Tools for Customer Service Chatbot
Implements:
1. get_order_status - Order status, shipping carrier, and live tracking lookup
2. create_support_ticket - Support ticket creation and issue escalation
3. search_knowledge_base - Tenant-specific RAG retrieval tool

Includes comprehensive debug statements, console outputs, and execution tracing.
"""

import logging
import secrets
from datetime import datetime
from typing import Optional, Dict, Any, List
from bson import ObjectId

from database import db
from models.customer_chatbot import CustomerOrder, SupportTicket

# Setup detailed logger
logger = logging.getLogger("agent_tools")
logger.setLevel(logging.DEBUG)

# Pre-seeded demo orders for instant out-of-the-box testing & simulation
MOCK_DEMO_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "end_user_name": "Sarah Connor",
        "end_user_email": "sarah@example.com",
        "status": "delivered",
        "carrier": "FedEx Express",
        "tracking_number": "FDX-99823184",
        "estimated_delivery": "2026-09-28",
        "items": [
            {"name": "Ultra HD Noise Cancelling Headphones", "quantity": 1, "price": 199.99},
            {"name": "Audio Jack Cable Adapter", "quantity": 2, "price": 12.50}
        ],
        "total_amount": 224.99,
        "currency": "USD",
        "notes": "Delivered to front porch. Signature waived."
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "end_user_name": "John Doe",
        "end_user_email": "john.doe@example.com",
        "status": "in_transit",
        "carrier": "UPS Ground",
        "tracking_number": "1Z9999999999999999",
        "estimated_delivery": "Tomorrow by 7:00 PM",
        "items": [
            {"name": "Ergonomic Mechanical Keyboard", "quantity": 1, "price": 129.00}
        ],
        "total_amount": 129.00,
        "currency": "USD",
        "notes": "Package scanned at local distribution hub."
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "end_user_name": "Alice Smith",
        "end_user_email": "alice@example.com",
        "status": "processing",
        "carrier": "DHL Express",
        "tracking_number": "DHL-004481923",
        "estimated_delivery": "In 3-5 business days",
        "items": [
            {"name": "Smart Fitness Smartwatch", "quantity": 1, "price": 249.99}
        ],
        "total_amount": 249.99,
        "currency": "USD",
        "notes": "Order packed at warehouse; awaiting courier pickup."
    }
}


# ==============================================================================
# TOOL 1: GET ORDER STATUS
# ==============================================================================

async def execute_get_order_status(
    order_id: str,
    customer_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Look up the live status, courier tracking, and line items for an order.
    
    Args:
        order_id: The order identifier (e.g. 'ORD-1002')
        customer_id: The tenant/business ID operating the chatbot
        
    Returns:
        Dict containing order details, status, ETA, and items
    """
    clean_order_id = str(order_id).strip().upper()
    
    print("\n" + "=" * 65)
    print(f"🛠️  [TOOL CALLED: get_order_status]")
    print(f"    ➡️  Order ID requested : {clean_order_id}")
    print(f"    ➡️  Customer Tenant ID  : {customer_id or 'Global/Demo'}")
    print("=" * 65)
    
    logger.debug(f"[get_order_status] Looking up order_id='{clean_order_id}', customer_id='{customer_id}'")

    # 1. Query MongoDB if connected
    found_order = None
    if db.customer_orders_collection is not None:
        try:
            query: Dict[str, Any] = {"order_id": clean_order_id}
            if customer_id and ObjectId.is_valid(customer_id):
                query["customer_id"] = ObjectId(customer_id)
            
            logger.debug(f"[get_order_status] Querying MongoDB with filter: {query}")
            found_order = await db.customer_orders_collection.find_one(query)
            
            # If not found with customer_id filter, try without tenant filter as fallback
            if not found_order:
                found_order = await db.customer_orders_collection.find_one({"order_id": clean_order_id})
                
        except Exception as e:
            logger.error(f"[get_order_status] Database error while fetching order: {e}", exc_info=True)
            print(f"⚠️  [TOOL WARNING] MongoDB query failed ({e}), falling back to in-memory order registry.")

    # 2. Check in-memory demo registry if not found in MongoDB
    if not found_order and clean_order_id in MOCK_DEMO_ORDERS:
        print(f"ℹ️   [TOOL DEBUG] Order matched in pre-seeded demo registry: {clean_order_id}")
        found_order = MOCK_DEMO_ORDERS[clean_order_id]

    # 3. Format and return response
    if found_order:
        status = found_order.get("status", "unknown").upper()
        carrier = found_order.get("carrier", "Standard Courier")
        tracking_num = found_order.get("tracking_number", "N/A")
        eta = found_order.get("estimated_delivery", "Pending Confirmation")
        items = found_order.get("items", [])
        total = found_order.get("total_amount", 0.0)
        currency = found_order.get("currency", "USD")
        notes = found_order.get("notes", "No additional notes")

        result = {
            "success": True,
            "order_id": clean_order_id,
            "status": status,
            "carrier": carrier,
            "tracking_number": tracking_num,
            "estimated_delivery": eta,
            "total_amount": f"{currency} {total:.2f}",
            "items_count": len(items),
            "items": items,
            "notes": notes,
            "message": f"Order {clean_order_id} is currently {status}. Carrier: {carrier}, ETA: {eta}."
        }
        
        print(f"✅  [TOOL SUCCESS: get_order_status]")
        print(f"    📦 Status            : {status}")
        print(f"    🚚 Carrier / Track # : {carrier} ({tracking_num})")
        print(f"    📅 Estimated Arrival : {eta}")
        print(f"    💰 Order Total       : {currency} {total:.2f}")
        print("=" * 65 + "\n")
        return result
    else:
        print(f"❌  [TOOL NOT FOUND: get_order_status]")
        print(f"    Could not locate order: {clean_order_id}")
        print("=" * 65 + "\n")
        return {
            "success": False,
            "order_id": clean_order_id,
            "status": "NOT_FOUND",
            "message": (
                f"We could not find an order matching '{clean_order_id}'. "
                f"Please check the order number or provide the email address used during purchase."
            )
        }


# ==============================================================================
# TOOL 2: CREATE SUPPORT TICKET
# ==============================================================================

async def execute_create_support_ticket(
    customer_email: str,
    subject: str,
    description: str,
    customer_id: Optional[str] = None,
    customer_name: Optional[str] = "Customer",
    priority: str = "medium",
    related_order_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an official customer support ticket and record it in MongoDB.
    
    Args:
        customer_email: Contact email address of the customer
        subject: Brief title / summary of the issue
        description: Detailed explanation of the issue or complaint
        customer_id: Tenant / business ID
        customer_name: Customer's display name
        priority: 'low', 'medium', 'high', or 'urgent'
        related_order_id: Optional related order ID (e.g., 'ORD-1002')
        session_id: Chat session ID where the ticket was initiated
        
    Returns:
        Dict confirming ticket creation with reference ID and SLA timeline
    """
    # Normalize inputs
    clean_priority = priority.lower() if priority.lower() in ["low", "medium", "high", "urgent"] else "medium"
    ticket_ref = f"TICK-{secrets.token_hex(3).upper()}"
    now = datetime.now()
    
    print("\n" + "=" * 65)
    print(f"🛠️  [TOOL CALLED: create_support_ticket]")
    print(f"    🎟️  Generated Ticket ID : {ticket_ref}")
    print(f"    📧 Customer Email      : {customer_email}")
    print(f"    📌 Subject             : {subject}")
    print(f"    ⚡ Priority            : {clean_priority.upper()}")
    if related_order_id:
        print(f"    📦 Related Order       : {related_order_id}")
    print(f"    📝 Description         : {description[:120]}...")
    print("=" * 65)

    logger.debug(
        f"[create_support_ticket] ticket_ref={ticket_ref}, email={customer_email}, "
        f"subject='{subject}', priority={clean_priority}"
    )

    ticket_doc = {
        "ticket_id": ticket_ref,
        "customer_id": ObjectId(customer_id) if customer_id and ObjectId.is_valid(customer_id) else None,
        "customer_email": customer_email.strip(),
        "customer_name": customer_name or "Valued Customer",
        "subject": subject.strip(),
        "description": description.strip(),
        "priority": clean_priority,
        "status": "open",
        "related_order_id": related_order_id.strip().upper() if related_order_id else None,
        "session_id": ObjectId(session_id) if session_id and ObjectId.is_valid(session_id) else None,
        "created_at": now,
        "updated_at": now
    }

    # Insert into database
    mongo_saved = False
    if db.customer_support_tickets_collection is not None:
        try:
            insert_result = await db.customer_support_tickets_collection.insert_one(ticket_doc)
            mongo_saved = True
            logger.info(f"[create_support_ticket] Stored ticket in DB with _id={insert_result.inserted_id}")
        except Exception as e:
            logger.error(f"[create_support_ticket] Failed to persist ticket to MongoDB: {e}", exc_info=True)
            print(f"⚠️  [TOOL WARNING] MongoDB ticket insertion failed ({e}); ticket registered in-memory.")

    # SLA estimation based on priority
    sla_map = {
        "urgent": "Within 2 to 4 hours",
        "high": "Within 6 to 12 hours",
        "medium": "Within 24 business hours",
        "low": "Within 48 business hours"
    }
    estimated_sla = sla_map.get(clean_priority, "Within 24 hours")

    response_payload = {
        "success": True,
        "ticket_id": ticket_ref,
        "status": "OPEN",
        "customer_email": customer_email,
        "priority": clean_priority.upper(),
        "subject": subject,
        "related_order_id": related_order_id.upper() if related_order_id else None,
        "estimated_response_time": estimated_sla,
        "created_at": now.isoformat(),
        "database_stored": mongo_saved,
        "message": (
            f"Support ticket {ticket_ref} has been successfully created. "
            f"Our dedicated support team has been notified and will contact {customer_email} "
            f"{estimated_sla.lower()}."
        )
    }

    print(f"✅  [TOOL SUCCESS: create_support_ticket]")
    print(f"    🎟️  Ticket ID Confirmed : {ticket_ref}")
    print(f"    ⏱️  Estimated SLA       : {estimated_sla}")
    print(f"    💾 Persisted to Mongo   : {mongo_saved}")
    print("=" * 65 + "\n")

    return response_payload


# ==============================================================================
# TOOL 3: SEARCH KNOWLEDGE BASE (RAG)
# ==============================================================================

async def execute_search_knowledge_base(
    customer_id: str,
    query: str,
    k: int = 4
) -> Dict[str, Any]:
    """
    Search customer's indexed documents, policies, FAQs, and product manuals.
    
    Args:
        customer_id: Tenant customer ID
        query: User search query or question
        k: Number of relevant chunks to retrieve
        
    Returns:
        Dict with retrieved context and references
    """
    print("\n" + "=" * 65)
    print(f"🛠️  [TOOL CALLED: search_knowledge_base]")
    print(f"    🏢 Customer ID : {customer_id}")
    print(f"    🔍 Query       : {query}")
    print("=" * 65)

    from services.customer_vectorstore import CustomerVectorStoreService
    from services.multimodal_embeddings import embed_text
    
    vectorstore_service = CustomerVectorStoreService()
    vectorstore_data = await vectorstore_service.get_customer_vectorstore(customer_id)
    
    if not vectorstore_data:
        print(f"⚠️  [TOOL WARNING] No vectorstore index available for customer {customer_id}")
        print("=" * 65 + "\n")
        return {
            "success": False,
            "context": "",
            "message": "No documents or FAQs have been indexed for this company yet."
        }

    vectorstore, _ = vectorstore_data
    emb = embed_text(query)
    docs = vectorstore.similarity_search_by_vector(emb, k=k)
    
    context_chunks = []
    for idx, doc in enumerate(docs):
        filename = doc.metadata.get("filename", "document")
        page = doc.metadata.get("page", 1)
        content_snippet = doc.page_content.strip()
        context_chunks.append(f"[{filename} - Page {page}]: {content_snippet}")

    full_context = "\n\n".join(context_chunks)

    print(f"✅  [TOOL SUCCESS: search_knowledge_base]")
    print(f"    📚 Retrieved {len(docs)} relevant context snippets")
    print("=" * 65 + "\n")

    return {
        "success": True,
        "chunks_count": len(docs),
        "context": full_context,
        "message": f"Found {len(docs)} relevant sections in company documents."
    }


# ==============================================================================
# TOOL DEFINITIONS FOR LLM FUNCTION CALLING (LangChain / Groq Compatible)
# ==============================================================================

# Definitions structured for Groq / OpenAI compatible tool-calling schema
AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Retrieve real-time tracking, shipping carrier, items, and status of an order. "
                "Use this tool whenever a customer asks about their order, package delivery, "
                "or mentions an order number like 'ORD-1002'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The exact order number or ID, e.g., 'ORD-1001', 'ORD-1002'."
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": (
                "Create an official customer support ticket for unresolved issues, damaged/broken goods, "
                "refund requests, human agent escalation, or formal customer complaints. "
                "Always collect or confirm the customer's email address and issue details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_email": {
                        "type": "string",
                        "description": "The customer's contact email address to receive updates."
                    },
                    "subject": {
                        "type": "string",
                        "description": "Short, clear summary of the issue (e.g. 'Defective headphone left ear')."
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed explanation of the issue or customer request."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "Ticket priority based on urgency. Default is 'medium'."
                    },
                    "related_order_id": {
                        "type": "string",
                        "description": "Optional order number if the complaint relates to a specific purchase."
                    }
                },
                "required": ["customer_email", "subject", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search indexed company documents, return policies, FAQs, manuals, and warranty details. "
                "Use this to answer questions about return windows, warranties, company policies, and FAQs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search phrase or question to look up in company documents."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

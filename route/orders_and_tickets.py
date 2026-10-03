"""
routes/orders_and_tickets.py - REST Endpoints for Order Tracking and Support Tickets
Provides APIs for:
1. GET  /api/v1/orders/{order_id} - Fetch order details & shipping status
2. POST /api/v1/orders           - Seed/Create an order for a tenant
3. GET  /api/v1/customers/{customer_id}/orders - List customer orders
4. POST /api/v1/tickets          - File a new support ticket
5. GET  /api/v1/tickets/{ticket_id} - Fetch ticket status & details
6. GET  /api/v1/customers/{customer_id}/tickets - List tickets for tenant

Includes extensive comments, execution outputs, and debug statements.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query, Header, Depends, status
from pydantic import BaseModel, EmailStr, Field

from database import db
from models.customer_chatbot import CustomerOrder, SupportTicket
from services.agent_tools import (
    execute_get_order_status,
    execute_create_support_ticket,
    MOCK_DEMO_ORDERS
)

logger = logging.getLogger("orders_and_tickets_api")
logger.setLevel(logging.DEBUG)

router = APIRouter(prefix="/api/v1", tags=["Orders & Support Tickets"])


# ==============================================================================
# PYDANTIC REQUEST SCHEMAS
# ==============================================================================

class CreateOrderRequest(BaseModel):
    """Schema for creating a new order"""
    order_id: str = Field(..., description="Unique order reference, e.g. ORD-1004")
    customer_id: str = Field(..., description="Business Tenant customer_id")
    end_user_name: Optional[str] = "Customer"
    end_user_email: Optional[str] = None
    status: str = Field("processing", description="pending, processing, in_transit, delivered, cancelled")
    carrier: Optional[str] = "FedEx Express"
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = "3-5 business days"
    items: List[Dict[str, Any]] = []
    total_amount: float = 0.0
    currency: str = "USD"
    notes: Optional[str] = None


class CreateTicketRequest(BaseModel):
    """Schema for creating a support ticket"""
    customer_id: Optional[str] = Field(None, description="Business Tenant customer_id")
    customer_email: str = Field(..., description="Customer contact email")
    customer_name: Optional[str] = "Customer"
    subject: str = Field(..., description="Brief summary of the issue")
    description: str = Field(..., description="Detailed explanation of the issue")
    priority: str = Field("medium", description="low, medium, high, or urgent")
    related_order_id: Optional[str] = Field(None, description="Optional associated order ID")
    session_id: Optional[str] = Field(None, description="Optional chat session ID")


# ==============================================================================
# 1. ORDER ENDPOINTS
# ==============================================================================

@router.get("/orders/{order_id}")
async def get_order_endpoint(
    order_id: str,
    customer_id: Optional[str] = Query(None, description="Tenant customer ID filter")
):
    """
    Look up live status, carrier, tracking number, and items for an order.
    Can be queried by end-users or called by frontend widgets.
    """
    print(f"\n📡 [API GET /api/v1/orders/{order_id}] Request received")
    logger.info(f"Received order lookup request for order_id: {order_id}")

    result = await execute_get_order_status(order_id=order_id, customer_id=customer_id)
    
    if not result.get("success"):
        print(f"⚠️  [API 404] Order {order_id} not found in database or demo registry")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.get("message")
        )

    print(f"✅ [API 200] Returning order details for {order_id}")
    return result


@router.post("/orders", status_code=status.HTTP_201_CREATED)
async def create_order_endpoint(payload: CreateOrderRequest):
    """
    Seed or create a new order in MongoDB under a business customer/tenant account.
    """
    print(f"\n📡 [API POST /api/v1/orders] Creating order: {payload.order_id}")
    logger.info(f"Creating new order {payload.order_id} for tenant {payload.customer_id}")

    try:
        # Check if order_id already exists for this tenant
        existing = await db.customer_orders_collection.find_one({
            "order_id": payload.order_id.upper()
        })
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Order '{payload.order_id}' already exists."
            )

        order_doc = {
            "customer_id": ObjectId(payload.customer_id) if ObjectId.is_valid(payload.customer_id) else None,
            "order_id": payload.order_id.upper().strip(),
            "end_user_name": payload.end_user_name,
            "end_user_email": payload.end_user_email,
            "status": payload.status.lower(),
            "items": payload.items,
            "total_amount": payload.total_amount,
            "currency": payload.currency,
            "carrier": payload.carrier,
            "tracking_number": payload.tracking_number,
            "estimated_delivery": payload.estimated_delivery,
            "notes": payload.notes,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        result = await db.customer_orders_collection.insert_one(order_doc)
        print(f"✅ [API 201] Order {payload.order_id} saved to MongoDB with _id={result.inserted_id}")

        return {
            "status": "success",
            "message": f"Order {payload.order_id} created successfully.",
            "order_id": payload.order_id.upper(),
            "db_id": str(result.inserted_id)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating order: {e}", exc_info=True)
        print(f"❌ [API ERROR] Failed to create order: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/customers/{customer_id}/orders")
async def list_customer_orders(
    customer_id: str,
    limit: int = Query(50, ge=1, le=100)
):
    """
    List all orders belonging to a specific business customer/tenant.
    """
    print(f"\n📡 [API GET /api/v1/customers/{customer_id}/orders] Fetching orders")
    try:
        query = {}
        if ObjectId.is_valid(customer_id):
            query["customer_id"] = ObjectId(customer_id)

        orders = await db.customer_orders_collection.find(query).sort("created_at", -1).to_list(limit)
        
        # Serialize ObjectIds
        formatted_orders = []
        for o in orders:
            o["_id"] = str(o["_id"])
            if o.get("customer_id"):
                o["customer_id"] = str(o["customer_id"])
            formatted_orders.append(o)

        print(f"✅ [API 200] Returning {len(formatted_orders)} orders for customer {customer_id}")
        return {
            "customer_id": customer_id,
            "total_count": len(formatted_orders),
            "orders": formatted_orders
        }
    except Exception as e:
        logger.error(f"Error listing orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# 2. SUPPORT TICKET ENDPOINTS
# ==============================================================================

@router.post("/tickets", status_code=status.HTTP_201_CREATED)
async def create_ticket_endpoint(payload: CreateTicketRequest):
    """
    Create a new support ticket and notify the support workflow.
    """
    print(f"\n📡 [API POST /api/v1/tickets] Creating ticket for: {payload.customer_email}")
    logger.info(f"Creating support ticket from API for: {payload.customer_email}, subject: {payload.subject}")

    result = await execute_create_support_ticket(
        customer_email=payload.customer_email,
        subject=payload.subject,
        description=payload.description,
        customer_id=payload.customer_id,
        customer_name=payload.customer_name,
        priority=payload.priority,
        related_order_id=payload.related_order_id,
        session_id=payload.session_id
    )

    print(f"✅ [API 201] Ticket created: {result.get('ticket_id')}")
    return result


@router.get("/tickets/{ticket_id}")
async def get_ticket_endpoint(ticket_id: str):
    """
    Fetch a specific support ticket by its human-readable ticket ID (e.g. 'TICK-A1B2C3').
    """
    clean_id = ticket_id.strip().upper()
    print(f"\n📡 [API GET /api/v1/tickets/{clean_id}] Fetching ticket status")

    try:
        ticket = await db.customer_support_tickets_collection.find_one({"ticket_id": clean_id})
        if not ticket:
            print(f"❌ [API 404] Support ticket {clean_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticket '{clean_id}' was not found in our records."
            )

        ticket["_id"] = str(ticket["_id"])
        if ticket.get("customer_id"):
            ticket["customer_id"] = str(ticket["customer_id"])
        if ticket.get("session_id"):
            ticket["session_id"] = str(ticket["session_id"])

        print(f"✅ [API 200] Found ticket {clean_id}: Status={ticket.get('status')}")
        return ticket
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ticket {clean_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/customers/{customer_id}/tickets")
async def list_customer_tickets(
    customer_id: str,
    status_filter: Optional[str] = Query(None, description="Filter by status: open, in_progress, resolved, closed"),
    limit: int = Query(50, ge=1, le=100)
):
    """
    List all support tickets created under a specific business tenant account.
    """
    print(f"\n📡 [API GET /api/v1/customers/{customer_id}/tickets] Listing tickets")
    try:
        query: Dict[str, Any] = {}
        if ObjectId.is_valid(customer_id):
            query["customer_id"] = ObjectId(customer_id)
        if status_filter:
            query["status"] = status_filter.lower()

        tickets = await db.customer_support_tickets_collection.find(query).sort("created_at", -1).to_list(limit)

        formatted_tickets = []
        for t in tickets:
            t["_id"] = str(t["_id"])
            if t.get("customer_id"):
                t["customer_id"] = str(t["customer_id"])
            if t.get("session_id"):
                t["session_id"] = str(t["session_id"])
            formatted_tickets.append(t)

        print(f"✅ [API 200] Found {len(formatted_tickets)} tickets for tenant {customer_id}")
        return {
            "customer_id": customer_id,
            "total_count": len(formatted_tickets),
            "tickets": formatted_tickets
        }
    except Exception as e:
        logger.error(f"Error fetching tickets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

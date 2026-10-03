"""
tests/test_agent_tools.py - Unit and Integration Tests for Agentic Tools & Endpoints
Validates:
1. execute_get_order_status tool execution (found & not found cases)
2. execute_create_support_ticket tool execution (creation, SLA mapping, output)
3. FastAPI router endpoints for orders & support tickets
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

venv_site_packages = project_root / "venv" / "lib" / "python3.12" / "site-packages"
if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
    sys.path.append(str(venv_site_packages))

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport

from main import app
from services.agent_tools import (
    execute_get_order_status,
    execute_create_support_ticket
)


@pytest.mark.anyio
async def test_get_order_status_success():
    """Verify live order status lookup for seeded demo order ORD-1002"""
    result = await execute_get_order_status(order_id="ORD-1002")
    
    assert result["success"] is True
    assert result["order_id"] == "ORD-1002"
    assert result["status"] == "IN_TRANSIT"
    assert result["carrier"] == "UPS Ground"
    assert "items" in result
    assert len(result["items"]) >= 1


@pytest.mark.anyio
async def test_get_order_status_not_found():
    """Verify appropriate response when querying non-existent order"""
    result = await execute_get_order_status(order_id="ORD-NON-EXISTENT-999")
    
    assert result["success"] is False
    assert result["status"] == "NOT_FOUND"
    assert "could not find" in result["message"].lower()


@pytest.mark.anyio
async def test_create_support_ticket_success():
    """Verify support ticket creation, ticket ID formatting, and SLA calculation"""
    result = await execute_create_support_ticket(
        customer_email="test.user@example.com",
        subject="Headphone left ear has no sound",
        description="I received the package yesterday but the left ear speaker is dead.",
        priority="high",
        related_order_id="ORD-1001"
    )
    
    assert result["success"] is True
    assert result["ticket_id"].startswith("TICK-")
    assert result["priority"] == "HIGH"
    assert result["related_order_id"] == "ORD-1001"
    assert "Within 6 to 12 hours" in result["estimated_response_time"]
    assert "successfully created" in result["message"]


@pytest.mark.anyio
async def test_rest_api_get_order():
    """Verify REST API endpoint GET /api/v1/orders/{order_id}"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/orders/ORD-1001")
        assert response.status_code == 200
        data = response.json()
        assert data["order_id"] == "ORD-1001"
        assert data["status"] == "DELIVERED"
        assert data["carrier"] == "FedEx Express"


@pytest.mark.anyio
async def test_rest_api_create_ticket():
    """Verify REST API endpoint POST /api/v1/tickets"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "customer_email": "api_tester@example.com",
            "subject": "Billing issue on monthly invoice",
            "description": "I was charged twice for the same subscription period.",
            "priority": "urgent"
        }
        response = await client.post("/api/v1/tickets", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["ticket_id"].startswith("TICK-")
        assert data["priority"] == "URGENT"

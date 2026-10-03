"""
scripts/demo_agent_tools.py - Interactive Demonstration of Order Status & Support Ticket Tools
Demonstrates:
1. Live execution of execute_get_order_status
2. Live execution of execute_create_support_ticket
3. Full formatted console debugging output and return schemas
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

import asyncio
import json
from services.agent_tools import (
    execute_get_order_status,
    execute_create_support_ticket
)


async def run_demonstration():
    print("\n" + "#" * 75)
    print("  CUSTOMER SERVICE CHATBOT - AGENTIC TOOLS LIVE DEMO")
    print("#" * 75)

    # -------------------------------------------------------------
    # 1. Order Status Lookup (Found)
    # -------------------------------------------------------------
    print("\n>>> DEMO 1: Looking up active order 'ORD-1002'...")
    order_result = await execute_get_order_status(order_id="ORD-1002")
    print("[RESULT DATA STRUCTURE]:")
    print(json.dumps(order_result, indent=2))

    # -------------------------------------------------------------
    # 2. Order Status Lookup (Not Found)
    # -------------------------------------------------------------
    print("\n>>> DEMO 2: Looking up unknown order 'ORD-99999'...")
    unknown_result = await execute_get_order_status(order_id="ORD-99999")
    print("[RESULT DATA STRUCTURE]:")
    print(json.dumps(unknown_result, indent=2))

    # -------------------------------------------------------------
    # 3. Create Support Ticket (Normal Priority)
    # -------------------------------------------------------------
    print("\n>>> DEMO 3: Creating a support ticket for defective product...")
    ticket_result = await execute_create_support_ticket(
        customer_email="sarah.connor@cyberdyne.org",
        subject="Headphone left earphone has static sound",
        description="The headphone arrived yesterday. The left driver produces buzzing audio.",
        customer_name="Sarah Connor",
        priority="high",
        related_order_id="ORD-1001"
    )
    print("[RESULT DATA STRUCTURE]:")
    print(json.dumps(ticket_result, indent=2))

    # -------------------------------------------------------------
    # 4. Create Support Ticket (Urgent Priority)
    # -------------------------------------------------------------
    print("\n>>> DEMO 4: Creating an URGENT support ticket for wrong delivery address...")
    urgent_ticket = await execute_create_support_ticket(
        customer_email="john.doe@acme.com",
        subject="Urgent: Package sent to wrong office branch",
        description="Please re-route ORD-1002 to Building B immediately before delivery.",
        customer_name="John Doe",
        priority="urgent",
        related_order_id="ORD-1002"
    )
    print("[RESULT DATA STRUCTURE]:")
    print(json.dumps(urgent_ticket, indent=2))

    print("\n" + "#" * 75)
    print("  DEMO COMPLETE - ALL TOOLS EXECUTED SUCCESSFULLY")
    print("#" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(run_demonstration())

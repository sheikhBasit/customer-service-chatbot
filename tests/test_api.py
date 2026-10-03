"""
Test Setup Script for Customer Chatbot Backend
This script creates test data in MongoDB for testing the frontend
"""

import asyncio
import os
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGO_DB", "cscb")


async def setup_test_data():
    """Create test plans and customers"""
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    
    print("🚀 Setting up test data...")
    
    # 1. Create Plans
    print("\n📋 Creating subscription plans...")
    
    plans = [
        {
            "name": "Starter",
            "price_monthly": 49.99,
            "max_documents": 10,
            "max_queries_per_month": 1000,
            "max_concurrent_users": 5,
            "websocket_enabled": True,
            "custom_branding": False,
            "api_rate_limit": 100
        },
        {
            "name": "Professional",
            "price_monthly": 149.99,
            "max_documents": 50,
            "max_queries_per_month": 5000,
            "max_concurrent_users": 20,
            "websocket_enabled": True,
            "custom_branding": True,
            "api_rate_limit": 200
        },
        {
            "name": "Enterprise",
            "price_monthly": 499.99,
            "max_documents": 200,
            "max_queries_per_month": 20000,
            "max_concurrent_users": 100,
            "websocket_enabled": True,
            "custom_branding": True,
            "api_rate_limit": 500
        }
    ]
    
    plan_ids = []
    for plan in plans:
        # Check if plan already exists
        existing = await db.chatbot_plans.find_one({"name": plan["name"]})
        if existing:
            print(f"   ✓ Plan '{plan['name']}' already exists")
            plan_ids.append(existing["_id"])
        else:
            result = await db.chatbot_plans.insert_one(plan)
            plan_ids.append(result.inserted_id)
            print(f"   ✓ Created plan: {plan['name']} (ID: {result.inserted_id})")
    
    # 2. Create Test Customers
    print("\n👥 Creating test customers...")
    
    customers = [
        {
            "company_name": "Acme Corporation",
            "email": "admin@acme.com",
            "api_key": f"cbk_{ObjectId()}",
            "plan_id": plan_ids[0],  # Starter
            "is_active": True,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "current_month_queries": 0,
            "total_queries": 0,
            "documents_count": 0,
            "chatbot_name": "Acme AI Assistant",
            "chatbot_greeting": "Hello! I'm the Acme AI Assistant. How can I help you today?",
            "brand_color": "#0066cc",
            "logo_url": None
        },
        {
            "company_name": "Tech Innovations Inc",
            "email": "contact@techinnovations.com",
            "api_key": f"cbk_{ObjectId()}",
            "plan_id": plan_ids[1],  # Professional
            "is_active": True,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "current_month_queries": 0,
            "total_queries": 0,
            "documents_count": 0,
            "chatbot_name": "TechBot",
            "chatbot_greeting": "Welcome! I'm TechBot, ready to assist you.",
            "brand_color": "#ff6600",
            "logo_url": None
        }
    ]
    
    customer_data = []
    for customer in customers:
        # Check if customer already exists
        existing = await db.customers.find_one({"email": customer["email"]})
        if existing:
            print(f"   ✓ Customer '{customer['company_name']}' already exists")
            customer_data.append({
                "id": str(existing["_id"]),
                "api_key": existing["api_key"],
                "company": existing["company_name"]
            })
        else:
            result = await db.customers.insert_one(customer)
            print(f"   ✓ Created customer: {customer['company_name']}")
            customer_data.append({
                "id": str(result.inserted_id),
                "api_key": customer["api_key"],
                "company": customer["company_name"]
            })
    
    # 3. Print credentials
    print("\n" + "="*60)
    print("✅ TEST DATA SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Use these credentials in your frontend:\n")
    
    for idx, cust in enumerate(customer_data, 1):
        print(f"Customer {idx}: {cust['company']}")
        print(f"  Customer ID: {cust['id']}")
        print(f"  API Key:     {cust['api_key']}")
        print()
    
    print("="*60)
    print("\n💡 Copy one of these credential sets into the frontend")
    print("   configuration screen to start testing!\n")
    
    # Close connection
    client.close()


if __name__ == "__main__":
    print("="*60)
    print("  Customer Chatbot - Test Data Setup")
    print("="*60)
    asyncio.run(setup_test_data())
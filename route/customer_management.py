"""
routes/customer_management.py - Admin routes for managing customers and documents
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from bson import ObjectId
from datetime import datetime
import shutil
from pathlib import Path

from models.customer_chatbot import (
    Customer, CustomerDocument, ChatbotPlan,
    MonthlyUsageReport
)
from services.customer_vectorstore import CustomerVectorStoreService
from utils.auth import get_customer_from_header, get_customer_usage_stats
from database import db
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/admin", tags=["Customer Management"])

vectorstore_service = CustomerVectorStoreService()


# ==================== CUSTOMER CRUD ====================

@router.post("/customers")
async def create_customer(
    company_name: str,
    email: str,
    plan_id: str,
    chatbot_name: Optional[str] = "AI Assistant"
):
    """Create a new customer account"""
    try:
        # Validate plan exists
        plan = await db.chatbot_plans_collection.find_one({"_id": ObjectId(plan_id)})
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        
        # Check if email already exists
        existing = await db.customers_collection.find_one({"email": email})
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create customer
        customer = Customer(
            company_name=company_name,
            email=email,
            plan_id=ObjectId(plan_id),
            chatbot_name=chatbot_name
        )
        
        result = await db.customers_collection.insert_one(
            customer.model_dump(by_alias=True)
        )
        
        customer.id = result.inserted_id
        
        return {
            "customer_id": str(customer.id),
            "api_key": customer.api_key,
            "company_name": customer.company_name,
            "plan": plan["name"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating customer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create customer")


@router.get("/customers/{customer_id}")
async def get_customer(customer_id: str):
    """Get customer details"""
    try:
        customer = await db.customers_collection.find_one({"_id": ObjectId(customer_id)})
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Get plan details
        plan = await db.chatbot_plans_collection.find_one({"_id": customer["plan_id"]})
        
        return {
            **customer,
            "_id": str(customer["_id"]),
            "plan_id": str(customer["plan_id"]),
            "plan_name": plan["name"] if plan else "Unknown",
            "api_key": customer["api_key"]  # In production, mask this
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching customer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch customer")


@router.patch("/customers/{customer_id}")
async def update_customer(
    customer_id: str,
    chatbot_name: Optional[str] = None,
    chatbot_greeting: Optional[str] = None,
    brand_color: Optional[str] = None,
    logo_url: Optional[str] = None,
    is_active: Optional[bool] = None
):
    """Update customer settings"""
    try:
        update_data = {}
        if chatbot_name is not None:
            update_data["chatbot_name"] = chatbot_name
        if chatbot_greeting is not None:
            update_data["chatbot_greeting"] = chatbot_greeting
        if brand_color is not None:
            update_data["brand_color"] = brand_color
        if logo_url is not None:
            update_data["logo_url"] = logo_url
        if is_active is not None:
            update_data["is_active"] = is_active
        
        update_data["updated_at"] = datetime.now()
        
        result = await db.customers_collection.update_one(
            {"_id": ObjectId(customer_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        return {"message": "Customer updated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating customer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update customer")


# ==================== DOCUMENT MANAGEMENT ====================

@router.post("/customers/{customer_id}/documents")
async def upload_document(
    customer_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload a document for a customer"""
    try:
        # Validate customer
        customer = await db.customers_collection.find_one({"_id": ObjectId(customer_id)})
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Check document limit
        plan = await db.chatbot_plans_collection.find_one({"_id": customer["plan_id"]})
        current_docs = customer.get("documents_count", 0)
        
        if current_docs >= plan["max_documents"]:
            raise HTTPException(
                status_code=400,
                detail=f"Document limit reached ({plan['max_documents']} documents)"
            )
        
        # Validate file type
        allowed_types = ["application/pdf", "text/plain", "text/markdown"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Allowed: PDF, TXT, MD"
            )
        
        # Create document record
        document = CustomerDocument(
            customer_id=ObjectId(customer_id),
            filename=file.filename,
            file_size=0,  # Will be updated after saving
            file_type=file.content_type.split("/")[-1]
        )
        
        result = await db.customer_documents_collection.insert_one(
            document.model_dump(by_alias=True)
        )
        document.id = result.inserted_id
        
        # Save file
        file_path = vectorstore_service._get_document_path(customer_id, document.id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with file_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Update file size
        file_size = file_path.stat().st_size
        await db.customer_documents_collection.update_one(
            {"_id": document.id},
            {"$set": {"file_size": file_size}}
        )
        
        # Update customer document count
        await db.customers_collection.update_one(
            {"_id": ObjectId(customer_id)},
            {"$inc": {"documents_count": 1}}
        )
        
        # Process document in background
        background_tasks.add_task(process_document_task, str(customer_id), str(document.id))
        
        return {
            "document_id": str(document.id),
            "filename": document.filename,
            "status": "processing",
            "message": "Document uploaded successfully and is being processed"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to upload document")


async def process_document_task(customer_id: str, document_id: str):
    """Background task to process uploaded document"""
    try:
        # Update status to processing
        await db.customer_documents_collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"processing_status": "processing"}}
        )
        
        # Process all documents for this customer (rebuild vectorstore)
        await vectorstore_service.process_customer_documents(customer_id, force_rebuild=True)
        
        # Update status to completed
        await db.customer_documents_collection.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "processing_status": "completed",
                    "processed": True
                }
            }
        )
        
        logger.info(f"Document {document_id} processed successfully")
    
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}", exc_info=True)
        await db.customer_documents_collection.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "processing_status": "failed",
                    "error_message": str(e)
                }
            }
        )


@router.get("/customers/{customer_id}/documents")
async def list_documents(customer_id: str):
    """List all documents for a customer"""
    try:
        documents = await db.customer_documents_collection.find({
            "customer_id": ObjectId(customer_id)
        }).to_list(None)
        
        return {
            "documents": [
                {
                    **doc,
                    "_id": str(doc["_id"]),
                    "customer_id": str(doc["customer_id"])
                }
                for doc in documents
            ],
            "total": len(documents)
        }
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.delete("/customers/{customer_id}/documents/{document_id}")
async def delete_document(customer_id: str, document_id: str):
    """Delete a document"""
    try:
        # Find document
        document = await db.customer_documents_collection.find_one({
            "_id": ObjectId(document_id),
            "customer_id": ObjectId(customer_id)
        })
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file
        file_path = vectorstore_service._get_document_path(customer_id, ObjectId(document_id))
        if file_path.exists():
            file_path.unlink()
        
        # Delete database record
        await db.customer_documents_collection.delete_one({"_id": ObjectId(document_id)})
        
        # Update customer document count
        await db.customers_collection.update_one(
            {"_id": ObjectId(customer_id)},
            {"$inc": {"documents_count": -1}}
        )
        
        # Rebuild vectorstore
        await vectorstore_service.process_customer_documents(customer_id, force_rebuild=True)
        
        return {"message": "Document deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete document")


# ==================== USAGE & ANALYTICS ====================

@router.get("/customers/{customer_id}/usage")
async def get_usage(customer_id: str, month_year: Optional[str] = None):
    """Get usage statistics for a customer"""
    try:
        stats = await get_customer_usage_stats(customer_id, month_year)
        
        # Get customer info for limits
        customer = await db.customers_collection.find_one({"_id": ObjectId(customer_id)})
        plan = await db.chatbot_plans_collection.find_one({"_id": customer["plan_id"]})
        
        return {
            "month_year": month_year or datetime.now().strftime("%Y-%m"),
            "current_month_queries": customer.get("current_month_queries", 0),
            "plan_limit": plan["max_queries_per_month"],
            "usage_percentage": (customer.get("current_month_queries", 0) / plan["max_queries_per_month"]) * 100,
            **stats
        }
    
    except Exception as e:
        logger.error(f"Error fetching usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch usage data")


# ==================== PLANS ====================

@router.get("/plans")
async def list_plans():
    """List all available plans"""
    try:
        plans = await db.chatbot_plans_collection.find().to_list(None)
        return {
            "plans": [
                {**plan, "_id": str(plan["_id"])}
                for plan in plans
            ]
        }
    except Exception as e:
        logger.error(f"Error listing plans: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list plans")
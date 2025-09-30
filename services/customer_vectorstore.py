"""
services/customer_vectorstore.py - Handle document processing for customers
"""
import hashlib
import pickle
import io
import base64
from pathlib import Path
from typing import Tuple, Optional, List
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bson import ObjectId
import logging

from services.multimodal_embeddings import embed_text, embed_image
from models.customer_chatbot import Customer, CustomerDocument, CustomerVectorStore
from database import db

logger = logging.getLogger(__name__)


class CustomerVectorStoreService:
    """Manage vectorstores for different customers"""
    
    def __init__(self, cache_base_dir: str = ".customer_vectorstores"):
        self.cache_base_dir = Path(cache_base_dir)
        self.cache_base_dir.mkdir(exist_ok=True)
    
    def _get_customer_cache_dir(self, customer_id: str) -> Path:
        """Get dedicated cache directory for a customer"""
        customer_dir = self.cache_base_dir / str(customer_id)
        customer_dir.mkdir(exist_ok=True)
        return customer_dir
    
    def _generate_cache_key(self, customer_id: str, document_ids: List[str]) -> str:
        """Generate unique cache key based on customer and their documents"""
        doc_string = "_".join(sorted(document_ids))
        hash_input = f"{customer_id}_{doc_string}".encode()
        return hashlib.md5(hash_input).hexdigest()
    
    async def process_customer_documents(
        self, 
        customer_id: str,
        force_rebuild: bool = False
    ) -> Tuple[FAISS, dict]:
        """
        Process all documents for a customer and create/update vectorstore
        
        Args:
            customer_id: Customer ID
            force_rebuild: Force rebuild even if cache exists
            
        Returns:
            Tuple of (vectorstore, image_data_store)
        """
        logger.info(f"Processing documents for customer {customer_id}")
        
        # Get all processed documents for this customer
        documents = await db.customer_documents_collection.find({
            "customer_id": ObjectId(customer_id),
            "processed": True,
            "processing_status": "completed"
        }).to_list(None)
        
        if not documents:
            raise ValueError(f"No processed documents found for customer {customer_id}")
        
        doc_ids = [str(doc["_id"]) for doc in documents]
        cache_key = self._generate_cache_key(customer_id, doc_ids)
        customer_cache_dir = self._get_customer_cache_dir(customer_id)
        
        # Check if vectorstore exists and is current
        if not force_rebuild and self._cache_exists(customer_cache_dir, cache_key):
            logger.info(f"Loading cached vectorstore for customer {customer_id}")
            return self._load_from_cache(customer_cache_dir, cache_key)
        
        # Build vectorstore from scratch
        logger.info(f"Building new vectorstore for customer {customer_id}")
        all_docs = []
        all_embeddings = []
        image_data_store = {}
        
        for doc_meta in documents:
            # Load document file and process
            file_path = self._get_document_path(customer_id, doc_meta["_id"])
            
            if doc_meta["file_type"] == "pdf":
                docs, embeddings, images = await self._process_pdf(file_path, doc_meta)
                all_docs.extend(docs)
                all_embeddings.extend(embeddings)
                image_data_store.update(images)
            elif doc_meta["file_type"] in ["txt", "md"]:
                docs, embeddings = await self._process_text_file(file_path, doc_meta)
                all_docs.extend(docs)
                all_embeddings.extend(embeddings)
        
        if not all_docs:
            raise ValueError(f"No content extracted from documents for customer {customer_id}")
        
        # Create FAISS vectorstore
        embeddings_array = np.array(all_embeddings)
        text_embeddings = [(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)]
        
        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=None,
            metadatas=[doc.metadata for doc in all_docs]
        )
        
        # Save to cache
        self._save_to_cache(customer_cache_dir, cache_key, vectorstore, image_data_store)
        
        # Update database record
        await self._update_vectorstore_record(customer_id, cache_key, len(all_embeddings))
        
        logger.info(f"Vectorstore created for customer {customer_id}: {len(all_embeddings)} embeddings")
        return vectorstore, image_data_store
    
    async def _process_pdf(
        self, 
        pdf_path: Path, 
        doc_meta: dict
    ) -> Tuple[List[Document], List[np.ndarray], dict]:
        """Process PDF file and extract text + images"""
        all_docs = []
        all_embeddings = []
        image_data_store = {}
        
        doc = fitz.open(str(pdf_path))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        
        try:
            for page_num, page in enumerate(doc):
                # Process text
                text = page.get_text()
                if text.strip():
                    temp_doc = Document(
                        page_content=text,
                        metadata={
                            "page": page_num,
                            "type": "text",
                            "document_id": str(doc_meta["_id"]),
                            "filename": doc_meta["filename"]
                        }
                    )
                    text_chunks = text_splitter.split_documents([temp_doc])
                    
                    for chunk in text_chunks:
                        embedding = embed_text(chunk.page_content)
                        all_embeddings.append(embedding)
                        all_docs.append(chunk)
                
                # Process images
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        image_id = f"doc_{doc_meta['_id']}_page_{page_num}_img_{img_index}"
                        
                        # Store as base64
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        image_data_store[image_id] = img_base64
                        
                        # Embed image
                        embedding = embed_image(pil_image)
                        all_embeddings.append(embedding)
                        
                        image_doc = Document(
                            page_content=f"[Image: {image_id}]",
                            metadata={
                                "page": page_num,
                                "type": "image",
                                "image_id": image_id,
                                "document_id": str(doc_meta["_id"]),
                                "filename": doc_meta["filename"]
                            }
                        )
                        all_docs.append(image_doc)
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")
                        continue
        finally:
            doc.close()
        
        return all_docs, all_embeddings, image_data_store
    
    async def _process_text_file(
        self, 
        file_path: Path, 
        doc_meta: dict
    ) -> Tuple[List[Document], List[np.ndarray]]:
        """Process plain text file"""
        all_docs = []
        all_embeddings = []
        
        text = file_path.read_text(encoding='utf-8')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        
        temp_doc = Document(
            page_content=text,
            metadata={
                "type": "text",
                "document_id": str(doc_meta["_id"]),
                "filename": doc_meta["filename"]
            }
        )
        
        chunks = text_splitter.split_documents([temp_doc])
        for chunk in chunks:
            embedding = embed_text(chunk.page_content)
            all_embeddings.append(embedding)
            all_docs.append(chunk)
        
        return all_docs, all_embeddings
    
    def _get_document_path(self, customer_id: str, document_id: ObjectId) -> Path:
        """Get file path for uploaded document"""
        customer_dir = self.cache_base_dir / str(customer_id) / "documents"
        customer_dir.mkdir(exist_ok=True)
        # Assume files are saved with their document ID
        return customer_dir / str(document_id)
    
    def _cache_exists(self, cache_dir: Path, cache_key: str) -> bool:
        """Check if cache files exist"""
        return all([
            (cache_dir / f"{cache_key}.faiss").exists(),
            (cache_dir / f"{cache_key}_images.pkl").exists()
        ])
    
    def _load_from_cache(self, cache_dir: Path, cache_key: str) -> Tuple[FAISS, dict]:
        """Load vectorstore from cache"""
        vectorstore = FAISS.load_local(
            str(cache_dir),
            index_name=cache_key,
            embeddings=None,
            allow_dangerous_deserialization=True
        )
        
        with open(cache_dir / f"{cache_key}_images.pkl", "rb") as f:
            image_data_store = pickle.load(f)
        
        return vectorstore, image_data_store
    
    def _save_to_cache(
        self, 
        cache_dir: Path, 
        cache_key: str, 
        vectorstore: FAISS, 
        image_data_store: dict
    ):
        """Save vectorstore to cache"""
        vectorstore.save_local(str(cache_dir), index_name=cache_key)
        
        with open(cache_dir / f"{cache_key}_images.pkl", "wb") as f:
            pickle.dump(image_data_store, f)
    
    async def _update_vectorstore_record(
        self, 
        customer_id: str, 
        cache_key: str, 
        total_embeddings: int
    ):
        """Update or create vectorstore record in database"""
        await db.customer_vectorstores_collection.update_one(
            {"customer_id": ObjectId(customer_id)},
            {
                "$set": {
                    "cache_key": cache_key,
                    "last_updated": datetime.now(),
                    "total_embeddings": total_embeddings
                },
                "$setOnInsert": {
                    "customer_id": ObjectId(customer_id),
                    "created_at": datetime.now()
                }
            },
            upsert=True
        )
    
    async def get_customer_vectorstore(
        self, 
        customer_id: str
    ) -> Optional[Tuple[FAISS, dict]]:
        """Get existing vectorstore for a customer"""
        record = await db.customer_vectorstores_collection.find_one({
            "customer_id": ObjectId(customer_id)
        })
        
        if not record:
            return None
        
        cache_dir = self._get_customer_cache_dir(customer_id)
        cache_key = record["cache_key"]
        
        if not self._cache_exists(cache_dir, cache_key):
            logger.warning(f"Cache files missing for customer {customer_id}, rebuilding...")
            return await self.process_customer_documents(customer_id)
        
        return self._load_from_cache(cache_dir, cache_key)
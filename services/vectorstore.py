import hashlib
import pickle
from datetime import datetime
from typing import Tuple
import time
import shutil
import json
from pathlib import Path
import os
import io
import base64
import fitz
import numpy as np
from PIL import Image
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from services.multimodal_embeddings import embed_image, embed_text
from services.vector_cache import CacheLoadError, VectorCache

def process_pdf_with_images(pdf_path: str, cache_dir: str = ".vector_cache", force_reprocess: bool = False) -> Tuple[FAISS, dict]:
    """Process PDF with enhanced caching and error handling"""
    cache = VectorCache(cache_dir)
    cache_key = cache.get_cache_key(pdf_path)
    
    if not force_reprocess and cache.cache_exists(cache_key):
        try:
            print("Loading from cache...")
            return cache.load_from_cache(cache_key)
        except CacheLoadError:
            print("Proceeding with reprocessing due to cache load error")
    
    print(f"Processing PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    all_docs = []
    all_embeddings = []
    image_data_store = {}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    try:
        for i, page in enumerate(doc):
            print(f"Processing page {i+1}")
            
            # Process text
            text = page.get_text()
            if text.strip():
                print(f"Found text on page {i+1}")
                temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
                text_chunks = text_splitter.split_documents([temp_doc])
                
                for chunk in text_chunks:
                    try:
                        embedding = embed_text(chunk.page_content)
                        all_embeddings.append(embedding)
                        all_docs.append(chunk)
                    except Exception as e:
                        print(f"Error embedding text chunk: {e}")
                        continue

            # Process images
            images = page.get_images(full=True)
            print(f"Found {len(images)} images on page {i+1}")
            
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    image_id = f"page_{i}_img_{img_index}"
                    
                    # Store as base64
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    image_data_store[image_id] = img_base64
                    
                    # Embed image using CLIP
                    embedding = embed_image(pil_image)
                    all_embeddings.append(embedding)
                    
                    image_doc = Document(
                        page_content=f"[Image: {image_id}]",
                        metadata={"page": i, "type": "image", "image_id": image_id}
                    )
                    all_docs.append(image_doc)
                    
                except Exception as e:
                    print(f"Error processing image {img_index} on page {i}: {e}")
                    continue

        # Validate we processed content
        if not all_docs:
            raise ValueError("PDF processing resulted in no content - document may be empty or corrupted")
            
        print(f"Processed {len(all_docs)} documents ({len([d for d in all_docs if d.metadata['type'] == 'text'])} text, "
              f"{len([d for d in all_docs if d.metadata['type'] == 'image'])} images)")
        
        # Convert to numpy arrays
        embeddings_array = np.array(all_embeddings)
        print(f"Created embeddings array of shape {embeddings_array.shape}")
        
        # Create FAISS index
        print("Creating FAISS index...")
        text_embeddings = [(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)]
        if not text_embeddings:
            raise ValueError("No valid text embeddings were created")
            
        vector_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=None,
            metadatas=[doc.metadata for doc in all_docs]
        )
        
        # Save to cache
        print("Saving to cache...")
        cache.save_to_cache(cache_key, vector_store, image_data_store)
        
        return vector_store, image_data_store
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        cache.clear_cache()
        raise ValueError(f"Failed to process PDF: {str(e)}")
    finally:
        doc.close()
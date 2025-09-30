import hashlib
import pickle
from datetime import datetime
from typing import Tuple
import shutil
import json
from pathlib import Path
import os
import io
import base64
import numpy as np
from PIL import Image
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from config import settings

class VectorCache:
    def __init__(self, cache_dir: str = ".vector_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.model_version = "clip-vit-base-patch32-v2"
    


    def get_cache_key(self, pdf_path: str) -> str:
        """Generate unique cache key based on file content and model version"""
        file_hash = hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()
        return f"{file_hash}_{self.model_version}"

    def cache_exists(self, cache_key: str) -> bool:
        """Check if all cache files exist"""
        required_files = [
            self.cache_dir / f"{cache_key}.faiss",
            self.cache_dir / f"{cache_key}_images.pkl",
            self.cache_dir / f"{cache_key}_meta.json"
        ]
        return all(f.exists() for f in required_files)

    def clear_cache(self):
        """Clear all cached vector stores"""
        shutil.rmtree(str(self.cache_dir), ignore_errors=True)
        print(f"Cleared cache directory: {self.cache_dir}")

    def get_cache_size(self) -> float:
        """Get cache directory size in MB"""
        return sum(f.stat().st_size for f in self.cache_dir.glob('**/*') if f.is_file()) / (1024 * 1024)

    def load_from_cache(self, cache_key: str) -> Tuple[FAISS, dict]:
        """Load cached embeddings and metadata"""
        try:
            # Load FAISS index WITHOUT embedding function
            vector_store = FAISS.load_local(
                str(self.cache_dir),
                index_name=cache_key,
                embeddings=None,  # ✅ This is correct for manually created indexes
                allow_dangerous_deserialization=True
            )

            # Load image store
            with open(self.cache_dir / f"{cache_key}_images.pkl", "rb") as f:
                image_data_store = pickle.load(f)

            # Load metadata
            with open(self.cache_dir / f"{cache_key}_meta.json", "r") as f:
                metadata = json.load(f)

            print(f"Loaded cached embeddings from {metadata['created_at']}")
            return vector_store, image_data_store

        except Exception as e:
            print(f"Cache loading failed: {e}")
            raise CacheLoadError("Failed to load cached embeddings")

    def save_to_cache(self, cache_key: str, vector_store: FAISS, image_data_store: dict):
        """Save embeddings to cache with metadata"""
        try:
            # Save FAISS index
            vector_store.save_local(
                str(self.cache_dir),
                index_name=cache_key
            )

            # Save image store
            with open(self.cache_dir / f"{cache_key}_images.pkl", "wb") as f:
                pickle.dump(image_data_store, f)

            # Save metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "model_version": self.model_version,
                "source_hash": cache_key.split("_")[0]
            }
            with open(self.cache_dir / f"{cache_key}_meta.json", "w") as f:
                json.dump(metadata, f)

        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")


class CacheLoadError(Exception):
    pass
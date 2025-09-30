import torch
from PIL import Image
import numpy as np
import threading


# Use global variables for the model and processor, initialized to None
clip_model = None
clip_processor = None
_model_lock = threading.Lock()

def initialize_clip_model():
    """Initializes the CLIP model and processor if they haven't been already."""
    global clip_model, clip_processor
    # Use a lock to ensure thread-safe initialization
    with _model_lock:
        if clip_model is None or clip_processor is None:
            print("[INFO] Initializing CLIP model for the first time (this may take a moment)...")
            from transformers import CLIPProcessor, CLIPModel
            try:
                clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print("[INFO] CLIP model initialized successfully.")
            except Exception as e:
                print(f"[CRITICAL] Failed to initialize CLIP model: {e}")
                raise

def embed_image(image_data):
    """Embed image using CLIP"""
    initialize_clip_model()  # Ensure model is loaded
    print(f"[DEBUG] Embedding image of type: {type(image_data)}")
    try:
        if isinstance(image_data, str):  # If path
            image = Image.open(image_data).convert("RGB")
        else:  # If PIL Image
            image = image_data

        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            result = features.squeeze().numpy()
            print(f"[DEBUG] Image embedding shape: {result.shape}")
            return result
    except Exception as e:
        print(f"[ERROR] Image embedding failed: {e}")
        raise

def embed_text(text):
    """Embed text using CLIP."""
    initialize_clip_model()  # Ensure model is loaded
    print(f"[DEBUG] Embedding text: {text[:50]}...")
    try:
        inputs = clip_processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        with torch.no_grad():
            features = clip_model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            result = features.squeeze().numpy()
            print(f"[DEBUG] Text embedding shape: {result.shape}")
            return result
    except Exception as e:
        print(f"[ERROR] Text embedding failed: {e}")
        raise
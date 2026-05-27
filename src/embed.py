# embed.py
# ==========================================================
# Phase2: Embeddings n Vector Database Construction
# ==========================================================
# - Loads preprocessed 224x224 images.
# - Generates 512-dimensional embeddings using CLIP.
# - Stores vectors, metadata, and file paths in ChromaDB.
# ==========================================================

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
import uuid
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, CHROMA_DB_PATH, COLLECTION_NAME, VALID_EXTNS

# -----------------------------
# Init models
# -----------------------------
print("CLIP Vision Transformer...")
model_id = "openai/clip-vit-base-patch32"
# Automatically use GPU if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# -----------------------------
# Init Vector Db
# -----------------------------
print(f"Initializing local ChromaDB at {CHROMA_DB_PATH}...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# We use 'cosine' similarity because we care about the angle between vectors 
# (visual semantic similarity) rather than raw magnitude.
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"} 
)

# -----------------------------
# Embedding Pipeline
# -----------------------------
def embed_and_store_dataset():
    success_count = 0
    
    for root, _, files in os.walk(PROCESSED_DATA_DIR):
        rel_path = os.path.relpath(root, PROCESSED_DATA_DIR)
        parts = rel_path.split(os.sep)

        # Ensure we are inside a Class/Region directory because here is actual meta data separated
        if len(parts) != 2:
            continue
            
        class_name, region_name = parts[0], parts[1]

        for fname in files:
            if not fname.lower().endswith(VALID_EXTNS):
                continue

            processed_img_path = os.path.join(root, fname)
            
            # reconstructs the path to the original raw image for the LLM payload later
            raw_img_path = os.path.join(RAW_DATA_DIR, class_name, region_name, fname)

            try:
                # Load image using PIL (native format for Hugging Face Transformers)
                image = Image.open(processed_img_path).convert("RGB")
                
                # Preprocess and pass through CLIP
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    # Extract the visual features
                    outputs = model.get_image_features(**inputs) if hasattr(model, 'get_image_features') else model(**inputs)
                    
                    # Foolproof extraction to isolate the raw math tensor
                    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                        image_features = outputs.image_embeds
                    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        image_features = outputs.pooler_output
                    elif isinstance(outputs, tuple):
                        # Some older HF versions return tuples
                        image_features = outputs[0] 
                    else:
                        # Fallback if it is already a raw tensor
                        image_features = outputs
                    
                # Normalize the vector (standard practice for cosine similarity)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # Convert the tensor to a flat Python list for ChromaDB
                embedding_list = image_features.cpu().numpy().tolist()[0]
                
                # Generate a unique ID for the database entry
                doc_id = str(uuid.uuid4())

                # Insert into ChromaDB
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding_list],
                    metadatas=[{
                        "class_label": class_name,
                        "region_label": region_name,
                        "raw_image_path": raw_img_path,
                        "processed_image_path": processed_img_path
                    }],
                    # ChromaDB requires a 'document' string, we can use the class as a fallback
                    documents=[f"{class_name} on {region_name}"] 
                )
                
                success_count += 1
                print(f"Stored: {class_name} | {region_name} | {fname}")

            except Exception as e:
                print(f"[ERROR] Failed to embed {processed_img_path}: {e}")

    print(f"\n[SUCCESS] Vector Database Built!")
    print(f"[INFO] Total embeddings stored: {success_count}")
    print(f"[INFO] Database saved locally to: {CHROMA_DB_PATH}")

if __name__ == "__main__":
    embed_and_store_dataset()
# ==========================================================
# Phase 3: Vector Retrieval Engine
# ==========================================================
# - Preprocesses unseen user images identically to Phase 1.
# - Generates a normalized CLIP embedding.
# - Queries the offline ChromaDB for top-K visually similar cases.
# ==========================================================

import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Import your existing configuration and preprocessing logic
from config import CHROMA_DB_PATH, COLLECTION_NAME
from preprocess import preprocess_single_image

# -----------------------------
# 1. Global Initialization
# -----------------------------
# We load the model and database outside the function so they 
# only initialize once when the server boots up, preventing lag.
print("[INFO] Booting Retrieval Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

try:
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # Notice we use get_collection here, not get_or_create, to strictly enforce reading
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print("[SUCCESS] CLIP and ChromaDB loaded and ready.")
except Exception as e:
    print(f"[CRITICAL ERROR] Engine failed to boot: {e}")

# -----------------------------
# 2. The Search Function
# -----------------------------
def retrieve_similar_cases(image_path: str, region_name: str = "cheek", top_k: int = 3):
    """
    Takes an unseen image, preprocesses it to enforce geometric integrity, 
    and fetches the top K similar cases from the vector database.
    """
    try:
        # Step 1: Preprocess identically to the training data
        # This returns a 224x224 RGB NumPy array
        processed_numpy_img = preprocess_single_image(image_path, region_name)

        # Step 2: Convert OpenCV NumPy array back to PIL Image for Hugging Face
        pil_image = Image.fromarray(processed_numpy_img)

        # Step 3: Pass through CLIP to extract the vector
        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs) if hasattr(model, 'get_image_features') else model(**inputs)

            # Foolproof extraction handling HF version differences
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                image_features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                image_features = outputs.pooler_output
            elif isinstance(outputs, tuple):
                image_features = outputs[0]
            else:
                image_features = outputs

        # Normalize the vector to match the database math (Cosine Similarity)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        query_embedding = image_features.cpu().numpy().tolist()[0]

        # Step 4: Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results

    except Exception as e:
        print(f"[ERROR] Retrieval failed for {image_path}: {e}")
        return None

# -----------------------------
# 3. Local Testing Block
# -----------------------------
if __name__ == "__main__":
    # To test this, temporarily drop a new, unseen image into your raw directory
    # and update this path to point to it.
    test_image = r"D:\SPARSH\Skyn-main (2)\data\raw_data\Acne vulgaris face Indian face skin\cheeks\test_image.jpg"
    
    print("\n[TEST] Running Vector Retrieval...")
    search_results = retrieve_similar_cases(test_image, region_name="cheeks", top_k=3)
    
    if search_results:
        print("\n🔍 TOP 3 MATCHES FOUND:")
        # ChromaDB returns a dictionary of lists. We unpack the first query's results [0].
        for i in range(len(search_results['ids'][0])):
            distance = search_results['distances'][0][i]
            metadata = search_results['metadatas'][0][i]
            
            print(f"\nMatch #{i+1} | Distance: {distance:.4f}")
            print(f"  -> Class: {metadata['class_label']}")
            print(f"  -> Original File: {metadata['raw_image_path']}")
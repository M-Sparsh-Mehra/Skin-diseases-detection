# test_pipeline.py
import os
from retriever import retrieve_similar_cases
from gemini_bridge import analyze_skin_condition

def run_test():
    # 1. Update this path to point exactly to your new separate test image
    test_image_path = "image.png"
    
    # 2. Set the region matching your image ("full" or specific region like "cheeks")
    region = "full" 

    if not os.path.exists(test_image_path):
        print(f"[ERROR] Test image not found at: {os.path.abspath(test_image_path)}")
        return

    print(f"Starting Inference for: {test_image_path}")
    print("--------------------------------------------------")
    
    # Step A: Preprocess image, create embedding, and query ChromaDB
    print("Step [1/2]: Querying ChromaDB Vector Index...")
    matched_cases = retrieve_similar_cases(test_image_path, region_name=region, top_k=3)
    
    # Step B: Pass the user image and the matched case metadata to Gemini
    print("Step [2/2]: Passing Image and Contextual RAG Matches to Gemini...")
    ai_analysis = analyze_skin_condition(test_image_path, matched_cases)
    
    print("\n==================== AI DIAGNOSTIC RESPONSE ====================")
    print(ai_analysis)
    print("================================================================")

if __name__ == "__main__":
    run_test()
# src/gemini_bridge.py
# ==========================================================
# Phase 3: Gemini Prompt Injection & Inference (Updated SDK)
# ==========================================================

import os
from google import genai
from PIL import Image

# -----------------------------
# 1. API Configuration
# -----------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("[WARNING] GEMINI_API_KEY environment variable not found. Inference will fail.")
    client = None
else:
    # Initialize the modern Google GenAI Client
    client = genai.Client(api_key=API_KEY)

# -----------------------------
# 2. Prompt Engineering & Injection
# -----------------------------
def analyze_skin_condition(user_image_path: str, retrieved_cases: dict):
    """
    Takes the user's uploaded image and the raw dictionary from ChromaDB,
    constructs a strict medical prompt, and queries Gemini.
    """
    if client is None:
        return "Inference failed: GEMINI_API_KEY is missing from your environment variables."

    try:
        # Load the user's raw image to send to the multimodal LLM
        user_image = Image.open(user_image_path).convert("RGB")
        
        # Parse the ChromaDB results into a readable context block
        rag_context = ""
        if retrieved_cases and 'metadatas' in retrieved_cases and len(retrieved_cases['metadatas'][0]) > 0:
            metadatas = retrieved_cases['metadatas'][0]
            distances = retrieved_cases['distances'][0]
            
            for i, meta in enumerate(metadatas):
                similarity = round((1.0 - distances[i]) * 100, 1)
                rag_context += f"Case {i+1}: Ground Truth Label = '{meta['class_label']}' (Visual Similarity: {similarity}%)\n"
        else:
            rag_context = "No highly similar cases found in the database.\n"

        # The System Prompt (Few-Shot Injection)
        prompt = f"""
You are an expert AI dermatology assistant. A user has provided a photo of a skin condition. 
To assist your analysis, our Visual RAG system has scanned a verified medical database and found the most visually similar historical cases.

--- RAG DATABASE RESULTS ---
{rag_context}
----------------------------

INSTRUCTIONS:
1. Carefully analyze the attached user image.
2. Consider the 'RAG Database Results' provided above. These are ground-truth verified historical cases that look mathematically similar to the user's image.
3. Provide a synthesized analysis. State if the user's image strongly aligns with the retrieved cases, or if it appears distinct.
4. Provide a disclaimer that this is an AI screening tool, not a formal medical diagnosis.

Output your response in a clear, highly structured format using markdown headings and bullet points.
"""

        print("[INFO] Sending multimodal payload to Gemini...")
        
        # Call the modern gemini-2.5-flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_image, prompt]
        )
        
        return response.text

    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e}")
        return "An error occurred while generating the analysis. Please check your API keys and try again."
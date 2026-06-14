# preprocess.py
# ==========================================================
# Vision preprocessing pipeline for RAG-ready embeddings
# ==========================================================
# - Relies on manual directory structure 
# - Applies Center Cropping to prevent aspect ratio distortion
# - Standardizes sizing (224x224) for embedding models
# ==========================================================

import os
import cv2
import numpy as np
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FACE_SIZE, REGION_SIZE, VALID_EXTNS



def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    # Convert BGR (OpenCV default) to RGB for accurate processing
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def center_crop(image: np.ndarray) -> np.ndarray:
    
    #crops the center square of an image to prevent distortion.
    
    h, w = image.shape[:2]
    # finds the shortest side to make a perfect square
    min_dim = min(h, w)
    
    # calculates the exact center coordinates
    start_x = (w // 2) - (min_dim // 2)
    start_y = (h // 2) - (min_dim // 2)
    
    # Slice the array to create the square mask
    return image[start_y:start_y+min_dim, start_x:start_x+min_dim]

def preprocess_single_image(image_path: str, region_name: str) -> np.ndarray:
    
    #Loads, crops, and resizes the image.
    
    image = load_image(image_path)
    
    #apply Center Crop to preserve geometric accuracy
    cropped_image = center_crop(image)
    
    # resize the perfect square to the target embedding size
    target_size = FACE_SIZE if region_name.lower() == "full" else REGION_SIZE
    resized = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_LINEAR)
    
    return resized

# -----------------------------
#data structuring (for metadata generation)
# -----------------------------
def run_offline_structuring(raw_dir: str = RAW_DATA_DIR, output_dir: str = PROCESSED_DATA_DIR):
    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    fail_count = 0

    for root, _, files in os.walk(raw_dir):
        rel_path = os.path.relpath(root, raw_dir)
        parts = rel_path.split(os.sep)

        if len(parts) != 2:
            continue
            
        class_name, region_name = parts[0], parts[1]

        for fname in files:
            if not fname.lower().endswith(VALID_EXTNS):
                continue

            input_path = os.path.join(root, fname)
            save_dir = os.path.join(output_dir, class_name, region_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, fname)

            try:
                processed_img = preprocess_single_image(input_path, region_name)
                
                # Convert back to BGR strictly so OpenCV saves it as a normal JPEG
                save_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, save_img_bgr)
                success_count += 1
                
            except Exception as e:
                print(f"[WARN] Failed processing {input_path}: {e}")
                fail_count += 1

    print(f"\n[INFO] Preprocessing Complete.")
    print(f"[INFO] Successfully processed: {success_count} images.")
    if fail_count > 0:
        print(f"[WARN] Failed to process: {fail_count} images.")

if __name__ == "__main__":
    print("[INFO] Starting offline structuring pipeline...")
    run_offline_structuring()
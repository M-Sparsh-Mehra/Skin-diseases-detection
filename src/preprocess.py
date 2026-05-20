import os
import cv2
import glob
import numpy as np
import config

def gray_world_color_constancy(image):
    """Applies the Gray World algorithm to balance clinical lighting variants."""
    img_float = image.astype(np.float32)
    b_mean = np.mean(img_float[:, :, 0])
    g_mean = np.mean(img_float[:, :, 1])
    r_mean = np.mean(img_float[:, :, 2])
    
    total_mean = (b_mean + g_mean + r_mean) / 3.0
    
    img_float[:, :, 0] *= (total_mean / (b_mean + 1e-8))
    img_float[:, :, 1] *= (total_mean / (g_mean + 1e-8))
    img_float[:, :, 2] *= (total_mean / (r_mean + 1e-8))
    
    return np.clip(img_float, 0, 255).astype(np.uint8)

def segment_lesion_roi(image, color_space='ycrcb'):
    """Generates an automated Otsu mask to extract the lesion area."""
    if color_space.lower() == 'ycrcb':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        target_channel = converted[:, :, 1]
    elif color_space.lower() == 'hsv':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        target_channel = converted[:, :, 1]
    else:
        raise ValueError("Invalid color space config choice.")

    blurred = cv2.GaussianBlur(target_channel, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return image[y:y+h, x:x+w], mask[y:y+h, x:x+w]
        
    return image, mask

def letterbox_resize(image, target_size=(224, 224)):
    """Resizes keeping aspect ratio uniform, padding deficits with black."""
    ih, iw = image.shape[:2]
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (th - nh) // 2
    bottom = th - nh - top
    left = (tw - nw) // 2
    right = tw - nw - left
    
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

def standardize_image(image):
    """Normalizes intensities to [0,1] and executes channel Z-Score standardization."""
    img_float = image.astype(np.float32) / 255.0
    mean, std = cv2.meanStdDev(img_float)
    return (img_float - mean.flatten()) / (std.flatten() + 1e-8)

def preprocess_pipeline(image_path, is_texture_or_diffuse=False):
    """
    Master execution pipeline.
    
    Args:
        image_path (str): Path to image file.
        is_texture_or_diffuse (bool): If True, bypasses localized ROI cropping 
                                      to preserve broad surface patterns (e.g., wrinkles).
    """
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Step 1: Always fix lighting variations across all classes
    color_corrected = gray_world_color_constancy(raw_img)
    
    # Step 2: Conditional Segmentation
    if is_texture_or_diffuse:
        # Bypass cropping; use the entire illumination-corrected image context
        processed_region = color_corrected
    else:
        # Isolate the specific object boundary (Best for distinct moles/isolated lesions)
        processed_region, _ = segment_lesion_roi(color_corrected, color_space=config.COLOR_SPACE)
    
    # Step 3 & 4: Resize and standardize uniformly
    resized_img = letterbox_resize(processed_region, target_size=config.TARGET_SIZE)
    standardized_tensor = standardize_image(resized_img)
    
    return standardized_tensor, resized_img


def run_offline_structuring():
    """Scans raw folders, detects condition type from folder names, and processes accordingly."""
    print("Initializing Smart Dataset Preprocessing Run...")
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.WEBP')
    
    if not os.path.exists(config.RAW_DATA_DIR):
        raise FileNotFoundError(f"Missing target configuration directory: {config.RAW_DATA_DIR}")
        
    for disease_folder in os.listdir(config.RAW_DATA_DIR):
        input_folder_path = os.path.join(config.RAW_DATA_DIR, disease_folder)
        
        if os.path.isdir(input_folder_path):
            output_folder_path = os.path.join(config.PROCESSED_DATA_DIR, disease_folder)
            os.makedirs(output_folder_path, exist_ok=True)
            
            # --- SMART SWITCH LOGIC ---
            # Automatically detect if the class relies on global skin texture or broad coverage
            folder_lower = disease_folder.lower()
            is_texture_or_diffuse = any(word in folder_lower for word in ['wrinkles', 'uneven', 'acne', 'atopic'])
            
            img_paths = []
            for ext in extensions:
                img_paths.extend(glob.glob(os.path.join(input_folder_path, ext)))
                
            print(f"Category: '{disease_folder}' | Global Mode: {is_texture_or_diffuse} | Files: {len(img_paths)}")
            
            for path in img_paths:
                filename = os.path.basename(path)
                try:
                    # Pass the automatic switch to the pipeline
                    _, visual_img = preprocess_pipeline(path, is_texture_or_diffuse=is_texture_or_diffuse)
                    cv2.imwrite(os.path.join(output_folder_path, filename), visual_img)
                except Exception as e:
                    print(f"Failed execution on file {filename}: {e}")
                    
    print("Preprocessing Complete. Structured directory mirrors successfully updated.")
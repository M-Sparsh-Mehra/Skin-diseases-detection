import os

# Base Directory Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Structured Dataset Directory Paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
VALID_EXTNS = (".jpg", ".jpeg", ".png")

# Vector Database Config
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_data") # Changed to absolute path for safety
COLLECTION_NAME = "dermato_rag"

# Core Preprocessing Hyperparameters
# Enforced at 224x224 to prevent Hugging Face from artificially scaling the image
FACE_SIZE = (224, 224)
REGION_SIZE = (224, 224) 

# Inference Configurations
BATCH_SIZE = 32  # Keep if using a dataloader for embedding generation
NUM_WORKERS = 2
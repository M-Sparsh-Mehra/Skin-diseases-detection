import os

# Base Directory Setup (Points to the project root folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Structured Dataset Directory Paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed_data")

# Core Preprocessing Hyperparameters
TARGET_SIZE = (224, 224)
COLOR_SPACE = "ycrcb"  # Options: 'ycrcb' (Recommended for Indian skin variants) or 'hsv'

# PyTorch Deep Learning Dataloader Configurations
BATCH_SIZE = 32
SHUFFLE_TRAIN = True
NUM_WORKERS = 2  # Adjust based on your CPU threads
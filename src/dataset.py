import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import config

class SkinDiseaseDataset(Dataset):
    """Custom PyTorch Dataset parsing preprocessed categorical skin anomaly subsets."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Map alphabetically sorted folder paths to index integer keys
        self.classes = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_samples = []
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.WEBP')
        
        # Crawl directories to index absolute image paths and numerical tags
        for cls_name in self.classes:
            cls_folder = os.path.join(data_dir, cls_name)
            for root, _, files in os.walk(cls_folder):
                for file in files:
                    if file.lower().endswith(extensions):
                        self.image_samples.append((os.path.join(root, file), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.image_samples)

    def __getitem__(self, idx):
        img_path, label = self.image_samples[idx]
        
        # Read the visual-preprocessed image from disk
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to standard RGB for PyTorch
        
        # Scale to [0, 1] and normalize shape channel format to (Channels, Height, Width)
        image_float = image.astype(np.float32) / 255.0
        tensor_image = torch.tensor(image_float).permute(2, 0, 1)
        
        return tensor_image, torch.tensor(label, dtype=torch.long)
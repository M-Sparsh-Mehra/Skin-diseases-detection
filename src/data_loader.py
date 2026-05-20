from torch.utils.data import DataLoader
from dataset import SkinDiseaseDataset
import config

def get_dataloader():
    """Builds and returns the structured training pipeline DataLoader instance."""
    print("Initializing Target Datastream Loaders...")
    
    dataset = SkinDiseaseDataset(data_dir=config.PROCESSED_DATA_DIR)
    
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_TRAIN,
        num_workers=config.NUM_WORKERS,
        drop_last=False
    )
    
    print(f"Total Dataset Samples Found: {len(dataset)}")
    print(f"Identified Classes Map: {dataset.class_to_idx}")
    return loader
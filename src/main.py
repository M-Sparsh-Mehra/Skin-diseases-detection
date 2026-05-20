import sys
import os

# Appending path variables dynamically to handle relative file calls cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import preprocess
import data_loader

def main():
    print("======================================================")
    print("🚀 SKIN DISEASE ML PIPELINE - MASTER ORCHESTRATION ENGINE")
    print("======================================================\n")
    
    # Step 1: Run the full dataset conversion loop matching subfolder matrices
    preprocess.run_offline_structuring()
    
    print("\n------------------------------------------------------")
    
    # Step 2: Initialize dataloader checks
    try:
        loader = data_loader.get_dataloader()
        
        # Pull a single sample batch to verify channels match target frameworks
        images, labels = next(iter(loader))
        print("\n⚡ [INTEGRITY CHECK PASSED]")
        print(f"   -> Mini-Batch Image Tensor Dimensions: {images.shape} (Format: [Batch, Channels, Height, Width])")
        print(f"   -> Mini-Batch Corresponding Labels Array: {labels.numpy()}")
        print("\n🎉 Setup validation successful! Ready for model training execution cycles.")
        
    except Exception as e:
        print(f"\n❌ [CRITICAL INTEGRITY FAILURE]: {e}")
        print("Ensure raw dataset classes contain valid non-corrupted images.")

if __name__ == "__main__":
    main()
import cv2
import matplotlib.pyplot as plt
import os

def visualize_preprocessing(raw_path, processed_path):
    # Load images in RGB for Matplotlib
    img_raw = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
    img_proc = cv2.cvtColor(cv2.imread(processed_path), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img_raw)
    axes[0].set_title(f"Raw: {img_raw.shape}")
    axes[0].axis('off')

    axes[1].imshow(img_proc)
    axes[1].set_title(f"Processed: {img_proc.shape}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Replace with an actual image path from your folders
raw_file = r"D:\SPARSH\Skyn-main (2)\data\raw\Acne vulgaris face Indian face skin\cheeks\5.Cost-Of-Acne-Treatment.png" 
proc_file = r"D:\SPARSH\Skyn-main (2)\data\processed\Acne vulgaris face Indian face skin\cheeks\5.Cost-Of-Acne-Treatment.png"

if os.path.exists(raw_file) and os.path.exists(proc_file):
    visualize_preprocessing(raw_file, proc_file)
else:
    print("Files not found. Check the paths!")
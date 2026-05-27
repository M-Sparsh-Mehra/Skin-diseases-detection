## ⚙️ Phase 1: Preprocessing Architecture
General computer vision preprocessing (like standard resizing) distorts the geometry of skin lesions, severely degrading the embedding model's ability to map them correctly. The preprocessing pipeline is engineered to preserve absolute geometric integrity.

### Key Preprocessing Steps:
1. **Manual Region Curation:** Sidestepped automated face-mesh libraries (e.g., MediaPipe) in favor of a strict manual directory structure (`raw/Class/Region/image.jpg`) to ensure 100% labeling accuracy for the foundational knowledge base.
2. **Color Space Standardization:** OpenCV native BGR loaded formats are strictly converted to RGB for accurate processing, and safely reverted to BGR for standard JPEG disk storage.
3. **Center Cropping (Anti-Distortion):** Replaced standard hard-resizing with dynamic center-cropping. This acts as a perfect square mask, discarding background noise while perfectly preserving the natural aspect ratio and geometry of the skin lesion.
4. **Standardized Resizing:** Final cropped squares are resized via Bilinear Interpolation to `224x224`, explicitly satisfying the input layer constraints of the downstream CLIP embedding architecture.

---

## 🚀 Deployment Strategy
Designed for a $0 deployment cost architecture suitable for portfolio showcasing:
* **Database:** ChromaDB (Pre-baked offline and stored within the repository).
* **Compute / Hosting:** Hugging Face Spaces (Handling the CLIP model and FastAPI/Streamlit UI).
* **Inference:** Gemini 2.5 Free Tier API via few-shot dynamic prompt injection.
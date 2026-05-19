# 📊 Dataset Strategy & RAG Taxonomy

## The Architecture: Why Visual RAG?

we will utilize a **Visual Retrieval-Augmented Generation (RAG)** architecture rather than traditional model fine-tuning. For specialized dermatological presentations on **melanin rich Indian skin** general vision LLMs often lack nuanced clinical accuracy and can hallucinate diagnoses. Thats why we will try to scrape images relavent to Indian skin type. 


---

## Classes & Subclasses

The dataset is explicitly curated for **Fitzpatrick Skin Types IV and V (South Asian / Indian skin tones)**. Dermatological presentations vary drastically based on baseline melanin and inflammation often presents as *violaceous (purplish)* or *hyperpigmented* rather than *erythematous (red)*, making generic datasets ineffective.

The vector database maps the following **5 core classes**, broken down into visually distinct subclasses:

### 1. Active Acne

**What it covers**  
Everything from whiteheads and blackheads to inflamed red pimples and cysts.

**Scraping terms**
- Acne vulgaris face Indian skin  
- Active acne breakouts brown skin  
- Pimple inflammation South Asian skin  

---

### 2. Acne Scars & Marks (Sequelae)

**What it covers**  
The residual dark spots (*post-inflammatory hyperpigmentation*) and indented scars left behind after acne resolution.

**Scraping terms**
- Post-inflammatory hyperpigmentation face Indian skin  
- Dark acne marks brown skin  
- Atrophic acne scars cheeks  

---

### 3. Pigmentation & Discoloration

**What it covers**  
Melasma, sun-induced hyperpigmentation, and dark patches such as *Acanthosis nigricans*.  
Hypopigmentation disorders (e.g., vitiligo) are intentionally excluded to avoid scraper confusion between light vs. dark lesions.

**Scraping terms**
- Melasma hyperpigmentation face Indian skin  
- Dark patches neck brown skin Acanthosis  
- Uneven skin tone pigmentation South Asian  

---

  
### 4. Fungal Infections (Ringworm / Tinea)

**What it covers**  
Classic scaly, annular, or discolored plaques typical of superficial fungal infections.

**Scraping terms**
- Tinea corporis ringworm Indian skin  
- Fungal infection skin patches brown skin  

---

### 5. Eczema & Psoriasis (Dry / Scaly Inflammation)

**What it covers**  
Chronic inflammatory dermatoses characterized by dry, itchy, scaly, or thickened skin, including violaceous presentations on darker skin tones.

**Scraping terms**
- Eczema dry scaly patches Indian skin  
- Atopic dermatitis flexural brown skin  
- Psoriasis plaque violaceous skin  


---

## Dataset Sizing: **“The 50/50 Rule”**

To optimize the high-dimensional latent space for retrieval accuracy, this database prioritizes **data quality and strict class balance** over raw volume.

### Target Instances
- **30–50 high-quality, clinically verified images per subclass**

**Why this works:**
- In a **512-dimensional vector space**, 30–50 heavily curated images accurately map the visual boundaries of a disease, accounting for:
  - Lighting variation  
  - Severity  
  - Skin tone differences
- Fewer than 30 instances creates a cluster too small for reliable nearest-neighbor retrieval.

---

## Class Balancing: Preventing Retrieval Bias

Classes are **strictly capped** to ensure equal representation in the vector database.

**Example problem:**
- 300 instances of Acne  
- 50 instances of Melanoma  

This imbalance would mathematically skew retrieval probability toward Acne, causing false negatives for critical conditions.

**Solution:**
- Enforce strict volume parity across all classes

This ensures retrieval is driven **purely by visual geometry**, not statistical dominance.

---

**Result:**  
A compact, mathematically grounded, clinically precise Visual RAG system optimized for melanin-rich dermatological analysis.